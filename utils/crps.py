"""
CRPS Decomposition following Hersbach (2000)

Hersbach, H. (2000). Decomposition of the Continuous Ranked Probability
Score for Ensemble Prediction Systems. Weather and Forecasting, 15(5), 559-570.

2-Component: CRPS = Reliability + CRPS_potential  (Eq. 35)
3-Component: CRPS = Reliability - Resolution + Uncertainty  (Eq. 39)
"""

import numpy as np
from typing import Dict


def crps_hersbach_decomposition(
    forecasts: np.ndarray,
    observations: np.ndarray,
    member_axis: int = -1
) -> Dict[str, float]:
    """
    Compute CRPS and its decomposition following Hersbach (2000).
    
    Parameters
    ----------
    forecasts : ndarray
        Ensemble forecasts with shape [..., n_members]
    observations : ndarray
        Observations with shape [...] matching forecasts[..., 0]
    member_axis : int
        Axis containing ensemble members (default: -1)
        
    Returns
    -------
    dict with keys:
        'crps' : Mean CRPS
        'reliability' : Reliability component (Eq. 36, lower is better)
        'crps_potential' : Potential CRPS (Eq. 37)
        'resolution' : Resolution component (Eq. 38, higher is better)
        'uncertainty' : Climatological uncertainty (Eq. 46)
        
    Notes
    -----
    The decomposition uses a bin-based formulation. For N ensemble members,
    there are N+1 bins:
    - Bin 0: (-inf, x_1) - outlier below ensemble
    - Bins 1 to N-1: (x_i, x_{i+1}) - internal bins between members
    - Bin N: (x_N, +inf) - outlier above ensemble
    
    Key equations:
    - CRPS = Reliability + CRPS_potential (Eq. 35)
    - CRPS = Reliability - Resolution + Uncertainty (Eq. 39)
    - CRPS_potential = Uncertainty - Resolution (Eq. 38)
    - Reliability = Σ g̅_i (o̅_i - p_i)² (Eq. 36)
    - CRPS_potential = Σ g̅_i o̅_i (1 - o̅_i) (Eq. 37)
    - o̅_i = β̅_i / g̅_i (Eq. 31)
    """
    # Move member axis to last position
    forecasts = np.moveaxis(forecasts, member_axis, -1)
    N = forecasts.shape[-1]
    
    # Validate shapes
    if forecasts.shape[:-1] != observations.shape:
        raise ValueError(
            f"Shape mismatch: forecasts[..., 0] shape {forecasts.shape[:-1]} "
            f"!= observations shape {observations.shape}"
        )
    
    # Flatten for processing
    f_sorted = np.sort(forecasts.reshape(-1, N), axis=-1)
    obs = observations.flatten()
    n_cases = len(obs)
    
    # Number of bins = N + 1
    # Bin 0: below minimum, Bins 1 to N-1: internal, Bin N: above maximum
    n_bins = N + 1
    
    # Accumulate alpha (obs above bin) and beta (obs below bin) for each case
    alpha_all = np.zeros((n_cases, n_bins))
    beta_all = np.zeros((n_cases, n_bins))
    
    for k in range(n_cases):
        x = f_sorted[k]  # sorted ensemble members
        o = obs[k]       # observation
        
        # Bin 0: outlier below x[0]
        if o < x[0]:
            beta_all[k, 0] = x[0] - o
        
        # Internal bins: bin i+1 corresponds to interval [x[i], x[i+1])
        for i in range(N - 1):
            x_lo, x_hi = x[i], x[i + 1]
            if o <= x_lo:
                # obs below this bin
                beta_all[k, i + 1] = x_hi - x_lo
            elif o >= x_hi:
                # obs above this bin
                alpha_all[k, i + 1] = x_hi - x_lo
            else:
                # obs inside this bin
                alpha_all[k, i + 1] = o - x_lo
                beta_all[k, i + 1] = x_hi - o
        
        # Bin N: outlier above x[N-1]
        if o > x[N - 1]:
            alpha_all[k, N] = o - x[N - 1]
    
    # Compute g = alpha + beta
    g_all = alpha_all + beta_all
    
    # Average over cases
    alpha_bar = np.mean(alpha_all, axis=0)
    beta_bar = np.mean(beta_all, axis=0)
    g_bar = np.mean(g_all, axis=0)
    
    # Observed frequency: o_bar = beta_bar / g_bar (Eq. 31)
    with np.errstate(divide='ignore', invalid='ignore'):
        o_bar = np.where(g_bar > 0, beta_bar / g_bar, 0.0)
    
    # Forecast probabilities for each bin
    p = np.concatenate([[0], np.arange(1, N) / N, [1]])
    
    # === Decomposition Components ===
    
    # Reliability (Eq. 36)
    reliability = float(np.sum(g_bar * (o_bar - p)**2))
    
    # CRPS_potential (Eq. 37)
    crps_potential = float(np.sum(g_bar * o_bar * (1 - o_bar)))
    
    # Total CRPS (Eq. 35)
    crps = reliability + crps_potential
    
    # Uncertainty (Eq. 46) - from observation climatology
    uncertainty = _compute_uncertainty(obs)
    
    # Resolution (Eq. 38)
    resolution = uncertainty - crps_potential
    
    return {
        'crps': crps,
        'reliability': reliability,
        'crps_potential': crps_potential,
        'resolution': resolution,
        'uncertainty': uncertainty
    }


def _compute_uncertainty(observations: np.ndarray) -> float:
    """
    Compute climatological uncertainty (Eq. 46).
    
    U = ∫ F_clim(x) [1 - F_clim(x)] dx
    
    where F_clim is the empirical CDF of observations.
    
    Parameters
    ----------
    observations : ndarray
        Flattened array of observations
        
    Returns
    -------
    float
        Climatological uncertainty
    """
    obs_sorted = np.sort(observations)
    M = len(obs_sorted)
    
    if M < 2:
        return 0.0
    
    # For sorted observations: F(y_j) = j/M
    # U = Σ (j/M)(1 - j/M)(y_{j+1} - y_j)
    j = np.arange(1, M)
    F_j = j / M
    dy = np.diff(obs_sorted)
    
    return float(np.sum(F_j * (1 - F_j) * dy))
