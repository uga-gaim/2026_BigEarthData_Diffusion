import numpy as np

from tqdm.auto import tqdm

# Hersbach (2000) formula
# https://hydrostats.readthedocs.io/en/latest/api/hydrostats.ens_metrics.ens_crps.html

def crps(forecasts, obs, disable_pbar=True):
    """
    Compute fair (unbiased) CRPS for ensemble forecasts in a memory-efficient way,
    showing progress with tqdm.

    Parameters
    ----------
    forecasts : np.ndarray
        Ensemble forecasts with shape (members, init_days, lead_time, height, width).
    obs : np.ndarray
        Observations with shape (init_days, lead_time, height, width).
    disable_pbar : bool
        Whether to disable (hide) the progress bar.

    Returns
    -------
    crps_score : np.ndarray
        Fair CRPS values with shape (init_days, lead_time, height, width).
    """
    if forecasts.shape[1:] != obs.shape:
        raise ValueError("Forecasts and observations have incompatible shapes.")
    
    M = forecasts.shape[0]
    shape = obs.shape  # (init_days, lead_time, height, width)
    
    # Term 1: mean absolute error between members and obs
    term1 = np.mean(np.abs(forecasts - obs[None, ...]), axis=0)
    
    # Term 2: unbiased spread term (sum over all distinct pairs of members)
    # Initialize accumulator
    term2 = np.zeros(shape, dtype=np.float64)
    pair_count = M * (M - 1)
    
    # Loop only over unique pairs (m < n), accumulate abs differences
    for m in tqdm(range(M), desc="Computing CRPS term2", disable=disable_pbar):
        f_m = forecasts[m]
        for n in range(m+1, M):
            f_n = forecasts[n]
            diff = np.abs(f_m - f_n)
            term2 += diff * 2
    
    term2 /= pair_count
    
    crps_score = term1 - term2
    return crps_score