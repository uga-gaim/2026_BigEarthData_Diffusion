import numpy as np
import matplotlib.pyplot as plt

def rel_calc(forecasts, observations, threshold, member_axis=-1, n_bins=10):
    """
    Calculate reliability diagram data for ensemble weather forecasts of arbitrary shape.
    
    Parameters:
        forecasts: array of shape [..., members, ...] - ensemble forecasts
        observations: array matching forecasts shape but without the member axis
        threshold: value above which event is considered to occur
        member_axis: axis along which ensemble members are stored (default: -1)
        n_bins: number of probability bins
    
    Returns:
        dict with keys: bin_edges, bin_centers, forecast_prob, observed_freq,
                        bin_counts, brier_score, brier_skill_score, climatology, threshold
    """
    # Move member axis to position 1 for consistent handling
    forecasts = np.moveaxis(forecasts, member_axis, 1)
    
    # Compute probability forecasts (fraction of members exceeding threshold)
    prob_forecast = np.mean(forecasts > threshold, axis=1)
    
    # Convert observations to binary
    obs_binary = (observations > threshold).astype(float)
    
    # Flatten everything for binning
    prob_flat = prob_forecast.ravel()
    obs_flat = obs_binary.ravel()
    
    # Define bins
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    # Calculate observed frequency and sample count for each bin
    observed_freq = np.zeros(n_bins)
    forecast_prob = np.zeros(n_bins)
    bin_counts = np.zeros(n_bins)
    
    for i in range(n_bins):
        if i == n_bins - 1:
            in_bin = (prob_flat >= bin_edges[i]) & (prob_flat <= bin_edges[i + 1])
        else:
            in_bin = (prob_flat >= bin_edges[i]) & (prob_flat < bin_edges[i + 1])
        
        bin_counts[i] = np.sum(in_bin)
        if bin_counts[i] > 0:
            observed_freq[i] = np.mean(obs_flat[in_bin])
            forecast_prob[i] = np.mean(prob_flat[in_bin])
    
    # Compute skill scores
    clim = np.mean(obs_flat)
    brier_score = np.mean((prob_flat - obs_flat) ** 2)
    brier_skill_score = 1 - brier_score / (clim * (1 - clim)) if clim > 0 and clim < 1 else np.nan
    
    results = {
        'bin_edges': bin_edges,
        'bin_centers': bin_centers,
        'forecast_prob': forecast_prob,
        'observed_freq': observed_freq,
        'bin_counts': bin_counts,
        'brier_score': brier_score,
        'brier_skill_score': brier_skill_score,
        'climatology': clim,
        'threshold': threshold
    }
    
    return results


def rel_plot(results, ax=None, show_inset=True, title=None):
    """
    Plot a reliability diagram from pre-calculated data.
    
    Parameters:
        results: dict returned by rel_calc
        ax: matplotlib axes to plot on (creates new figure if None)
        show_inset: whether to show the sharpness histogram inset
        title: custom title (uses default if None)
    
    Returns:
        fig, ax: matplotlib figure and axes objects
    """
    # Extract data from results
    bin_centers = results['bin_centers']
    forecast_prob = results['forecast_prob']
    observed_freq = results['observed_freq']
    bin_counts = results['bin_counts']
    brier_score = results['brier_score']
    brier_skill_score = results['brier_skill_score']
    clim = results['climatology']
    threshold = results['threshold']
    n_bins = len(bin_centers)
    
    # Mask empty bins
    mask = bin_counts > 0
    
    # Create figure if no axes provided
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))
    else:
        fig = ax.get_figure()
    
    # Reliability diagram
    ax.plot([0, 1], [0, 1], 'k--', label='Perfect reliability')
    ax.axhline(clim, color='gray', linestyle=':', alpha=0.7, label=f'Climatology ({clim:.2f})')
    ax.axvline(clim, color='gray', linestyle=':', alpha=0.7)
    ax.plot(forecast_prob[mask], observed_freq[mask], 'o-', markersize=8, label='Forecast')
    
    ax.set_xlabel('Forecast probability')
    ax.set_ylabel('Observed frequency')
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    
    if title is None:
        title = (f'Reliability Diagram (threshold={threshold})\n'
                 f'Brier Score: {brier_score:.4f}, BSS: {brier_skill_score:.4f}')
    ax.set_title(title)
    
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    
    # Inset histogram (sharpness) in upper left
    if show_inset:
        ax_inset = ax.inset_axes([0.05, 0.6, 0.35, 0.25])
        ax_inset.bar(bin_centers, bin_counts, width=1/n_bins, edgecolor='black', alpha=0.7)
        ax_inset.set_xlabel('Forecast prob.', fontsize=8)
        ax_inset.set_ylabel('Count', fontsize=8)
        ax_inset.set_xlim([0, 1])
        ax_inset.set_title(f'Sharpness (n={int(np.sum(bin_counts))})', fontsize=9)
        ax_inset.tick_params(axis='both', labelsize=7)
    
    plt.tight_layout()
    
    return fig, ax
