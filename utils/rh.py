import numpy as np

def rh(forecasts, observations):
    """
    Missing Rate Error (MRE) is the fraction of observations lower (higher) than
    the lowest (highest) ranked prediction above or below the expected missing rate, 2/(n + 1)
    
    forecasts: np.ndarray, shape [members, init_times, lead_times, height, width]
    observations: np.ndarray, shape [init_times, lead_times, height, width]
    """
    
    M = forecasts.shape[0]
    
    # Flatten over init_time, lead_time, height, width
    f_flat = forecasts.reshape(M, -1)           # [M, N]
    o_flat = observations.reshape(-1)           # [N]
    
    # Sort forecasts along member dimension
    sorted_f = np.sort(f_flat, axis=0)           # [M, N]
    
    # Rank: number of forecasts <= observation
    ranks = np.sum(sorted_f <= o_flat, axis=0)   # [N], values 0..M
    
    # Histogram (M+1 bins for ranks 0..M)
    hist, _ = np.histogram(ranks, bins=np.arange(M+2)-0.5)
    hist = hist / hist.sum()
    
    # Missing Rate Error
    # https://www.sciencedirect.com/science/article/pii/S2589004221011044
    #
    mre = hist[0] + hist[-1] - 2 / (15 + 1)
    return hist, mre