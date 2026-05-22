import numpy as np

from scipy import stats

def calc_kge(sim, obs):
    """
    Code provided by Qian Cao on 2023/04/10
    She suggested the following references:
    1. https://rdrr.io/cran/hydroGOF/man/KGE.html
    2. Gupta et al. 2009. Decomposition of the mean
    squared error and NSE performance criteria:
    Implications for improving hydrological modelling.
    """

    data_temp1_1 = np.ma.masked_invalid(obs)
    data_temp1 = data_temp1_1[~data_temp1_1.mask]
    data_temp2 = sim[~data_temp1_1.mask]
    r2 = (np.corrcoef(data_temp1, data_temp2)[0][1]) ** 2
    
    tempr1 = 0
    tempr2 = 0
    
    tempr1 = np.sum(np.square(data_temp2 - data_temp1))
    tempr2 = np.sum(np.square(data_temp1 - np.mean(data_temp1)))
    r1 = stats.pearsonr(data_temp1, data_temp2)[0]
    beta1 = np.mean(data_temp2)/np.mean(data_temp1)
    alpha1 = np.std(data_temp2)/np.std(data_temp1)
    KGE1 = 1-((1-r1)**2+(1-alpha1)**2+(1-beta1)**2)**0.5
    # KGE1_1 = "{:.2f}".format(KGE1)

    return {
        'kge': KGE1.item(),
        'r': r1.item(),
        'beta': beta1.item(),
        'alpha': alpha1.item()
    }


def calc_rmse(y_pred, y):
    return np.sqrt(np.nanmean((y_pred - y) ** 2))


def calc_bias(y_pred, y):
    return np.nanmean(y_pred - y)
