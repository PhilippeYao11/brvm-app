import numpy as np
import pandas as pd

from config import ANNUAL_FACTOR


def compute_sharpe_sortino(port_ret: pd.Series):
    """
    Compute annualised mean, volatility, Sharpe and Sortino ratios
    for a portfolio daily return series.
    """
    r = port_ret.dropna()
    mu_d = r.mean()
    sigma_d = r.std()

    if sigma_d > 0:
        sharpe = (mu_d / sigma_d) * np.sqrt(ANNUAL_FACTOR)
    else:
        sharpe = np.nan

    downside = r[r < 0]
    downsigma_d = downside.std()

    if downsigma_d > 0:
        sortino = (mu_d / downsigma_d) * np.sqrt(ANNUAL_FACTOR)
    else:
        sortino = np.nan

    mu_a = mu_d * ANNUAL_FACTOR
    sigma_a = sigma_d * np.sqrt(ANNUAL_FACTOR)

    return {
        "mu_ann": mu_a,
        "sigma_ann": sigma_a,
        "sharpe": sharpe,
        "sortino": sortino,
    }
