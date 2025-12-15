# core/models.py
import numpy as np
import pandas as pd
from typing import Dict, Tuple

try:
    from statsmodels.tsa.arima.model import ARIMA
    HAS_STATSMODELS = True
except Exception:
    HAS_STATSMODELS = False


def _var_cvar_from_terminal_values(V0: float, V_T: np.ndarray, var_level: float) -> Tuple[float, float]:
    """
    Loss L = V0 - V_T.
    VaR = quantile_{var_level}(L)
    CVaR = E[L | L >= VaR]
    """
    if V_T.ndim != 1 or len(V_T) < 2:
        return float("nan"), float("nan")

    L = float(V0) - V_T
    q = float(np.quantile(L, var_level))
    tail = L[L >= q]
    cvar = float(tail.mean()) if len(tail) > 0 else float("nan")
    return q, cvar


def _fit_arima_on_log_returns(logret: pd.Series, order=(1, 0, 0)):
    """
    Fit ARIMA on log-returns.
    trend='n' avoids constant/drift that can bias forecasts.
    """
    model = ARIMA(
        logret,
        order=order,
        trend="n",
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    return model.fit()


def _simulate_arima_logret_paths(
    price_series: pd.Series,
    horizon_days: int,
    n_sims: int,
    order: tuple = (1, 0, 0),   # ✅ simple ARIMA on log-returns
    seed: int = 42,
) -> Tuple[np.ndarray, float]:
    """
    Returns:
      - log-return paths: shape (n_sims, horizon_days)
      - last observed price
    """
    s = price_series.dropna().astype(float)
    s = s[s > 0]

    if len(s) == 0:
        return np.zeros((n_sims, horizon_days), dtype=float), 0.0

    last_price = float(s.iloc[-1])

    logp = np.log(s)
    logret = logp.diff().dropna()

    rng = np.random.default_rng(seed)

    # If statsmodels missing or too little data -> Gaussian fallback
    if (not HAS_STATSMODELS) or (len(logret) < 60):
        mu = float(logret.mean()) if len(logret) > 0 else 0.0
        sig = float(logret.std(ddof=1)) if len(logret) > 1 else 0.01
        sig = max(sig, 1e-6)
        lr = rng.normal(mu, sig, size=(n_sims, horizon_days))
        return lr, last_price

    try:
        fit = _fit_arima_on_log_returns(logret, order=order)
    except Exception:
        mu = float(logret.mean()) if len(logret) > 0 else 0.0
        sig = float(logret.std(ddof=1)) if len(logret) > 1 else 0.01
        sig = max(sig, 1e-6)
        lr = rng.normal(mu, sig, size=(n_sims, horizon_days))
        return lr, last_price

    lr_paths = np.zeros((n_sims, horizon_days), dtype=float)
    for k in range(n_sims):
        try:
            sim_lr = fit.simulate(
                nsimulations=horizon_days,
                anchor="end",
                random_state=rng,
            )
            lr_paths[k, :] = sim_lr.values
        except Exception:
            # fallback Gaussian for this path
            mu = float(logret.mean())
            sig = float(logret.std(ddof=1)) if logret.std(ddof=1) > 0 else 1e-6
            lr_paths[k, :] = rng.normal(mu, sig, size=horizon_days)

    return lr_paths, last_price


def simulate_prices_and_portfolio(
    prices_wide: pd.DataFrame,
    returns: pd.DataFrame,          # kept for compatibility, not used
    weights: pd.Series,
    horizon_days: int,
    start_portfolio_value: float,
    n_sims: int = 300,
    var_level: float = 0.95,
    seed: int = 42,
    arima_order: tuple = (1, 0, 0),  # ✅ ARIMA on log-returns
) -> Dict:
    """
    Simulate prices + portfolio using ARIMA on log-returns, then rebuild prices.

    Outputs:
      - "portfolio_path": ONE simulated portfolio path (sim #0) for charts
      - "price_paths": ONE simulated price path per asset (sim #0) for charts
      - "mean_price_paths": mean price path per asset across all sims (for rebal plan)
      - "VaR", "CVaR": computed from terminal portfolio value distribution
    """
    tickers = list(weights.index)

    prices_sel = prices_wide[tickers].ffill().dropna(how="all", axis=0)
    if prices_sel.empty:
        raise ValueError("No valid price data available for the selected assets.")

    horizon_days = int(horizon_days)
    if horizon_days <= 0:
        raise ValueError("horizon_days must be positive.")

    n_sims = int(max(50, n_sims))
    var_level = float(var_level)

    w = weights.copy()
    if w.sum() <= 0:
        raise ValueError("Weights sum to zero.")
    w = w / w.sum()

    # --- simulate per asset log-return paths ---
    lr_paths: Dict[str, np.ndarray] = {}
    last_prices: Dict[str, float] = {}

    for t in tickers:
        lr, last_p = _simulate_arima_logret_paths(
            prices_sel[t],
            horizon_days=horizon_days,
            n_sims=n_sims,
            order=arima_order,
            seed=seed + 11,
        )
        lr_paths[t] = lr
        last_prices[t] = last_p

    # --- rebuild price paths: shape (n_sims, horizon_days+1) ---
    price_paths_arr: Dict[str, np.ndarray] = {}
    for t in tickers:
        P = np.zeros((n_sims, horizon_days + 1), dtype=float)
        P[:, 0] = last_prices[t]

        cum_lr = np.cumsum(lr_paths[t], axis=1)              # (n_sims, horizon_days)
        P[:, 1:] = last_prices[t] * np.exp(cum_lr)           # rebuild prices
        price_paths_arr[t] = P

    # ONE path for charting (sim #0)
    price_paths = {
        t: pd.Series(price_paths_arr[t][0, :], index=np.arange(horizon_days + 1))
        for t in tickers
    }

    # mean price paths for planning/rebalancing
    mean_price_paths = {
        t: pd.Series(price_paths_arr[t].mean(axis=0), index=np.arange(horizon_days + 1))
        for t in tickers
    }

    # --- portfolio simulation across all sims ---
    p0 = pd.Series({t: price_paths_arr[t][0, 0] for t in tickers}).replace(0.0, np.nan)
    shares = (w * float(start_portfolio_value) / p0).fillna(0.0)

    V = np.zeros((n_sims, horizon_days + 1), dtype=float)
    for k in range(n_sims):
        total = np.zeros(horizon_days + 1, dtype=float)
        for t in tickers:
            total += float(shares[t]) * price_paths_arr[t][k, :]
        V[k, :] = total

    portfolio_path = pd.Series(V[0, :], index=np.arange(horizon_days + 1), name="Portfolio")

    V_T = V[:, -1]
    VaR, CVaR = _var_cvar_from_terminal_values(float(start_portfolio_value), V_T, var_level)

    return {
        "model_used": f"ARIMA{arima_order} on log-returns (trend='n')",
        "portfolio_path": portfolio_path,
        "price_paths": price_paths,
        "mean_price_paths": mean_price_paths,
        "final_expected": float(V_T.mean()),
        "VaR": VaR,
        "CVaR": CVaR,
    }
