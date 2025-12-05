import numpy as np
import pandas as pd
from typing import Dict

# Try to use statsmodels ARIMA. If not available, we fall back to a manual AR(1).
try:
    from statsmodels.tsa.arima.model import ARIMA
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False


# =========================================================
#   Fallback AR(1) on log-returns (in case statsmodels is missing)
# =========================================================

def _estimate_ar1_params(r: pd.Series):
    """
    Estimate r_{t+1} = alpha + beta * r_t + eps on a return series.
    Returns (alpha, beta, sigma_eps, last_r) or None if not enough data.
    """
    r = r.dropna().values
    if len(r) < 40:
        return None

    y = r[1:]
    x = r[:-1]
    X = np.vstack([np.ones_like(x), x]).T
    theta, *_ = np.linalg.lstsq(X, y, rcond=None)
    alpha, beta = theta
    resid = y - (alpha + beta * x)
    sigma_eps = resid.std(ddof=1)
    last_r = r[-1]
    return alpha, beta, sigma_eps, last_r


def _simulate_ar1_price_path(price_series: pd.Series, horizon_days: int) -> pd.Series:
    """
    Simple AR(1) on log-returns, used only as a fallback when statsmodels
    (ARIMA) is not available.
    """
    rng = np.random.default_rng(42)

    s = price_series.dropna().astype(float)
    if len(s) < 5:
        last_price = float(s.iloc[-1])
        return pd.Series(
            [last_price] * (horizon_days + 1),
            index=np.arange(horizon_days + 1),
        )

    logp = np.log(s)
    r_series = logp.diff().dropna()

    params = _estimate_ar1_params(r_series)
    if params is None:
        mu = r_series.mean()
        sigma = r_series.std(ddof=1) if r_series.std(ddof=1) > 0 else 1e-6
        r_path = rng.normal(mu, sigma, size=horizon_days)
    else:
        alpha, beta, sigma_eps, last_r = params
        if not np.isfinite(sigma_eps) or sigma_eps <= 0:
            sigma_eps = 1e-6
        r_path = np.zeros(horizon_days)
        r_prev = last_r
        for t in range(horizon_days):
            eps = rng.normal(0.0, sigma_eps)
            r_curr = alpha + beta * r_prev + eps
            r_path[t] = r_curr
            r_prev = r_curr

    last_price = float(s.iloc[-1])
    log_p = np.log(last_price)
    prices_forecast = [last_price]
    for r in r_path:
        log_p += r
        prices_forecast.append(float(np.exp(log_p)))

    return pd.Series(prices_forecast, index=np.arange(horizon_days + 1))


# =========================================================
#   ARIMA(5,2,1) on log-prices (more complex model)
# =========================================================

def _simulate_arima_price_path(price_series: pd.Series, horizon_days: int) -> pd.Series:
    """
    Use ARIMA(5,2,1) on log-prices to simulate a future price path.

    - We fit ARIMA(5,2,1) to log(P_t).
    - We then simulate `horizon_days` new points with random shocks
      (so the path is not smooth).
    - We exponentiate to get prices and prepend the last observed price.
    """
    # If statsmodels is not available, fall back to AR(1) on log-returns
    if not HAS_STATSMODELS:
        return _simulate_ar1_price_path(price_series, horizon_days)

    rng = np.random.default_rng(42)

    s = price_series.dropna().astype(float)
    if len(s) < 20:
        # not enough data for ARIMA → simple flat extension
        last_price = float(s.iloc[-1])
        return pd.Series(
            [last_price] * (horizon_days + 1),
            index=np.arange(horizon_days + 1),
        )

    logp = np.log(s)

    try:
        # ARIMA(5,2,1) on log-prices
        model = ARIMA(
            logp,
            order=(5, 2, 1),
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        fit = model.fit()

        # simulate `horizon_days` new log-prices, continuing from the end
        sim_log = fit.simulate(
            nsimulations=horizon_days,
            anchor="end",
            random_state=rng,
        )

        # build full price path: last observed price + future simulated prices
        last_price = float(s.iloc[-1])
        prices_forecast = [last_price]
        prices_forecast.extend(np.exp(sim_log.values))

        return pd.Series(
            prices_forecast,
            index=np.arange(horizon_days + 1),
        )

    except Exception:
        # If ARIMA fails for some reason, fall back to AR(1)
        return _simulate_ar1_price_path(price_series, horizon_days)


# =========================================================
#   Public API used by app.py
# =========================================================

def simulate_prices_and_portfolio(
    prices_wide: pd.DataFrame,
    returns: pd.DataFrame,         # kept for compatibility, not used here
    weights: pd.Series,
    horizon_days: int,
    start_portfolio_value: float,
    lookback: int = 60,            # unused, kept for API compatibility
) -> Dict:
    """
    Simulate prices and portfolio value using ARIMA(5,2,1) (with randomness)
    on log-prices for each asset.

    - One realistic (non-smooth) path per asset.
    - Buy & hold portfolio with given weights.
    - Returns a dict with:
        'mean_portfolio_path' : pd.Series of length horizon_days+1
        'mean_price_paths'    : dict[ticker -> pd.Series]
        'final_expected'      : float final portfolio value
        'VaR', 'CVaR'         : NaN (kept for compatibility)
    """
    tickers = list(weights.index)

    # Keep only selected assets, ffill missing, drop rows where all are NaN
    prices_sel = prices_wide[tickers].ffill().dropna(how="all", axis=0)
    if prices_sel.empty:
        raise ValueError("No valid price data available for the selected assets.")

    horizon_days = int(horizon_days)
    if horizon_days <= 0:
        raise ValueError("horizon_days must be a positive integer.")

    price_paths: Dict[str, pd.Series] = {}

    # --- simulate one ARIMA path per asset ---
    for ticker in tickers:
        series = prices_sel[ticker].dropna()
        if series.empty:
            path = pd.Series(
                [0.0] * (horizon_days + 1),
                index=np.arange(horizon_days + 1),
            )
        else:
            path = _simulate_arima_price_path(series, horizon_days)
        price_paths[ticker] = path

    # --- portfolio path (buy & hold) ---
    idx = next(iter(price_paths.values())).index  # 0..horizon_days

    # initial prices at t=0
    p0 = pd.Series({tic: price_paths[tic].iloc[0] for tic in tickers}).replace(0.0, np.nan)
    n_shares = weights * start_portfolio_value / p0
    n_shares = n_shares.fillna(0.0)

    portfolio_values = []
    for t in idx:
        total = 0.0
        for ticker in tickers:
            total += float(n_shares[ticker]) * float(price_paths[ticker].iloc[t])
        portfolio_values.append(total)

    portfolio_series = pd.Series(portfolio_values, index=idx, name="Portfolio")
    final_value = float(portfolio_series.iloc[-1])

    return {
        "mean_portfolio_path": portfolio_series,
        "mean_price_paths": price_paths,
        "final_expected": final_value,
        "VaR": float("nan"),
        "CVaR": float("nan"),
    }
