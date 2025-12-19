# core/models.py
import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional

# statsmodels for ARIMA
try:
    from statsmodels.tsa.arima.model import ARIMA
    HAS_STATSMODELS = True
except Exception:
    HAS_STATSMODELS = False

# sklearn for Neural Net (MLP)
try:
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.neural_network import MLPRegressor
    HAS_SKLEARN = True
except Exception:
    HAS_SKLEARN = False


# =========================================================
# Utilities
# =========================================================

def _safe_log_returns(price_series: pd.Series) -> pd.Series:
    s = price_series.dropna().astype(float).replace(0.0, np.nan).dropna()
    if len(s) < 5:
        return pd.Series(dtype=float)
    return np.log(s).diff().dropna()


def _simple_returns_from_logret(lr: np.ndarray) -> np.ndarray:
    # r = exp(lr) - 1
    return np.expm1(lr)


def _build_price_path_from_returns(p0: float, simple_returns: np.ndarray) -> np.ndarray:
    # path length = len(simple_returns)+1 with p0 at t=0
    path = np.empty(len(simple_returns) + 1, dtype=float)
    path[0] = float(p0)
    for t in range(len(simple_returns)):
        path[t + 1] = path[t] * (1.0 + float(simple_returns[t]))
        if not np.isfinite(path[t + 1]) or path[t + 1] <= 0:
            path[t + 1] = max(1e-6, path[t])  # keep positive
    return path


def _compute_var_cvar_from_final_values(
    final_values: np.ndarray,
    start_value: float,
    var_level: float,
) -> Tuple[float, float]:
    """
    VaR/CVaR computed on LOSS distribution over horizon:
      loss = -(final - start) = start - final
    VaR at confidence var_level (e.g. 0.95) => 95% quantile of loss.
    We clamp to >= 0 for readability.
    """
    final_values = np.asarray(final_values, dtype=float)
    pnl = final_values - float(start_value)
    losses = -pnl  # positive = loss

    q = float(np.quantile(losses, var_level))
    var = max(0.0, q)

    tail = losses[losses >= q]
    if len(tail) == 0:
        cvar = var
    else:
        cvar = float(np.mean(tail))
        cvar = max(0.0, cvar)

    return var, cvar


# =========================================================
# ARIMA simulation on LOG-RETURNS
# =========================================================

def _fit_arima_on_logret(lr: pd.Series, order: Tuple[int, int, int]):
    if not HAS_STATSMODELS:
        return None
    try:
        model = ARIMA(
            lr,
            order=order,
            trend="n",
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        return model.fit()
    except Exception:
        return None


def _simulate_arima_logret_paths(
    lr: pd.Series,
    order: Tuple[int, int, int],
    horizon_days: int,
    n_sims: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Returns array shape (n_sims, horizon_days) of simulated log-returns.
    """
    if not HAS_STATSMODELS:
        # fallback: bootstrap from empirical distribution
        arr = lr.values
        if len(arr) < 20:
            mu = float(np.mean(arr)) if len(arr) else 0.0
            sig = float(np.std(arr, ddof=1)) if len(arr) > 2 else 1e-6
            return rng.normal(mu, sig, size=(n_sims, horizon_days))
        idx = rng.integers(0, len(arr), size=(n_sims, horizon_days))
        return arr[idx]

    fit = _fit_arima_on_logret(lr, order)
    if fit is None:
        # fallback: bootstrap
        arr = lr.values
        if len(arr) < 20:
            mu = float(np.mean(arr)) if len(arr) else 0.0
            sig = float(np.std(arr, ddof=1)) if len(arr) > 2 else 1e-6
            return rng.normal(mu, sig, size=(n_sims, horizon_days))
        idx = rng.integers(0, len(arr), size=(n_sims, horizon_days))
        return arr[idx]

    sims = np.empty((n_sims, horizon_days), dtype=float)
    for k in range(n_sims):
        try:
            # simulate future log-returns with randomness
            sim_lr = fit.simulate(
                nsimulations=horizon_days,
                anchor="end",
                random_state=np.random.default_rng(int(rng.integers(0, 2**31 - 1))),
            )
            sims[k, :] = np.asarray(sim_lr.values, dtype=float)
        except Exception:
            # fallback bootstrap
            arr = lr.values
            idx = rng.integers(0, len(arr), size=horizon_days)
            sims[k, :] = arr[idx]
    return sims


# =========================================================
# Neural Net (MLP) simulation on LOG-RETURNS
# =========================================================

def _make_supervised_lagged(lr: np.ndarray, lags: int):
    """
    lr: 1D array of log-returns
    X[t] = [lr[t-1], lr[t-2], ..., lr[t-lags]]
    y[t] = lr[t]
    """
    n = len(lr)
    if n <= lags + 2:
        return None, None
    X = []
    y = []
    for t in range(lags, n):
        X.append(lr[t - lags:t][::-1])  # most recent first
        y.append(lr[t])
    return np.asarray(X, dtype=float), np.asarray(y, dtype=float)


def _fit_mlp_logret_model(
    lr: pd.Series,
    lags: int,
    hidden_layer_sizes: Tuple[int, ...],
    max_iter: int,
    rng: np.random.Generator,
):
    """
    Fit MLP to predict next log-return from lagged log-returns.
    Returns (pipeline_model, residuals_array, last_lags_vector)
    """
    if not HAS_SKLEARN:
        return None, None, None

    arr = np.asarray(lr.values, dtype=float)
    X, y = _make_supervised_lagged(arr, lags)
    if X is None or len(y) < 50:
        return None, None, None

    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("mlp", MLPRegressor(
                hidden_layer_sizes=hidden_layer_sizes,
                activation="relu",
                solver="adam",
                alpha=1e-4,
                learning_rate_init=1e-3,
                max_iter=max_iter,
                random_state=int(rng.integers(0, 2**31 - 1)),
                early_stopping=True,
                n_iter_no_change=10,
                validation_fraction=0.15,
            )),
        ]
    )

    try:
        model.fit(X, y)
        y_hat = model.predict(X)
        resid = (y - y_hat).astype(float)
        last_lags = arr[-lags:][::-1].astype(float)  # most recent first
        return model, resid, last_lags
    except Exception:
        return None, None, None


def _simulate_mlp_logret_paths(
    lr: pd.Series,
    horizon_days: int,
    n_sims: int,
    lags: int,
    hidden_layer_sizes: Tuple[int, ...],
    max_iter: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Simulate log-return paths using MLP mean + bootstrap residuals.
    Returns array shape (n_sims, horizon_days)
    """
    model, resid, last_lags = _fit_mlp_logret_model(
        lr=lr,
        lags=lags,
        hidden_layer_sizes=hidden_layer_sizes,
        max_iter=max_iter,
        rng=rng,
    )

    # fallback if NN not available or insufficient data
    arr = np.asarray(lr.values, dtype=float)
    if model is None or resid is None or last_lags is None or len(arr) < 20:
        mu = float(np.mean(arr)) if len(arr) else 0.0
        sig = float(np.std(arr, ddof=1)) if len(arr) > 2 else 1e-6
        return rng.normal(mu, sig, size=(n_sims, horizon_days))

    resid = resid[np.isfinite(resid)]
    if len(resid) < 20:
        resid = np.array([0.0], dtype=float)

    sims = np.empty((n_sims, horizon_days), dtype=float)

    for k in range(n_sims):
        lags_vec = last_lags.copy()  # shape (lags,)
        for t in range(horizon_days):
            x = lags_vec.reshape(1, -1)
            mu_hat = float(model.predict(x)[0])
            eps = float(resid[int(rng.integers(0, len(resid)))])
            lr_next = mu_hat + eps
            sims[k, t] = lr_next
            # update lags
            lags_vec = np.roll(lags_vec, shift=1)
            lags_vec[0] = lr_next

    return sims


# =========================================================
# Public API used by app.py
# =========================================================

def simulate_prices_and_portfolio(
    prices_wide: pd.DataFrame,
    returns: pd.DataFrame,  # kept for compatibility (not required)
    weights: pd.Series,
    horizon_days: int,
    start_portfolio_value: float,
    n_sims: int = 300,
    var_level: float = 0.95,
    model_choice: str = "ARIMA",                 # "ARIMA" or "NN"
    arima_order: Tuple[int, int, int] = (1, 0, 1),
    nn_lags: int = 10,
    nn_hidden: Tuple[int, ...] = (32, 16),
    nn_max_iter: int = 200,
    lookback: int = 504,
) -> Dict:
    """
    Simulate asset prices and portfolio value over horizon_days.

    - model_choice="ARIMA": ARIMA on LOG-RETURNS (per asset), simulate n_sims paths
    - model_choice="NN": MLP on LOG-RETURNS (per asset), simulate n_sims paths by bootstrapping residuals

    Returns dict keys (used by app.py):
      price_paths      : dict[ticker -> pd.Series]    # one representative path (sim 0)
      mean_price_paths : dict[ticker -> pd.Series]    # mean over simulations
      portfolio_path   : pd.Series                    # representative portfolio path
      mean_portfolio_path : pd.Series                 # mean portfolio path over sims
      final_expected   : float                        # mean final portfolio value over sims
      VaR, CVaR        : float                        # horizon loss VaR/CVaR in XOF
      model_used       : str
    """
    tickers = list(weights.index)
    horizon_days = int(horizon_days)
    n_sims = int(max(50, n_sims))
    rng = np.random.default_rng(42)

    prices_sel = prices_wide[tickers].ffill().dropna(how="any", axis=0)
    if prices_sel.empty:
        raise ValueError("No valid price data available for the selected assets.")

    # last observed prices (t=0)
    p0 = prices_sel.iloc[-1].astype(float).replace(0.0, np.nan).fillna(method="ffill")
    if p0.isna().any():
        p0 = p0.fillna(p0.mean())
    p0 = p0.replace([np.inf, -np.inf], np.nan).fillna(1e-6)

    # shares (buy & hold) from start_portfolio_value
    w = weights.copy().astype(float)
    if w.sum() <= 0:
        raise ValueError("Weights sum to zero.")
    w = w / w.sum()
    n_shares = (w * float(start_portfolio_value) / p0).fillna(0.0)

    # containers
    price_paths_rep: Dict[str, pd.Series] = {}
    mean_price_paths: Dict[str, pd.Series] = {}

    # store portfolio paths for all simulations to compute mean + VaR/CVaR
    port_paths = np.empty((n_sims, horizon_days + 1), dtype=float)

    # per-asset simulation
    all_asset_price_sims = {}  # ticker -> ndarray(n_sims, horizon_days+1)

    for tic in tickers:
        lr = _safe_log_returns(prices_sel[tic])
        if len(lr) > lookback:
            lr = lr.iloc[-lookback:]

        if len(lr) < 40:
            # fallback: tiny noise
            lr_arr = lr.values
            mu = float(np.mean(lr_arr)) if len(lr_arr) else 0.0
            sig = float(np.std(lr_arr, ddof=1)) if len(lr_arr) > 2 else 1e-6
            lr_sims = rng.normal(mu, sig, size=(n_sims, horizon_days))
        else:
            if model_choice.upper() == "NN":
                lr_sims = _simulate_mlp_logret_paths(
                    lr=lr,
                    horizon_days=horizon_days,
                    n_sims=n_sims,
                    lags=int(nn_lags),
                    hidden_layer_sizes=tuple(nn_hidden),
                    max_iter=int(nn_max_iter),
                    rng=rng,
                )
            else:
                lr_sims = _simulate_arima_logret_paths(
                    lr=lr,
                    order=tuple(arima_order),
                    horizon_days=horizon_days,
                    n_sims=n_sims,
                    rng=rng,
                )

        # convert to simple returns and build price paths
        p0_t = float(p0[tic])
        price_sims = np.empty((n_sims, horizon_days + 1), dtype=float)
        for k in range(n_sims):
            r_simple = _simple_returns_from_logret(lr_sims[k, :])
            price_sims[k, :] = _build_price_path_from_returns(p0_t, r_simple)

        all_asset_price_sims[tic] = price_sims

        # representative (sim 0) + mean path
        rep = pd.Series(price_sims[0, :], index=np.arange(horizon_days + 1))
        meanp = pd.Series(price_sims.mean(axis=0), index=np.arange(horizon_days + 1))
        price_paths_rep[tic] = rep
        mean_price_paths[tic] = meanp

    # portfolio paths
    for k in range(n_sims):
        total = np.zeros(horizon_days + 1, dtype=float)
        for tic in tickers:
            total += float(n_shares[tic]) * all_asset_price_sims[tic][k, :]
        port_paths[k, :] = total

    port_rep = pd.Series(port_paths[0, :], index=np.arange(horizon_days + 1), name="Portfolio")
    port_mean = pd.Series(port_paths.mean(axis=0), index=np.arange(horizon_days + 1), name="Portfolio_mean")

    final_values = port_paths[:, -1]
    final_expected = float(np.mean(final_values))
    VaR, CVaR = _compute_var_cvar_from_final_values(final_values, start_portfolio_value, var_level)

    return {
        "price_paths": price_paths_rep,
        "mean_price_paths": mean_price_paths,
        "portfolio_path": port_rep,
        "mean_portfolio_path": port_mean,
        "final_expected": final_expected,
        "VaR": VaR,
        "CVaR": CVaR,
        "model_used": "NN (MLP on log-returns)" if model_choice.upper() == "NN" else f"ARIMA{tuple(arima_order)} on log-returns",
    }
