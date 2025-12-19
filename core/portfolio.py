import numpy as np
import pandas as pd
from scipy.optimize import minimize

from config import ANNUAL_FACTOR


def portfolio_returns_series(returns: pd.DataFrame, weights: pd.Series) -> pd.Series:
    sub = returns[weights.index]
    return (sub * weights.values).sum(axis=1)


def portfolio_equity_curve(port_ret: pd.Series, initial_capital: float) -> pd.Series:
    return initial_capital * (1.0 + port_ret).cumprod()


def _solve_markowitz(
    mu_d_vec: np.ndarray,
    cov: np.ndarray,
    mode: str,
    target_mu_daily: float | None = None,
    target_var_daily: float | None = None,
    allow_short: bool = False,
):
    n = len(mu_d_vec)
    x0 = np.ones(n) / n

    cov = np.asarray(cov, dtype=float)
    mu_d_vec = np.asarray(mu_d_vec, dtype=float)

    lambda_l2 = 1e-4

    def port_stats(w: np.ndarray):
        mu_d = float(mu_d_vec @ w)
        var_d = float(w @ cov @ w)
        sigma_d = float(np.sqrt(max(var_d, 0.0)))
        return mu_d, var_d, sigma_d

    def obj_min_var(w: np.ndarray) -> float:
        _, var_d, _ = port_stats(w)
        reg = lambda_l2 * float((w**2).sum())
        return var_d + reg

    def obj_max_ret(w: np.ndarray) -> float:
        mu_d, _, _ = port_stats(w)
        reg = lambda_l2 * float((w**2).sum())
        return -mu_d + reg

    def obj_max_sharpe(w: np.ndarray) -> float:
        mu_d, _, sigma_d = port_stats(w)
        if sigma_d <= 0:
            return 1e6
        reg = lambda_l2 * float((w**2).sum())
        return -(mu_d / sigma_d) + reg

    if allow_short:
        bounds = [(-1.0, 1.0)] * n
    else:
        bounds = [(0.0, 1.0)] * n

    cons: list[dict] = [
        {"type": "eq", "fun": lambda w: np.sum(w) - 1.0},
    ]

    if mode == "min_var_target_mu" and target_mu_daily is not None:
        cons.append({"type": "ineq", "fun": lambda w: float(mu_d_vec @ w) - target_mu_daily})

    if mode == "max_ret_target_var" and target_var_daily is not None:
        cons.append({"type": "ineq", "fun": lambda w: float(target_var_daily - (w @ cov @ w))})

    if mode == "min_var":
        objective = obj_min_var
    elif mode == "max_ret":
        objective = obj_max_ret
    elif mode == "min_var_target_mu":
        objective = obj_min_var
    elif mode == "max_ret_target_var":
        objective = obj_max_ret
    elif mode == "max_sharpe":
        objective = obj_max_sharpe
    else:
        raise ValueError(f"Unknown Markowitz mode: {mode}")

    res = minimize(
        objective,
        x0,
        method="SLSQP",
        bounds=bounds,
        constraints=cons,
        options={"maxiter": 1000},
    )

    w_opt = x0.copy() if (not res.success) else res.x

    mu_d, var_d, sigma_d = port_stats(w_opt)
    mu_a = mu_d * ANNUAL_FACTOR
    sigma_a = sigma_d * np.sqrt(ANNUAL_FACTOR)

    return w_opt, mu_a, sigma_a


def markowitz_min_variance(returns: pd.DataFrame, tickers: list[str], allow_short: bool = False):
    sub = returns[tickers]
    cov = sub.cov().values
    mu_d_vec = sub.mean().values

    w_opt, mu_a, sigma_a = _solve_markowitz(mu_d_vec, cov, mode="min_var", allow_short=allow_short)
    return pd.Series(w_opt, index=tickers), mu_a, sigma_a


def markowitz_max_return(returns: pd.DataFrame, tickers: list[str], allow_short: bool = False):
    sub = returns[tickers]
    cov = sub.cov().values
    mu_d_vec = sub.mean().values

    w_opt, mu_a, sigma_a = _solve_markowitz(mu_d_vec, cov, mode="max_ret", allow_short=allow_short)
    return pd.Series(w_opt, index=tickers), mu_a, sigma_a


def markowitz_min_var_target_return(
    returns: pd.DataFrame,
    tickers: list[str],
    target_return_ann: float,
    allow_short: bool = False,
):
    sub = returns[tickers]
    cov = sub.cov().values
    mu_d_vec = sub.mean().values
    target_mu_daily = target_return_ann / ANNUAL_FACTOR

    w_opt, mu_a, sigma_a = _solve_markowitz(
        mu_d_vec,
        cov,
        mode="min_var_target_mu",
        target_mu_daily=target_mu_daily,
        allow_short=allow_short,
    )
    return pd.Series(w_opt, index=tickers), mu_a, sigma_a


def markowitz_max_return_target_var(
    returns: pd.DataFrame,
    tickers: list[str],
    target_vol_ann: float,
    allow_short: bool = False,
):
    sub = returns[tickers]
    cov = sub.cov().values
    mu_d_vec = sub.mean().values

    target_sigma_daily = target_vol_ann / np.sqrt(ANNUAL_FACTOR)
    target_var_daily = target_sigma_daily**2

    w_opt, mu_a, sigma_a = _solve_markowitz(
        mu_d_vec,
        cov,
        mode="max_ret_target_var",
        target_var_daily=target_var_daily,
        allow_short=allow_short,
    )
    return pd.Series(w_opt, index=tickers), mu_a, sigma_a


def erc_portfolio(returns: pd.DataFrame, tickers: list[str], allow_short: bool = False):
    sub = returns[tickers]
    cov = sub.cov().values
    mu_d_vec = sub.mean().values
    n = len(tickers)
    x0 = np.ones(n) / n

    def port_stats(w: np.ndarray):
        mu_d = float(mu_d_vec @ w)
        var_d = float(w @ cov @ w)
        sigma_d = float(np.sqrt(max(var_d, 0.0)))
        return mu_d, var_d, sigma_d

    def objective(w: np.ndarray) -> float:
        if (not allow_short) and np.any(w < -1e-8):
            return 1e6
        _, var_d, _ = port_stats(w)
        if var_d <= 0:
            return 1e6
        marginal = cov @ w
        rc = w * marginal
        total_rc = rc.sum()
        if total_rc <= 0:
            return 1e6
        rc_frac = rc / total_rc
        return float(((rc_frac - 1.0 / n) ** 2).sum())

    bounds = [(-1.0, 1.0)] * n if allow_short else [(0.0, 1.0)] * n
    cons = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]

    res = minimize(objective, x0, method="SLSQP", bounds=bounds, constraints=cons, options={"maxiter": 1000})
    w_opt = x0.copy() if (not res.success) else res.x

    mu_d, var_d, sigma_d = port_stats(w_opt)
    mu_a = mu_d * ANNUAL_FACTOR
    sigma_a = sigma_d * np.sqrt(ANNUAL_FACTOR)

    return pd.Series(w_opt, index=tickers), mu_a, sigma_a


def max_sharpe_portfolio(returns: pd.DataFrame, tickers: list[str], allow_short: bool = False):
    sub = returns[tickers]
    cov = sub.cov().values
    mu_d_vec = sub.mean().values

    w_opt, mu_a, sigma_a = _solve_markowitz(mu_d_vec, cov, mode="max_sharpe", allow_short=allow_short)
    sharpe = mu_a / sigma_a if sigma_a > 0 else np.nan
    return pd.Series(w_opt, index=tickers), mu_a, sigma_a, sharpe


def compute_discrete_allocation(
    prices_wide: pd.DataFrame,
    weights: pd.Series,
    initial_capital: float,
    fee_rate: float = 0.0,
    trade_on: str = "last",  # "first" (backtest) or "last" (orders live)
):
    """
    Convert continuous weights into integer share counts, including transaction fees.

    Returns
    -------
    df_allocation : DataFrame
    cash_remaining : float
    total_fees : float
    """
    if prices_wide.empty:
        return None, 0.0, 0.0

    if fee_rate < 0:
        fee_rate = 0.0

    # Discrete allocation cannot implement short positions
    if (weights < -1e-12).any():
        return None, float(initial_capital), 0.0

    tickers = list(weights.index)
    prices_sel = prices_wide[tickers].ffill().dropna(how="all", axis=0)
    if prices_sel.empty:
        return None, float(initial_capital), 0.0

    if trade_on not in {"first", "last"}:
        raise ValueError("trade_on must be 'first' or 'last'.")

    trade_date = prices_sel.index[0] if trade_on == "first" else prices_sel.index[-1]
    p_exec = prices_sel.loc[trade_date].astype(float)

    valid = p_exec.notna() & (p_exec > 0)
    if valid.sum() < 2:
        return None, float(initial_capital), 0.0

    p_exec = p_exec[valid]
    w = weights.reindex(p_exec.index).fillna(0.0).clip(lower=0.0)
    if w.sum() <= 0:
        return None, float(initial_capital), 0.0
    w = w / w.sum()

    alloc_capital = initial_capital * w
    effective_capital = alloc_capital / (1.0 + fee_rate)

    n_shares = np.floor(effective_capital / p_exec).astype(int)

    trade_value = n_shares * p_exec
    fees = fee_rate * trade_value
    cash_used = trade_value + fees

    total_fees = float(fees.sum())
    total_cash_used = float(cash_used.sum())
    cash_remaining = float(initial_capital - total_cash_used)

    # ✅ Correct cash handling: cash counted once
    v_assets = trade_value.astype(float)
    total_port = float(v_assets.sum() + cash_remaining)
    if total_port <= 0:
        return None, float(initial_capital), 0.0

    w_realised = v_assets / total_port

    df = pd.DataFrame(
        {
            "Target_weight": w,
            "Initial_price": p_exec,
            "Shares_to_buy": n_shares,
            "Trade_value_XOF": trade_value,
            "Fees_XOF": fees,
            "Total_cost_XOF": cash_used,
            "Final_weight_no_rebal": w_realised,
            "Weight_diff": w_realised - w,
        }
    )

    return df, cash_remaining, total_fees
