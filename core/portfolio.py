import numpy as np
import pandas as pd
from scipy.optimize import minimize

from config import ANNUAL_FACTOR


# =======================================================
# Basic portfolio helpers
# =======================================================


def portfolio_returns_series(returns: pd.DataFrame, weights: pd.Series) -> pd.Series:
    """
    Daily portfolio returns for given weights (long-only, constant weights).

    Parameters
    ----------
    returns : DataFrame
        Daily returns of all assets (dates x tickers).
    weights : Series
        Portfolio weights indexed by ticker (must be a subset of columns of `returns`).

    Returns
    -------
    Series
        Daily portfolio returns.
    """
    sub = returns[weights.index]
    return (sub * weights.values).sum(axis=1)


def portfolio_equity_curve(port_ret: pd.Series, initial_capital: float) -> pd.Series:
    """
    Portfolio value over time given daily returns and initial capital.

    Parameters
    ----------
    port_ret : Series
        Daily portfolio returns.
    initial_capital : float
        Initial amount invested.

    Returns
    -------
    Series
        Portfolio value path.
    """
    return initial_capital * (1.0 + port_ret).cumprod()


# =======================================================
# Generic Markowitz-style solver (internal helper)
# =======================================================

def _solve_markowitz(
    mu_d_vec: np.ndarray,
    cov: np.ndarray,
    mode: str,
    target_mu_daily: float | None = None,
    target_var_daily: float | None = None,
    allow_short: bool = False,
):
    """
    Generic solver for mean-variance-style problems.

    mode ∈ {"min_var", "max_ret", "min_var_target_mu",
            "max_ret_target_var", "max_sharpe"}
    """
    n = len(mu_d_vec)
    x0 = np.ones(n) / n

    cov = np.asarray(cov, dtype=float)
    mu_d_vec = np.asarray(mu_d_vec, dtype=float)

    # small L2 regularisation (fixed, not exposed to the UI)
    lambda_l2 = 1e-4

    def port_stats(w: np.ndarray):
        """Return (mu_d, var_d, sigma_d) for daily returns."""
        mu_d = float(mu_d_vec @ w)
        var_d = float(w @ cov @ w)
        sigma_d = float(np.sqrt(max(var_d, 0.0)))
        return mu_d, var_d, sigma_d

    # Objectives
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

    # Bounds
    if allow_short:
        # allow some limited shorting
        bounds = [(-1.0, 1.0)] * n
    else:
        bounds = [(0.0, 1.0)] * n

    # Constraints (budget + optional target)
    cons: list[dict] = [
        {"type": "eq", "fun": lambda w: np.sum(w) - 1.0},  # budget
    ]

    if mode == "min_var_target_mu" and target_mu_daily is not None:
        cons.append(
            {"type": "ineq", "fun": lambda w: float(mu_d_vec @ w) - target_mu_daily}
        )

    if mode == "max_ret_target_var" and target_var_daily is not None:
        cons.append(
            {
                "type": "ineq",
                "fun": lambda w: float(target_var_daily - (w @ cov @ w)),
            }
        )

    # Select objective
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

    if not res.success:
        # Fallback: equal-weight
        w_opt = x0.copy()
    else:
        w_opt = res.x

    mu_d, var_d, sigma_d = port_stats(w_opt)
    mu_a = mu_d * ANNUAL_FACTOR
    sigma_a = sigma_d * np.sqrt(ANNUAL_FACTOR)

    return w_opt, mu_a, sigma_a


# =======================================================
# Public Markowitz wrappers
# =======================================================

def markowitz_min_variance(
    returns: pd.DataFrame,
    tickers: list[str],
    allow_short: bool = False,
):
    """
    Pure minimum-variance portfolio.
    """
    sub = returns[tickers]
    cov = sub.cov().values
    mu_d_vec = sub.mean().values

    w_opt, mu_a, sigma_a = _solve_markowitz(
        mu_d_vec,
        cov,
        mode="min_var",
        allow_short=allow_short,
    )
    return pd.Series(w_opt, index=tickers), mu_a, sigma_a


def markowitz_max_return(
    returns: pd.DataFrame,
    tickers: list[str],
    allow_short: bool = False,
):
    """
    Maximum expected return (daily mean) with sum-to-one constraint.
    """
    sub = returns[tickers]
    cov = sub.cov().values
    mu_d_vec = sub.mean().values

    w_opt, mu_a, sigma_a = _solve_markowitz(
        mu_d_vec,
        cov,
        mode="max_ret",
        allow_short=allow_short,
    )
    return pd.Series(w_opt, index=tickers), mu_a, sigma_a


def markowitz_min_var_target_return(
    returns: pd.DataFrame,
    tickers: list[str],
    target_return_ann: float,
    allow_short: bool = False,
):
    """
    Minimise variance for a given *annual* target return.
    """
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
    """
    Maximise expected return for a given *annual* target volatility.
    """
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


# =======================================================
# Equal Risk Contribution (ERC) portfolio
# =======================================================

def erc_portfolio(
    returns: pd.DataFrame,
    tickers: list[str],
    allow_short: bool = False,
):
    """
    Equal Risk Contribution (risk parity) portfolio.

    Each asset's percentage contribution to total risk is (approximately)
    the same.

    Parameters
    ----------
    returns : DataFrame
        Daily returns of selected assets.
    tickers : list[str]
        Assets to include.
    allow_short : bool
        If False (default), weights are constrained to be >= 0.

    Returns
    -------
    weights : Series
    mu_ann : float
    sigma_ann : float
    """
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
        # Penalise violating long-only if needed
        if (not allow_short) and np.any(w < -1e-8):
            return 1e6

        _, var_d, _ = port_stats(w)
        if var_d <= 0:
            return 1e6

        # Risk contribution of each asset
        marginal = cov @ w
        rc = w * marginal  # absolute contribution
        total_rc = rc.sum()
        if total_rc <= 0:
            return 1e6

        rc_frac = rc / total_rc
        # Target: 1/n each
        return float(((rc_frac - 1.0 / n) ** 2).sum())

    if allow_short:
        bounds = [(-1.0, 1.0)] * n
    else:
        bounds = [(0.0, 1.0)] * n

    cons = [
        {"type": "eq", "fun": lambda w: np.sum(w) - 1.0},
    ]

    res = minimize(
        objective,
        x0,
        method="SLSQP",
        bounds=bounds,
        constraints=cons,
        options={"maxiter": 1000},
    )

    if not res.success:
        w_opt = x0.copy()
    else:
        w_opt = res.x

    mu_d, var_d, sigma_d = port_stats(w_opt)
    mu_a = mu_d * ANNUAL_FACTOR
    sigma_a = sigma_d * np.sqrt(ANNUAL_FACTOR)

    return pd.Series(w_opt, index=tickers), mu_a, sigma_a


# =======================================================
# Max Sharpe portfolio
# =======================================================

def max_sharpe_portfolio(
    returns: pd.DataFrame,
    tickers: list[str],
    allow_short: bool = False,
):
    """
    Maximise the (theoretical) Sharpe ratio using daily mean/vol.

    We assume risk-free rate = 0 for simplicity.
    """
    sub = returns[tickers]
    cov = sub.cov().values
    mu_d_vec = sub.mean().values

    w_opt, mu_a, sigma_a = _solve_markowitz(
        mu_d_vec,
        cov,
        mode="max_sharpe",
        allow_short=allow_short,
    )

    sharpe = mu_a / sigma_a if sigma_a > 0 else np.nan
    return pd.Series(w_opt, index=tickers), mu_a, sigma_a, sharpe


# =======================================================
# Discrete allocation (integer number of shares + fees)
# =======================================================

def compute_discrete_allocation(
    prices_wide: pd.DataFrame,
    weights: pd.Series,
    initial_capital: float,
    fee_rate: float = 0.0,
):
    """
    Convert continuous weights into integer share counts, including transaction fees.

    fee_rate is the percentage fee (e.g. 0.005 for 0.5%) charged on the traded amount.

    Returns
    -------
    df_allocation : DataFrame with orders and weights
    cash_remaining : float
    total_fees : float
    """
    if prices_wide.empty:
        return None, 0.0, 0.0

    tickers = list(weights.index)
    prices_sel = prices_wide[tickers].ffill().dropna(how="all", axis=0)

    start_date = prices_sel.index[0]
    end_date = prices_sel.index[-1]

    p0 = prices_sel.loc[start_date]
    pT = prices_sel.loc[end_date]

    if fee_rate < 0:
        fee_rate = 0.0

    # Capital allocated per asset before fees
    alloc_capital = initial_capital * weights

    # Maximum trade value per asset so that trade_value * (1 + fee_rate) <= alloc_capital
    effective_capital = alloc_capital / (1.0 + fee_rate)

    # Integer number of shares
    n_shares = np.floor(effective_capital / p0).astype(int)

    # Monetary values
    trade_value = n_shares * p0              # value of shares bought
    fees = fee_rate * trade_value           # fees on those trades
    cash_used = trade_value + fees          # total cash spent per asset

    total_trade_value = float(trade_value.sum())
    total_fees = float(fees.sum())
    total_cash_used = float(cash_used.sum())
    cash_remaining = float(initial_capital - total_cash_used)

    # Final weights if we do not rebalance afterwards
    vT = n_shares * pT + cash_remaining
    total_T = float(vT.sum())
    w_final = vT / total_T

    df = pd.DataFrame(
        {
            "Target_weight": weights,
            "Initial_price": p0,
            "Shares_to_buy": n_shares,
            "Trade_value_XOF": trade_value,
            "Fees_XOF": fees,
            "Total_cost_XOF": cash_used,
            "Final_weight_no_rebal": w_final,
            "Weight_diff": w_final - weights,
        }
    )

    return df, cash_remaining, total_fees
