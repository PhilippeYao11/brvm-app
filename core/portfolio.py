import numpy as np
import pandas as pd
from scipy.optimize import minimize

from config import ANNUAL_FACTOR


# =======================================================
#  Helpers de base
# =======================================================

def portfolio_returns_series(returns: pd.DataFrame, weights: pd.Series) -> pd.Series:
    """
    Rendements journaliers du portefeuille pour des pondérations données.
    """
    sub = returns[weights.index]
    return (sub * weights.values).sum(axis=1)


def portfolio_equity_curve(port_ret: pd.Series, initial_capital: float) -> pd.Series:
    """
    Valeur du portefeuille dans le temps à partir des rendements journaliers.
    """
    return initial_capital * (1.0 + port_ret).cumprod()


# =======================================================
#  Solveur générique moyenne–variance
# =======================================================

def _solve_mean_variance(
    mu_d_vec: np.ndarray,
    cov: np.ndarray,
    mode: str,
    target_mu_daily: float | None = None,
    target_var_daily: float | None = None,
    allow_short: bool = False,
    min_weight: float | None = None,
    max_weight: float | None = None,
):
    """
    Solveur générique pour problèmes de type moyenne–variance.

    mode ∈ {"min_var_target_mu", "max_ret_target_var", "max_sharpe"}
    min_weight / max_weight : bornes par actif (utilisées en long-only).
    """
    n = len(mu_d_vec)
    x0 = np.ones(n) / n

    cov = np.asarray(cov, dtype=float)
    mu_d_vec = np.asarray(mu_d_vec, dtype=float)

    # Régularisation L2 fixée (éviter portefeuilles trop concentrés)
    L2_REG = 0.05

    def port_stats(w: np.ndarray):
        mu_d = float(mu_d_vec @ w)
        var_d = float(w @ cov @ w)
        sigma_d = float(np.sqrt(max(var_d, 0.0)))
        return mu_d, var_d, sigma_d

    def obj_min_var(w: np.ndarray) -> float:
        _, var_d, _ = port_stats(w)
        reg = L2_REG * float((w**2).sum())
        return var_d + reg

    def obj_max_ret(w: np.ndarray) -> float:
        mu_d, _, _ = port_stats(w)
        reg = L2_REG * float((w**2).sum())
        return -mu_d + reg

    def obj_max_sharpe(w: np.ndarray) -> float:
        mu_d, var_d, sigma_d = port_stats(w)
        if sigma_d <= 0:
            return 1e6
        reg = L2_REG * float((w**2).sum())
        return -(mu_d / sigma_d) + reg

    # Bornes : long-only ou autorisation de short
    if allow_short:
        # On garde la logique "sum(w)=1" sans bornes individuelles,
        # les contraintes de poids max sont appliquées seulement en long-only.
        bounds = None
    else:
        low = 0.0 if min_weight is None else max(0.0, float(min_weight))
        high = 1.0 if max_weight is None else min(1.0, float(max_weight))
        if high <= low:
            # Sécurité : si réglage incohérent, on revient à [0,1]
            low, high = 0.0, 1.0
        bounds = [(low, high)] * n

    # Contraintes
    cons = [
        {"type": "eq", "fun": lambda w: np.sum(w) - 1.0},  # budget
    ]

    if mode == "min_var_target_mu" and target_mu_daily is not None:
        # μ(w) >= target_mu_daily
        cons.append(
            {"type": "ineq", "fun": lambda w: float(mu_d_vec @ w) - target_mu_daily}
        )
    if mode == "max_ret_target_var" and target_var_daily is not None:
        # variance(w) <= target_var_daily
        cons.append(
            {
                "type": "ineq",
                "fun": lambda w: float(target_var_daily - (w @ cov @ w)),
            }
        )

    # Choix de l’objectif
    if mode == "min_var_target_mu":
        objective = obj_min_var
    elif mode == "max_ret_target_var":
        objective = obj_max_ret
    elif mode == "max_sharpe":
        objective = obj_max_sharpe
    else:
        raise ValueError(f"Unknown optimisation mode: {mode}")

    res = minimize(
        objective,
        x0,
        method="SLSQP",
        bounds=bounds,
        constraints=cons,
        options={"maxiter": 1000},
    )

    if not res.success:
        w_opt = x0.copy()  # fallback 1/N
    else:
        w_opt = res.x

    mu_d, var_d, sigma_d = port_stats(w_opt)
    mu_a = mu_d * ANNUAL_FACTOR
    sigma_a = sigma_d * np.sqrt(ANNUAL_FACTOR)

    return w_opt, mu_a, sigma_a, mu_d, sigma_d


# =======================================================
#  Portefeuilles moyenne–variance demandés
# =======================================================

def min_variance_target_return_portfolio(
    returns: pd.DataFrame,
    tickers: list[str],
    target_return_ann: float,
    allow_short: bool = False,
):
    """
    Minimise la variance pour un rendement annuel cible donné.
    """
    sub = returns[tickers].dropna()
    cov = sub.cov().values
    mu_d_vec = sub.mean().values

    target_mu_daily = target_return_ann / ANNUAL_FACTOR

    w_opt, mu_a, sigma_a, _, _ = _solve_mean_variance(
        mu_d_vec,
        cov,
        mode="min_var_target_mu",
        target_mu_daily=target_mu_daily,
        allow_short=allow_short,
    )
    return pd.Series(w_opt, index=tickers), mu_a, sigma_a


def max_return_target_vol_portfolio(
    returns: pd.DataFrame,
    tickers: list[str],
    target_vol_ann: float,
    allow_short: bool = False,
):
    """
    Maximiser le rendement pour une volatilité annuelle cible donnée.
    """
    sub = returns[tickers].dropna()
    cov = sub.cov().values
    mu_d_vec = sub.mean().values

    target_sigma_daily = target_vol_ann / np.sqrt(ANNUAL_FACTOR)
    target_var_daily = target_sigma_daily**2

    w_opt, mu_a, sigma_a, _, _ = _solve_mean_variance(
        mu_d_vec,
        cov,
        mode="max_ret_target_var",
        target_var_daily=target_var_daily,
        allow_short=allow_short,
    )
    return pd.Series(w_opt, index=tickers), mu_a, sigma_a


def max_sharpe_portfolio(
    returns: pd.DataFrame,
    tickers: list[str],
    allow_short: bool = False,
    min_weight: float | None = None,
    max_weight: float | None = None,
):
    """
    Portefeuille qui maximise le Sharpe ratio (taux sans risque ≈ 0).

    min_weight / max_weight : bornes par actif (utilisées seulement si allow_short=False).
    """
    sub = returns[tickers].dropna()
    cov = sub.cov().values
    mu_d_vec = sub.mean().values

    w_opt, mu_a, sigma_a, mu_d, sigma_d = _solve_mean_variance(
        mu_d_vec,
        cov,
        mode="max_sharpe",
        allow_short=allow_short,
        min_weight=min_weight,
        max_weight=max_weight,
    )

    sharpe_ann = mu_a / sigma_a if sigma_a > 0 else np.nan
    return pd.Series(w_opt, index=tickers), mu_a, sigma_a, sharpe_ann


# =======================================================
#  Allocation discrète (nombre entier de titres)
# =======================================================

def compute_discrete_allocation(
    prices_wide: pd.DataFrame,
    weights: pd.Series,
    initial_capital: float,
    fee_rate: float = 0.0,
):
    """
    Convertit des pondérations continues en nombres entiers d'actions,
    en prenant en compte des frais de transaction proportionnels.
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

    # Capital alloué par actif avant frais
    alloc_capital = initial_capital * weights

    # Capital effectif max pour que trade_value * (1 + fee_rate) <= alloc_capital
    effective_capital = alloc_capital / (1.0 + fee_rate)

    # Nombre de titres (entiers)
    n_shares = np.floor(effective_capital / p0).astype(int)

    trade_value = n_shares * p0
    fees = fee_rate * trade_value
    cash_used = trade_value + fees

    total_trade_value = float(trade_value.sum())
    total_fees = float(fees.sum())
    total_cash_used = float(cash_used.sum())
    cash_remaining = float(initial_capital - total_cash_used)

    # Poids finaux si on ne rééquilibre plus
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


# =======================================================
#  Risk parity exact : Equal Risk Contribution (ERC)
# =======================================================

def equal_risk_contribution(
    returns: pd.DataFrame,
    tickers: list[str],
    max_weight: float = 0.5,
    min_weight: float = 0.0,
):
    """
    Equal Risk Contribution (ERC) :
    minimise la dispersion des contributions au risque.
    """
    sub = returns[tickers].dropna()
    cov = sub.cov().values
    n = len(tickers)
    x0 = np.ones(n) / n

    def port_var(w: np.ndarray) -> float:
        return float(w @ cov @ w)

    def risk_contrib(w: np.ndarray):
        v = port_var(w)
        if v <= 0:
            return np.zeros_like(w), 0.0
        mrc = cov @ w      # marginal risk contribution
        rc = w * mrc       # risk contribution par actif
        return rc, v

    def obj(w: np.ndarray) -> float:
        rc, v = risk_contrib(w)
        if v <= 0:
            return 1e6
        avg = rc.mean()
        return float(((rc - avg) ** 2).sum())

    bounds = [(min_weight, max_weight)] * n
    cons = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]

    res = minimize(
        obj,
        x0,
        method="SLSQP",
        bounds=bounds,
        constraints=cons,
        options={"maxiter": 1000},
    )

    if not res.success:
        w = pd.Series(x0, index=tickers)
    else:
        w = pd.Series(res.x, index=tickers)

    mu_d_vec = sub.mean().values
    w_vec = w.values
    mu_d = float(mu_d_vec @ w_vec)
    sigma_d = float(np.sqrt(w_vec @ cov @ w_vec))

    mu_a = mu_d * ANNUAL_FACTOR
    sigma_a = sigma_d * np.sqrt(ANNUAL_FACTOR)

    return w, mu_a, sigma_a
