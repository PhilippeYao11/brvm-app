import numpy as np
import pandas as pd


def build_monthly_rebalancing_plan(
    prices_uni: pd.DataFrame,
    mean_price_paths: dict,
    weights: pd.Series,
    discrete_alloc: pd.DataFrame,
    initial_cash: float,
    fee_rate: float,
    horizon_days: int,
    rebalance_every_days: int = 21,
    strategy: str = "full",
    threshold_weight_diff: float = 0.0,
):
    """
    Build a simple rebalancing plan based on:
      - expected future prices (mean_price_paths from simulation)
      - target weights (weights)
      - initial discrete allocation (discrete_alloc["Shares_to_buy"])
      - initial cash
      - transaction fee rate (fee_rate, e.g. 0.005 for 0.5%)

    Hypotheses:
      - At t=0 we hold the discrete allocation & cash.
      - Every 'rebalance date' (e.g. every 21 trading days), we trade
        to go back to target weights using expected prices.
      - This is a *planning* tool: we ignore the strict "no negative cash"
        constraint to keep things simple.

    strategy:
      - "full": rebalance all assets back to target weights at each date
      - "threshold": only trade assets whose weight deviation from target
                     is larger than threshold_weight_diff (absolute).

    Returns
    -------
    plan : pd.DataFrame with rows per (date, asset, action) and columns:
        Date, Asset, Action, Shares, Trade_value_XOF, Fees_XOF, Net_cash_flow_XOF,
        Portfolio_value_after_XOF
    """
    tickers = list(weights.index)

    if discrete_alloc is None or discrete_alloc.empty:
        # no starting position -> nothing to rebalance
        return pd.DataFrame(
            columns=[
                "Date",
                "Asset",
                "Action",
                "Shares",
                "Trade_value_XOF",
                "Fees_XOF",
                "Net_cash_flow_XOF",
                "Portfolio_value_after_XOF",
            ]
        )

    if rebalance_every_days is None or rebalance_every_days <= 0:
        # explicit "no rebalancing" case
        return pd.DataFrame(
            columns=[
                "Date",
                "Asset",
                "Action",
                "Shares",
                "Trade_value_XOF",
                "Fees_XOF",
                "Net_cash_flow_XOF",
                "Portfolio_value_after_XOF",
            ]
        )

    # initial shares
    shares = discrete_alloc["Shares_to_buy"].reindex(tickers).fillna(0).astype(int)
    cash = float(initial_cash)

    # future dates: we approximate with business days from last historical date
    last_hist_date = prices_uni.index[-1]
    future_dates = pd.bdate_range(start=last_hist_date, periods=horizon_days + 1)

    # mean_price_paths: dict[asset -> Series(index=0..horizon_days)]
    price_paths_df = pd.DataFrame(
        {asset: mean_price_paths[asset].values for asset in tickers},
        index=np.arange(horizon_days + 1),
    )

    rebal_indices = list(
        range(rebalance_every_days, horizon_days + 1, rebalance_every_days)
    )
    records = []

    for idx in rebal_indices:
        if idx > horizon_days:
            break

        date = future_dates[idx]
        prices_t = price_paths_df.loc[idx, tickers]

        # current position values
        pos_values = shares * prices_t
        port_value = float(pos_values.sum() + cash)

        if port_value <= 0:
            # portfolio is dead → nothing more to plan
            continue

        # target value per asset
        target_values = weights * port_value
        diff_values = target_values - pos_values

        # trades to move toward target
        for asset in tickers:
            diff = diff_values[asset]
            price = prices_t[asset]
            if price <= 0 or abs(diff) < 1e-6:
                continue

            # Threshold strategy: rebalance only if weight deviation is large enough
            if strategy == "threshold" and port_value > 0:
                current_weight = float(pos_values[asset] / port_value)
                target_weight = float(weights[asset])
                if abs(current_weight - target_weight) < threshold_weight_diff:
                    continue

            # continuous number of shares
            q = diff / price
            # integer number of shares
            q_int = int(np.floor(abs(q))) * np.sign(q)

            if q_int == 0:
                continue

            trade_value = q_int * price
            fee = fee_rate * abs(trade_value)
            # from the cash point of view:
            # - buy: we pay trade_value + fee (negative cash flow)
            # - sell: we receive |trade_value| - fee (positive cash flow)
            net_cash = -trade_value - fee

            # update state
            shares[asset] += q_int
            cash += net_cash

            action = "Buy" if q_int > 0 else "Sell"

            records.append(
                {
                    "Date": date,
                    "Asset": asset,
                    "Action": action,
                    "Shares": int(q_int),
                    "Trade_value_XOF": float(trade_value),
                    "Fees_XOF": float(fee),
                    "Net_cash_flow_XOF": float(net_cash),
                    "Portfolio_value_after_XOF": float(
                        (shares * prices_t).sum() + cash
                    ),
                }
            )

    if not records:
        return pd.DataFrame(
            columns=[
                "Date",
                "Asset",
                "Action",
                "Shares",
                "Trade_value_XOF",
                "Fees_XOF",
                "Net_cash_flow_XOF",
                "Portfolio_value_after_XOF",
            ]
        )

    plan = pd.DataFrame(records).sort_values(["Date", "Asset"]).reset_index(drop=True)
    return plan
