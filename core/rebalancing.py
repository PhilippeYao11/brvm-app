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
    tickers = list(weights.index)

    empty_cols = [
        "Date",
        "Asset",
        "Action",
        "Shares",
        "Trade_value_XOF",
        "Fees_XOF",
        "Net_cash_flow_XOF",
        "Portfolio_value_after_XOF",
    ]

    if discrete_alloc is None or discrete_alloc.empty:
        return pd.DataFrame(columns=empty_cols)

    if rebalance_every_days is None or rebalance_every_days <= 0:
        return pd.DataFrame(columns=empty_cols)

    shares = discrete_alloc["Shares_to_buy"].reindex(tickers).fillna(0).astype(int)
    cash = float(initial_cash)

    last_hist_date = prices_uni.index[-1]
    # ✅ start on next business day to avoid overlap ambiguity
    future_dates = pd.bdate_range(start=last_hist_date + pd.tseries.offsets.BDay(1), periods=horizon_days + 1)

    # mean_price_paths: dict[asset -> Series(index=0..horizon_days)]
    price_paths_df = pd.DataFrame(
        {asset: mean_price_paths[asset].values for asset in tickers if asset in mean_price_paths},
        index=np.arange(horizon_days + 1),
    )

    # keep only assets present in price_paths_df
    tickers_eff = [t for t in tickers if t in price_paths_df.columns]
    if len(tickers_eff) < 1:
        return pd.DataFrame(columns=empty_cols)

    rebal_indices = list(range(rebalance_every_days, horizon_days + 1, rebalance_every_days))
    records = []

    for idx in rebal_indices:
        if idx > horizon_days:
            break

        date = future_dates[idx]
        prices_t = price_paths_df.loc[idx, tickers_eff]

        pos_values = shares.reindex(tickers_eff).fillna(0).values * prices_t.values
        pos_values = pd.Series(pos_values, index=tickers_eff)

        port_value = float(pos_values.sum() + cash)
        if port_value <= 0:
            continue

        target_values = weights.reindex(tickers_eff).fillna(0.0) * port_value
        diff_values = target_values - pos_values

        for asset in tickers_eff:
            diff = float(diff_values[asset])
            price = float(prices_t[asset])
            if price <= 0 or abs(diff) < 1e-6:
                continue

            if strategy == "threshold" and port_value > 0:
                current_weight = float(pos_values[asset] / port_value)
                target_weight = float(weights.reindex(tickers_eff).fillna(0.0)[asset])
                if abs(current_weight - target_weight) < float(threshold_weight_diff):
                    continue

            q = diff / price
            q_int = int(np.floor(abs(q))) * (1 if q >= 0 else -1)
            if q_int == 0:
                continue

            trade_value = q_int * price
            fee = fee_rate * abs(trade_value)
            net_cash = -trade_value - fee

            shares[asset] = int(shares.get(asset, 0) + q_int)
            cash += float(net_cash)

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
                    "Portfolio_value_after_XOF": float((shares.reindex(tickers_eff).fillna(0) * prices_t).sum() + cash),
                }
            )

    if not records:
        return pd.DataFrame(columns=empty_cols)

    return pd.DataFrame(records).sort_values(["Date", "Asset"]).reset_index(drop=True)
