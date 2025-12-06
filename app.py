import numpy as np
import pandas as pd
import altair as alt
import streamlit as st

from config import RAW_DIR, ANNUAL_FACTOR, DEFAULT_MC_SIMS, DEFAULT_VAR_LEVEL
from core.data_loader import (
    build_prices_from_sika_folder,
    build_prices_from_uploaded,
    slice_history,
    compute_returns,
    compute_asset_stats,
)
from core.portfolio import (
    portfolio_returns_series,
    portfolio_equity_curve,
    markowitz_min_variance,
    markowitz_max_return,
    markowitz_min_var_target_return,
    markowitz_max_return_target_var,
    erc_portfolio,
    max_sharpe_portfolio,
    compute_discrete_allocation,
)
from core.models import simulate_prices_and_portfolio
from core.risk import compute_sharpe_sortino
from core.technical import compute_indicators
from core.rebalancing import build_monthly_rebalancing_plan


def main():
    st.set_page_config(
        page_title="BRVM Quant – Portfolio Lab",
        page_icon="📊",
        layout="wide",
    )

    # Light blue theme
    st.markdown(
        """
        <style>
        .stApp {
            background-color: #e0f2fe;
            color: #0f172a;
        }
        .block-container {
            padding-top: 1.5rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.title("📊 BRVM Quant – Portfolio Lab (BRVM)")

    # ======================================================
    # Sidebar: global parameters
    # ======================================================
    st.sidebar.header("Investment settings")

    initial_capital = st.sidebar.number_input(
        "Initial capital (XOF)",
        min_value=1.0,
        value=1_000_000.0,
        step=50_000.0,
    )

    transaction_fee_pct = st.sidebar.number_input(
        "Transaction fee per trade (% of trade value)",
        min_value=0.0,
        max_value=5.0,
        value=0.5,
        step=0.1,
    )
    fee_rate = transaction_fee_pct / 100.0

    lookback_choice = st.sidebar.selectbox(
        "Historical lookback used for estimation",
        ("All history", "Last 6 months", "Last 12 months", "Last 24 months"),
        index=0,
    )

    var_level = (
        st.sidebar.slider(
            "VaR / CVaR confidence level (%)",
            min_value=90,
            max_value=99,
            value=int(DEFAULT_VAR_LEVEL * 100),
            step=1,
        )
        / 100.0
    )

    st.sidebar.markdown("---")
    st.sidebar.subheader("Data source")

    uploaded_files = st.sidebar.file_uploader(
        "Raw SIKA CSV files (you can select multiple)",
        type=["csv"],
        accept_multiple_files=True,
    )
    st.sidebar.caption(
        f"If you don't upload anything, the app reads CSV files from:\n`{RAW_DIR}`"
    )

    # ======================================================
    # Load data
    # ======================================================
    if uploaded_files:
        data_long, prices_wide, volumes_wide = build_prices_from_uploaded(uploaded_files)
        st.info(f"✅ Data loaded from uploaded files ({len(uploaded_files)} CSV).")
    else:
        data_long, prices_wide, volumes_wide = build_prices_from_sika_folder(RAW_DIR)
        st.info(f"✅ Data loaded from folder: `{RAW_DIR}`")

    # Convert lookback label to French label used in slice_history
    label_to_choice = {
        "All history": "Tout",
        "Last 6 months": "6 derniers mois",
        "Last 12 months": "12 derniers mois",
        "Last 24 months": "24 derniers mois",
    }
    prices_wide, volumes_wide = slice_history(
        prices_wide, volumes_wide, label_to_choice[lookback_choice]
    )

    # Compute returns & per-asset statistics
    returns = compute_returns(prices_wide)
    asset_stats = compute_asset_stats(returns)

    if len(asset_stats) < 2:
        st.error("You need at least 2 assets with valid returns.")
        return

    all_tickers = list(asset_stats.index)

    # ======================================================
    # Sidebar: selection of assets + optimisation method
    # ======================================================
    st.sidebar.subheader("Optimisation universe")

    selected_assets = st.sidebar.multiselect(
        "Assets used for optimisation",
        options=all_tickers,
        default=all_tickers[: min(5, len(all_tickers))],
    )

    if len(selected_assets) < 2:
        st.error("Please select at least two assets for optimisation.")
        return

    prices_uni = prices_wide[selected_assets]
    returns_uni = returns[selected_assets]
    stats_uni = asset_stats.loc[selected_assets]
    universe_tickers = selected_assets

    st.sidebar.subheader("Optimisation method")

    opt_method = st.sidebar.selectbox(
        "Choose optimisation method",
        (
            "Min variance",
            "Max return",
            "Min variance (target return)",
            "Max return (target volatility)",
            "Equal Risk Contribution (ERC)",
            "Max Sharpe ratio",
        ),
        index=0,
    )

    allow_short = st.sidebar.checkbox("Allow short selling", value=False)

    target_return_ann = None
    target_vol_ann = None

    if opt_method == "Min variance (target return)":
        target_return_ann = (
            st.sidebar.number_input(
                "Target annual return (%)",
                min_value=0.0,
                max_value=100.0,
                value=10.0,
                step=1.0,
            )
            / 100.0
        )

    if opt_method == "Max return (target volatility)":
        target_vol_ann = (
            st.sidebar.number_input(
                "Target annual volatility (%)",
                min_value=0.0,
                max_value=100.0,
                value=15.0,
                step=1.0,
            )
            / 100.0
        )

    # ======================================================
    # Run optimisation
    # ======================================================
    sharpe_theo = np.nan

    if opt_method == "Min variance":
        w_opt, mu_opt, sigma_opt = markowitz_min_variance(
            returns_uni, universe_tickers, allow_short=allow_short
        )
        mode_label = "Min variance"

    elif opt_method == "Max return":
        w_opt, mu_opt, sigma_opt = markowitz_max_return(
            returns_uni, universe_tickers, allow_short=allow_short
        )
        mode_label = "Max return"

    elif opt_method == "Min variance (target return)":
        w_opt, mu_opt, sigma_opt = markowitz_min_var_target_return(
            returns_uni,
            universe_tickers,
            target_return_ann=target_return_ann,
            allow_short=allow_short,
        )
        mode_label = f"Min variance for target return ({target_return_ann:.0%})"

    elif opt_method == "Max return (target volatility)":
        w_opt, mu_opt, sigma_opt = markowitz_max_return_target_var(
            returns_uni,
            universe_tickers,
            target_vol_ann=target_vol_ann,
            allow_short=allow_short,
        )
        mode_label = f"Max return for target volatility ({target_vol_ann:.0%})"

    elif opt_method == "Equal Risk Contribution (ERC)":
        w_opt, mu_opt, sigma_opt = erc_portfolio(
            returns_uni, universe_tickers, allow_short=allow_short
        )
        mode_label = "Equal Risk Contribution (ERC)"

    else:  # "Max Sharpe ratio"
        w_opt, mu_opt, sigma_opt, sharpe_theo = max_sharpe_portfolio(
            returns_uni, universe_tickers, allow_short=allow_short
        )
        mode_label = "Max Sharpe ratio"

    # ======================================================
    # Portfolio returns, equity curve & discrete allocation
    # ======================================================
    port_ret = portfolio_returns_series(returns_uni, w_opt)
    risk_measures = compute_sharpe_sortino(port_ret)

    discrete_alloc, leftover_cash, total_fees = compute_discrete_allocation(
        prices_uni, w_opt, initial_capital, fee_rate=fee_rate
    )

    if discrete_alloc is not None:
        total_trade_value = float(discrete_alloc["Trade_value_XOF"].sum())
        initial_invested = total_trade_value
        initial_portfolio_value = initial_invested + leftover_cash
        equity_invested = portfolio_equity_curve(port_ret, initial_invested)
        equity_curve = equity_invested + leftover_cash
    else:
        initial_portfolio_value = initial_capital
        equity_curve = portfolio_equity_curve(port_ret, initial_capital)
        leftover_cash = 0.0
        total_fees = 0.0

    # ======================================================
    # Tabs
    # ======================================================
    tab_data, tab_portfolio, tab_proj, tab_tech = st.tabs(
        [
            "📊 Data & universe",
            "🧮 Optimised portfolio",
            "🔮 Projections & risk",
            "📈 Technical signals",
        ]
    )

    # ------------------------------------------------------
    # TAB 1: DATA & UNIVERSE
    # ------------------------------------------------------
    with tab_data:
        st.subheader("Universe overview")

        c1, c2, c3 = st.columns(3)
        c1.metric("Total assets loaded", f"{len(all_tickers)}")
        c2.metric("Assets selected for optimisation", f"{len(universe_tickers)}")
        c3.metric(
            "Date range",
            f"{prices_wide.index.min().date()} → {prices_wide.index.max().date()}",
        )

        st.markdown("**Global universe statistics (all assets)**")
        st.dataframe(
            asset_stats[["mu_ann", "sigma_ann", "mu_3m"]]
            .rename(
                columns={
                    "mu_ann": "Annual return",
                    "sigma_ann": "Annual volatility",
                    "mu_3m": "3-month return (approx.)",
                }
            )
            .style.format("{:.2%}")
        )

        st.markdown("**Current optimisation universe (selected assets)**")
        st.dataframe(
            stats_uni[["mu_ann", "sigma_ann", "mu_3m"]]
            .rename(
                columns={
                    "mu_ann": "Annual return",
                    "sigma_ann": "Annual volatility",
                    "mu_3m": "3-month return (approx.)",
                }
            )
            .style.format("{:.2%}")
        )

        st.subheader("Daily prices (selected assets)")

        assets_to_plot = st.multiselect(
            "Assets to display",
            options=universe_tickers,
            default=universe_tickers[: min(5, len(universe_tickers))],
            key="hist_assets_multiselect",
        )

        if assets_to_plot:
            df_prices = (
                prices_uni[assets_to_plot]
                .reset_index()
                .melt(id_vars="date", var_name="Asset", value_name="Price")
            )

            chart_prices = (
                alt.Chart(df_prices)
                .mark_line()
                .encode(
                    x=alt.X("date:T", title="Date"),
                    y=alt.Y("Price:Q", title="Closing price"),
                    color=alt.Color("Asset:N", title="Asset"),
                )
                .properties(height=350)
                .interactive()
            )
            st.altair_chart(chart_prices, use_container_width=True)

    # ------------------------------------------------------
    # TAB 2: OPTIMISED PORTFOLIO
    # ------------------------------------------------------
    with tab_portfolio:
        st.subheader("Optimised portfolio (selected universe only)")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Optimisation method", mode_label)
        c2.metric("Expected annual return", f"{mu_opt:.2%}")
        c3.metric("Expected annual volatility", f"{sigma_opt:.2%}")
        c4.metric(
            "Historical Sharpe (annualised)",
            f"{risk_measures['sharpe']:.2f}"
            if not np.isnan(risk_measures["sharpe"])
            else "N/A",
        )
        st.caption(
            "Historical Sortino ratio (annualised): "
            + (
                f"{risk_measures['sortino']:.2f}"
                if not np.isnan(risk_measures["sortino"])
                else "N/A"
            )
        )

        amounts = (w_opt * initial_capital).rename("Amount_XOF")
        df_weights = pd.concat([w_opt.rename("Weight"), amounts], axis=1)

        st.markdown("**Continuous weights and invested amounts**")
        st.dataframe(
            df_weights[df_weights["Weight"] > 1e-4]
            .sort_values("Weight", ascending=False)
            .style.format({"Weight": "{:.2%}", "Amount_XOF": "{:,.0f}"})
        )

        st.markdown("**Discrete allocation – orders for SIKA virtual portfolio**")
        if discrete_alloc is not None:
            st.caption(
                f"Total transaction fees: ~ {total_fees:,.0f} XOF  •  "
                f"Cash remaining after trades: ~ {leftover_cash:,.0f} XOF"
            )
            st.dataframe(
                discrete_alloc[
                    [
                        "Target_weight",
                        "Initial_price",
                        "Shares_to_buy",
                        "Trade_value_XOF",
                        "Fees_XOF",
                        "Total_cost_XOF",
                        "Final_weight_no_rebal",
                        "Weight_diff",
                    ]
                ]
                .sort_values("Target_weight", ascending=False)
                .style.format(
                    {
                        "Target_weight": "{:.2%}",
                        "Initial_price": "{:,.0f}",
                        "Shares_to_buy": "{:d}",
                        "Trade_value_XOF": "{:,.0f}",
                        "Fees_XOF": "{:,.0f}",
                        "Total_cost_XOF": "{:,.0f}",
                        "Final_weight_no_rebal": "{:.2%}",
                        "Weight_diff": "{:.2%}",
                    }
                )
            )
        else:
            st.info("No discrete allocation available.")

        st.markdown("**Historical portfolio value (including cash)**")
        st.line_chart(equity_curve, height=280)

    # ------------------------------------------------------
    # TAB 3: PROJECTIONS & RISK
    # ------------------------------------------------------
    with tab_proj:
        st.subheader("Daily projections and portfolio risk")

        horizon_years = st.slider(
            "Projection horizon (years)",
            min_value=0.5,
            max_value=10.0,
            value=1.0,
            step=0.5,
        )
        horizon_days = int(252 * horizon_years)

        start_value = float(equity_curve.iloc[-1])

        sim = simulate_prices_and_portfolio(
        prices_uni,
        returns_uni,
        w_opt,
        horizon_days,
        start_value,
    )



        # Monthly rebalancing plan, based on expected future prices
        rebal_plan = build_monthly_rebalancing_plan(
            prices_uni=prices_uni,
            mean_price_paths=sim["mean_price_paths"],
            weights=w_opt,
            discrete_alloc=discrete_alloc,
            initial_cash=leftover_cash,
            fee_rate=fee_rate,
            horizon_days=horizon_days,
            rebalance_every_days=21,  # ~ 1 month (21 trading days)
            strategy="full",
            threshold_weight_diff=0.0,
        )

        # Portfolio value: historical + projected mean
        # --- Portfolio value: historical + projected mean ---
        # Historical part
        hist_df = pd.DataFrame(
            {
                "Date": equity_curve.index,                         # same length as values
                "Value": equity_curve.values,
                "Type": ["Historical"] * len(equity_curve),         # list with same length
            }
        )

        # Future dates for the projected path
        future_dates = pd.bdate_range(
            start=equity_curve.index[-1], 
            periods=len(sim["mean_portfolio_path"])
        )

        fut_df = pd.DataFrame(
            {
                "Date": future_dates,
                "Value": sim["mean_portfolio_path"].values,
                "Type": ["Projected mean"] * len(sim["mean_portfolio_path"]),
            }
        )

        chart_df = pd.concat([hist_df, fut_df], ignore_index=True)


        chart_port = (
            alt.Chart(chart_df)
            .mark_line()
            .encode(
                x=alt.X("Date:T", title="Date"),
                y=alt.Y("Value:Q", title="Portfolio value (XOF)"),
                color=alt.Color("Type:N", title="Series"),
            )
            .properties(height=350)
            .interactive()
        )
        st.altair_chart(chart_port, use_container_width=True)

        # Predicted daily prices (mean) per asset
        st.markdown("**Predicted daily prices per asset (mean over simulations)**")

        future_assets = st.multiselect(
            "Assets to display (forecasted prices)",
            options=universe_tickers,
            default=universe_tickers[: min(3, len(universe_tickers))],
            key="future_assets_multiselect",
        )

        if future_assets:
            df_list = []
            for asset in future_assets:
                series = sim["mean_price_paths"][asset]
                df_tmp = pd.DataFrame(
                    {
                        "Day": series.index,
                        "Price": series.values,
                        "Asset": asset,
                    }
                )
                df_list.append(df_tmp)

            df_future_prices = pd.concat(df_list, axis=0)
            df_future_prices["Date"] = hist_df["Date"].iloc[-1] + pd.to_timedelta(
                df_future_prices["Day"], unit="D"
            )

            chart_future_prices = (
                alt.Chart(df_future_prices)
                .mark_line()
                .encode(
                    x=alt.X("Date:T", title="Date"),
                    y=alt.Y("Price:Q", title="Predicted price (mean)"),
                    color=alt.Color("Asset:N", title="Asset"),
                )
                .properties(height=350)
                .interactive()
            )
            st.altair_chart(chart_future_prices, use_container_width=True)

        st.markdown("### Monthly rebalancing plan (expected trades)")
        if rebal_plan is None or rebal_plan.empty:
            st.info("No planned rebalancing trades over the projection horizon.")
        else:
            st.dataframe(
                rebal_plan.style.format(
                    {
                        "Trade_value_XOF": "{:,.0f}",
                        "Fees_XOF": "{:,.0f}",
                        "Net_cash_flow_XOF": "{:,.0f}",
                        "Portfolio_value_after_XOF": "{:,.0f}",
                    }
                )
            )

        # Risk summary
        df_risk = pd.DataFrame(
            {
                "Current portfolio value (XOF)": [start_value],
                f"Expected value in {horizon_years:.1f} years (XOF)": [
                    sim["final_expected"]
                ],
                f"VaR {int(var_level * 100)}% (XOF)": [sim["VaR"]],
                f"CVaR {int(var_level * 100)}% (XOF)": [sim["CVaR"]],
                "Annualised Sharpe": [risk_measures["sharpe"]],
                "Annualised Sortino": [risk_measures["sortino"]],
            }
        )

        df_risk["Expected return over horizon (%)"] = (
            df_risk[f"Expected value in {horizon_years:.1f} years (XOF)"]
            / df_risk["Current portfolio value (XOF)"]
            - 1.0
        )
        df_risk[f"VaR {int(var_level * 100)}% (%)"] = (
            df_risk[f"VaR {int(var_level * 100)}% (XOF)"]
            / df_risk["Current portfolio value (XOF)"]
        )
        df_risk[f"CVaR {int(var_level * 100)}% (%)"] = (
            df_risk[f"CVaR {int(var_level * 100)}% (XOF)"]
            / df_risk["Current portfolio value (XOF)"]
        )

        st.dataframe(
            df_risk.style.format(
                {
                    "Current portfolio value (XOF)": "{:,.0f}",
                    f"Expected value in {horizon_years:.1f} years (XOF)": "{:,.0f}",
                    f"VaR {int(var_level * 100)}% (XOF)": "{:,.0f}",
                    f"CVaR {int(var_level * 100)}% (XOF)": "{:,.0f}",
                    "Expected return over horizon (%)": "{:.2%}",
                    f"VaR {int(var_level * 100)}% (%)": "{:.2%}",
                    f"CVaR {int(var_level * 100)}% (%)": "{:.2%}",
                    "Annualised Sharpe": "{:.2f}",
                    "Annualised Sortino": "{:.2f}",
                }
            )
        )

    # ------------------------------------------------------
    # TAB 4: TECHNICAL INDICATORS & SIGNALS
    # ------------------------------------------------------
    with tab_tech:
        st.subheader("Technical indicators and trading signals (historical)")

        asset_sig = st.selectbox(
            "Asset for technical analysis",
            options=universe_tickers,
            index=0,
        )

        price_series = prices_uni[asset_sig]
        tech_df = compute_indicators(price_series)

        # Drop early NaNs in indicators
        tech_df_clean = tech_df.dropna(subset=["RSI"])

        if tech_df_clean.empty:
            st.info("Not enough data to compute indicators for this asset.")
        else:
            last_row = tech_df_clean.iloc[-1]
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Last close", f"{last_row['Price']:.0f} XOF")
            c2.metric("RSI (14)", f"{last_row['RSI']:.1f}")
            c3.metric("MACD", f"{last_row['MACD']:.4f}")
            c4.metric("Current signal", last_row["Signal"])

            tech_plot = tech_df_clean.reset_index()
            # Ensure we have a 'date' column for Altair
            if "date" not in tech_plot.columns:
                tech_plot.rename(columns={tech_plot.columns[0]: "date"}, inplace=True)

            # Price + moving averages + Buy/Sell markers
            price_ma_long = tech_plot.melt(
                id_vars="date",
                value_vars=["Price", "MA_short", "MA_long"],
                var_name="Series",
                value_name="Value",
            )
            price_ma_chart = (
                alt.Chart(price_ma_long)
                .mark_line()
                .encode(
                    x="date:T",
                    y="Value:Q",
                    color="Series:N",
                )
                .properties(height=300)
                .interactive()
            )

            sig_points = tech_plot[tech_plot["Signal"].isin(["Buy", "Sell"])]
            if not sig_points.empty:
                sig_chart = (
                    alt.Chart(sig_points)
                    .mark_point(size=80, filled=True)
                    .encode(
                        x="date:T",
                        y="Price:Q",
                        color="Signal:N",
                        shape="Signal:N",
                    )
                )
                price_ma_chart = price_ma_chart + sig_chart

            st.altair_chart(price_ma_chart, use_container_width=True)

            # RSI chart
            rsi_chart = (
                alt.Chart(tech_plot)
                .mark_line()
                .encode(
                    x="date:T",
                    y=alt.Y("RSI:Q", title="RSI (14)"),
                )
                .properties(height=150)
                .interactive()
            )
            st.altair_chart(rsi_chart, use_container_width=True)

            # MACD chart
            macd_long = tech_plot.melt(
                id_vars="date",
                value_vars=["MACD", "MACD_signal"],
                var_name="Series",
                value_name="Value",
            )
            macd_chart = (
                alt.Chart(macd_long)
                .mark_line()
                .encode(
                    x="date:T",
                    y="Value:Q",
                    color="Series:N",
                )
                .properties(height=150)
                .interactive()
            )
            st.altair_chart(macd_chart, use_container_width=True)


if __name__ == "__main__":
    main()
