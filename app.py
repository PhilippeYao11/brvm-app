import numpy as np
import pandas as pd
import altair as alt
import streamlit as st

from config import RAW_DIR
from core.data_loader import (
    build_prices_from_sika_folder,
    build_prices_from_uploaded,
    slice_history,
    compute_returns,
    compute_asset_stats,
)
from core.portfolio import (
    min_variance_target_return_portfolio,   # toujours dispo même si non utilisé
    max_return_target_vol_portfolio,
    max_sharpe_portfolio,
    equal_risk_contribution,
    portfolio_returns_series,
    portfolio_equity_curve,
    compute_discrete_allocation,
)
from core.models import simulate_prices_and_portfolio
from core.risk import compute_sharpe_sortino
from core.technical import compute_indicators
from core.rebalancing import build_monthly_rebalancing_plan


def main():
    st.set_page_config(
        page_title="BRVM Quant – Global Portfolio",
        page_icon="📊",
        layout="wide",
    )

    # --- Light blue theme ---
    st.markdown(
        """
        <style>
        .stApp { background-color: #e0f2fe; color: #0f172a; }
        .block-container { padding-top: 1.2rem; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.title("📊 BRVM Quant – Global Portfolio")
    st.caption(
        "Cette application vous aide à construire un portefeuille BRVM diversifié. "
        "Les calculs sont réalisés par un moteur quantitatif en arrière-plan. "
        "Les résultats sont indicatifs et ne constituent pas un conseil en investissement."
    )

    # =====================================================
    # Sidebar
    # =====================================================
    st.sidebar.header("Investment settings")

    initial_capital = st.sidebar.number_input(
        "Investment amount (XOF)",
        min_value=1.0,
        value=1_000_000.0,
        step=50_000.0,
    )

    transaction_fee_pct = st.sidebar.number_input(
        "Estimated trading cost per trade (%)",
        min_value=0.0,
        max_value=5.0,
        value=0.5,
        step=0.1,
    )
    fee_rate = transaction_fee_pct / 100.0

    st.sidebar.subheader("Portfolio optimisation")

    allow_short = st.sidebar.checkbox(
        "Advanced: allow short selling",
        value=False,
        help=(
            "Laissez décoché pour un portefeuille classique 'achat uniquement'. "
            "À cocher uniquement pour des usages professionnels."
        ),
    )

    max_weight_pct = st.sidebar.slider(
        "Max % of capital in a single asset",
        min_value=5.0,
        max_value=100.0,
        value=40.0,
        step=5.0,
        help="Limite la concentration sur un seul titre (en mode achat uniquement).",
    )
    max_weight = max_weight_pct / 100.0

    lookback_choice = st.sidebar.selectbox(
        "History used for the analysis",
        ("All history", "Last 6 months", "Last 12 months", "Last 24 months"),
        index=0,
    )

    n_assets = st.sidebar.slider(
        "Number of assets in the portfolio",
        min_value=3,
        max_value=15,
        value=5,
        step=1,
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

    # =====================================================
    # Load and prepare data
    # =====================================================
    if uploaded_files:
        data_long, prices_wide, volumes_wide = build_prices_from_uploaded(uploaded_files)
        st.info(f"✅ Data loaded from uploaded files ({len(uploaded_files)} CSV).")
    else:
        data_long, prices_wide, volumes_wide = build_prices_from_sika_folder(RAW_DIR)
        st.info(f"✅ Data loaded from folder: `{RAW_DIR}`")

    label_to_choice = {
        "All history": "Tout",
        "Last 6 months": "6 derniers mois",
        "Last 12 months": "12 derniers mois",
        "Last 24 months": "24 derniers mois",
    }
    prices_wide, volumes_wide = slice_history(
        prices_wide, volumes_wide, label_to_choice[lookback_choice]
    )

    returns = compute_returns(prices_wide)
    asset_stats = compute_asset_stats(returns)

    if len(asset_stats) < 2:
        st.error("You need at least 2 assets to build a portfolio.")
        return

    # =====================================================
    # Asset universe selection
    # =====================================================
    # Portefeuille orienté performance : on garde les titres au meilleur rendement historique
    universe_tickers = (
        asset_stats.sort_values("mu_ann", ascending=False)
        .head(n_assets)
        .index
        .tolist()
    )

    prices_uni = prices_wide[universe_tickers]
    returns_uni = returns[universe_tickers]
    stats_uni = asset_stats.loc[universe_tickers]

    # Last historical date (used to build future calendar dates)
    last_hist_date = prices_uni.index[-1]

    # =====================================================
    # Optimisation (moteur "meilleur compromis rendement/risque")
    # =====================================================
    w_opt, mu_opt, sigma_opt, sharpe_theo = max_sharpe_portfolio(
        returns_uni,
        universe_tickers,
        allow_short=allow_short,
        max_weight=None if allow_short else max_weight,
    )

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

    # =====================================================
    # Tabs
    # =====================================================
    tab_data, tab_portfolio, tab_models, tab_proj, tab_tech = st.tabs(
        [
            "📊 Market overview",
            "🧮 Portfolio",
            "🔍 Strategy",
            "🔮 Projections & rebalancing",
            "📈 Signals",
        ]
    )

    # -----------------------------------------------------
    # Tab 1 - Data
    # -----------------------------------------------------
    with tab_data:
        st.subheader("Selected investment universe")

        c1, c2, c3 = st.columns(3)
        c1.metric("Assets in universe", f"{len(universe_tickers)}")
        c2.metric("First date", str(prices_uni.index.min().date()))
        c3.metric("Last date", str(prices_uni.index.max().date()))

        st.markdown("**Key figures per asset**")
        st.dataframe(
            stats_uni[["mu_ann", "sigma_ann", "mu_3m"]]
            .rename(
                columns={
                    "mu_ann": "Average yearly return",
                    "sigma_ann": "Yearly risk (volatility)",
                    "mu_3m": "Return over last 3 months",
                }
            )
            .style.format("{:.2%}")
        )

        st.subheader("Price history")
        assets_to_plot = st.multiselect(
            "Assets to display:",
            options=universe_tickers,
            default=universe_tickers[: min(5, len(universe_tickers))],
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
                .encode(x="date:T", y="Price:Q", color="Asset:N")
                .properties(height=350)
                .interactive()
            )
            st.altair_chart(chart_prices, use_container_width=True)

    # -----------------------------------------------------
    # Tab 2 - Portfolio
    # -----------------------------------------------------
    with tab_portfolio:
        st.subheader("Portfolio summary")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Optimisation engine", "Balanced risk / return")
        c2.metric("Expected yearly return", f"{mu_opt:.2%}")
        c3.metric("Yearly risk (volatility)", f"{sigma_opt:.2%}")
        c4.metric(
            "Risk/return score",
            f"{risk_measures['sharpe']:.2f}"
            if not np.isnan(risk_measures["sharpe"])
            else "N/A",
        )

        df_weights = pd.DataFrame(
            {"Weight": w_opt, "Amount_XOF": w_opt * initial_capital}
        )
        st.markdown("**Target allocation and invested amounts**")
        st.dataframe(
            df_weights[df_weights["Weight"] > 1e-4]
            .sort_values("Weight", ascending=False)
            .style.format({"Weight": "{:.2%}", "Amount_XOF": "{:,.0f}"})
        )

        st.markdown("**Number of shares to buy (rounded)**")
        if discrete_alloc is not None:
            st.caption(
                f"Estimated fees ≈ {total_fees:,.0f} XOF | Remaining cash ≈ {leftover_cash:,.0f} XOF"
            )
            st.dataframe(discrete_alloc.style.format("{:,.0f}"))

        st.markdown("**Portfolio value over time (including cash)**")
        st.line_chart(equity_curve, height=280)

    # -----------------------------------------------------
    # Tab 3 - Strategy (high-level explanation)
    # -----------------------------------------------------
    with tab_models:
        st.subheader("How the engine works (high level)")
        st.info(
            "Le moteur combine l'historique des prix pour construire un portefeuille "
            "diversifié qui cherche un bon compromis entre rendement et risque. "
            "Il utilise des modèles statistiques pour simuler des scénarios futurs et "
            "estimer l’évolution possible du portefeuille. "
            "Ces résultats restent des projections théoriques et ne sont pas garantis."
        )

    # -----------------------------------------------------
    # Tab 4 - Projections + rebalancing
    # -----------------------------------------------------
    with tab_proj:
        st.subheader("Price & portfolio projections")

        horizon_years = st.slider(
            "Projection horizon (years)",
            min_value=0.5,
            max_value=10.0,
            value=1.0,
            step=0.5,
        )
        horizon_days = int(252 * horizon_years)

        # On démarre la projection à partir de la dernière valeur observée du portefeuille
        sim_res = simulate_prices_and_portfolio(
            prices_wide=prices_uni,
            returns=returns_uni,
            weights=w_opt,
            horizon_days=horizon_days,
            start_portfolio_value=float(equity_curve.iloc[-1]),
        )

        # ---- Portfolio path with DATE index ----
        port_path = sim_res["mean_portfolio_path"].copy()
        future_dates = pd.bdate_range(
            start=last_hist_date, periods=len(port_path)
        )
        port_path.index = future_dates

        # HISTORIQUE + PROJECTION SUR LE MÊME GRAPHE
        df_port_hist = pd.DataFrame(
            {
                "Date": equity_curve.index,
                "Value": equity_curve.values,
                "Type": "Historical",
            }
        )
        df_port_proj = pd.DataFrame(
            {
                "Date": port_path.index,
                "Value": port_path.values,
                "Type": "Projected",
            }
        )
        df_port = pd.concat([df_port_hist, df_port_proj], ignore_index=True)

        st.markdown("**Portfolio value: history and projection**")
        chart_port = (
            alt.Chart(df_port)
            .mark_line()
            .encode(
                x="Date:T",
                y="Value:Q",
                color="Type:N",
            )
            .properties(height=320)
            .interactive()
        )
        st.altair_chart(chart_port, use_container_width=True)
        st.caption(
            f"Projected portfolio value at the end of the horizon: {sim_res['final_expected']:.0f} XOF "
            "(non-guaranteed scenario)."
        )

        # ---- Projected prices with DATE index ----
        st.subheader("Prices: history and projection")
        assets_to_plot = st.multiselect(
            "Assets to display:",
            options=universe_tickers,
            default=universe_tickers[: min(3, len(universe_tickers))],
        )
        if assets_to_plot:
            # Historique
            df_hist = (
                prices_uni[assets_to_plot]
                .reset_index()
                .melt(id_vars="date", var_name="Asset", value_name="Price")
                .rename(columns={"date": "Date"})
            )
            df_hist["Type"] = "Historical"

            # Projections
            df_proj_list = []
            for a in assets_to_plot:
                s = sim_res["mean_price_paths"][a]
                dates_a = pd.bdate_range(start=last_hist_date, periods=len(s))
                df_proj_list.append(
                    pd.DataFrame(
                        {"Date": dates_a, "Asset": a, "Price": s.values, "Type": "Projected"}
                    )
                )
            df_proj_all = pd.concat(df_proj_list, ignore_index=True)

            df_prices_all = pd.concat([df_hist, df_proj_all], ignore_index=True)

            chart_prices = (
                alt.Chart(df_prices_all)
                .mark_line()
                .encode(
                    x="Date:T",
                    y="Price:Q",
                    color="Asset:N",
                    strokeDash="Type:N",
                )
                .properties(height=350)
                .interactive()
            )
            st.altair_chart(chart_prices, use_container_width=True)

        # ---- Rebalancing plan ----
        st.subheader("Rebalancing plan based on projections")
        strategy_choice = st.selectbox(
            "Rebalancing strategy",
            ["Full", "Threshold-based"],
            index=0,
        )
        if strategy_choice == "Full":
            strategy = "full"
            threshold = 0.0
        else:
            strategy = "threshold"
            threshold = (
                st.slider(
                    "Threshold for deviation from target weights (%)",
                    0.0,
                    10.0,
                    2.0,
                    0.5,
                )
                / 100.0
            )

        rebal_freq = st.slider(
            "Rebalancing frequency (days)",
            5,
            63,
            21,
            1,
        )

        rebal_plan = build_monthly_rebalancing_plan(
            prices_uni=prices_uni,
            mean_price_paths=sim_res["mean_price_paths"],
            weights=w_opt,
            discrete_alloc=discrete_alloc,
            initial_cash=leftover_cash,
            fee_rate=fee_rate,
            horizon_days=horizon_days,
            rebalance_every_days=rebal_freq,
            strategy=strategy,
            threshold_weight_diff=threshold,
        )

        if rebal_plan.empty:
            st.info("No rebalancing trades planned over the selected horizon.")
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

    # -----------------------------------------------------
    # Tab 5 - Technical signals (on projected prices)
    # -----------------------------------------------------
    with tab_tech:
        st.subheader("Signals based on projected prices")

        # We reuse sim_res computed in Tab 4
        asset_sig = st.selectbox("Select asset:", options=universe_tickers, index=0)

        proj_prices = sim_res["mean_price_paths"][asset_sig].copy()
        future_dates = pd.bdate_range(
            start=last_hist_date, periods=len(proj_prices)
        )
        proj_prices.index = future_dates

        tech_df = compute_indicators(proj_prices).dropna(subset=["RSI"])

        if tech_df.empty:
            st.info("Not enough data for indicators.")
        else:
            last_row = tech_df.iloc[-1]
            last_date = tech_df.index[-1]
            signal = str(last_row["Signal"])
            date_str = last_date.strftime("%Y-%m-%d")

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Last projected close", f"{last_row['Price']:.0f} XOF")
            c2.metric("RSI (14)", f"{last_row['RSI']:.1f}")
            c3.metric("MACD", f"{last_row['MACD']:.4f}")
            c4.metric("Current signal", signal)

            if signal == "Buy":
                msg = f"**Suggested action:** BUY {asset_sig} on **{date_str}** (based on projected indicators)."
            elif signal == "Sell":
                msg = f"**Suggested action:** SELL {asset_sig} on **{date_str}** (based on projected indicators)."
            else:  # Hold
                msg = (
                    f"**Suggested action:** HOLD your position in {asset_sig} "
                    f"at least until **{date_str}** (no strong Buy/Sell signal)."
                )
            st.markdown(msg)

            st.markdown("**Projected price and moving averages**")
            st.line_chart(tech_df[["Price", "MA_short", "MA_long"]], height=300)

            st.markdown("**Projected RSI (14)**")
            st.line_chart(tech_df["RSI"], height=150)

            st.markdown("**Projected MACD and signal line**")
            st.line_chart(tech_df[["MACD", "MACD_signal"]], height=150)


if __name__ == "__main__":
    main()
