# app.py
import io
import json
import numpy as np
import pandas as pd
import altair as alt
import streamlit as st

import matplotlib.pyplot as plt

from config import RAW_DIR, ANNUAL_FACTOR, DEFAULT_VAR_LEVEL, DEFAULT_MC_SIMS
from core.data_loader import (
    build_prices_from_sika_folder,
    build_prices_from_uploaded,
    slice_history,
    compute_returns,
    compute_asset_stats,
    update_history_from_sika_online,
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

# PDF (reportlab)
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage, Table, TableStyle, PageBreak
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors


# ======================================================
# Backtesting helper (ARIMA 1-day ahead on LOG-RETURNS)
# ======================================================
def backtest_portfolio_arima_logret_one_step(
    prices_uni: pd.DataFrame,
    weights: pd.Series,
    train_window: int = 252,
    test_days: int = 60,
    order: tuple = (1, 0, 0),
):
    """
    Rolling 1-day-ahead ARIMA backtest on LOG-RETURNS (portfolio level).
    Predict next-day returns using ARIMA per asset on log-returns.

    Returns:
      df_bt: Pred_ret, Real_ret, Pred_equity, Real_equity
      metrics: MAE, RMSE, Directional accuracy
    """
    try:
        from statsmodels.tsa.arima.model import ARIMA
    except Exception as e:
        raise ImportError("statsmodels is required for ARIMA backtesting.") from e

    tickers = list(weights.index)

    px = prices_uni[tickers].ffill().dropna(how="any")
    if len(px) < train_window + 10:
        raise ValueError("Not enough data for the chosen train_window.")

    w = weights.copy()
    if w.sum() <= 0:
        raise ValueError("Weights sum to zero.")
    w = w / w.sum()

    logp = np.log(px.replace(0.0, np.nan)).dropna(how="any")
    lr = logp.diff().dropna(how="any")  # log returns

    if len(lr) < train_window + 5:
        raise ValueError("Not enough log-return history after cleaning.")

    test_days = int(max(10, test_days))
    end_idx = len(lr) - 1
    start_test_idx = max(train_window, end_idx - test_days)

    out_dates, pred_rets, real_rets = [], [], []

    # initial fit per asset
    results = {}
    for t in tickers:
        train = lr[t].iloc[start_test_idx - train_window : start_test_idx].dropna()
        if len(train) < 60:
            results[t] = None
            continue
        try:
            results[t] = ARIMA(
                train,
                order=order,
                trend="n",
                enforce_stationarity=False,
                enforce_invertibility=False,
            ).fit()
        except Exception:
            results[t] = None

    for i in range(start_test_idx, end_idx):
        pred_asset_ret = {}

        for t in tickers:
            res = results.get(t)
            if res is None:
                pred_asset_ret[t] = 0.0
                continue
            try:
                pred_lr = float(res.forecast(steps=1).iloc[0])
                pred_asset_ret[t] = float(np.exp(pred_lr) - 1.0)
            except Exception:
                pred_asset_ret[t] = 0.0

        pred_port_ret = float(sum(w[t] * pred_asset_ret[t] for t in tickers))

        real_lr_vec = lr.iloc[i + 1][tickers]
        real_r_vec = np.exp(real_lr_vec) - 1.0
        real_port_ret = float((real_r_vec.values * w.values).sum())

        pred_rets.append(pred_port_ret)
        real_rets.append(real_port_ret)
        out_dates.append(lr.index[i + 1])

        # update models with realized obs (append if possible, else refit)
        for t in tickers:
            res = results.get(t)
            if res is None:
                continue
            try:
                new_obs = pd.Series([float(lr[t].iloc[i + 1])], index=[lr.index[i + 1]])
                results[t] = res.append(new_obs, refit=False)
            except Exception:
                train = lr[t].iloc[i + 1 - train_window + 1 : i + 2].dropna()
                if len(train) >= 60:
                    try:
                        results[t] = ARIMA(
                            train,
                            order=order,
                            trend="n",
                            enforce_stationarity=False,
                            enforce_invertibility=False,
                        ).fit()
                    except Exception:
                        results[t] = None
                else:
                    results[t] = None

    df_bt = pd.DataFrame({"Date": out_dates, "Pred_ret": pred_rets, "Real_ret": real_rets}).set_index("Date")

    err = df_bt["Pred_ret"] - df_bt["Real_ret"]
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err**2)))
    dir_acc = float(np.mean(np.sign(df_bt["Pred_ret"]) == np.sign(df_bt["Real_ret"])))

    df_bt["Pred_equity"] = (1.0 + df_bt["Pred_ret"]).cumprod()
    df_bt["Real_equity"] = (1.0 + df_bt["Real_ret"]).cumprod()

    metrics = {"MAE": mae, "RMSE": rmse, "Directional accuracy": dir_acc}
    return df_bt, metrics


# ======================================================
# Projected technical signals helper (thresholds exposed)
# ======================================================
def compute_projected_signal_last(
    full_price: pd.Series,
    short_ma: int,
    long_ma: int,
    rsi_period: int,
    macd_fast: int,
    macd_slow: int,
    macd_signal: int,
    rsi_buy_below: float,
    rsi_sell_above: float,
) -> str:
    """
    Compute indicators using existing compute_indicators, then apply
    thresholds (projected signal) on the last row.
    """
    df = compute_indicators(
        full_price,
        short_ma=short_ma,
        long_ma=long_ma,
        rsi_period=rsi_period,
        macd_fast=macd_fast,
        macd_slow=macd_slow,
        macd_signal=macd_signal,
    ).dropna(subset=["RSI", "MACD", "MACD_signal"])

    if df.empty:
        return "N/A"

    last = df.iloc[-1]
    prev = df.iloc[-2] if len(df) >= 2 else last

    macd_cross_up = (last["MACD"] > last["MACD_signal"]) and (prev["MACD"] <= prev["MACD_signal"])
    macd_cross_down = (last["MACD"] < last["MACD_signal"]) and (prev["MACD"] >= prev["MACD_signal"])

    if macd_cross_up and float(last["RSI"]) <= rsi_buy_below:
        return "Buy"
    if macd_cross_down and float(last["RSI"]) >= rsi_sell_above:
        return "Sell"
    return "Hold"


# ======================================================
# PDF report generation
# ======================================================
def _fig_to_png_bytes(fig) -> bytes:
    bio = io.BytesIO()
    fig.savefig(bio, format="png", bbox_inches="tight", dpi=160)
    plt.close(fig)
    return bio.getvalue()


def generate_pdf_report(
    title: str,
    prices_uni: pd.DataFrame,
    equity_curve: pd.Series,
    w_opt: pd.Series,
    mu_opt: float,
    sigma_opt: float,
    risk_measures: dict,
    var_level: float,
    sim: dict,
    tech_asset: str,
    tech_params: dict,
) -> bytes:
    """
    Create a PDF report (bytes) with graphs and explanations.
    """
    styles = getSampleStyleSheet()
    styleN = styles["BodyText"]
    styleH = styles["Heading1"]
    styleH2 = styles["Heading2"]

    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, rightMargin=36, leftMargin=36, topMargin=36, bottomMargin=36)

    story = []
    story.append(Paragraph(title, styleH))
    story.append(Spacer(1, 10))
    story.append(Paragraph(
        "This report summarises the portfolio optimisation, historical performance, projected scenarios, "
        "risk metrics (VaR/CVaR) and technical indicators usage.", styleN
    ))
    story.append(Spacer(1, 12))

    # Summary table
    summary_data = [
        ["Optimised expected annual return", f"{mu_opt:.2%}"],
        ["Optimised expected annual volatility", f"{sigma_opt:.2%}"],
        ["Historical Sharpe (annualised)", f"{risk_measures.get('sharpe', np.nan):.2f}"],
        ["Historical Sortino (annualised)", f"{risk_measures.get('sortino', np.nan):.2f}"],
        [f"VaR ({int(var_level*100)}%)", f"{sim.get('VaR', np.nan):,.0f} XOF"],
        [f"CVaR ({int(var_level*100)}%)", f"{sim.get('CVaR', np.nan):,.0f} XOF"],
        ["Projection model", str(sim.get("model_used", "ARIMA"))],
    ]
    t = Table(summary_data, colWidths=[260, 240])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), colors.whitesmoke),
        ("GRID", (0,0), (-1,-1), 0.25, colors.grey),
        ("FONTNAME", (0,0), (-1,-1), "Helvetica"),
        ("FONTSIZE", (0,0), (-1,-1), 9),
        ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
    ]))
    story.append(Paragraph("Key metrics", styleH2))
    story.append(t)
    story.append(Spacer(1, 12))

    # Weights table
    w_tbl = [["Asset", "Weight"]] + [[idx, f"{val:.2%}"] for idx, val in w_opt.sort_values(ascending=False).items() if val > 1e-6]
    wt = Table(w_tbl, colWidths=[260, 240])
    wt.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), colors.whitesmoke),
        ("GRID", (0,0), (-1,-1), 0.25, colors.grey),
        ("FONTSIZE", (0,0), (-1,-1), 9),
    ]))
    story.append(Paragraph("Optimised weights", styleH2))
    story.append(wt)
    story.append(Spacer(1, 12))

    # Figure: historical equity
    fig1 = plt.figure()
    plt.plot(equity_curve.index, equity_curve.values)
    plt.title("Historical portfolio value (including cash)")
    plt.xlabel("Date")
    plt.ylabel("Value (XOF)")
    img1 = _fig_to_png_bytes(fig1)
    story.append(Paragraph("Historical performance", styleH2))
    story.append(RLImage(io.BytesIO(img1), width=500, height=280))
    story.append(Spacer(1, 12))

    # Figure: portfolio projection (historical + projected)
    proj = sim.get("portfolio_path", None)
    if proj is not None and len(proj) > 1:
        future_dates = pd.bdate_range(start=equity_curve.index[-1], periods=len(proj))
        fig2 = plt.figure()
        plt.plot(equity_curve.index, equity_curve.values, label="Historical")
        plt.plot(future_dates, proj.values, label="Projected (ARIMA path)")
        plt.title("Portfolio projection (historical + projected)")
        plt.xlabel("Date")
        plt.ylabel("Value (XOF)")
        plt.legend()
        img2 = _fig_to_png_bytes(fig2)
        story.append(Paragraph("Projections", styleH2))
        story.append(RLImage(io.BytesIO(img2), width=500, height=280))
        story.append(Spacer(1, 12))

    # Technical indicators explanation + plot
    story.append(Paragraph("How to use technical indicators (RSI / MACD / Moving Averages)", styleH2))
    story.append(Paragraph(
        "RSI is a momentum indicator (0–100). Typical interpretation: "
        "oversold when RSI is low (often ~30) and overbought when RSI is high (often ~70). "
        "MACD crossovers are often used as trend/momentum signals; "
        "a bullish crossover occurs when MACD crosses above its signal line, and bearish is the opposite. "
        "Moving averages provide trend context (short vs long).", styleN
    ))
    story.append(Spacer(1, 10))

    if tech_asset in prices_uni.columns:
        hist_price = prices_uni[tech_asset].dropna()
        proj_price = sim.get("price_paths", {}).get(tech_asset, None)

        if proj_price is not None and len(proj_price) > 1:
            proj_dates = pd.bdate_range(start=hist_price.index[-1], periods=len(proj_price))
            full_series = pd.concat(
                [hist_price, pd.Series(proj_price.values, index=proj_dates)],
                axis=0
            )
        else:
            full_series = hist_price.copy()

        tech_df = compute_indicators(
            full_series,
            short_ma=tech_params["short_ma"],
            long_ma=tech_params["long_ma"],
            rsi_period=tech_params["rsi_period"],
            macd_fast=tech_params["macd_fast"],
            macd_slow=tech_params["macd_slow"],
            macd_signal=tech_params["macd_signal"],
        ).dropna(subset=["RSI"])

        fig3 = plt.figure()
        plt.plot(tech_df.index, tech_df["Price"].values, label="Price")
        plt.plot(tech_df.index, tech_df["MA_short"].values, label="MA short")
        plt.plot(tech_df.index, tech_df["MA_long"].values, label="MA long")
        plt.title(f"{tech_asset} – Price + Moving Averages (historical + projected)")
        plt.xlabel("Date")
        plt.ylabel("Price")
        plt.legend()
        img3 = _fig_to_png_bytes(fig3)
        story.append(RLImage(io.BytesIO(img3), width=500, height=280))
        story.append(Spacer(1, 10))

    story.append(Paragraph(
        f"Signal thresholds used in the app (projection signals): "
        f"RSI Buy if RSI ≤ {tech_params['rsi_buy_below']}, "
        f"RSI Sell if RSI ≥ {tech_params['rsi_sell_above']}.",
        styleN
    ))

    doc.build(story)
    return buf.getvalue()


# ======================================================
# Main app
# ======================================================
def main():
    st.set_page_config(page_title="BRVM Quant – Portfolio Lab", page_icon="📊", layout="wide")

    # Light blue theme
    st.markdown(
        """
        <style>
        .stApp { background-color: #e0f2fe; color: #0f172a; }
        .block-container { padding-top: 1.5rem; }
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

    data_source_mode = st.sidebar.radio(
        "Choose input method",
        ["Upload CSVs", "Local folder", "Online (Sika update)"],
        index=0,
    )

    uploaded_files = None
    update_online = False
    online_hint = None

    if data_source_mode == "Upload CSVs":
        uploaded_files = st.sidebar.file_uploader(
            "Raw SIKA CSV files (you can select multiple)",
            type=["csv"],
            accept_multiple_files=True,
        )
        st.sidebar.caption("Upload CSV files exported from SikaFinance.")
    elif data_source_mode == "Local folder":
        st.sidebar.caption(f"App will read CSV files from: `{RAW_DIR}`")
    else:
        # This uses your existing online update mechanism based on SIKA_URL_MAP.
        update_online = st.sidebar.checkbox("Fetch latest data online (requires SIKA_URL_MAP)", value=True)
        online_hint = st.sidebar.text_area(
            "Optional: paste a JSON dict symbol->url to override runtime (advanced)",
            value="",
            height=120,
        )
        st.sidebar.caption(
            "Online update uses SIKA_URL_MAP in core/data_loader.py. "
            "If Sika blocks automated downloads, this may fail."
        )

        # Runtime override for SIKA_URL_MAP if provided
        if online_hint.strip():
            try:
                from core import data_loader as _dl
                runtime_map = json.loads(online_hint)
                if isinstance(runtime_map, dict):
                    _dl.SIKA_URL_MAP.clear()
                    _dl.SIKA_URL_MAP.update(runtime_map)
                    st.sidebar.success("✅ Runtime SIKA_URL_MAP updated.")
            except Exception as e:
                st.sidebar.error(f"Invalid JSON / override failed: {e}")

    # ======================================================
    # Load data
    # ======================================================
    if data_source_mode == "Upload CSVs":
        if uploaded_files:
            data_long, prices_wide, volumes_wide = build_prices_from_uploaded(uploaded_files)
            st.info(f"✅ Data loaded from uploaded files ({len(uploaded_files)} CSV).")
        else:
            st.warning("Please upload at least one CSV file, or switch to another data source.")
            return
    else:
        data_long, prices_wide, volumes_wide = build_prices_from_sika_folder(RAW_DIR, update_online=False)

        if data_source_mode == "Online (Sika update)" and update_online:
            try:
                data_long = update_history_from_sika_online(data_long)
                # rebuild wide tables
                prices_wide = data_long.pivot(index="date", columns="symbol", values="close").sort_index()
                volumes_wide = data_long.pivot(index="date", columns="symbol", values="volume").sort_index()
                st.info("✅ Local data loaded + online update applied.")
            except Exception as e:
                st.warning(f"Online update failed; using local data only. Details: {e}")
        else:
            st.info(f"✅ Data loaded from folder: `{RAW_DIR}`")

    # Convert lookback label to French label used in slice_history
    label_to_choice = {
        "All history": "Tout",
        "Last 6 months": "6 derniers mois",
        "Last 12 months": "12 derniers mois",
        "Last 24 months": "24 derniers mois",
    }
    prices_wide, volumes_wide = slice_history(prices_wide, volumes_wide, label_to_choice[lookback_choice])

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
        target_return_ann = st.sidebar.number_input(
            "Target annual return (%)",
            min_value=0.0,
            max_value=100.0,
            value=10.0,
            step=1.0,
        ) / 100.0

    if opt_method == "Max return (target volatility)":
        target_vol_ann = st.sidebar.number_input(
            "Target annual volatility (%)",
            min_value=0.0,
            max_value=100.0,
            value=15.0,
            step=1.0,
        ) / 100.0

    # ======================================================
    # Run optimisation
    # ======================================================
    sharpe_theo = np.nan

    if opt_method == "Min variance":
        w_opt, mu_opt, sigma_opt = markowitz_min_variance(returns_uni, universe_tickers, allow_short=allow_short)
        mode_label = "Min variance"

    elif opt_method == "Max return":
        w_opt, mu_opt, sigma_opt = markowitz_max_return(returns_uni, universe_tickers, allow_short=allow_short)
        mode_label = "Max return"

    elif opt_method == "Min variance (target return)":
        w_opt, mu_opt, sigma_opt = markowitz_min_var_target_return(
            returns_uni, universe_tickers, target_return_ann=target_return_ann, allow_short=allow_short
        )
        mode_label = f"Min variance for target return ({target_return_ann:.0%})"

    elif opt_method == "Max return (target volatility)":
        w_opt, mu_opt, sigma_opt = markowitz_max_return_target_var(
            returns_uni, universe_tickers, target_vol_ann=target_vol_ann, allow_short=allow_short
        )
        mode_label = f"Max return for target volatility ({target_vol_ann:.0%})"

    elif opt_method == "Equal Risk Contribution (ERC)":
        w_opt, mu_opt, sigma_opt = erc_portfolio(returns_uni, universe_tickers, allow_short=allow_short)
        mode_label = "Equal Risk Contribution (ERC)"

    else:  # Max Sharpe
        w_opt, mu_opt, sigma_opt, sharpe_theo = max_sharpe_portfolio(returns_uni, universe_tickers, allow_short=allow_short)
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
        equity_invested = portfolio_equity_curve(port_ret, initial_invested)
        equity_curve = equity_invested + leftover_cash
    else:
        equity_curve = portfolio_equity_curve(port_ret, initial_capital)
        leftover_cash = 0.0
        total_fees = 0.0

    # ======================================================
    # Tabs (✅ include tab_bt to avoid your error)
    # ======================================================
    tab_data, tab_portfolio, tab_proj, tab_tech, tab_bt = st.tabs(
        ["📊 Data & universe", "🧮 Optimised portfolio", "🔮 Projections & risk", "📈 Technical signals", "🧪 Backtesting"]
    )

    # ------------------------------------------------------
    # TAB 1: DATA & UNIVERSE
    # ------------------------------------------------------
    with tab_data:
        st.subheader("Universe overview")

        c1, c2, c3 = st.columns(3)
        c1.metric("Total assets loaded", f"{len(all_tickers)}")
        c2.metric("Assets selected for optimisation", f"{len(universe_tickers)}")
        c3.metric("Date range", f"{prices_wide.index.min().date()} → {prices_wide.index.max().date()}")

        st.markdown("**Global universe statistics (all assets)**")
        st.dataframe(
            asset_stats[["mu_ann", "sigma_ann", "mu_3m"]]
            .rename(columns={"mu_ann": "Annual return", "sigma_ann": "Annual volatility", "mu_3m": "3-month return (approx.)"})
            .style.format("{:.2%}")
        )

        st.markdown("**Current optimisation universe (selected assets)**")
        st.dataframe(
            stats_uni[["mu_ann", "sigma_ann", "mu_3m"]]
            .rename(columns={"mu_ann": "Annual return", "sigma_ann": "Annual volatility", "mu_3m": "3-month return (approx.)"})
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
            df_prices = prices_uni[assets_to_plot].reset_index().melt(id_vars="date", var_name="Asset", value_name="Price")

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
            f"{risk_measures['sharpe']:.2f}" if not np.isnan(risk_measures["sharpe"]) else "N/A",
        )
        st.caption(
            "Historical Sortino ratio (annualised): "
            + (f"{risk_measures['sortino']:.2f}" if not np.isnan(risk_measures["sortino"]) else "N/A")
        )

        amounts = (w_opt * initial_capital).rename("Amount_XOF")
        df_weights = pd.concat([w_opt.rename("Weight"), amounts], axis=1)

        st.markdown("**Continuous weights and invested amounts**")
        st.dataframe(
            df_weights[df_weights["Weight"] > 1e-4].sort_values("Weight", ascending=False)
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
                    ["Target_weight", "Initial_price", "Shares_to_buy", "Trade_value_XOF", "Fees_XOF", "Total_cost_XOF", "Final_weight_no_rebal", "Weight_diff"]
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

        st.markdown("---")
        st.subheader("PDF report")
        st.caption("Generate a PDF report including charts and explanations (weights, performance, projections, VaR/CVaR, technical indicators).")

        # Report inputs
        report_title = st.text_input("Report title", value="BRVM Quant – Portfolio Analysis Report")
        tech_asset_report = st.selectbox("Asset for technical section", options=universe_tickers, index=0, key="tech_asset_report")

        # technical thresholds for report
        r1, r2, r3, r4 = st.columns(4)
        short_ma = r1.number_input("MA short", min_value=5, max_value=200, value=20, step=1, key="rep_short_ma")
        long_ma = r2.number_input("MA long", min_value=10, max_value=400, value=50, step=1, key="rep_long_ma")
        rsi_buy_below = r3.number_input("RSI buy below", min_value=0.0, max_value=100.0, value=30.0, step=1.0, key="rep_rsi_buy")
        rsi_sell_above = r4.number_input("RSI sell above", min_value=0.0, max_value=100.0, value=70.0, step=1.0, key="rep_rsi_sell")

        if st.button("Generate PDF report"):
            # run a small simulation to include VaR/CVaR and projections in report
            horizon_days_rep = 252
            sim_rep = simulate_prices_and_portfolio(
                prices_wide=prices_uni,
                returns=returns_uni,
                weights=w_opt,
                horizon_days=horizon_days_rep,
                start_portfolio_value=float(equity_curve.iloc[-1]),
                n_sims=300,
                var_level=var_level,
                arima_order=(1, 0, 0),
            )
            tech_params = dict(
                short_ma=int(short_ma),
                long_ma=int(long_ma),
                rsi_period=14,
                macd_fast=12,
                macd_slow=26,
                macd_signal=9,
                rsi_buy_below=float(rsi_buy_below),
                rsi_sell_above=float(rsi_sell_above),
            )

            pdf_bytes = generate_pdf_report(
                title=report_title,
                prices_uni=prices_uni,
                equity_curve=equity_curve,
                w_opt=w_opt,
                mu_opt=mu_opt,
                sigma_opt=sigma_opt,
                risk_measures=risk_measures,
                var_level=var_level,
                sim=sim_rep,
                tech_asset=tech_asset_report,
                tech_params=tech_params,
            )

            st.download_button(
                "Download PDF",
                data=pdf_bytes,
                file_name="brvm_portfolio_report.pdf",
                mime="application/pdf",
            )

    # ------------------------------------------------------
    # TAB 3: PROJECTIONS & RISK
    # ------------------------------------------------------
    with tab_proj:
        st.subheader("Daily projections and portfolio risk")

        horizon_years = st.slider("Projection horizon (years)", min_value=0.5, max_value=10.0, value=1.0, step=0.5)
        horizon_days = int(252 * horizon_years)

        start_value = float(equity_curve.iloc[-1])

        # ARIMA-only simulation (log-returns)
        sim = simulate_prices_and_portfolio(
            prices_wide=prices_uni,
            returns=returns_uni,
            weights=w_opt,
            horizon_days=horizon_days,
            start_portfolio_value=start_value,
            n_sims=300,
            var_level=var_level,
            arima_order=(1, 1, 1),
        )

        # Monthly rebalancing plan based on mean expected future prices
        rebal_plan = build_monthly_rebalancing_plan(
            prices_uni=prices_uni,
            mean_price_paths=sim["mean_price_paths"],
            weights=w_opt,
            discrete_alloc=discrete_alloc,
            initial_cash=leftover_cash,
            fee_rate=fee_rate,
            horizon_days=horizon_days,
            rebalance_every_days=21,
            strategy="full",
            threshold_weight_diff=0.0,
        )

        # --- Portfolio value chart: historical + projected PATH (not bands) ---
        proj_path = sim["portfolio_path"]
        future_dates = pd.bdate_range(start=equity_curve.index[-1], periods=len(proj_path))

        hist_df = pd.DataFrame({"Date": equity_curve.index, "Value": equity_curve.values, "Type": ["Historical"] * len(equity_curve)})
        fut_df = pd.DataFrame({"Date": future_dates, "Value": proj_path.values, "Type": ["Projected (ARIMA path)"] * len(proj_path)})
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

        # --- Predicted daily prices per asset (ARIMA path) + overlay with history ---
        st.markdown("**Projected prices per asset (ARIMA path) – overlaid with history**")
        future_assets = st.multiselect(
            "Assets to display (forecasted prices)",
            options=universe_tickers,
            default=universe_tickers[: min(3, len(universe_tickers))],
            key="future_assets_multiselect",
        )

        if future_assets:
            rows = []
            for asset in future_assets:
                # historical
                s_hist = prices_uni[asset].dropna()
                for d, v in s_hist.items():
                    rows.append({"Date": d, "Price": float(v), "Asset": asset, "Type": "Historical"})
                # projected
                s_proj = sim["price_paths"][asset]
                d_proj = pd.bdate_range(start=s_hist.index[-1], periods=len(s_proj))
                for d, v in zip(d_proj, s_proj.values):
                    rows.append({"Date": d, "Price": float(v), "Asset": asset, "Type": "Projected (ARIMA path)"})

            df_future_prices = pd.DataFrame(rows)

            chart_future_prices = (
                alt.Chart(df_future_prices)
                .mark_line()
                .encode(
                    x=alt.X("Date:T", title="Date"),
                    y=alt.Y("Price:Q", title="Price"),
                    color=alt.Color("Type:N", title="Series"),
                    strokeDash=alt.StrokeDash("Type:N", title="Series"),
                    facet=alt.Facet("Asset:N", columns=1, title=None),
                )
                .properties(height=180)
                .interactive()
            )
            st.altair_chart(chart_future_prices, use_container_width=True)

        # --- Projected signal thresholds (what you "watch") ---
        st.markdown("### Projected signal thresholds (what is watched to send signals)")
        cA, cB, cC, cD = st.columns(4)
        rsi_buy_below = cA.number_input("RSI buy if ≤", min_value=0.0, max_value=100.0, value=30.0, step=1.0, key="proj_rsi_buy")
        rsi_sell_above = cB.number_input("RSI sell if ≥", min_value=0.0, max_value=100.0, value=70.0, step=1.0, key="proj_rsi_sell")
        short_ma = cC.number_input("MA short", min_value=5, max_value=200, value=20, step=1, key="proj_short_ma")
        long_ma = cD.number_input("MA long", min_value=10, max_value=400, value=50, step=1, key="proj_long_ma")

        sig_asset = st.selectbox("Asset for projected signal check", options=universe_tickers, index=0, key="proj_sig_asset")
        s_hist = prices_uni[sig_asset].dropna()
        s_proj = sim["price_paths"][sig_asset]
        d_proj = pd.bdate_range(start=s_hist.index[-1], periods=len(s_proj))
        s_full = pd.concat([s_hist, pd.Series(s_proj.values, index=d_proj)])

        last_sig = compute_projected_signal_last(
            full_price=s_full,
            short_ma=int(short_ma),
            long_ma=int(long_ma),
            rsi_period=14,
            macd_fast=12,
            macd_slow=26,
            macd_signal=9,
            rsi_buy_below=float(rsi_buy_below),
            rsi_sell_above=float(rsi_sell_above),
        )
        st.info(f"Projected latest signal for **{sig_asset}** (based on historical + projected path): **{last_sig}**")

        # Rebalancing plan
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

        # Risk summary (always show VaR/CVaR)
        df_risk = pd.DataFrame(
            {
                "Current portfolio value (XOF)": [start_value],
                f"Expected value in {horizon_years:.1f} years (XOF)": [sim["final_expected"]],
                f"VaR {int(var_level * 100)}% (XOF)": [sim["VaR"]],
                f"CVaR {int(var_level * 100)}% (XOF)": [sim["CVaR"]],
                "Annualised Sharpe": [risk_measures["sharpe"]],
                "Annualised Sortino": [risk_measures["sortino"]],
            }
        )

        df_risk["Expected return over horizon (%)"] = (
            df_risk[f"Expected value in {horizon_years:.1f} years (XOF)"] / df_risk["Current portfolio value (XOF)"] - 1.0
        )
        df_risk[f"VaR {int(var_level * 100)}% (%)"] = df_risk[f"VaR {int(var_level * 100)}% (XOF)"] / df_risk["Current portfolio value (XOF)"]
        df_risk[f"CVaR {int(var_level * 100)}% (%)"] = df_risk[f"CVaR {int(var_level * 100)}% (XOF)"] / df_risk["Current portfolio value (XOF)"]

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

        asset_sig = st.selectbox("Asset for technical analysis", options=universe_tickers, index=0)

        price_series = prices_uni[asset_sig]
        tech_df = compute_indicators(price_series)

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
            if "date" not in tech_plot.columns:
                tech_plot.rename(columns={tech_plot.columns[0]: "date"}, inplace=True)

            price_ma_long = tech_plot.melt(
                id_vars="date",
                value_vars=["Price", "MA_short", "MA_long"],
                var_name="Series",
                value_name="Value",
            )
            price_ma_chart = (
                alt.Chart(price_ma_long)
                .mark_line()
                .encode(x="date:T", y="Value:Q", color="Series:N")
                .properties(height=300)
                .interactive()
            )

            sig_points = tech_plot[tech_plot["Signal"].isin(["Buy", "Sell"])]
            if not sig_points.empty:
                sig_chart = (
                    alt.Chart(sig_points)
                    .mark_point(size=80, filled=True)
                    .encode(x="date:T", y="Price:Q", color="Signal:N", shape="Signal:N")
                )
                price_ma_chart = price_ma_chart + sig_chart

            st.altair_chart(price_ma_chart, use_container_width=True)

            rsi_chart = (
                alt.Chart(tech_plot)
                .mark_line()
                .encode(x="date:T", y=alt.Y("RSI:Q", title="RSI (14)"))
                .properties(height=150)
                .interactive()
            )
            st.altair_chart(rsi_chart, use_container_width=True)

            macd_long = tech_plot.melt(
                id_vars="date",
                value_vars=["MACD", "MACD_signal"],
                var_name="Series",
                value_name="Value",
            )
            macd_chart = (
                alt.Chart(macd_long)
                .mark_line()
                .encode(x="date:T", y="Value:Q", color="Series:N")
                .properties(height=150)
                .interactive()
            )
            st.altair_chart(macd_chart, use_container_width=True)

    # ------------------------------------------------------
    # TAB 5: BACKTESTING
    # ------------------------------------------------------
    with tab_bt:
        st.subheader("Backtesting (ARIMA – 1-day ahead) on returns")

        c1, c2, c3, c4 = st.columns(4)
        train_window = c1.number_input("Train window (days)", min_value=60, max_value=1500, value=252, step=21)
        test_days = c2.number_input("Test length (days)", min_value=20, max_value=400, value=60, step=10)

        order_choice = c3.selectbox(
            "ARIMA order on log-returns",
            options=[(1, 1, 1), (1, 0, 0), (1, 0, 1)],
            index=1,
            format_func=lambda x: f"ARIMA{x}",
        )

        run_bt = c4.button("Run backtest")

        st.caption(
            "Forecasts **next-day returns** using ARIMA on log-returns per asset, "
            "then compares predicted vs realized portfolio returns."
        )

        if run_bt:
            with st.spinner("Running backtest..."):
                try:
                    df_bt, metrics = backtest_portfolio_arima_logret_one_step(
                        prices_uni=prices_uni,
                        weights=w_opt,
                        train_window=int(train_window),
                        test_days=int(test_days),
                        order=order_choice,
                    )
                except Exception as e:
                    st.error(f"Backtest error: {e}")
                    st.stop()

            m1, m2, m3 = st.columns(3)
            m1.metric("MAE (daily return)", f"{metrics['MAE']:.6f}")
            m2.metric("RMSE (daily return)", f"{metrics['RMSE']:.6f}")
            m3.metric("Directional accuracy", f"{metrics['Directional accuracy']:.1%}")

            st.markdown("### Predicted vs realized returns")
            df_plot = df_bt.reset_index().melt(
                id_vars="Date",
                value_vars=["Pred_ret", "Real_ret"],
                var_name="Series",
                value_name="Return",
            )
            chart_ret = (
                alt.Chart(df_plot)
                .mark_line()
                .encode(
                    x=alt.X("Date:T", title="Date"),
                    y=alt.Y("Return:Q", title="Daily return"),
                    color=alt.Color("Series:N", title="Series"),
                )
                .properties(height=300)
                .interactive()
            )
            st.altair_chart(chart_ret, use_container_width=True)

            st.markdown("### Equity curve (normalized)")
            df_eq = df_bt.reset_index().melt(
                id_vars="Date",
                value_vars=["Pred_equity", "Real_equity"],
                var_name="Series",
                value_name="Equity",
            )
            chart_eq = (
                alt.Chart(df_eq)
                .mark_line()
                .encode(
                    x=alt.X("Date:T", title="Date"),
                    y=alt.Y("Equity:Q", title="Equity (start=1)"),
                    color=alt.Color("Series:N", title="Series"),
                )
                .properties(height=300)
                .interactive()
            )
            st.altair_chart(chart_eq, use_container_width=True)

            st.markdown("### Backtest table")
            st.dataframe(
                df_bt.style.format(
                    {
                        "Pred_ret": "{:.4%}",
                        "Real_ret": "{:.4%}",
                        "Pred_equity": "{:.4f}",
                        "Real_equity": "{:.4f}",
                    }
                )
            )


if __name__ == "__main__":
    main()
