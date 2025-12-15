import io
import numpy as np
import pandas as pd
import requests
from pathlib import Path

from config import ANNUAL_FACTOR

# =========================
#  URLs CSV Sika par symbole
# =========================
# Exemple:
# SIKA_URL_MAP = {
#     "SNTS": "https://www.sikafinance.com/...csv...",
#     "SGBC": "https://www.sikafinance.com/...csv...",
# }
SIKA_URL_MAP: dict[str, str] = {}


def standardize_sika_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean a raw SIKA CSV:
      - keep expected columns
      - rename to unified English names
      - parse dates
      - normalise symbol names (SNTS.sn -> SNTS)
      - convertir les colonnes de prix/volume en float
    """
    cols_map = {
        "symbole": "symbol",
        "date": "date",
        "ouverture": "open",
        "haut": "high",
        "bas": "low",
        "cloture": "close",
        "volume": "volume",
    }

    # Strict: keep only expected columns
    df = df[list(cols_map.keys())].rename(columns=cols_map)

    # Symbol: keep everything before the dot (SNTS.sn -> SNTS)
    df["symbol"] = df["symbol"].astype(str).str.split(".").str[0]

    # Day-first date format on SIKA
    df["date"] = pd.to_datetime(df["date"], dayfirst=True, errors="coerce")

    # Convert numeric cols
    num_cols = ["open", "high", "low", "close", "volume"]
    for col in num_cols:
        df[col] = (
            df[col]
            .astype(str)
            .str.replace("\u00a0", "", regex=False)
            .str.replace(" ", "", regex=False)
            .str.replace(",", ".", regex=False)
        )
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def _build_wide_tables(all_rows: list[pd.DataFrame]):
    """
    From a list of standardised long DataFrames, build:
      - data_long (symbol, date, OHLC, volume)
      - prices_wide (dates x symbols, close)
      - volumes_wide (dates x symbols, volume)
    """
    data_long = (
        pd.concat(all_rows, ignore_index=True)
        .drop_duplicates(subset=["symbol", "date"])
        .sort_values(["symbol", "date"])
    )

    prices_wide = (
        data_long
        .pivot(index="date", columns="symbol", values="close")
        .sort_index()
    )

    volumes_wide = (
        data_long
        .pivot(index="date", columns="symbol", values="volume")
        .sort_index()
    )

    return data_long, prices_wide, volumes_wide


def fetch_sika_history_for_symbol(symbol: str, url_map: dict[str, str] | None = None) -> pd.DataFrame:
    """
    Télécharge l'historique complet d'un symbole sur Sika Finance
    en utilisant l'URL explicite définie dans url_map (ou SIKA_URL_MAP).

    Si le symbole n'est pas dans la map, on renvoie un DataFrame vide.
    """
    if url_map is None:
        url_map = SIKA_URL_MAP

    url = url_map.get(symbol)
    if not url:
        return pd.DataFrame(columns=["symbol", "date", "open", "high", "low", "close", "volume"])

    resp = requests.get(url, timeout=20)
    resp.raise_for_status()

    raw_csv = io.StringIO(resp.text)
    tmp = pd.read_csv(raw_csv, sep=";", encoding="utf-8")
    tmp = standardize_sika_df(tmp)
    return tmp


def update_history_from_sika_online(data_long: pd.DataFrame, url_map: dict[str, str] | None = None) -> pd.DataFrame:
    """
    Pour chaque symbole présent dans data_long:
      - on télécharge l'historique Sika via fetch_sika_history_for_symbol()
      - on ajoute uniquement les lignes dont la date est > dernière date locale.
    """
    if url_map is None:
        url_map = SIKA_URL_MAP

    if data_long.empty:
        return data_long

    symbols = data_long["symbol"].unique()
    new_parts: list[pd.DataFrame] = []

    for sym in symbols:
        if not isinstance(sym, str) or sym.strip() == "":
            continue

        try:
            remote_hist = fetch_sika_history_for_symbol(sym, url_map=url_map)
        except Exception:
            continue

        if remote_hist.empty:
            continue

        last_local_date = data_long.loc[data_long["symbol"] == sym, "date"].max()
        df_new = remote_hist[remote_hist["date"] > last_local_date]

        if not df_new.empty:
            new_parts.append(df_new)

    if not new_parts:
        return data_long

    updated = pd.concat([data_long] + new_parts, ignore_index=True)
    updated = (
        updated
        .drop_duplicates(subset=["symbol", "date"])
        .sort_values(["symbol", "date"])
    )
    return updated


def build_prices_from_sika_folder(raw_dir: Path, update_online: bool = False, url_map: dict[str, str] | None = None):
    """
    Read all *.csv files from raw_dir and build wide price/volume tables.
    If update_online=True, tries to update latest quotes from Sika using url_map / SIKA_URL_MAP.
    """
    csv_files = list(raw_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {raw_dir}")

    all_rows: list[pd.DataFrame] = []
    for f in csv_files:
        tmp = pd.read_csv(f, sep=";", encoding="utf-8")
        tmp = standardize_sika_df(tmp)
        all_rows.append(tmp)

    data_long = (
        pd.concat(all_rows, ignore_index=True)
        .drop_duplicates(subset=["symbol", "date"])
        .sort_values(["symbol", "date"])
    )

    if update_online:
        data_long = update_history_from_sika_online(data_long, url_map=url_map)

    prices_wide = (
        data_long
        .pivot(index="date", columns="symbol", values="close")
        .sort_index()
    )

    volumes_wide = (
        data_long
        .pivot(index="date", columns="symbol", values="volume")
        .sort_index()
    )

    return data_long, prices_wide, volumes_wide


def build_prices_from_uploaded(uploaded_files):
    """
    Same as build_prices_from_sika_folder but using Streamlit uploaded files.
    """
    if not uploaded_files:
        raise ValueError("No uploaded files")

    all_rows: list[pd.DataFrame] = []
    for f in uploaded_files:
        tmp = pd.read_csv(f, sep=";", encoding="utf-8")
        tmp = standardize_sika_df(tmp)
        all_rows.append(tmp)

    return _build_wide_tables(all_rows)


def build_prices_from_sika_online_urls(url_map: dict[str, str]):
    """
    Charge directement depuis Sika Finance (URLs CSV explicites).
    url_map = {"SNTS": "https://....", "SGBC": "..."}
    """
    if not url_map:
        raise ValueError("url_map is empty. Provide at least one SYMBOL->URL mapping.")

    all_rows: list[pd.DataFrame] = []
    for sym in url_map.keys():
        try:
            df = fetch_sika_history_for_symbol(sym, url_map=url_map)
            if not df.empty:
                all_rows.append(df)
        except Exception:
            continue

    if not all_rows:
        raise ValueError("No data fetched from Sika (check URLs / symbols).")

    return _build_wide_tables(all_rows)


def slice_history(
    prices_wide: pd.DataFrame,
    volumes_wide: pd.DataFrame,
    lookback_choice: str,
):
    """
    Restrict history according to a French label:
      - "Tout"
      - "6 derniers mois"
      - "12 derniers mois"
      - "24 derniers mois"
    """
    if prices_wide is None or prices_wide.empty:
        return prices_wide, volumes_wide

    if lookback_choice == "Tout":
        return prices_wide, volumes_wide

    last_date = prices_wide.index.max()

    if lookback_choice == "6 derniers mois":
        start = last_date - pd.DateOffset(months=6)
    elif lookback_choice == "12 derniers mois":
        start = last_date - pd.DateOffset(years=1)
    else:  # "24 derniers mois"
        start = last_date - pd.DateOffset(years=2)

    prices = prices_wide.loc[prices_wide.index >= start]
    volumes = volumes_wide.loc[volumes_wide.index >= start]
    return prices, volumes


def compute_returns(prices_wide: pd.DataFrame) -> pd.DataFrame:
    """
    Simple daily returns: P_t / P_{t-1} - 1.
    Ensures numeric columns.
    """
    prices_num = prices_wide.apply(pd.to_numeric, errors="coerce")
    prices_num = prices_num.replace([np.inf, -np.inf], np.nan)
    prices_num = prices_num.dropna(axis=1, how="all")

    returns = prices_num.pct_change()
    returns = returns.dropna(how="all")

    return returns


def compute_asset_stats(returns: pd.DataFrame) -> pd.DataFrame:
    """
    Compute per-asset statistics: daily & annual mean/vol, approx. 3-month return.
    """
    mu_d = returns.mean()
    sigma_d = returns.std()

    mu_a = mu_d * ANNUAL_FACTOR
    sigma_a = sigma_d * np.sqrt(ANNUAL_FACTOR)
    mu_3m = mu_d * 63  # ~ 3 months

    stats = pd.DataFrame(
        {
            "mu_daily": mu_d,
            "sigma_daily": sigma_d,
            "mu_ann": mu_a,
            "sigma_ann": sigma_a,
            "mu_3m": mu_3m,
        }
    )
    return stats
