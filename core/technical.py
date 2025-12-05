import numpy as np
import pandas as pd

def compute_indicators(
    price: pd.Series,
    short_ma: int = 20,
    long_ma: int = 50,
    rsi_period: int = 14,
    macd_fast: int = 12,
    macd_slow: int = 26,
    macd_signal: int = 9,
) -> pd.DataFrame:
    """
    Compute basic technical indicators on a price series:
      - short & long moving averages
      - RSI
      - MACD (line, signal, histogram)
      - simple Buy / Sell / Hold signals using MACD cross + RSI filter.

    Returns a DataFrame indexed by date with columns:
      Price, MA_short, MA_long, RSI, MACD, MACD_signal, MACD_hist, Signal
    """
    s = price.sort_index().dropna()
    df = pd.DataFrame({"Price": s})

    # Moving averages
    df["MA_short"] = df["Price"].rolling(short_ma).mean()
    df["MA_long"] = df["Price"].rolling(long_ma).mean()

    # RSI
    delta = df["Price"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    roll_up = gain.rolling(rsi_period).mean()
    roll_down = loss.rolling(rsi_period).mean()
    rs = roll_up / roll_down
    df["RSI"] = 100 - (100 / (1 + rs))

    # MACD
    ema_fast = df["Price"].ewm(span=macd_fast, adjust=False).mean()
    ema_slow = df["Price"].ewm(span=macd_slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=macd_signal, adjust=False).mean()
    hist = macd_line - signal_line

    df["MACD"] = macd_line
    df["MACD_signal"] = signal_line
    df["MACD_hist"] = hist

    # Trading signals (very simple rule)
    df["Signal"] = "Hold"
    buy_cond = (
        (df["MACD"] > df["MACD_signal"])
        & (df["MACD"].shift(1) <= df["MACD_signal"].shift(1))
        & (df["RSI"] < 70)
    )
    sell_cond = (
        (df["MACD"] < df["MACD_signal"])
        & (df["MACD"].shift(1) >= df["MACD_signal"].shift(1))
        & (df["RSI"] > 30)
    )

    df.loc[buy_cond, "Signal"] = "Buy"
    df.loc[sell_cond, "Signal"] = "Sell"

    return df
