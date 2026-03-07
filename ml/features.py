import pandas as pd
import pandas_ta as ta
import numpy as np


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Takes OHLCV dataframe, returns feature matrix.
    Works for both Equity and F&O candles.
    """
    f = pd.DataFrame(index=df.index)

    # ── Price action ──────────────────────────────────────────
    f["returns_1"]  = df["close"].pct_change(1)
    f["returns_3"]  = df["close"].pct_change(3)
    f["returns_5"]  = df["close"].pct_change(5)
    f["returns_10"] = df["close"].pct_change(10)
    f["hl_ratio"]   = (df["high"] - df["low"]) / df["close"]
    f["co_ratio"]   = (df["close"] - df["open"]) / df["open"]

    # ── Trend ─────────────────────────────────────────────────
    f["ema_9"]   = ta.ema(df["close"], 9)
    f["ema_21"]  = ta.ema(df["close"], 21)
    f["ema_50"]  = ta.ema(df["close"], 50)
    f["ema_200"] = ta.ema(df["close"], 200)
    f["ema_9_cross_21"]  = (f["ema_9"] > f["ema_21"]).astype(int)
    f["ema_21_cross_50"] = (f["ema_21"] > f["ema_50"]).astype(int)
    macd = ta.macd(df["close"])
    if macd is not None:
        f["macd"]      = macd["MACD_12_26_9"]
        f["macd_sig"]  = macd["MACDs_12_26_9"]
        f["macd_hist"] = macd["MACDh_12_26_9"]

    # ── Momentum ──────────────────────────────────────────────
    f["rsi_14"]  = ta.rsi(df["close"], 14)
    f["rsi_7"]   = ta.rsi(df["close"], 7)
    stoch = ta.stoch(df["high"], df["low"], df["close"])
    if stoch is not None:
        f["stoch_k"] = stoch["STOCHk_14_3_3"]
        f["stoch_d"] = stoch["STOCHd_14_3_3"]

    # ── Volatility ────────────────────────────────────────────
    bb = ta.bbands(df["close"], length=20)
    if bb is not None:
        f["bb_upper"] = bb["BBU_20_2.0"]
        f["bb_lower"] = bb["BBL_20_2.0"]
        f["bb_width"] = (bb["BBU_20_2.0"] - bb["BBL_20_2.0"]) / df["close"]
        f["bb_pos"]   = (df["close"] - bb["BBL_20_2.0"]) / \
                        (bb["BBU_20_2.0"] - bb["BBL_20_2.0"] + 1e-9)
    f["atr_14"] = ta.atr(df["high"], df["low"], df["close"], 14)

    # ── Volume ────────────────────────────────────────────────
    f["volume_ma_20"] = df["volume"].rolling(20).mean()
    f["volume_ratio"] = df["volume"] / (f["volume_ma_20"] + 1e-9)
    f["obv"] = ta.obv(df["close"], df["volume"])

    # ── Regime ───────────────────────────────────────────────
    f["adx"] = ta.adx(df["high"], df["low"], df["close"])["ADX_14"]

    # ── Time features ────────────────────────────────────────
    if "timestamp" in df.columns:
        f["hour"]        = pd.to_datetime(df["timestamp"]).dt.hour
        f["minute"]      = pd.to_datetime(df["timestamp"]).dt.minute
        f["day_of_week"] = pd.to_datetime(df["timestamp"]).dt.dayofweek

    return f.dropna()


def build_labels(df: pd.DataFrame, horizon: int = 3,
                 threshold: float = 0.005) -> pd.Series:
    """
    Label each candle:
      1 = price rises > threshold% in next `horizon` candles
      0 = price falls > threshold%
      (rows where neither — filtered out during training)
    """
    future_return = df["close"].shift(-horizon) / df["close"] - 1
    labels = pd.Series(index=df.index, dtype=float)
    labels[future_return >  threshold] = 1
    labels[future_return < -threshold] = 0
    return labels