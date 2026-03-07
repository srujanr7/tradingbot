import pandas as pd
import numpy as np
import ta


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Takes OHLCV dataframe, returns feature matrix.
    Works for both Equity and F&O candles.
    Uses the 'ta' library instead of pandas-ta for
    Python 3.11 compatibility on Render.
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
    f["ema_9"]   = ta.trend.ema_indicator(df["close"], window=9)
    f["ema_21"]  = ta.trend.ema_indicator(df["close"], window=21)
    f["ema_50"]  = ta.trend.ema_indicator(df["close"], window=50)
    f["ema_200"] = ta.trend.ema_indicator(df["close"], window=200)
    f["ema_9_cross_21"]  = (f["ema_9"]  > f["ema_21"]).astype(int)
    f["ema_21_cross_50"] = (f["ema_21"] > f["ema_50"]).astype(int)

    macd_ind       = ta.trend.MACD(df["close"])
    f["macd"]      = macd_ind.macd()
    f["macd_sig"]  = macd_ind.macd_signal()
    f["macd_hist"] = macd_ind.macd_diff()

    # ── Momentum ──────────────────────────────────────────────
    f["rsi_14"] = ta.momentum.rsi(df["close"], window=14)
    f["rsi_7"]  = ta.momentum.rsi(df["close"], window=7)

    stoch        = ta.momentum.StochasticOscillator(
        df["high"], df["low"], df["close"]
    )
    f["stoch_k"] = stoch.stoch()
    f["stoch_d"] = stoch.stoch_signal()

    # ── Volatility ────────────────────────────────────────────
    bb            = ta.volatility.BollingerBands(df["close"], window=20)
    f["bb_upper"] = bb.bollinger_hband()
    f["bb_lower"] = bb.bollinger_lband()
    f["bb_width"] = (f["bb_upper"] - f["bb_lower"]) / df["close"]
    f["bb_pos"]   = (df["close"] - f["bb_lower"]) / (
        f["bb_upper"] - f["bb_lower"] + 1e-9
    )
    f["atr_14"]   = ta.volatility.average_true_range(
        df["high"], df["low"], df["close"], window=14
    )

    # ── Volume ────────────────────────────────────────────────
    f["volume_ma_20"] = df["volume"].rolling(20).mean()
    f["volume_ratio"] = df["volume"] / (f["volume_ma_20"] + 1e-9)
    f["obv"]          = ta.volume.on_balance_volume(
        df["close"], df["volume"]
    )

    # ── Regime ────────────────────────────────────────────────
    f["adx"] = ta.trend.adx(
        df["high"], df["low"], df["close"], window=14
    )

    # ── Time features ─────────────────────────────────────────
    if "timestamp" in df.columns:
        ts               = pd.to_datetime(df["timestamp"])
        f["hour"]        = ts.dt.hour
        f["minute"]      = ts.dt.minute
        f["day_of_week"] = ts.dt.dayofweek

    return f.dropna()


def build_labels(df: pd.DataFrame, horizon: int = 3,
                 threshold: float = 0.005) -> pd.Series:
    """
    Label each candle:
      1 = price rises > threshold% in next horizon candles
      0 = price falls > threshold%
      rows where neither — filtered out during training
    """
    future_return = df["close"].shift(-horizon) / df["close"] - 1
    labels        = pd.Series(index=df.index, dtype=float)
    labels[future_return >  threshold] = 1
    labels[future_return < -threshold] = 0
    return labels
