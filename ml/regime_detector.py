import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)


class RegimeDetector:
    """
    Detects current market regime.
    TRENDING  → follow momentum signals
    RANGING   → use mean-reversion signals
    VOLATILE  → reduce position size or skip
    BEAR      → only short or skip
    """

    def detect(self, df: pd.DataFrame) -> dict:
        if len(df) < 50:
            return {"regime": "UNKNOWN", "confidence": 0.0}

        close = df["close"]
        high  = df["high"]
        low   = df["low"]

        # ADX for trend strength
        try:
            import pandas_ta as ta
            adx_df = ta.adx(high, low, close)
            adx    = float(adx_df["ADX_14"].iloc[-1])
        except Exception:
            adx = 25.0

        # Volatility — ATR as % of price
        atr     = float((high - low).rolling(14).mean().iloc[-1])
        atr_pct = atr / float(close.iloc[-1]) * 100

        # Trend direction
        ema_20  = float(close.ewm(span=20).mean().iloc[-1])
        ema_50  = float(close.ewm(span=50).mean().iloc[-1])
        current = float(close.iloc[-1])

        # Regime classification
        if adx > 35 and current > ema_20 > ema_50:
            regime     = "TRENDING_UP"
            confidence = min(adx / 50, 1.0)
        elif adx > 35 and current < ema_20 < ema_50:
            regime     = "TRENDING_DOWN"
            confidence = min(adx / 50, 1.0)
        elif adx < 20:
            regime     = "RANGING"
            confidence = 1.0 - (adx / 20)
        elif atr_pct > 2.5:
            regime     = "VOLATILE"
            confidence = min(atr_pct / 5, 1.0)
        else:
            regime     = "NEUTRAL"
            confidence = 0.5

        return {
            "regime":     regime,
            "confidence": round(confidence, 3),
            "adx":        round(adx, 1),
            "atr_pct":    round(atr_pct, 2),
            "trend":      "UP" if current > ema_50 else "DOWN"
        }

    def should_trade(self, regime: dict,
                     signal: str) -> tuple:
        """
        Returns (should_trade, reason, size_multiplier)
        size_multiplier reduces position size in bad conditions
        """
        r = regime["regime"]

        if r == "TRENDING_UP" and signal == "BUY":
            return True, "Trend confirms BUY", 1.0
        elif r == "TRENDING_UP" and signal == "SELL":
            return False, "Selling against uptrend", 0.0
        elif r == "TRENDING_DOWN" and signal == "SELL":
            return True, "Trend confirms SELL", 1.0
        elif r == "TRENDING_DOWN" and signal == "BUY":
            return False, "Buying against downtrend", 0.0
        elif r == "VOLATILE":
            return True, "Volatile — reduced size", 0.5
        elif r == "RANGING":
            return True, "Ranging market", 0.7
        else:
            return True, "Neutral regime", 0.8
