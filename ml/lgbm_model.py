import lightgbm as lgb
import numpy as np
import pandas as pd
import logging
import os
import joblib
import pandas_ta as ta

logger = logging.getLogger(__name__)


class LGBMModel:
    """
    LightGBM price direction predictor.
    Faster than XGBoost, handles more features better.
    """

    MODEL_DIR = "ml/models"

    def __init__(self, scrip_code: str):
        self.scrip_code = scrip_code
        self.model      = None
        self.trained    = False
        self._load()

    def _path(self):
        return f"{self.MODEL_DIR}/lgbm_{self.scrip_code}.pkl"

    def _load(self):
        if os.path.exists(self._path()):
            try:
                self.model   = joblib.load(self._path())
                self.trained = True
                logger.info(f"✅ LGBM loaded for {self.scrip_code}")
            except Exception as e:
                logger.warning(f"LGBM load error: {e}")

    def _build_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        df["ema_9"]  = ta.ema(df["close"], length=9)
        df["ema_21"] = ta.ema(df["close"], length=21)
        df["ema_50"] = ta.ema(df["close"], length=50)
        df["ema_cross"] = (df["ema_9"] > df["ema_21"]).astype(int)

        df["rsi"]    = ta.rsi(df["close"], length=14)
        df["roc"]    = ta.roc(df["close"], length=10)
        df["mom"]    = ta.mom(df["close"], length=10)

        macd         = ta.macd(df["close"])
        df["macd"]   = macd["MACD_12_26_9"]
        df["macd_s"] = macd["MACDs_12_26_9"]
        df["macd_h"] = macd["MACDh_12_26_9"]

        bb           = ta.bbands(df["close"])
        df["bb_pct"] = bb["BBP_5_2.0"]
        df["atr"]    = ta.atr(df["high"], df["low"], df["close"])

        df["vol_ratio"] = df["volume"] / df["volume"].rolling(20).mean()
        df["obv"]       = ta.obv(df["close"], df["volume"])

        stoch           = ta.stoch(df["high"], df["low"], df["close"])
        df["stoch_k"]   = stoch["STOCHk_14_3_3"]
        df["stoch_d"]   = stoch["STOCHd_14_3_3"]

        df["hl_ratio"]  = (df["high"] - df["low"]) / df["close"]
        df["oc_ratio"]  = (df["close"] - df["open"]) / df["close"]

        # Target — next candle direction
        df["target"] = (df["close"].shift(-1) > df["close"]).astype(int)

        df.dropna(inplace=True)
        return df

    FEATURE_COLS = [
        "ema_cross", "rsi", "roc", "mom",
        "macd", "macd_s", "macd_h",
        "bb_pct", "atr", "vol_ratio",
        "stoch_k", "stoch_d",
        "hl_ratio", "oc_ratio"
    ]

    def train(self, df: pd.DataFrame):
        if len(df) < 50:
            return
        try:
            df = self._build_features(df)
            X  = df[self.FEATURE_COLS].values
            y  = df["target"].values

            self.model = lgb.LGBMClassifier(
                n_estimators=200,
                learning_rate=0.05,
                max_depth=6,
                num_leaves=31,
                subsample=0.8,
                colsample_bytree=0.8,
                verbose=-1
            )
            self.model.fit(X, y)

            os.makedirs(self.MODEL_DIR, exist_ok=True)
            joblib.dump(self.model, self._path())
            self.trained = True
            logger.info(f"✅ LGBM trained for {self.scrip_code}")

        except Exception as e:
            logger.error(f"LGBM train error: {e}")

    def predict(self, df: pd.DataFrame) -> dict:
        if not self.trained or self.model is None:
            return {"signal": "HOLD", "confidence": 0.0}

        try:
            df   = self._build_features(df)
            X    = df[self.FEATURE_COLS].values[-1:]
            prob = self.model.predict_proba(X)[0]
            conf = float(max(prob))
            pred = int(self.model.predict(X)[0])

            # Below confidence threshold → HOLD regardless
            if conf < 0.60:
                return {"signal": "HOLD", "confidence": conf}

            # pred==1 → price going UP → BUY
            # pred==0 → price going DOWN → SELL
            signal = "BUY" if pred == 1 else "SELL"
            return {"signal": signal, "confidence": conf}

        except Exception as e:
            logger.error(f"LGBM predict error: {e}")
            return {"signal": "HOLD", "confidence": 0.0}
