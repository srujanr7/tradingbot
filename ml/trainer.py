import logging
import time
import schedule
import threading
import pandas as pd
from datetime import datetime, timedelta
from api import INDstocksAPI
from ml.features import build_features, build_labels
from ml.model import XGBSignalModel, RLAgent

logger = logging.getLogger(__name__)


class AutoTrainer:
    """
    Fetches fresh historical data and retrains both
    XGBoost and PPO models on a schedule.
    Runs in a background thread — never blocks the live bot.
    """
    def __init__(self, scrip_code: str, security_id: str,
                 interval: str = "15minute",
                 retrain_days: int = 90):
        self.api         = INDstocksAPI()
        self.scrip_code  = scrip_code
        self.security_id = security_id
        self.interval    = interval
        self.retrain_days = retrain_days

        self.xgb = XGBSignalModel()
        self.rl  = RLAgent()

        # Try loading existing models first
        self.xgb.load()
        self.rl.load()

        self._lock = threading.Lock()   # Thread-safe model swaps

    def _fetch_training_data(self) -> pd.DataFrame:
        """Fetch last N days of candles in chunks (API max: 7 days per call)"""
        all_candles = []
        end_ms   = int(time.time() * 1000)
        chunk_ms = 6 * 24 * 60 * 60 * 1000    # 6-day chunks
        total_ms = self.retrain_days * 24 * 60 * 60 * 1000
        start_ms = end_ms - total_ms

        current_start = start_ms
        while current_start < end_ms:
            current_end = min(current_start + chunk_ms, end_ms)
            df_chunk = self.api.get_historical(
                self.scrip_code, self.interval,
                current_start, current_end
            )
            if df_chunk is not None and len(df_chunk) > 0:
                all_candles.append(df_chunk)
            current_start = current_end
            time.sleep(0.5)   # Respect rate limits

        if not all_candles:
            return None

        df = pd.concat(all_candles).drop_duplicates("timestamp")
        df = df.sort_values("timestamp").reset_index(drop=True)
        logger.info(f"Fetched {len(df)} candles for training")
        return df

    def retrain(self):
        logger.info(f"🔄 Retraining started at {datetime.now()}")
        df = self._fetch_training_data()
        if df is None or len(df) < 200:
            logger.warning("Not enough data to retrain. Skipping.")
            return

        features = build_features(df)
        labels   = build_labels(df, horizon=3, threshold=0.005)

        # ── Retrain XGBoost ───────────────────────────────────
        new_xgb = XGBSignalModel()
        new_xgb.train(features, labels)

        # ── Retrain RL ────────────────────────────────────────
        aligned = features.copy()
        aligned["close"] = df["close"].loc[features.index]
        new_rl = RLAgent()
        new_rl.train(
            aligned.drop(columns=["close"]),
            aligned["close"],
            timesteps=300_000
        )

        # ── Atomic model swap ─────────────────────────────────
        with self._lock:
            self.xgb = new_xgb
            self.rl  = new_rl

        logger.info("✅ Models retrained and swapped live")

    def get_signal(self, df: pd.DataFrame) -> dict:
        """
        Combine XGBoost + RL votes into final signal.
        Called by the live bot on every cycle.
        """
        with self._lock:
            features = build_features(df)
            if features.empty:
                return {"signal": "HOLD", "confidence": 0.0, "source": "none"}

            # XGBoost vote
            xgb_result = self.xgb.predict(features)

            # RL vote
            obs = features.iloc[-1].values
            import numpy as np
            rl_action = self.rl.predict(
                np.append(obs, [0.0, 0.0]).astype(np.float32)
            )
            rl_signal = {0: "HOLD", 1: "BUY", 2: "SELL"}.get(rl_action, "HOLD")

            # Weighted consensus: XGB 60%, RL 40%
            votes = {"BUY": 0.0, "SELL": 0.0, "HOLD": 0.0}
            votes[xgb_result["signal"]] += 0.6 * xgb_result["confidence"]
            votes[rl_signal] += 0.4

            final = max(votes, key=votes.get)
            return {
                "signal":     final,
                "confidence": votes[final],
                "xgb":        xgb_result["signal"],
                "rl":         rl_signal,
                "votes":      votes
            }

    def start_schedule(self, retrain_time: str = "18:30"):
        """
        Schedule weekly retraining after market close.
        Runs in background thread.
        """
        schedule.every().sunday.at(retrain_time).do(self.retrain)

        # Also retrain immediately on first startup if no model exists
        if not self.xgb.is_trained:
            logger.info("No model found. Running initial training...")
            threading.Thread(target=self.retrain, daemon=True).start()

        def _loop():
            while True:
                schedule.run_pending()
                time.sleep(60)

        thread = threading.Thread(target=_loop, daemon=True)
        thread.start()
        logger.info(f"✅ Auto-trainer scheduled every Sunday at {retrain_time}")
        return thread