import numpy as np  # FIX 5: moved to top
import logging
import time
import schedule
import threading
import pandas as pd
from datetime import datetime
from api import INDstocksAPI
from ml.features import build_features, build_labels
from ml.model import XGBSignalModel, RLAgent

logger = logging.getLogger(__name__)


class AutoTrainer:
    """
    Fetches fresh historical data and retrains
    XGBoost model on a schedule.
    RL model trained only if libraries available.
    Runs in background thread — never blocks live bot.
    """

    def __init__(self, scrip_code: str, security_id: str,
                 interval: str = "15minute",
                 retrain_days: int = 90):
        self.api          = INDstocksAPI()
        self.scrip_code   = scrip_code
        self.security_id  = security_id
        self.interval     = interval
        self.retrain_days = retrain_days

        self.xgb = XGBSignalModel(model_path=f"ml/xgb_{scrip_code}.pkl")
        self.rl  = RLAgent(model_path=f"ml/ppo_{scrip_code}")

        self.xgb.load()
        self.rl.load()

        self._lock = threading.Lock()

    # ── Data fetching ─────────────────────────────────────────

    def _fetch_training_data(self) -> pd.DataFrame:
        all_candles = []
        end_ms      = int(time.time() * 1000)
        chunk_ms    = 6 * 24 * 60 * 60 * 1000
        total_ms    = self.retrain_days * 24 * 60 * 60 * 1000
        start_ms    = end_ms - total_ms
        current     = start_ms

        while current < end_ms:
            chunk_end = min(current + chunk_ms, end_ms)
            df_chunk  = self.api.get_historical(
                self.scrip_code,
                self.interval,
                current,
                chunk_end
            )
            if df_chunk is not None and len(df_chunk) > 0:
                all_candles.append(df_chunk)
            current = chunk_end
            time.sleep(0.5)

        if not all_candles:
            logger.warning(f"No training data for {self.scrip_code}")
            return None

        df = pd.concat(all_candles).drop_duplicates("timestamp")
        df = df.sort_values("timestamp").reset_index(drop=True)
        logger.info(f"Fetched {len(df)} candles for {self.scrip_code}")
        return df

    # ── Retraining ────────────────────────────────────────────

    def retrain(self):
        start_time = time.time()
        logger.info(f"Retraining started for {self.scrip_code} at {datetime.now()}")

        df = self._fetch_training_data()
        if df is None or len(df) < 200:
            logger.warning(f"Not enough data for {self.scrip_code}. Skipping retrain.")
            return

        features = build_features(df)
        labels   = build_labels(df, horizon=3, threshold=0.005)

        if features.empty:
            logger.warning(f"Empty features for {self.scrip_code}. Skipping.")
            return

        # ── Retrain XGBoost ───────────────────────────────────
        new_xgb = XGBSignalModel(model_path=f"ml/xgb_{self.scrip_code}.pkl")
        try:
            new_xgb.train(features, labels)
        except Exception as e:
            logger.error(f"XGB training failed: {e}")
            return

        # ── Retrain RL (only if available) ────────────────────
        new_rl = RLAgent(model_path=f"ml/ppo_{self.scrip_code}")
        if new_rl.available:
            try:
                aligned          = features.copy()
                aligned["close"] = df["close"].loc[features.index]
                new_rl.train(
                    aligned.drop(columns=["close"]),
                    aligned["close"],
                    timesteps=300_000
                )
                logger.info(f"RL retrain complete for {self.scrip_code}")
            except Exception as e:
                logger.error(f"RL training failed: {e}")
                new_rl = self.rl
        else:
            new_rl = self.rl
            logger.info("Skipping RL retrain — using XGBoost only")

        # ── Atomic model swap ─────────────────────────────────
        with self._lock:
            self.xgb = new_xgb
            self.rl  = new_rl

        duration = time.time() - start_time
        logger.info(f"Retrain complete for {self.scrip_code} in {duration:.0f}s")

        try:
            from notifier import TelegramNotifier
            notifier = TelegramNotifier()
            notifier.model_retrained(
                instrument=self.scrip_code,
                samples=len(features),
                duration_seconds=duration,
                next_retrain="Next Sunday 18:30"
            )
        except Exception:
            pass

    # ── Signal generation ─────────────────────────────────────

    def get_signal(self, df: pd.DataFrame) -> dict:
        with self._lock:
            if df is None or len(df) < 50:
                return {
                    "signal": "HOLD", "confidence": 0.0,
                    "xgb": "HOLD", "rl": "HOLD",
                    "votes": {"BUY": 0, "SELL": 0, "HOLD": 1}
                }

            features = build_features(df)
            if features.empty:
                return {
                    "signal": "HOLD", "confidence": 0.0,
                    "xgb": "HOLD", "rl": "HOLD",
                    "votes": {"BUY": 0, "SELL": 0, "HOLD": 1}
                }

            xgb_result = self.xgb.predict(features)
            xgb_signal = xgb_result["signal"]
            xgb_conf   = xgb_result["confidence"]

            try:
                obs = features.iloc[-1].values
                rl_action = self.rl.predict(
                    np.append(obs, [0.0, 0.0]).astype(np.float32)
                    if self.rl.available else obs
                )
                rl_signal = {0: "HOLD", 1: "BUY", 2: "SELL"}.get(rl_action, "HOLD")
            except Exception:
                rl_signal = "HOLD"

            votes = {"BUY": 0.0, "SELL": 0.0, "HOLD": 0.0}
            if self.rl.available and self.rl.model is not None:
                votes[xgb_signal] += 0.6 * xgb_conf
                votes[rl_signal]  += 0.4
            else:
                votes[xgb_signal] += xgb_conf

            final = max(votes, key=votes.get)
            return {
                "signal":     final,
                "confidence": votes[final],
                "xgb":        xgb_signal,
                "rl":         rl_signal,
                "votes":      votes
            }

    # ── Scheduling ────────────────────────────────────────────

    def start_schedule(self, retrain_time: str = "18:30"):
        schedule.every().sunday.at(retrain_time).do(self.retrain)

        if not self.xgb.is_trained:
            logger.info(f"No model for {self.scrip_code}. Running initial training...")
            threading.Thread(target=self.retrain, daemon=True).start()

        def _loop():
            while True:
                schedule.run_pending()
                time.sleep(60)

        thread = threading.Thread(target=_loop, daemon=True)
        thread.start()
        logger.info(
            f"Auto-trainer scheduled for {self.scrip_code} "
            f"every Sunday at {retrain_time}"
        )
        return thread
