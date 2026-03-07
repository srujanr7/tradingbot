import numpy as np
import logging
import time
import schedule
import threading
import pandas as pd
import pandas_ta as ta
from datetime import datetime
from api import INDstocksAPI
from ml.features import build_features, build_labels
from ml.model import XGBSignalModel, RLAgent

# ── New model imports ─────────────────────────────────────────
from ml.lgbm_model      import LGBMModel
from ml.lstm_model      import LSTMModel
from ml.regime_detector import RegimeDetector
from ml.pattern_detector import PatternDetector
from ml.time_filter     import TimeFilter
from ml.meta_model      import MetaModel
from ml.trade_memory    import TradeMemory

logger = logging.getLogger(__name__)

# ── Shared singletons (one instance across all AutoTrainers) ──
_regime       = RegimeDetector()
_pattern      = PatternDetector()
_time_filter  = TimeFilter()
_meta_model   = MetaModel()
_trade_memory = TradeMemory()


class AutoTrainer:
    """
    Full ML orchestrator for one instrument.

    Model stack:
      Layer 1 — Price prediction : XGBoost + LightGBM + LSTM
      Layer 2 — RL agents        : PPO (via existing RLAgent)
      Layer 3 — Market analysis  : RegimeDetector + PatternDetector
                                   + TimeFilter
      Layer 4 — Aggregation      : MetaModel (weighted votes,
                                   self-updating weights)

    Self-improves:
      - TradeMemory records every trade outcome
      - MetaModel.update_weights() shifts weight toward
        whichever model was correct
      - Weekly retrain on Sunday 18:30
      - Interval auto-selected via IntervalSelector (optional)
    """

    def __init__(self, scrip_code: str, security_id: str,
                 interval: str = "5minute",
                 retrain_days: int = 90):

        self.api          = INDstocksAPI()
        self.scrip_code   = scrip_code
        self.security_id  = security_id
        self.retrain_days = retrain_days
        self._lock        = threading.Lock()
        self._last_df     = None   # kept for scheduled retrain

        # ── Auto-select interval if requested ─────────────────
        if interval == "auto":
            self.interval = self._auto_select_interval()
        else:
            self.interval = interval

        # ── Initialise all models ─────────────────────────────
        self.xgb  = XGBSignalModel(
            model_path=f"ml/models/xgb_{scrip_code}.pkl"
        )
        self.lgbm = LGBMModel(scrip_code)
        self.lstm = LSTMModel(scrip_code)
        self.rl   = RLAgent(
            model_path=f"ml/models/ppo_{scrip_code}"
        )

        # Load saved weights
        self.xgb.load()
        self.rl.load()
        # LGBMModel and LSTMModel load in __init__ automatically

        # Per-instrument trade history for RL context
        self._instrument_history = \
            _trade_memory.get_instrument_stats(scrip_code)

        logger.info(
            f"🧠 AutoTrainer ready | {scrip_code} "
            f"| interval={self.interval}"
        )

    # ─────────────────────────────────────────────────────────
    # INTERVAL AUTO-SELECTION
    # ─────────────────────────────────────────────────────────

    def _auto_select_interval(self) -> str:
        """
        Try to import IntervalSelector.
        Falls back to '5minute' if file not present yet.
        """
        try:
            from ml.interval_selector import IntervalSelector
            selector = IntervalSelector(
                self.api, self.scrip_code, self.security_id
            )
            interval = selector.find_best()
            logger.info(
                f"🎯 Auto-selected interval: {interval} "
                f"for {self.scrip_code}"
            )
            return interval
        except Exception as e:
            logger.warning(
                f"IntervalSelector failed ({e}), "
                f"defaulting to 5minute"
            )
            return "5minute"

    # ─────────────────────────────────────────────────────────
    # INDICATOR BUILDER  (shared by train + predict paths)
    # ─────────────────────────────────────────────────────────

    def _add_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Adds all technical indicators in-place on a copy.
        Safe — drops NaN rows after calculation.
        """
        df = df.copy()
        try:
            # Trend
            df["ema_9"]     = ta.ema(df["close"], length=9)
            df["ema_20"]    = ta.ema(df["close"], length=20)
            df["ema_50"]    = ta.ema(df["close"], length=50)
            df["ema_cross"] = (
                df["ema_9"] > df["ema_20"]
            ).astype(int)

            # Momentum
            df["rsi"]    = ta.rsi(df["close"], length=14)
            df["roc"]    = ta.roc(df["close"], length=10)
            df["mom"]    = ta.mom(df["close"], length=10)
            df["cci"]    = ta.cci(
                df["high"], df["low"], df["close"]
            )

            # MACD
            macd          = ta.macd(df["close"])
            df["macd"]    = macd["MACD_12_26_9"]
            df["macd_s"]  = macd["MACDs_12_26_9"]
            df["macd_h"]  = macd["MACDh_12_26_9"]

            # Bollinger Bands
            bb             = ta.bbands(df["close"])
            df["bb_pct"]   = bb["BBP_5_2.0"]
            df["bb_upper"] = bb["BBU_5_2.0"]
            df["bb_lower"] = bb["BBL_5_2.0"]

            # Volatility
            df["atr"]     = ta.atr(
                df["high"], df["low"], df["close"]
            )
            df["atr_pct"] = df["atr"] / df["close"]

            # ADX
            adx_df    = ta.adx(
                df["high"], df["low"], df["close"]
            )
            df["adx"] = adx_df["ADX_14"]

            # Stochastic
            stoch          = ta.stoch(
                df["high"], df["low"], df["close"]
            )
            df["stoch_k"]  = stoch["STOCHk_14_3_3"]
            df["stoch_d"]  = stoch["STOCHd_14_3_3"]

            # Volume
            df["vol_ratio"] = (
                df["volume"] / df["volume"].rolling(20).mean()
            )
            df["obv"] = ta.obv(df["close"], df["volume"])

            # Price structure
            df["hl_ratio"] = (
                df["high"] - df["low"]
            ) / df["close"]
            df["oc_ratio"] = (
                df["close"] - df["open"]
            ) / df["close"]

            df.dropna(inplace=True)

        except Exception as e:
            logger.warning(f"Indicator error ({self.scrip_code}): {e}")

        return df

    # ─────────────────────────────────────────────────────────
    # DATA FETCHING
    # ─────────────────────────────────────────────────────────

    def _fetch_training_data(self) -> pd.DataFrame | None:
        """
        Fetches retrain_days of OHLCV candles in 6-day chunks
        to stay within API limits.
        """
        all_candles = []
        end_ms      = int(time.time() * 1000)
        chunk_ms    = 6 * 24 * 60 * 60 * 1000
        total_ms    = self.retrain_days * 24 * 60 * 60 * 1000
        start_ms    = end_ms - total_ms
        current     = start_ms

        while current < end_ms:
            chunk_end = min(current + chunk_ms, end_ms)
            try:
                df_chunk = self.api.get_historical(
                    self.scrip_code,
                    self.interval,
                    current,
                    chunk_end
                )
                if df_chunk is not None and len(df_chunk) > 0:
                    all_candles.append(df_chunk)
            except Exception as e:
                logger.warning(
                    f"Chunk fetch error ({self.scrip_code}): {e}"
                )
            current = chunk_end
            time.sleep(0.5)

        if not all_candles:
            logger.warning(
                f"No training data fetched for {self.scrip_code}"
            )
            return None

        df = (
            pd.concat(all_candles)
            .drop_duplicates("timestamp")
            .sort_values("timestamp")
            .reset_index(drop=True)
        )
        logger.info(
            f"Fetched {len(df)} candles "
            f"({self.interval}) for {self.scrip_code}"
        )
        return df

    # ─────────────────────────────────────────────────────────
    # TRAINING
    # ─────────────────────────────────────────────────────────

    def retrain(self):
        """
        Full retrain of all models.
        Called automatically every Sunday at 18:30,
        or manually via retrain().
        Runs all model trains in parallel threads.
        """
        start_time = time.time()
        logger.info(
            f"🔄 Retrain started: {self.scrip_code} "
            f"@ {datetime.now().strftime('%H:%M:%S')}"
        )

        # ── Fetch data ────────────────────────────────────────
        df = self._fetch_training_data()
        if df is None or len(df) < 200:
            logger.warning(
                f"Not enough data for {self.scrip_code}. "
                f"Skipping retrain."
            )
            return

        self._last_df   = df
        df_with_ind     = self._add_indicators(df)

        # ── XGBoost features/labels (existing pipeline) ───────
        features = build_features(df)
        labels   = build_labels(df, horizon=3, threshold=0.005)

        # ── Prepare new model instances ───────────────────────
        new_xgb  = XGBSignalModel(
            model_path=f"ml/models/xgb_{self.scrip_code}.pkl"
        )
        new_lgbm = LGBMModel(self.scrip_code)
        new_lstm = LSTMModel(self.scrip_code)
        new_rl   = RLAgent(
            model_path=f"ml/models/ppo_{self.scrip_code}"
        )

        # ── Train XGBoost ─────────────────────────────────────
        def _train_xgb():
            if features.empty:
                logger.warning(
                    f"Empty features for XGB ({self.scrip_code})"
                )
                return
            try:
                new_xgb.train(features, labels)
            except Exception as e:
                logger.error(f"XGB train error: {e}")

        # ── Train LGBM ────────────────────────────────────────
        def _train_lgbm():
            try:
                new_lgbm.train(df_with_ind)
            except Exception as e:
                logger.error(f"LGBM train error: {e}")

        # ── Train LSTM ────────────────────────────────────────
        def _train_lstm():
            try:
                new_lstm.train(df_with_ind)
            except Exception as e:
                logger.error(f"LSTM train error: {e}")

        # ── Train RL ──────────────────────────────────────────
        def _train_rl():
            if not new_rl.available:
                logger.info(
                    f"RL not available — skipping "
                    f"({self.scrip_code})"
                )
                return
            try:
                aligned          = features.copy()
                aligned["close"] = df["close"].loc[features.index]
                new_rl.train(
                    aligned.drop(columns=["close"]),
                    aligned["close"],
                    timesteps=300_000
                )
            except Exception as e:
                logger.error(f"RL train error: {e}")

        # ── Run all in parallel ───────────────────────────────
        threads = [
            threading.Thread(target=_train_xgb,  daemon=True),
            threading.Thread(target=_train_lgbm, daemon=True),
            threading.Thread(target=_train_lstm, daemon=True),
            threading.Thread(target=_train_rl,   daemon=True),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=180)   # max 3 min per model

        # ── Atomic swap — never leaves bot modelless ──────────
        with self._lock:
            self.xgb  = new_xgb
            self.lgbm = new_lgbm
            self.lstm = new_lstm
            self.rl   = new_rl  if new_rl.available else self.rl

        duration = time.time() - start_time
        logger.info(
            f"✅ Retrain complete: {self.scrip_code} "
            f"in {duration:.0f}s"
        )

        # ── Telegram notification ─────────────────────────────
        try:
            from notifier import TelegramNotifier
            TelegramNotifier().model_retrained(
                instrument      = self.scrip_code,
                samples         = len(features),
                duration_seconds= duration,
                next_retrain    = "Next Sunday 18:30"
            )
        except Exception:
            pass

    # ─────────────────────────────────────────────────────────
    # SIGNAL GENERATION  — full pipeline
    # ─────────────────────────────────────────────────────────

    def get_signal(self, df: pd.DataFrame,
                   sentiment: float = 0.0) -> dict:
        """
        Full multi-model signal pipeline.

        Flow:
          1. Time filter        → skip bad hours
          2. Market regime      → filter direction
          3. All model votes    → XGB + LGBM + LSTM + RL
          4. Pattern bonus      → candlestick patterns
          5. MetaModel          → weighted aggregation
          6. Confidence gate    → < 55% → HOLD
          7. Regime gate        → counter-trend → HOLD

        Returns dict with final signal + all sub-signals
        for transparency and logging.
        """
        with self._lock:

            # ── Guard: not enough data ─────────────────────────
            if df is None or len(df) < 50:
                return self._hold_response("Not enough candles")

            self._last_df  = df
            df_ind         = self._add_indicators(df)

            if df_ind.empty:
                return self._hold_response("Indicators empty")

            # ── 1. Time filter ────────────────────────────────
            time_info = _time_filter.get_multiplier()
            if not time_info["should_trade"]:
                return self._hold_response(
                    time_info["label"],
                    time=time_info["label"]
                )

            # ── 2. Market regime ──────────────────────────────
            regime = _regime.detect(df_ind)

            # ── 3. Candlestick patterns ───────────────────────
            pattern = _pattern.detect_all(df_ind)

            # ── 4. Model predictions ──────────────────────────
            # XGBoost (existing feature pipeline)
            features   = build_features(df)
            xgb_result = (
                self.xgb.predict(features)
                if not features.empty
                else {"signal": "HOLD", "confidence": 0.0}
            )

            # LightGBM
            lgbm_result = self.lgbm.predict(df_ind)

            # LSTM
            lstm_result = self.lstm.predict(df_ind)

            # RL Agent
            rl_result = self._get_rl_signal(features)

            # ── 5. Weighted vote aggregation ──────────────────
            #
            #  Weights reflect typical reliability ranking.
            #  MetaModel.update_weights() shifts these over time
            #  based on which model was actually correct.
            #
            votes = {"BUY": 0.0, "SELL": 0.0, "HOLD": 0.0}

            votes[xgb_result["signal"]]  += 3.0 * xgb_result["confidence"]
            votes[lgbm_result["signal"]] += 2.5 * lgbm_result["confidence"]
            votes[lstm_result["signal"]] += 2.0 * lstm_result["confidence"]
            votes[rl_result["signal"]]   += 2.0 * rl_result["confidence"]

            # Pattern bonus (no confidence multiplier needed,
            # patterns already encode their own win-rate)
            if pattern["pattern"] != "NONE":
                votes[pattern["signal"]] += 1.5 * pattern["confidence"]

            # ── 6. MetaModel aggregation + sentiment ──────────
            meta = _meta_model.predict(
                xgb_signal = xgb_result["signal"],
                xgb_conf   = xgb_result["confidence"],
                rl_result  = rl_result,
                sentiment  = sentiment
            )

            # Blend raw votes with meta output (meta acts as
            # a tiebreaker and applies sentiment hard-blocks)
            votes[meta["signal"]] += 2.0 * meta["confidence"]

            # Determine winner
            total  = sum(votes.values())
            final  = max(votes, key=votes.get)
            conf   = votes[final] / total if total > 0 else 0.0

            # ── 7. Regime gate ────────────────────────────────
            should_trade, regime_reason, size_mult = \
                _regime.should_trade(regime, final)

            if not should_trade:
                return self._hold_response(
                    regime_reason,
                    xgb     = xgb_result["signal"],
                    lgbm    = lgbm_result["signal"],
                    lstm    = lstm_result["signal"],
                    rl      = rl_result["signal"],
                    pattern = pattern["pattern"],
                    regime  = regime["regime"]
                )

            # ── 8. Apply multipliers ──────────────────────────
            # Time quality × regime quality → scale confidence
            conf *= time_info["multiplier"] * size_mult

            # ── 9. Minimum confidence gate ────────────────────
            if conf < 0.55:
                final = "HOLD"
                conf  = 0.0

            return {
                "signal":     final,
                "confidence": round(conf, 3),

                # Sub-signals for logging / dashboard
                "xgb":        xgb_result["signal"],
                "lgbm":       lgbm_result["signal"],
                "lstm":       lstm_result["signal"],
                "rl":         rl_result["signal"],
                "pattern":    pattern["pattern"],
                "regime":     regime["regime"],
                "time":       time_info["label"],
                "size_mult":  size_mult,
                "votes":      votes,
                "meta":       meta["signal"],
            }

    # ─────────────────────────────────────────────────────────
    # FEEDBACK LOOP  — call after trade closes
    # ─────────────────────────────────────────────────────────

    def record_trade_outcome(
        self,
        signal:    str,
        entry:     float,
        exit_price:float,
        xgb_was:   str,
        rl_was:    str,
        sentiment: float
    ):
        """
        Call this when a trade closes.
        Updates MetaModel weights so better models
        get more vote weight over time.

        Example:
            trainer.record_trade_outcome(
                signal     = "BUY",
                entry      = 2450.0,
                exit_price = 2468.0,
                xgb_was    = "BUY",
                rl_was     = "HOLD",
                sentiment  = 0.2
            )
        """
        profitable = exit_price > entry if signal == "BUY" \
                     else exit_price < entry

        pnl_pct = (
            (exit_price - entry) / entry * 100
            if signal == "BUY"
            else (entry - exit_price) / entry * 100
        )

        # Did each model agree with the winning direction?
        xgb_correct  = (xgb_was == signal) and profitable
        rl_correct   = (rl_was  == signal) and profitable
        sent_correct = (
            (sentiment > 0.3 and signal == "BUY") or
            (sentiment < -0.3 and signal == "SELL")
        ) and profitable

        _meta_model.update_weights(
            xgb_correct       = xgb_correct,
            rl_correct        = rl_correct,
            sentiment_correct = sent_correct
        )

        # Persist to trade memory
        _trade_memory.record(
            scrip_code  = self.scrip_code,
            signal      = signal,
            entry       = entry,
            exit_price  = exit_price,
            pnl_pct     = pnl_pct,
            profitable  = profitable
        )

        logger.info(
            f"📊 Trade recorded: {self.scrip_code} | "
            f"{signal} | PnL={pnl_pct:+.2f}% | "
            f"{'✅' if profitable else '❌'}"
        )

    # ─────────────────────────────────────────────────────────
    # SCHEDULING
    # ─────────────────────────────────────────────────────────

    def start_schedule(self, retrain_time: str = "18:30"):
        """
        Schedules:
          - Weekly retrain every Sunday at retrain_time
          - Weekly interval re-evaluation every Sunday at 17:00
          - Initial train if no model exists yet
        """
        # Weekly retrain
        schedule.every().sunday.at(retrain_time).do(self.retrain)

        # Weekly interval re-evaluation
        schedule.every().sunday.at("17:00").do(
            self._reselect_interval
        )

        # Initial train if no model saved
        if not self.xgb.is_trained:
            logger.info(
                f"No existing model for {self.scrip_code}. "
                f"Running initial train..."
            )
            threading.Thread(
                target=self.retrain, daemon=True
            ).start()

        def _loop():
            while True:
                schedule.run_pending()
                time.sleep(60)

        thread = threading.Thread(target=_loop, daemon=True)
        thread.start()
        logger.info(
            f"📅 Scheduler started: {self.scrip_code} | "
            f"retrain every Sunday @ {retrain_time}"
        )
        return thread

    def _reselect_interval(self):
        """
        Re-evaluates best interval weekly.
        Swaps only if a meaningfully better interval is found.
        """
        try:
            from ml.interval_selector import IntervalSelector
            selector     = IntervalSelector(
                self.api, self.scrip_code, self.security_id
            )
            new_interval = selector.find_best()

            if new_interval != self.interval:
                logger.info(
                    f"📊 Interval updated: "
                    f"{self.interval} → {new_interval} "
                    f"({self.scrip_code})"
                )
                self.interval = new_interval
            else:
                logger.info(
                    f"Interval unchanged: {self.interval} "
                    f"({self.scrip_code})"
                )
        except Exception as e:
            logger.warning(f"Interval reselect failed: {e}")

    # ─────────────────────────────────────────────────────────
    # HELPERS
    # ─────────────────────────────────────────────────────────

    def _get_rl_signal(self, features: pd.DataFrame) -> dict:
        """
        Wraps the existing RLAgent.predict() into the
        standard {signal, confidence} dict format.
        """
        if features.empty:
            return {"signal": "HOLD", "confidence": 0.0}
        try:
            obs    = features.iloc[-1].values
            obs_in = (
                np.append(obs, [0.0, 0.0]).astype(np.float32)
                if self.rl.available
                else obs
            )
            action = self.rl.predict(obs_in)
            signal = {0: "HOLD", 1: "BUY", 2: "SELL"}.get(
                action, "HOLD"
            )
            return {
                "signal":     signal,
                # RL gives no probability — use fixed conf
                # (MetaModel will adjust weight over time)
                "confidence": 0.6 if signal != "HOLD" else 0.0
            }
        except Exception as e:
            logger.error(f"RL predict error: {e}")
            return {"signal": "HOLD", "confidence": 0.0}

    @staticmethod
    def _hold_response(reason: str = "", **extra) -> dict:
        """
        Returns a standard HOLD dict.
        Accepts optional kwargs for sub-signal logging.
        """
        base = {
            "signal":     "HOLD",
            "confidence": 0.0,
            "xgb":        "HOLD",
            "lgbm":       "HOLD",
            "lstm":       "HOLD",
            "rl":         "HOLD",
            "pattern":    "NONE",
            "regime":     "UNKNOWN",
            "time":       "",
            "size_mult":  0.0,
            "votes":      {"BUY": 0.0, "SELL": 0.0, "HOLD": 1.0},
            "reason":     reason,
        }
        base.update(extra)
        return base
