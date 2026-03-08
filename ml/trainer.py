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
from ml.lgbm_model       import LGBMModel
from ml.lstm_model       import LSTMModel
from ml.regime_detector  import RegimeDetector
from ml.pattern_detector import PatternDetector
from ml.time_filter      import TimeFilter
from ml.meta_model       import MetaModel
from ml.trade_memory     import TradeMemory
from ml.reward_engine    import RewardEngine
from ml.pattern_memory   import PatternMemory
from ml.risk_reward      import (
    RiskRewardCalculator,
    TrailingStop,
    ExpectedValueFilter,
    KellyCriterion,
)

logger = logging.getLogger(__name__)

# ── Shared singletons (one instance across all AutoTrainers) ──
_regime        = RegimeDetector()
_pattern       = PatternDetector()
_time_filter   = TimeFilter()
_meta_model    = MetaModel()
_trade_memory  = TradeMemory()
_reward_engine = RewardEngine()
_rr_calc       = RiskRewardCalculator()
_ev_filter     = ExpectedValueFilter()
_kelly         = KellyCriterion()

# PatternMemory is initialised after _trade_memory
_pattern_mem   = PatternMemory(_trade_memory)


class AutoTrainer:
    """
    Full ML orchestrator for one instrument.

    Model stack:
      Layer 1 — Price prediction : XGBoost + LightGBM + LSTM
      Layer 2 — RL agent         : PPO (via RLAgent)
      Layer 3 — Market analysis  : RegimeDetector + PatternDetector
                                   + TimeFilter
      Layer 4 — Aggregation      : MetaModel (weighted votes,
                                   self-updating weights)

    Risk management:
      - ATR-based dynamic SL/TP on every BUY signal
      - Expected Value filter (negative EV → HOLD)
      - Kelly Criterion position sizing hint
      - TrailingStop tracked per open position
      - Adaptive confidence gate (tightens on loss streaks)

    Self-improves:
      - RewardEngine scores every trade outcome
      - MetaModel weights shift toward correct models
      - PatternMemory tracks per-setup historical win rates
      - Weekly full retrain every Sunday 18:30
      - Interval auto-selected via IntervalSelector
    """

    def __init__(self, scrip_code: str, security_id: str,
                 interval: str = "5minute",
                 retrain_days: int = 90):

        self.api          = INDstocksAPI()
        self.scrip_code   = scrip_code
        self.security_id  = security_id
        self.retrain_days = retrain_days
        self._lock        = threading.Lock()
        self._last_df     = None

        # Adaptive confidence gate — starts at 0.55,
        # tightens to max 0.80 after consecutive losses,
        # relaxes 0.01 per winning trade
        self._confidence_gate = 0.55

        if interval == "auto":
            self.interval = self._auto_select_interval()
        else:
            self.interval = interval

        self.xgb  = XGBSignalModel(
            model_path=f"ml/models/xgb_{scrip_code}.pkl"
        )
        self.lgbm = LGBMModel(scrip_code)
        self.lstm = LSTMModel(scrip_code)
        self.rl   = RLAgent(
            model_path=f"ml/models/ppo_{scrip_code}"
        )

        self.xgb.load()
        self.rl.load()

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
    # INDICATOR BUILDER
    # ─────────────────────────────────────────────────────────

    def _add_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        try:
            df["ema_9"]     = ta.ema(df["close"], length=9)
            df["ema_20"]    = ta.ema(df["close"], length=20)
            df["ema_50"]    = ta.ema(df["close"], length=50)
            df["ema_cross"] = (df["ema_9"] > df["ema_20"]).astype(int)

            df["rsi"]  = ta.rsi(df["close"], length=14)
            df["roc"]  = ta.roc(df["close"], length=10)
            df["mom"]  = ta.mom(df["close"], length=10)
            df["cci"]  = ta.cci(df["high"], df["low"], df["close"])

            macd         = ta.macd(df["close"])
            df["macd"]   = macd["MACD_12_26_9"]
            df["macd_s"] = macd["MACDs_12_26_9"]
            df["macd_h"] = macd["MACDh_12_26_9"]

            bb             = ta.bbands(df["close"])
            df["bb_pct"]   = bb["BBP_5_2.0"]
            df["bb_upper"] = bb["BBU_5_2.0"]
            df["bb_lower"] = bb["BBL_5_2.0"]

            df["atr"]     = ta.atr(df["high"], df["low"], df["close"])
            df["atr_pct"] = df["atr"] / df["close"]

            adx_df    = ta.adx(df["high"], df["low"], df["close"])
            df["adx"] = adx_df["ADX_14"]

            stoch         = ta.stoch(df["high"], df["low"], df["close"])
            df["stoch_k"] = stoch["STOCHk_14_3_3"]
            df["stoch_d"] = stoch["STOCHd_14_3_3"]

            df["vol_ratio"] = (
                df["volume"] / df["volume"].rolling(20).mean()
            )
            df["obv"] = ta.obv(df["close"], df["volume"])

            df["hl_ratio"] = (df["high"] - df["low"]) / df["close"]
            df["oc_ratio"] = (df["close"] - df["open"]) / df["close"]

            df.dropna(inplace=True)

        except Exception as e:
            logger.warning(
                f"Indicator error ({self.scrip_code}): {e}"
            )
        return df

    # ─────────────────────────────────────────────────────────
    # DATA FETCHING
    # ─────────────────────────────────────────────────────────

    def _fetch_training_data(self) -> pd.DataFrame | None:
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
                    self.scrip_code, self.interval,
                    current, chunk_end
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
                f"No training data for {self.scrip_code}"
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
        start_time = time.time()
        logger.info(
            f"🔄 Retrain started: {self.scrip_code} "
            f"@ {datetime.now().strftime('%H:%M:%S')}"
        )

        df = self._fetch_training_data()
        if df is None or len(df) < 200:
            logger.warning(
                f"Not enough data for {self.scrip_code}. "
                f"Skipping retrain."
            )
            return

        self._last_df = df
        df_with_ind   = self._add_indicators(df)
        features      = build_features(df)
        labels        = build_labels(df, horizon=3, threshold=0.005)

        new_xgb  = XGBSignalModel(
            model_path=f"ml/models/xgb_{self.scrip_code}.pkl"
        )
        new_lgbm = LGBMModel(self.scrip_code)
        new_lstm = LSTMModel(self.scrip_code)
        new_rl   = RLAgent(
            model_path=f"ml/models/ppo_{self.scrip_code}"
        )

        def _train_xgb():
            if features.empty:
                return
            try:
                new_xgb.train(features, labels)
            except Exception as e:
                logger.error(f"XGB train error: {e}")

        def _train_lgbm():
            try:
                new_lgbm.train(df_with_ind)
            except Exception as e:
                logger.error(f"LGBM train error: {e}")

        def _train_lstm():
            try:
                new_lstm.train(df_with_ind)
            except Exception as e:
                logger.error(f"LSTM train error: {e}")

        def _train_rl():
            if not new_rl.available:
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

        threads = [
            threading.Thread(target=_train_xgb,  daemon=True),
            threading.Thread(target=_train_lgbm, daemon=True),
            threading.Thread(target=_train_lstm, daemon=True),
            threading.Thread(target=_train_rl,   daemon=True),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=180)

        with self._lock:
            self.xgb  = new_xgb
            self.lgbm = new_lgbm
            self.lstm = new_lstm
            self.rl   = new_rl if new_rl.available else self.rl

        duration = time.time() - start_time
        logger.info(
            f"✅ Retrain complete: {self.scrip_code} "
            f"in {duration:.0f}s"
        )

        try:
            from notifier import TelegramNotifier
            TelegramNotifier().model_retrained(
                instrument       = self.scrip_code,
                samples          = len(features),
                duration_seconds = duration,
                next_retrain     = "Next Sunday 18:30"
            )
        except Exception:
            pass

    # ─────────────────────────────────────────────────────────
    # SIGNAL GENERATION
    # ─────────────────────────────────────────────────────────

    def get_signal(self, df: pd.DataFrame,
                   sentiment: float = 0.0) -> dict:
        """
        Full multi-model signal pipeline.

        Flow:
          1.  Time filter
          2.  Market regime
          3.  Candlestick patterns
          4.  Model votes (XGB + LGBM + LSTM + RL)
          5.  MetaModel aggregation + sentiment
          6.  Regime gate
          7.  Time + regime multipliers
          8.  Adaptive confidence gate
          9.  PatternMemory confidence adjustment
          10. ATR-based SL/TP (BUY only)
          11. Expected Value check (BUY only)
          12. Kelly position sizing hint (BUY only)
        """
        with self._lock:
            if df is None or len(df) < 50:
                return self._hold_response("Not enough candles")

            self._last_df = df
            df_ind        = self._add_indicators(df)

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
            features    = build_features(df)
            xgb_result  = (
                self.xgb.predict(features)
                if not features.empty
                else {"signal": "HOLD", "confidence": 0.0}
            )
            lgbm_result = self.lgbm.predict(df_ind)
            lstm_result = self.lstm.predict(df_ind)
            rl_result   = self._get_rl_signal(features)

            # ── 5. Weighted vote aggregation ──────────────────
            votes = {"BUY": 0.0, "SELL": 0.0, "HOLD": 0.0}
            votes[xgb_result["signal"]]  += 3.0 * xgb_result["confidence"]
            votes[lgbm_result["signal"]] += 2.5 * lgbm_result["confidence"]
            votes[lstm_result["signal"]] += 2.0 * lstm_result["confidence"]
            votes[rl_result["signal"]]   += 2.0 * rl_result["confidence"]

            if pattern["pattern"] != "NONE":
                votes[pattern["signal"]] += 1.5 * pattern["confidence"]

            # ── 6. MetaModel aggregation + sentiment ──────────
            meta = _meta_model.predict(
                xgb_signal = xgb_result["signal"],
                xgb_conf   = xgb_result["confidence"],
                rl_result  = rl_result,
                sentiment  = sentiment
            )
            votes[meta["signal"]] += 2.0 * meta["confidence"]

            total = sum(votes.values())
            final = max(votes, key=votes.get)
            conf  = votes[final] / total if total > 0 else 0.0

            # ── 7. Regime gate ────────────────────────────────
            should_trade, regime_reason, size_mult = \
                _regime.should_trade(regime, final)

            if not should_trade:
                return self._hold_response(
                    regime_reason,
                    xgb=xgb_result["signal"],
                    lgbm=lgbm_result["signal"],
                    lstm=lstm_result["signal"],
                    rl=rl_result["signal"],
                    pattern=pattern["pattern"],
                    regime=regime["regime"]
                )

            # ── 8. Apply time + regime multipliers ────────────
            conf *= time_info["multiplier"] * size_mult

            # ── 9. Adaptive confidence gate ───────────────────
            if conf < self._confidence_gate:
                return self._hold_response(
                    f"Confidence {conf:.1%} below gate "
                    f"{self._confidence_gate:.1%}",
                    xgb=xgb_result["signal"],
                    lgbm=lgbm_result["signal"],
                    lstm=lstm_result["signal"],
                    rl=rl_result["signal"],
                    pattern=pattern["pattern"],
                    regime=regime["regime"]
                )

            # ── 10. PatternMemory confidence adjustment ────────
            signal_dict = {
                "signal":     final,
                "confidence": conf,
                "pattern":    pattern["pattern"],
                "regime":     regime["regime"],
                "interval":   self.interval,
            }
            signal_dict = _pattern_mem.adjust_confidence(
                signal_dict, self.scrip_code
            )
            conf  = signal_dict["confidence"]
            final = signal_dict["signal"]

            # Re-check gate after PatternMemory adjustment
            if conf < self._confidence_gate:
                return self._hold_response(
                    signal_dict.get(
                        "reason",
                        "PatternMemory reduced confidence"
                    ),
                    pattern=pattern["pattern"],
                    regime=regime["regime"]
                )

            # ── 11. ATR-based SL/TP (BUY signals only) ────────
            rr     = {}
            ev     = {}
            kelly  = {}
            ltp_est = float(df_ind["close"].iloc[-1])

            if final == "BUY":
                rr = _rr_calc.calculate(ltp_est, df_ind)

                if not rr["acceptable"]:
                    return self._hold_response(
                        f"RR ratio {rr['rr_ratio']:.1f} "
                        f"below 2:1 minimum",
                        xgb=xgb_result["signal"],
                        rl=rl_result["signal"],
                        regime=regime["regime"]
                    )

                # ── 12. Expected Value check ───────────────────
                win_rate = _pattern_mem.get_historical_winrate(
                    self.scrip_code,
                    pattern["pattern"],
                    regime["regime"],
                    self.interval
                )
                # Blend historical WR with model confidence
                # so early on (no history) we still trade
                blended_wr = (
                    win_rate * 0.4 + conf * 0.6
                    if win_rate != 0.5
                    else conf
                )
                ev = _ev_filter.calculate(
                    win_rate      = blended_wr,
                    risk_amount   = rr["risk_per_share"],
                    reward_amount = rr["reward_per_share"]
                )
                if not ev["positive"]:
                    return self._hold_response(
                        f"Negative EV ({ev['ev_pct']:+.1f}%)",
                        xgb=xgb_result["signal"],
                        rl=rl_result["signal"],
                        regime=regime["regime"]
                    )

                # ── 13. Kelly position sizing ──────────────────
                kelly = _kelly.calculate(
                    win_rate = blended_wr,
                    rr_ratio = rr["rr_ratio"],
                    balance  = 0.0   # bot.py fills real balance
                )

            return {
                "signal":     final,
                "confidence": round(conf, 3),

                # Sub-signals
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

                # Risk management levels
                "stop_loss":   rr.get("stop_loss"),
                "take_profit": rr.get("take_profit"),
                "rr_ratio":    rr.get("rr_ratio"),
                "sl_pct":      rr.get("sl_pct"),
                "tp_pct":      rr.get("tp_pct"),
                "atr":         rr.get("atr"),
                "ev_quality":  ev.get("quality", ""),
                "kelly_pct":   kelly.get("safe_pct", 0.0),
                "pattern_wr":  signal_dict.get("pattern_wr"),
            }

    # ─────────────────────────────────────────────────────────
    # FEEDBACK LOOP
    # ─────────────────────────────────────────────────────────

    def record_trade_outcome(self,
                             signal:           str,
                             entry:            float,
                             exit_price:       float,
                             xgb_was:          str,
                             rl_was:           str,
                             sentiment:        float,
                             held_candles:     int   = 1,
                             max_drawdown_pct: float = 0.0,
                             signal_meta:      dict  = None):
        """
        Call when a trade closes.
          1. Calculates reward via RewardEngine
          2. Updates MetaModel weights
          3. Tightens/relaxes adaptive confidence gate
          4. Persists enriched record to TradeMemory
        """
        signal_meta = signal_meta or {}
        profitable  = (
            exit_price > entry if signal == "BUY"
            else exit_price < entry
        )
        pnl_pct = (
            (exit_price - entry) / entry * 100
            if signal == "BUY"
            else (entry - exit_price) / entry * 100
        )

        # ── RewardEngine ──────────────────────────────────────
        reward = _reward_engine.calculate(
            entry            = entry,
            exit_price       = exit_price,
            held_candles     = held_candles,
            signal           = signal_meta,
            max_drawdown_pct = max_drawdown_pct
        )
        logger.info(
            f"🎯 Reward: {reward:+.3f} | "
            f"PnL: {pnl_pct:+.2f}% | {self.scrip_code}"
        )

        # ── MetaModel weight update ───────────────────────────
        xgb_correct  = (xgb_was == signal) and profitable
        rl_correct   = (rl_was  == signal) and profitable
        sent_correct = (
            (sentiment > 0.3  and signal == "BUY") or
            (sentiment < -0.3 and signal == "SELL")
        ) and profitable

        _meta_model.update_weights(
            xgb_correct       = xgb_correct,
            rl_correct        = rl_correct,
            sentiment_correct = sent_correct
        )

        # ── Adaptive confidence gate ──────────────────────────
        recent = _trade_memory.get_last_n(self.scrip_code, n=3)
        if (len(recent) == 3
                and all(not t.get("profitable", True)
                        for t in recent)):
            old_gate = self._confidence_gate
            self._confidence_gate = min(
                0.80, self._confidence_gate + 0.05
            )
            logger.warning(
                f"⚠️ 3 losses in a row on {self.scrip_code} "
                f"— gate {old_gate:.0%} → "
                f"{self._confidence_gate:.0%}"
            )
        elif profitable:
            self._confidence_gate = max(
                0.55, self._confidence_gate - 0.01
            )

        # ── Persist trade ─────────────────────────────────────
        _trade_memory.record({
            "scrip_code":       self.scrip_code,
            "signal":           signal,
            "entry":            entry,
            "exit":             exit_price,
            "pnl_pct":          round(pnl_pct, 3),
            "profitable":       profitable,
            "reward":           reward,
            "held_candles":     held_candles,
            "max_drawdown_pct": round(max_drawdown_pct, 3),
            "xgb":              xgb_was,
            "rl":               rl_was,
            "sentiment":        sentiment,
            "pattern":          signal_meta.get("pattern", ""),
            "regime":           signal_meta.get("regime", ""),
            "interval":         self.interval,
            "confidence_gate":  self._confidence_gate,
        })

        logger.info(
            f"📊 Trade recorded: {self.scrip_code} | "
            f"{signal} | PnL={pnl_pct:+.2f}% | "
            f"{'✅' if profitable else '❌'} | "
            f"Gate={self._confidence_gate:.0%}"
        )

    # ─────────────────────────────────────────────────────────
    # SCHEDULING
    # ─────────────────────────────────────────────────────────

    def start_schedule(self, retrain_time: str = "18:30"):
        schedule.every().sunday.at(retrain_time).do(self.retrain)
        schedule.every().sunday.at("17:00").do(
            self._reselect_interval
        )

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
        if features.empty:
            return {"signal": "HOLD", "confidence": 0.0}
        try:
            obs    = features.iloc[-1].values
            obs_in = (
                np.append(obs, [0.0, 0.0]).astype(np.float32)
                if self.rl.available else obs
            )
            action = self.rl.predict(obs_in)
            signal = {0: "HOLD", 1: "BUY", 2: "SELL"}.get(
                action, "HOLD"
            )
            return {
                "signal":     signal,
                "confidence": 0.6 if signal != "HOLD" else 0.0
            }
        except Exception as e:
            logger.error(f"RL predict error: {e}")
            return {"signal": "HOLD", "confidence": 0.0}

    @staticmethod
    def _hold_response(reason: str = "", **extra) -> dict:
        base = {
            "signal":      "HOLD",
            "confidence":  0.0,
            "xgb":         "HOLD",
            "lgbm":        "HOLD",
            "lstm":        "HOLD",
            "rl":          "HOLD",
            "pattern":     "NONE",
            "regime":      "UNKNOWN",
            "time":        "",
            "size_mult":   0.0,
            "votes":       {"BUY": 0.0, "SELL": 0.0, "HOLD": 1.0},
            "reason":      reason,
            "stop_loss":   None,
            "take_profit": None,
            "rr_ratio":    None,
            "sl_pct":      None,
            "tp_pct":      None,
            "atr":         None,
            "ev_quality":  "",
            "kelly_pct":   0.0,
            "pattern_wr":  None,
        }
        base.update(extra)
        return base
