import sys
import time
import logging
import schedule
import threading
from datetime import datetime
from api import INDstocksAPI
from risk import RiskManager
from websocket_feed import PriceFeed, OrderFeed
from ml.trainer import AutoTrainer
from ml.trade_memory import TradeMemory
from ml.risk_reward import TrailingStop
from notifier import TelegramNotifier
from watchlist import FullMarketScanner
from position_manager import PositionManager
import config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(config.LOG_FILE),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

import requests as _req
try:
    _ip = _req.get("https://api.ipify.org", timeout=5).text
    logger.info(f"🌐 Outbound IP: {_ip}")
except Exception:
    logger.warning("Could not detect outbound IP")

# ── Runtime config ────────────────────────────────────────────
TRADE_MODE     = "MIS"
TRADE_CAPITAL  = None
MAX_EQUITY_POS = config.MAX_EQUITY_POS
MAX_FNO_POS    = config.MAX_FNO_POS
MIN_CONFIDENCE = config.TRADE_CONFIDENCE

# ── Core services ─────────────────────────────────────────────
api      = INDstocksAPI()
risk     = RiskManager(
    max_position_pct     = config.MAX_POSITION_PCT,
    daily_loss_limit_pct = config.DAILY_LOSS_LIMIT_PCT
)
notifier    = TelegramNotifier()
scanner     = FullMarketScanner(
    api, top_n_equity=10, top_n_fno=5, top_n_index=2
)
posmgr      = PositionManager(
    max_equity     = MAX_EQUITY_POS,
    max_fno        = MAX_FNO_POS,
    min_confidence = MIN_CONFIDENCE
)
trade_memory = TradeMemory()

# ── ML Trainers ───────────────────────────────────────────────
trainers      = {}
_trainer_lock = threading.Lock()


def get_trainer(cfg: dict) -> AutoTrainer:
    key = cfg["scrip_code"]
    if key not in trainers:
        with _trainer_lock:
            if key not in trainers:
                trainers[key] = AutoTrainer(
                    scrip_code   = cfg["scrip_code"],
                    security_id  = cfg["security_id"],
                    interval     = config.CANDLE_INTERVAL,
                    retrain_days = config.RETRAIN_DAYS
                )
                trainers[key].start_schedule(
                    retrain_time=config.RETRAIN_TIME
                )
                logger.info(
                    f"🧠 Trainer created: {cfg['name']} | "
                    f"interval={trainers[key].interval}"
                )
                time.sleep(1.0)
    return trainers[key]


# ── Token refresh ─────────────────────────────────────────────

def refresh_token():
    notifier.send(
        "🔑 <b>Action Required: Daily Token Refresh</b>\n\n"
        "INDstocks token expired at 6AM.\n\n"
        "1️⃣ SSH into your Google Cloud VM\n"
        "2️⃣ Run: <code>cd tradingbot && ./update_token.sh</code>\n"
        "3️⃣ Paste your new token when prompted\n"
        "4️⃣ Bot restarts automatically\n\n"
        "⏰ Complete before 9:15AM"
    )


# ── WebSocket feeds ───────────────────────────────────────────

def on_price_tick(token, ltp):
    logger.debug(f"Tick → {token}: ₹{ltp}")


def on_order_update(order):
    status   = order["order_status"]
    order_id = order["order_id"]
    logger.info(f"📬 {order_id} → {status}")
    if status == "FAILED":
        notifier.error("OrderFailed", f"Order {order_id} rejected")


price_feed = PriceFeed(
    instruments=[], mode="ltp", on_tick=on_price_tick
)
order_feed = OrderFeed(on_update=on_order_update)


# ── Helpers ───────────────────────────────────────────────────

def is_market_open() -> bool:
    now = datetime.now().strftime("%H:%M")
    return config.MARKET_OPEN <= now <= config.MARKET_CLOSE


def get_candles(cfg: dict):
    trainer  = trainers.get(cfg["scrip_code"])
    interval = trainer.interval if trainer else config.CANDLE_INTERVAL
    end      = int(time.time() * 1000)
    start    = end - (5 * 24 * 60 * 60 * 1000)
    return api.get_historical(cfg["scrip_code"], interval, start, end)


def get_balance(segment: str) -> float:
    funds = api.get_funds()
    if not funds:
        return 0.0
    avl = funds.get("detailed_avl_balance", {})
    if segment == "EQUITY":
        if TRADE_MODE == "MIS":
            raw = avl.get("eq_mis", 0.0)
        elif TRADE_MODE == "MTF":
            raw = avl.get("eq_mtf", 0.0)
        else:
            raw = avl.get("eq_cnc", 0.0)
    elif segment == "DERIVATIVE":
        raw = avl.get("future", 0.0) if TRADE_MODE == "MARGIN" \
              else avl.get("option_buy", 0.0)
    else:
        raw = 0.0
    return min(raw, TRADE_CAPITAL) if TRADE_CAPITAL else raw


def get_all_balances() -> dict:
    funds = api.get_funds()
    return funds if funds else {}


def _interval_minutes(interval: str) -> int:
    mapping = {
        "1minute": 1, "2minute": 2, "3minute": 3,
        "5minute": 5, "10minute": 10, "15minute": 15,
        "30minute": 30, "60minute": 60, "1day": 375
    }
    return mapping.get(interval, 5)


# ── SL/TP Monitor — runs first every cycle ────────────────────

def _monitor_open_positions(active: list):
    """
    Checks every open position for:
      1. Hard stop loss
      2. Take profit
      3. Trailing stop
      4. Time exit (> 2 hours)
    Exits immediately if any condition is met.
    Runs BEFORE scanning for new entries.
    """
    for scrip_code, pos in list(posmgr.positions.items()):
        meta = pos.get("signal_meta", {})

        cfg = next(
            (s for s in active if s["scrip_code"] == scrip_code),
            None
        )
        if not cfg:
            continue

        token = cfg["ws_token"].split(":")[1]
        ltp   = price_feed.get_ltp(token)
        if ltp is None:
            data = api.get_ltp(scrip_code)
            ltp  = data.get(scrip_code)
        if not ltp:
            continue

        ltp         = float(ltp)
        entry       = pos["entry"]
        stop_loss   = meta.get("stop_loss",   entry * 0.99)
        take_profit = meta.get("take_profit", entry * 1.02)
        trailing    = meta.get("trailing")

        # Track max drawdown live
        current_dd = max(0.0, (entry - ltp) / entry * 100)
        meta["max_drawdown"] = max(
            meta.get("max_drawdown", 0.0), current_dd
        )

        # Trailing stop update
        trail_exit = False
        locked_pnl = 0.0
        if trailing:
            t_info     = trailing.update(ltp)
            trail_exit = t_info["exit_now"]
            locked_pnl = t_info.get("locked_pnl", 0.0)

        hard_sl   = ltp <= stop_loss
        tp_hit    = ltp >= take_profit

        opened_at    = datetime.fromisoformat(
            pos.get("opened_at", datetime.now().isoformat())
        )
        held_minutes = (
            datetime.now() - opened_at
        ).total_seconds() / 60
        time_exit = held_minutes > 120

        if hard_sl or tp_hit or trail_exit or time_exit:
            reason = (
                "HARD_STOP_LOSS" if hard_sl    else
                "TAKE_PROFIT"    if tp_hit     else
                "TRAILING_STOP"  if trail_exit else
                "TIME_EXIT_2HR"
            )
            pnl_pct = (ltp - entry) / entry * 100
            logger.info(
                f"🚨 {reason}: {cfg['name']} | "
                f"LTP=₹{ltp} Entry=₹{entry} "
                f"PnL={pnl_pct:+.2f}%"
                + (f" Locked={locked_pnl:+.2f}%"
                   if trail_exit else "")
            )
            process_exit(cfg, ltp, reason=reason)


# ── Core cycle ────────────────────────────────────────────────

def run_cycle():
    if not is_market_open():
        return
    if notifier.bot_paused:
        logger.info("Bot paused via Telegram.")
        return
    if not api.is_token_valid():
        logger.warning("⚠️ Token invalid — skipping cycle.")
        return

    active = scanner.get_active()
    if not active:
        logger.warning("No active instruments.")
        return

    # SL/TP monitoring runs first — protect capital
    _monitor_open_positions(active)

    signals      = []
    exit_signals = []

    def scan_instrument(cfg):
        try:
            if posmgr.has_position(cfg["scrip_code"]):
                token = cfg["ws_token"].split(":")[1]
                ltp   = price_feed.get_ltp(token)
                if ltp is None:
                    data = api.get_ltp(cfg["scrip_code"])
                    ltp  = data.get(cfg["scrip_code"])
                if ltp:
                    df = get_candles(cfg)
                    if df is not None and len(df) >= 30:
                        result = get_trainer(cfg).get_signal(df)
                        if result["signal"] == "SELL":
                            exit_signals.append({
                                "cfg":    cfg,
                                "result": result,
                                "ltp":    ltp,
                                "reason": "MODEL_SELL_SIGNAL"
                            })
                return

            token = cfg["ws_token"].split(":")[1]
            ltp   = price_feed.get_ltp(token)
            if ltp is None:
                data = api.get_ltp(cfg["scrip_code"])
                ltp  = data.get(cfg["scrip_code"])
            if not ltp:
                return

            df = get_candles(cfg)
            if df is None or len(df) < 30:
                time.sleep(2)
                df = get_candles(cfg)
            if df is None or len(df) < 30:
                logger.warning(
                    f"[{cfg['name']}] Insufficient candles, skipping."
                )
                return

            result = get_trainer(cfg).get_signal(df)

            sl_tp_str = ""
            if result.get("stop_loss"):
                sl_tp_str = (
                    f" | SL=₹{result['stop_loss']} "
                    f"TP=₹{result['take_profit']} "
                    f"RR={result.get('rr_ratio', 0):.1f}"
                )

            logger.info(
                f"[{cfg['name']}] ₹{ltp} | "
                f"{result['signal']} {result['confidence']:.1%} | "
                f"XGB={result.get('xgb','')} "
                f"LGBM={result.get('lgbm','')} "
                f"LSTM={result.get('lstm','')} "
                f"RL={result.get('rl','')} | "
                f"Pattern={result.get('pattern','NONE')} "
                f"Regime={result.get('regime','')}"
                f"{sl_tp_str}"
            )

            if result["signal"] == "BUY":
                signals.append({
                    "cfg": cfg, "result": result, "ltp": ltp
                })

        except Exception as e:
            logger.error(f"Scan error [{cfg['name']}]: {e}")

    threads = [
        threading.Thread(
            target=scan_instrument, args=(cfg,), daemon=True
        )
        for cfg in active
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=15)

    for entry in exit_signals:
        process_exit(
            entry["cfg"], entry["ltp"],
            reason=entry.get("reason", "MODEL_SELL_SIGNAL")
        )

    ranked = posmgr.rank_signals(signals)
    logger.info(
        f"📊 {len(ranked)} actionable | "
        f"EQ={posmgr.equity_slots_free()} free | "
        f"FNO={posmgr.fno_slots_free()} free"
    )

    for entry in ranked:
        can_enter, reason = posmgr.can_enter(
            entry["cfg"]["segment"],
            entry["result"]["confidence"]
        )
        if not can_enter:
            logger.info(
                f"[{entry['cfg']['name']}] Skipped: {reason}"
            )
            continue
        process_entry(
            entry["cfg"], entry["result"], entry["ltp"]
        )

    status = posmgr.status()
    logger.info(
        f"📁 Open={status['open_positions']} | "
        f"EQ={status['equity_open']}/{MAX_EQUITY_POS} | "
        f"FNO={status['fno_open']}/{MAX_FNO_POS} | "
        f"WR={status['win_rate']:.1f}%"
    )


# ── Entry ─────────────────────────────────────────────────────

def process_entry(cfg: dict, result: dict, ltp: float):
    effective_segment = (
        "DERIVATIVE" if TRADE_MODE == "MARGIN"
        else cfg["segment"]
    )
    balance = get_balance(effective_segment)

    if not risk.can_trade(balance):
        notifier.kill_switch(
            risk.daily_pnl, risk.get_daily_limit(balance)
        )
        return

    limit = risk.get_daily_limit(balance)
    if risk.daily_pnl < -(limit * 0.8):
        notifier.risk_warning(risk.daily_pnl, limit, balance)

    # ── Fully automatic sizing — Kelly first, confidence fallback
    qty = posmgr.position_size(
        balance    = balance,
        price      = ltp,
        segment    = effective_segment,
        confidence = result["confidence"],
        kelly_pct  = result.get("kelly_pct", 0.0),
        rr_ratio   = result.get("rr_ratio",  0.0),
    )
    qty = risk.apply_per_trade_limit(qty, ltp)

    margin = api.check_margin(
        cfg["security_id"], qty, ltp,
        segment = effective_segment,
        product = TRADE_MODE
    )
    if margin and margin["total_margin"] > balance:
        notifier.insufficient_margin(
            cfg["name"], margin["total_margin"], balance
        )
        return

    order = api.place_order(
        txn_type    = "BUY",
        security_id = cfg["security_id"],
        qty         = qty,
        order_type  = "LIMIT",
        limit_price = round(
            ltp * (1 + config.LIMIT_ORDER_SLIPPAGE), 2
        ),
        segment     = effective_segment,
        product     = TRADE_MODE,
        exchange    = cfg["exchange"]
    )

    if order:
        fill      = order_feed.wait_for_fill(
            order["order_id"],
            timeout = config.ORDER_FILL_TIMEOUT,
            segment = effective_segment
        )
        avg_price = fill.get("average_price", ltp)

        if fill.get("order_status") == "FAILED":
            notifier.order_failed(
                "BUY", cfg["name"], qty,
                fill.get("reason", "Unknown")
            )
            return

        # ATR-based SL/TP — prefer signal levels, fall back fixed
        stop_loss   = result.get("stop_loss")  or round(avg_price * 0.99, 2)
        take_profit = result.get("take_profit") or round(avg_price * 1.02, 2)
        atr_val     = result.get("atr") or avg_price * 0.01

        trailing = TrailingStop(
            entry            = avg_price,
            atr              = atr_val,
            trail_multiplier = 1.5
        )

        sl_pct = result.get("sl_pct") or (
            (avg_price - stop_loss) / avg_price * 100
        )
        tp_pct = result.get("tp_pct") or (
            (take_profit - avg_price) / avg_price * 100
        )

        posmgr.open_position(
            scrip_code  = cfg["scrip_code"],
            name        = cfg["name"],
            segment     = effective_segment,
            qty         = qty,
            entry_price = avg_price,
            order_id    = order["order_id"],
            signal_meta = {
                **result,
                "stop_loss":    stop_loss,
                "take_profit":  take_profit,
                "trailing":     trailing,
                "max_drawdown": 0.0,
            }
        )

        notifier.trade_executed(
            side        = "BUY",
            name        = cfg["name"],
            segment     = effective_segment,
            qty         = qty,
            price       = avg_price,
            confidence  = result["confidence"],
            xgb         = result.get("xgb", ""),
            rl          = result.get("rl", ""),
            lgbm        = result.get("lgbm", ""),
            lstm        = result.get("lstm", ""),
            pattern     = result.get("pattern", "NONE"),
            regime      = result.get("regime", ""),
            stop_loss   = stop_loss,
            take_profit = take_profit,
            rr_ratio    = result.get("rr_ratio", 0),
            sl_pct      = sl_pct,
            tp_pct      = tp_pct,
        )


# ── Exit ──────────────────────────────────────────────────────

def process_exit(cfg: dict, ltp: float,
                 reason: str = "MODEL_SELL_SIGNAL"):
    pos = posmgr.positions.get(cfg["scrip_code"])
    if not pos:
        return

    order = api.place_order(
        txn_type    = "SELL",
        security_id = cfg["security_id"],
        qty         = pos["qty"],
        order_type  = "MARKET",
        segment     = pos["segment"],
        product     = TRADE_MODE,
        exchange    = cfg["exchange"]
    )

    if order:
        fill       = order_feed.wait_for_fill(
            order["order_id"],
            timeout = config.ORDER_FILL_TIMEOUT,
            segment = pos["segment"]
        )
        exit_price = fill.get("average_price", ltp)
        closed     = posmgr.close_position(
            cfg["scrip_code"], exit_price
        )
        pnl     = closed.get("pnl", 0)
        pnl_pct = (
            (exit_price - pos["entry"]) / pos["entry"] * 100
            if pos["entry"] else 0
        )
        risk.update_pnl(pnl)

        meta = pos.get("signal_meta", {})

        # Held candles calculation
        opened_at    = datetime.fromisoformat(
            pos.get("opened_at", datetime.now().isoformat())
        )
        held_minutes = (
            datetime.now() - opened_at
        ).total_seconds() / 60
        interval_key = (
            trainers[cfg["scrip_code"]].interval
            if cfg["scrip_code"] in trainers
            else "5minute"
        )
        held_candles = max(
            1, int(held_minutes / _interval_minutes(interval_key))
        )

        # Feedback loop
        trainer = trainers.get(cfg["scrip_code"])
        if trainer:
            trainer.record_trade_outcome(
                signal           = "BUY",
                entry            = pos["entry"],
                exit_price       = exit_price,
                xgb_was          = meta.get("xgb",  "HOLD"),
                rl_was           = meta.get("rl",   "HOLD"),
                sentiment        = meta.get("sentiment", 0.0),
                held_candles     = held_candles,
                max_drawdown_pct = meta.get("max_drawdown", 0.0),
                signal_meta      = meta,
            )

        # Persist full trade record
        trade_memory.record({
            "scrip_code":       cfg["scrip_code"],
            "name":             cfg["name"],
            "segment":          pos["segment"],
            "side":             "BUY",
            "entry":            pos["entry"],
            "exit":             exit_price,
            "qty":              pos["qty"],
            "pnl":              pnl,
            "pnl_pct":          round(pnl_pct, 3),
            "profitable":       pnl >= 0,
            "held_candles":     held_candles,
            "max_drawdown_pct": round(
                meta.get("max_drawdown", 0.0), 3
            ),
            "exit_reason":      reason,
            "stop_loss":        meta.get("stop_loss"),
            "take_profit":      meta.get("take_profit"),
            "confidence":       meta.get("confidence", 0),
            "xgb":              meta.get("xgb", ""),
            "lgbm":             meta.get("lgbm", ""),
            "lstm":             meta.get("lstm", ""),
            "rl":               meta.get("rl", ""),
            "pattern":          meta.get("pattern", ""),
            "regime":           meta.get("regime", ""),
            "sentiment":        meta.get("sentiment", 0),
            "rr_ratio":         meta.get("rr_ratio", 0),
            "ev_quality":       meta.get("ev_quality", ""),
            "interval":         interval_key,
        })

        notifier.trade_closed(
            name        = cfg["name"],
            qty         = pos["qty"],
            entry       = pos["entry"],
            exit_price  = exit_price,
            pnl         = pnl,
            daily_pnl   = risk.daily_pnl,
            reason      = reason,
        )


# ── Scheduled jobs ────────────────────────────────────────────

def square_off_all():
    logger.info("⏰ Auto square-off triggered")
    notifier.squareoff_alert(posmgr.positions)
    for scrip_code, pos in list(posmgr.positions.items()):
        cfg = next(
            (s for s in
             scanner.universe_equity + scanner.universe_fno
             if s["scrip_code"] == scrip_code),
            None
        )
        if cfg:
            ltp = (
                price_feed.get_ltp(
                    cfg["ws_token"].split(":")[1]
                ) or pos["entry"]
            )
            process_exit(cfg, ltp, reason="AUTO_SQUAREOFF")


def daily_reset():
    risk.reset_daily()
    logger.info("🔄 Daily counters reset")


def send_daily_summary():
    status  = posmgr.status()
    balance = get_balance("EQUITY") + get_balance("DERIVATIVE")
    notifier.daily_summary(
        total_pnl = risk.daily_pnl,
        trades    = risk.trades_today,
        wins      = status["win_count"],
        balance   = balance
    )


def refresh_ws_subscriptions():
    new_tokens = scanner.get_ws_tokens()
    if price_feed.ws:
        try:
            import json
            price_feed.ws.send(json.dumps({
                "action":      "subscribe",
                "mode":        "ltp",
                "instruments": new_tokens
            }))
            logger.info(
                f"🔄 WS updated: {len(new_tokens)} instruments"
            )
        except Exception as e:
            logger.error(f"WS subscription error: {e}")


# ── Boot ──────────────────────────────────────────────────────

if __name__ == "__main__":
    logger.info("🚀 Booting algo bot — Full Market Mode...")

    scanner.load_universe()
    scanner.tier1_scan()

    price_feed.instruments = scanner.get_ws_tokens()
    price_feed.start()
    order_feed.start()
    time.sleep(2)

    scanner.start_background_refresh()
    notifier.start_command_listener(bot_ref=sys.modules[__name__])

    schedule.every(config.CYCLE_INTERVAL_SECONDS).seconds.do(
        run_cycle
    )
    schedule.every(30).minutes.do(refresh_ws_subscriptions)
    schedule.every().day.at("06:30").do(refresh_token)
    schedule.every().day.at(config.DAILY_RESET_AT).do(daily_reset)
    schedule.every().day.at(config.SQUAREOFF_AT).do(square_off_all)
    schedule.every().day.at("15:30").do(send_daily_summary)
    schedule.every().day.at("00:01").do(scanner.load_universe)

    notifier.bot_started()

    logger.info("✅ All systems running")
    logger.info("⏳ Waiting 3 minutes for initial model training...")
    time.sleep(180)

    while True:
        try:
            schedule.run_pending()
            time.sleep(1)
        except KeyboardInterrupt:
            logger.info("Shutting down...")
            notifier.bot_stopped("Manual shutdown")
            price_feed.stop()
            order_feed.stop()
            notifier.stop_listener()
            break
        except Exception as e:
            logger.error(f"Main loop error: {e}")
            notifier.error("MainLoopError", str(e))
            time.sleep(5)
