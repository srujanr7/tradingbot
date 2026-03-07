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

# ── Log outbound IP ───────────────────────────────────────────
import requests as _req
try:
    _ip = _req.get("https://api.ipify.org", timeout=5).text
    logger.info(f"🌐 Outbound IP: {_ip}")
except Exception:
    logger.warning("Could not detect outbound IP")

# ── Runtime config (adjustable via Telegram) ──────────────────
TRADE_MODE     = "MIS"
TRADE_CAPITAL  = None
MAX_EQUITY_POS = config.MAX_EQUITY_POS
MAX_FNO_POS    = config.MAX_FNO_POS
MIN_CONFIDENCE = config.TRADE_CONFIDENCE

# ── Core services ─────────────────────────────────────────────
api      = INDstocksAPI()
risk     = RiskManager(
    max_position_pct    = config.MAX_POSITION_PCT,
    daily_loss_limit_pct= config.DAILY_LOSS_LIMIT_PCT
)
notifier = TelegramNotifier()
scanner  = FullMarketScanner(
    api,
    top_n_equity = 10,
    top_n_fno    = 5,
    top_n_index  = 2
)
posmgr = PositionManager(
    max_equity     = MAX_EQUITY_POS,
    max_fno        = MAX_FNO_POS,
    min_confidence = MIN_CONFIDENCE
)
trade_memory = TradeMemory()

# ── ML Trainers — one per instrument ─────────────────────────
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
    logger.info("🔑 Daily token refresh alert sending...")
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
    instruments=[],
    mode="ltp",
    on_tick=on_price_tick
)
order_feed = OrderFeed(on_update=on_order_update)


# ── Helpers ───────────────────────────────────────────────────

def is_market_open() -> bool:
    now = datetime.now().strftime("%H:%M")
    return config.MARKET_OPEN <= now <= config.MARKET_CLOSE


def get_candles(cfg: dict):
    """
    Fetches candles using each instrument's own trainer interval.
    Falls back to config.CANDLE_INTERVAL if trainer not yet created.
    """
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
        if TRADE_MODE == "MARGIN":
            raw = avl.get("future", 0.0)
        else:
            raw = avl.get("option_buy", 0.0)
    else:
        raw = 0.0
    return min(raw, TRADE_CAPITAL) if TRADE_CAPITAL else raw


def get_all_balances() -> dict:
    funds = api.get_funds()
    return funds if funds else {}


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

    signals      = []
    exit_signals = []

    def scan_instrument(cfg):
        try:
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
                    f"[{cfg['name']}] Insufficient candles "
                    f"({len(df) if df is not None else 0}), skipping."
                )
                return

            result = get_trainer(cfg).get_signal(df)

            logger.info(
                f"[{cfg['name']}] ₹{ltp} | "
                f"{result['signal']} {result['confidence']:.1%} | "
                f"XGB={result['xgb']} LGBM={result.get('lgbm','')} "
                f"LSTM={result.get('lstm','')} RL={result['rl']} | "
                f"Pattern={result.get('pattern','NONE')} "
                f"Regime={result.get('regime','')}"
            )

            entry = {"cfg": cfg, "result": result, "ltp": ltp}

            if (posmgr.has_position(cfg["scrip_code"])
                    and result["signal"] == "SELL"):
                exit_signals.append(entry)
            elif result["signal"] == "BUY":
                signals.append(entry)

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
        process_exit(entry["cfg"], entry["ltp"])

    ranked = posmgr.rank_signals(signals)
    logger.info(
        f"📊 {len(ranked)} actionable signals | "
        f"EQ slots: {posmgr.equity_slots_free()} | "
        f"FNO slots: {posmgr.fno_slots_free()}"
    )

    for entry in ranked:
        can_enter, reason = posmgr.can_enter(
            entry["cfg"]["segment"],
            entry["result"]["confidence"]
        )
        if not can_enter:
            logger.info(f"[{entry['cfg']['name']}] Skipped: {reason}")
            continue
        process_entry(entry["cfg"], entry["result"], entry["ltp"])

    status = posmgr.status()
    logger.info(
        f"📁 Open: {status['open_positions']} | "
        f"EQ: {status['equity_open']}/{MAX_EQUITY_POS} | "
        f"FNO: {status['fno_open']}/{MAX_FNO_POS} | "
        f"Win rate: {status['win_rate']:.1f}%"
    )


# ── Entry ─────────────────────────────────────────────────────

def process_entry(cfg: dict, result: dict, ltp: float):
    effective_segment = (
        "DERIVATIVE" if TRADE_MODE == "MARGIN" else cfg["segment"]
    )
    balance = get_balance(effective_segment)

    if not risk.can_trade(balance):
        notifier.kill_switch(risk.daily_pnl,
                             risk.get_daily_limit(balance))
        return

    limit = risk.get_daily_limit(balance)
    if risk.daily_pnl < -(limit * 0.8):
        notifier.risk_warning(risk.daily_pnl, limit, balance)

    qty = posmgr.position_size(
        balance, ltp,
        effective_segment,
        result["confidence"]
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
        limit_price = round(ltp * (1 + config.LIMIT_ORDER_SLIPPAGE), 2),
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

        posmgr.open_position(
            scrip_code  = cfg["scrip_code"],
            name        = cfg["name"],
            segment     = effective_segment,
            qty         = qty,
            entry_price = avg_price,
            order_id    = order["order_id"],
            signal_meta = result
        )
        notifier.trade_executed(
            side       = "BUY",
            name       = cfg["name"],
            segment    = effective_segment,
            qty        = qty,
            price      = avg_price,
            confidence = result["confidence"],
            xgb        = result.get("xgb", ""),
            rl         = result.get("rl", ""),
            lgbm       = result.get("lgbm", ""),
            lstm       = result.get("lstm", ""),
            pattern    = result.get("pattern", "NONE"),
            regime     = result.get("regime", "")
        )


# ── Exit ──────────────────────────────────────────────────────

def process_exit(cfg: dict, ltp: float):
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
        closed     = posmgr.close_position(cfg["scrip_code"], exit_price)
        pnl        = closed.get("pnl", 0)
        pnl_pct    = (
            (exit_price - pos["entry"]) / pos["entry"] * 100
            if pos["entry"] else 0
        )
        risk.update_pnl(pnl)

        meta = pos.get("signal_meta", {})

        # ── Feedback loop: update MetaModel weights ───────────
        trainer = trainers.get(cfg["scrip_code"])
        if trainer:
            trainer.record_trade_outcome(
                signal     = "BUY",
                entry      = pos["entry"],
                exit_price = exit_price,
                xgb_was    = meta.get("xgb", "HOLD"),
                rl_was     = meta.get("rl",  "HOLD"),
                sentiment  = meta.get("sentiment", 0.0)
            )

        # ── Persist to trade memory ───────────────────────────
        hold_candles = max(
            1,
            int(
                (datetime.now() -
                 datetime.fromisoformat(pos["opened_at"])
                ).total_seconds() / 60
                / _interval_minutes(
                    trainers[cfg["scrip_code"]].interval
                    if cfg["scrip_code"] in trainers
                    else "5minute"
                )
            )
        )
        trade_memory.record({
            "scrip_code":    cfg["scrip_code"],
            "name":          cfg["name"],
            "segment":       pos["segment"],
            "side":          "BUY",
            "entry":         pos["entry"],
            "exit":          exit_price,
            "qty":           pos["qty"],
            "pnl":           pnl,
            "pnl_pct":       pnl_pct,
            "hold_candles":  hold_candles,
            "confidence":    meta.get("confidence", 0),
            "xgb":           meta.get("xgb", ""),
            "lgbm":          meta.get("lgbm", ""),
            "lstm":          meta.get("lstm", ""),
            "rl":            meta.get("rl", ""),
            "pattern":       meta.get("pattern", ""),
            "regime":        meta.get("regime", ""),
            "sentiment":     meta.get("sentiment", 0),
            "interval":      (
                trainers[cfg["scrip_code"]].interval
                if cfg["scrip_code"] in trainers
                else ""
            ),
        })

        notifier.trade_closed(
            name       = cfg["name"],
            qty        = pos["qty"],
            entry      = pos["entry"],
            exit_price = exit_price,
            pnl        = pnl,
            daily_pnl  = risk.daily_pnl
        )


def _interval_minutes(interval: str) -> int:
    """Convert interval string to minutes for hold time calc."""
    mapping = {
        "1minute": 1, "2minute": 2, "3minute": 3,
        "5minute": 5, "10minute": 10, "15minute": 15,
        "30minute": 30, "60minute": 60, "1day": 375
    }
    return mapping.get(interval, 5)


# ── Scheduled jobs ────────────────────────────────────────────

def square_off_all():
    logger.info("⏰ Auto square-off triggered")
    notifier.squareoff_alert(posmgr.positions)
    for scrip_code, pos in list(posmgr.positions.items()):
        cfg = next(
            (
                s for s in
                scanner.universe_equity + scanner.universe_fno
                if s["scrip_code"] == scrip_code
            ),
            None
        )
        if cfg:
            ltp = (
                price_feed.get_ltp(cfg["ws_token"].split(":")[1])
                or pos["entry"]
            )
            process_exit(cfg, ltp)


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
