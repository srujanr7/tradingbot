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
from notifier import TelegramNotifier
from watchlist import FullMarketScanner
from position_manager import PositionManager

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("bot.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────
INTERVAL       = "15minute"
MARKET_OPEN    = "09:15"
MARKET_CLOSE   = "15:20"
SQUAREOFF_AT   = "15:10"
MAX_EQUITY_POS = 3
MAX_FNO_POS    = 2
MIN_CONFIDENCE = 0.70
# ─────────────────────────────────────────────────────────────

api      = INDstocksAPI()
risk     = RiskManager(max_position_pct=0.02, daily_loss_limit_pct=0.03)
notifier = TelegramNotifier()
scanner  = FullMarketScanner(
    api,
    top_n_equity=10,
    top_n_fno=5,
    top_n_index=2
)
posmgr = PositionManager(
    max_equity=MAX_EQUITY_POS,
    max_fno=MAX_FNO_POS,
    min_confidence=MIN_CONFIDENCE
)

# ── ML Trainers ───────────────────────────────────────────────
trainers      = {}
_trainer_lock = threading.Lock()

def get_trainer(cfg: dict) -> AutoTrainer:
    """Get or create ML trainer for an instrument. Thread-safe."""
    key = cfg["scrip_code"]
    if key not in trainers:
        with _trainer_lock:
            if key not in trainers:
                trainers[key] = AutoTrainer(
                    cfg["scrip_code"],
                    cfg["security_id"],
                    INTERVAL
                )
                trainers[key].start_schedule(retrain_time="18:30")
                logger.info(f"🧠 Trainer created for {cfg['name']}")
                time.sleep(1.0)
    return trainers[key]

# ── WebSocket Price Feed ──────────────────────────────────────

def on_price_tick(token, ltp):
    logger.debug(f"Tick → {token}: ₹{ltp}")

def on_order_update(order):
    status   = order["order_status"]
    order_id = order["order_id"]
    logger.info(f"📬 {order_id} → {status}")
    if status == "FAILED":
        notifier.error("OrderFailed", f"Order {order_id} rejected by exchange")

price_feed = PriceFeed(
    instruments=[],
    mode="ltp",
    on_tick=on_price_tick
)
# REST-only order feed — no WebSocket needed at 15min timeframe
order_feed = OrderFeed(on_update=on_order_update)

# ── Helpers ───────────────────────────────────────────────────

def is_market_open() -> bool:
    now = datetime.now().strftime("%H:%M")
    return MARKET_OPEN <= now <= MARKET_CLOSE

def get_candles(scrip_code: str):
    end   = int(time.time() * 1000)
    start = end - (7 * 24 * 60 * 60 * 1000)
    return api.get_historical(scrip_code, INTERVAL, start, end)

def get_balance(segment: str) -> float:
    funds = api.get_funds()
    if not funds:
        return 0.0
    avl = funds.get("detailed_avl_balance", {})
    return avl.get("eq_cnc" if segment == "EQUITY" else "future", 0.0)

# ── Core cycle ────────────────────────────────────────────────

def run_cycle():
    if not is_market_open():
        return

    if notifier.bot_paused:
        logger.info("Bot paused via Telegram.")
        return

    active = scanner.get_active()
    if not active:
        logger.warning("No active instruments. Waiting for scanner.")
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

            df = get_candles(cfg["scrip_code"])
            if df is None or len(df) < 50:
                logger.warning(
                    f"[{cfg['name']}] Insufficient candle data, skipping."
                )
                return

            result = get_trainer(cfg).get_signal(df)

            logger.info(
                f"[{cfg['name']}] ₹{ltp} | "
                f"{result['signal']} {result['confidence']:.1%} | "
                f"XGB={result['xgb']} RL={result['rl']}"
            )

            entry = {"cfg": cfg, "result": result, "ltp": ltp}

            if (posmgr.has_position(cfg["scrip_code"]) and
                    result["signal"] == "SELL"):
                exit_signals.append(entry)
            elif result["signal"] == "BUY":
                signals.append(entry)

        except Exception as e:
            logger.error(f"Scan error [{cfg['name']}]: {e}")

    threads = [
        threading.Thread(target=scan_instrument, args=(cfg,), daemon=True)
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
        f"Equity slots: {posmgr.equity_slots_free()} | "
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
        f"Equity: {status['equity_open']}/{MAX_EQUITY_POS} | "
        f"FNO: {status['fno_open']}/{MAX_FNO_POS} | "
        f"Win rate: {status['win_rate']:.1f}%"
    )

# ── Entry ─────────────────────────────────────────────────────

def process_entry(cfg: dict, result: dict, ltp: float):
    balance = get_balance(cfg["segment"])

    if not risk.can_trade(balance):
        limit = balance * risk.daily_loss_limit_pct
        notifier.kill_switch(risk.daily_pnl, limit)
        return

    limit = balance * risk.daily_loss_limit_pct
    if risk.daily_pnl < -(limit * 0.8):
        notifier.risk_warning(risk.daily_pnl, limit, balance)

    qty = posmgr.position_size(
        balance, ltp,
        cfg["segment"],
        result["confidence"]
    )

    margin = api.check_margin(
        cfg["security_id"], qty, ltp,
        segment=cfg["segment"],
        product=cfg["product"]
    )
    if margin and margin["total_margin"] > balance:
        notifier.insufficient_margin(
            cfg["name"],
            margin["total_margin"],
            balance
        )
        return

    order = api.place_order(
        txn_type="BUY",
        security_id=cfg["security_id"],
        qty=qty,
        order_type="LIMIT",
        limit_price=round(ltp * 1.001, 2),
        segment=cfg["segment"],
        product=cfg["product"],
        exchange=cfg["exchange"]
    )

    if order:
        fill      = order_feed.wait_for_fill(
            order["order_id"],
            timeout=30,
            segment=cfg["segment"]   # pass segment for REST poll
        )
        avg_price = fill.get("average_price", ltp)

        if fill.get("order_status") == "FAILED":
            notifier.order_failed(
                "BUY", cfg["name"], qty,
                fill.get("reason", "Unknown")
            )
            return

        posmgr.open_position(
            scrip_code=cfg["scrip_code"],
            name=cfg["name"],
            segment=cfg["segment"],
            qty=qty,
            entry_price=avg_price,
            order_id=order["order_id"],
            signal_meta=result
        )
        notifier.trade_executed(
            side="BUY",
            name=cfg["name"],
            segment=cfg["segment"],
            qty=qty,
            price=avg_price,
            confidence=result["confidence"],
            xgb=result["xgb"],
            rl=result["rl"]
        )

# ── Exit ──────────────────────────────────────────────────────

def process_exit(cfg: dict, ltp: float):
    pos = posmgr.positions.get(cfg["scrip_code"])
    if not pos:
        return

    order = api.place_order(
        txn_type="SELL",
        security_id=cfg["security_id"],
        qty=pos["qty"],
        order_type="MARKET",
        segment=cfg["segment"],
        product=cfg["product"],
        exchange=cfg["exchange"]
    )

    if order:
        fill       = order_feed.wait_for_fill(
            order["order_id"],
            timeout=30,
            segment=cfg["segment"]   # pass segment for REST poll
        )
        exit_price = fill.get("average_price", ltp)
        closed     = posmgr.close_position(cfg["scrip_code"], exit_price)
        pnl        = closed.get("pnl", 0)
        risk.update_pnl(pnl)

        notifier.trade_closed(
            name=cfg["name"],
            qty=pos["qty"],
            entry=pos["entry"],
            exit_price=exit_price,
            pnl=pnl,
            daily_pnl=risk.daily_pnl
        )

# ── Scheduled jobs ────────────────────────────────────────────

def square_off_all():
    logger.info("⏰ Auto square-off triggered")
    notifier.squareoff_alert(posmgr.positions)
    for scrip_code, pos in list(posmgr.positions.items()):
        cfg = next(
            (s for s in scanner.universe_equity + scanner.universe_fno
             if s["scrip_code"] == scrip_code),
            None
        )
        if cfg:
            ltp = price_feed.get_ltp(
                cfg["ws_token"].split(":")[1]
            ) or pos["entry"]
            process_exit(cfg, ltp)

def daily_reset():
    risk.reset_daily()
    logger.info("🔄 Daily counters reset")

def send_daily_summary():
    status  = posmgr.status()
    balance = get_balance("EQUITY") + get_balance("DERIVATIVE")
    notifier.daily_summary(
        total_pnl=risk.daily_pnl,
        trades=risk.trades_today,
        wins=status["win_count"],
        balance=balance
    )

def refresh_ws_subscriptions():
    """Update WebSocket subscriptions after each Tier 1 scan."""
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
                f"🔄 WebSocket updated: {len(new_tokens)} instruments"
            )
        except Exception as e:
            logger.error(f"WS subscription update error: {e}")

# ── Boot ──────────────────────────────────────────────────────

if __name__ == "__main__":
    logger.info("🚀 Booting algo bot — Full Market Mode...")

    # Step 1: Load full instrument universe
    scanner.load_universe()

    # Step 2: First Tier 1 scan — shortlist top instruments
    scanner.tier1_scan()

    # Step 3: Start price feed WebSocket with shortlisted tokens
    price_feed.instruments = scanner.get_ws_tokens()
    price_feed.start()

    # Step 4: Start order feed (REST polling mode — no WebSocket)
    order_feed.start()

    time.sleep(2)

    # Step 5: Start background Tier 1 refresh every 30 min
    scanner.start_background_refresh()

    # Step 6: Start Telegram command listener
    notifier.start_command_listener(bot_ref=sys.modules[__name__])

    # Step 7: Schedule all jobs
    schedule.every(60).seconds.do(run_cycle)
    schedule.every(30).minutes.do(refresh_ws_subscriptions)
    schedule.every().day.at("09:14").do(daily_reset)
    schedule.every().day.at(SQUAREOFF_AT).do(square_off_all)
    schedule.every().day.at("15:30").do(send_daily_summary)
    schedule.every().day.at("00:01").do(scanner.load_universe)

    # Step 8: Startup alert
    notifier.bot_started()

    logger.info("✅ All systems running — scanning full NSE market")

    # Step 9: Main loop
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
