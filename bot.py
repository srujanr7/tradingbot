import time
import logging
import schedule
from datetime import datetime
from api import INDstocksAPI
from strategy import MACrossRSIStrategy
from risk import RiskManager

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
SCRIP_CODE  = "NSE_2885"      # Reliance — change to your stock
SECURITY_ID = "2885"          # From instruments CSV
INTERVAL    = "15minute"      # Candle timeframe
CHECK_EVERY = 60              # Seconds between checks
MARKET_OPEN  = "09:15"
MARKET_CLOSE = "15:20"
# ─────────────────────────────────────────────────────────────

api      = INDstocksAPI()
strategy = MACrossRSIStrategy(fast=9, slow=21)
risk     = RiskManager(max_position_pct=0.02, daily_loss_limit_pct=0.03)

position = None   # Tracks open position: None or dict

def is_market_open() -> bool:
    now = datetime.now().strftime("%H:%M")
    return MARKET_OPEN <= now <= MARKET_CLOSE

def get_candles():
    """Fetch last 7 days of 15-min candles"""
    import time as t
    end   = int(t.time() * 1000)
    start = end - (7 * 24 * 60 * 60 * 1000)
    return api.get_historical(SCRIP_CODE, INTERVAL, start, end)

def run_cycle():
    global position

    if not is_market_open():
        logger.info("Market closed. Skipping cycle.")
        return

    # 1. Get account balance
    funds = api.get_funds()
    if not funds:
        logger.error("Could not fetch funds.")
        return
    balance = funds.get("detailed_avl_balance", {}).get("eq_cnc", 0)
    logger.info(f"Available balance (CNC): ₹{balance:.2f}")

    # 2. Kill switch check
    if not risk.can_trade(balance):
        return

    # 3. Get market data + generate signal
    df = get_candles()
    signal = strategy.generate_signal(df)
    ltp    = api.get_ltp(SCRIP_CODE).get(SCRIP_CODE, 0)
    logger.info(f"Signal: {signal} | LTP: ₹{ltp}")

    # 4. Execute
    if signal == "BUY" and position is None:
        qty = risk.position_size(balance, ltp)

        # Check margin first
        margin = api.check_margin(SECURITY_ID, qty, ltp)
        if margin and margin["total_margin"] > balance:
            logger.warning("Insufficient margin. Skipping.")
            return

        order = api.place_order(
            txn_type="BUY",
            security_id=SECURITY_ID,
            qty=qty,
            order_type="LIMIT",
            limit_price=round(ltp * 1.001, 2)  # 0.1% above LTP
        )
        if order:
            position = {"order_id": order["order_id"], "qty": qty, "entry": ltp}
            logger.info(f"✅ BUY | qty={qty} @ ₹{ltp}")

    elif signal == "SELL" and position is not None:
        order = api.place_order(
            txn_type="SELL",
            security_id=SECURITY_ID,
            qty=position["qty"],
            order_type="MARKET"
        )
        if order:
            pnl = (ltp - position["entry"]) * position["qty"]
            risk.update_pnl(pnl)
            logger.info(f"✅ SELL | PnL: ₹{pnl:.2f}")
            position = None

def square_off_all():
    """Force close all positions at 3:15 PM"""
    global position
    if position:
        logger.info("⏰ Auto square-off triggered")
        api.place_order("SELL", SECURITY_ID, position["qty"], order_type="MARKET")
        position = None

def daily_reset():
    risk.reset_daily()
    logger.info("🔄 Daily counters reset")

# ── Scheduler ────────────────────────────────────────────────
schedule.every(CHECK_EVERY).seconds.do(run_cycle)
schedule.every().day.at("09:14").do(daily_reset)
schedule.every().day.at("15:15").do(square_off_all)

if __name__ == "__main__":
    logger.info("🚀 Bot started")
    while True:
        schedule.run_pending()
        time.sleep(1)
