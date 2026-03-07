import os
from dotenv import load_dotenv

load_dotenv()

# ── API Credentials ───────────────────────────────────────────
INDSTOCKS_TOKEN    = os.getenv("INDSTOCKS_TOKEN")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID   = os.getenv("TELEGRAM_CHAT_ID")

# ── Instruments ───────────────────────────────────────────────
# Change these to whichever stocks/contracts you want to trade
# Get security_id from the instruments CSV via /market/instruments
EQUITY = {
    "name":        "Reliance",
    "scrip_code":  "NSE_2885",     # Format: EXCHANGE_TOKEN
    "security_id": "2885",         # From instruments CSV
    "ws_token":    "NSE:2885",     # Format for WebSocket
    "segment":     "EQUITY",
    "product":     "CNC",          # CNC = delivery trade
    "exchange":    "NSE",
}

FNO = {
    "name":        "Nifty Futures",
    "scrip_code":  "NFO_51011",
    "security_id": "51011",
    "ws_token":    "NFO:51011",
    "segment":     "DERIVATIVE",
    "product":     "MARGIN",
    "exchange":    "NSE",
}

# ── Trading Schedule (IST) ────────────────────────────────────
MARKET_OPEN      = "09:15"   # When bot starts looking for signals
MARKET_CLOSE     = "15:20"   # When bot stops new entries
SQUAREOFF_AT     = "15:10"   # Force close all positions
DAILY_RESET_AT   = "09:14"   # Reset daily counters before open
RETRAIN_DAY      = "sunday"  # Day of week to retrain ML models
RETRAIN_TIME     = "18:30"   # Time to retrain after market close

# ── Bot Cycle ─────────────────────────────────────────────────
CYCLE_INTERVAL_SECONDS = 60  # How often bot checks for signals

# ── ML Settings ───────────────────────────────────────────────
CANDLE_INTERVAL   = "15minute"  # Timeframe for candles
                                # Options: 1minute, 5minute,
                                # 15minute, 60minute, 1day
RETRAIN_DAYS      = 90          # Days of history to train on
LABEL_HORIZON     = 3           # Candles ahead to predict
LABEL_THRESHOLD   = 0.005       # 0.5% move = valid signal
XGB_WEIGHT        = 0.6         # XGBoost vote weight
RL_WEIGHT         = 0.4         # PPO agent vote weight
MIN_CONFIDENCE    = 0.60        # Below this = HOLD, no trade

# ── Risk Management ───────────────────────────────────────────
MAX_POSITION_PCT      = 0.02   # Max 2% of balance per trade
DAILY_LOSS_LIMIT_PCT  = 0.03   # Stop trading after 3% daily loss
MAX_TRADES_PER_DAY    = 10     # Hard cap on number of trades
ORDER_FILL_TIMEOUT    = 30     # Seconds to wait for fill confirm

# ── Order Settings ────────────────────────────────────────────
LIMIT_ORDER_SLIPPAGE  = 0.001  # Place limit 0.1% above LTP on buys
ALGO_ID               = "99999" # Required by INDstocks for all orders

# ── Model File Paths ──────────────────────────────────────────
XGB_MODEL_PATH   = "ml/xgb_model.pkl"
PPO_MODEL_PATH   = "ml/ppo_model"
TENSORBOARD_PATH = "./tensorboard/"

# ── Logging ───────────────────────────────────────────────────
LOG_FILE         = "bot.log"
LOG_LEVEL        = "INFO"    # DEBUG for more detail, INFO for normal