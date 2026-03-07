import os
from dotenv import load_dotenv

load_dotenv()

# ── API Credentials ───────────────────────────────────────────
INDSTOCKS_TOKEN    = os.getenv("INDSTOCKS_TOKEN")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID   = os.getenv("TELEGRAM_CHAT_ID")

# ── Trading Schedule (IST) ────────────────────────────────────
MARKET_OPEN      = "09:15"
MARKET_CLOSE     = "15:20"
SQUAREOFF_AT     = "15:10"
DAILY_RESET_AT   = "09:14"
RETRAIN_DAY      = "sunday"
RETRAIN_TIME     = "18:30"

# ── Bot Cycle ─────────────────────────────────────────────────
CYCLE_INTERVAL_SECONDS = 60

# ── ML Settings ───────────────────────────────────────────────
CANDLE_INTERVAL   = "auto"   # "auto" = per-instrument best interval
                              # or set fixed: "5minute", "15minute" etc.
RETRAIN_DAYS      = 90
LABEL_HORIZON     = 3
LABEL_THRESHOLD   = 0.005

# Model vote weights (MetaModel self-adjusts these over time)
XGB_WEIGHT        = 3.0
LGBM_WEIGHT       = 2.5
LSTM_WEIGHT       = 2.0
RL_WEIGHT         = 2.0
PATTERN_WEIGHT    = 1.5

# Confidence gates
MIN_CONFIDENCE    = 0.55    # Below this → HOLD (trainer pipeline)
TRADE_CONFIDENCE  = 0.70    # Below this → skip order (position manager)

# ── Risk Management ───────────────────────────────────────────
MAX_POSITION_PCT      = 0.02
DAILY_LOSS_LIMIT_PCT  = 0.03
MAX_TRADES_PER_DAY    = 10
ORDER_FILL_TIMEOUT    = 30

# ── Order Settings ────────────────────────────────────────────
LIMIT_ORDER_SLIPPAGE  = 0.001
ALGO_ID               = "99999"

# ── Model File Paths ──────────────────────────────────────────
MODEL_DIR        = "ml/models"
XGB_MODEL_PATH   = f"{MODEL_DIR}/xgb_model.pkl"
PPO_MODEL_PATH   = f"{MODEL_DIR}/ppo_model"
TENSORBOARD_PATH = "./tensorboard/"

# ── Logging ───────────────────────────────────────────────────
LOG_FILE  = "bot.log"
LOG_LEVEL = "INFO"

# ── Position Slots ────────────────────────────────────────────
MAX_EQUITY_POS = 3
MAX_FNO_POS    = 2
