import logging
logger = logging.getLogger(__name__)

class RiskManager:
    def __init__(self, max_position_pct=0.02, daily_loss_limit_pct=0.03):
        self.max_position_pct = max_position_pct    # 2% of balance per trade
        self.daily_loss_limit_pct = daily_loss_limit_pct  # 3% daily max loss
        self.daily_pnl = 0.0
        self.trades_today = 0
        self.max_trades_per_day = 10

    def can_trade(self, balance: float) -> bool:
        if self.daily_pnl < -(balance * self.daily_loss_limit_pct):
            logger.warning("🔴 Daily loss limit hit. Kill switch activated.")
            return False
        if self.trades_today >= self.max_trades_per_day:
            logger.warning("🔴 Max daily trades reached.")
            return False
        return True

    def position_size(self, balance: float, price: float) -> int:
        """Returns number of shares to buy based on % of balance"""
        allocation = balance * self.max_position_pct
        qty = int(allocation / price)
        return max(1, qty)

    def update_pnl(self, pnl: float):
        self.daily_pnl += pnl
        self.trades_today += 1
        logger.info(f"Daily PnL: ₹{self.daily_pnl:.2f} | Trades: {self.trades_today}")

    def reset_daily(self):
        """Call this at start of each trading day"""
        self.daily_pnl = 0.0
        self.trades_today = 0