import logging
logger = logging.getLogger(__name__)


class RiskManager:
    def __init__(self, max_position_pct=0.02, daily_loss_limit_pct=0.03):
        self.max_position_pct     = max_position_pct
        self.daily_loss_limit_pct = daily_loss_limit_pct
        self.daily_pnl            = 0.0
        self.trades_today         = 0
        self.max_trades_per_day   = 10
        self.per_trade_limit      = None  # ₹ cap per trade — set via /setlimit
        self.daily_loss_cap       = None  # ₹ absolute daily loss — set via /setdaily

    def can_trade(self, balance: float) -> bool:
        limit = self.get_daily_limit(balance)
        if self.daily_pnl < -limit:
            logger.warning("🔴 Daily loss limit hit. Kill switch activated.")
            return False
        if self.trades_today >= self.max_trades_per_day:
            logger.warning("🔴 Max daily trades reached.")
            return False
        return True

    def get_daily_limit(self, balance: float) -> float:
        """Returns active daily loss limit in ₹."""
        if self.daily_loss_cap is not None:
            return self.daily_loss_cap
        return balance * self.daily_loss_limit_pct

    def apply_per_trade_limit(self, qty: int, ltp: float) -> int:
        """Clamp qty so trade value stays within per_trade_limit."""
        if self.per_trade_limit is None:
            return qty
        max_qty = int(self.per_trade_limit / ltp)
        clamped = max(1, min(qty, max_qty))
        if clamped < qty:
            logger.info(
                f"Per-trade limit ₹{self.per_trade_limit:,.0f} applied: "
                f"qty {qty} → {clamped}"
            )
        return clamped

    def position_size(self, balance: float, price: float) -> int:
        """Returns number of shares based on % of balance."""
        allocation = balance * self.max_position_pct
        qty = int(allocation / price)
        return max(1, qty)

    def update_pnl(self, pnl: float):
        self.daily_pnl    += pnl
        self.trades_today += 1
        logger.info(
            f"Daily PnL: ₹{self.daily_pnl:.2f} | "
            f"Trades: {self.trades_today}"
        )

    def reset_daily(self):
        """Call at start of each trading day."""
        self.daily_pnl    = 0.0
        self.trades_today = 0
        logger.info("🔄 Risk counters reset")
