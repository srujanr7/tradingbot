import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class PositionManager:
    """
    Manages how many positions are open at once.
    Dynamically allocates capital across slots.
    Enforces max equity and FNO positions separately.
    """

    def __init__(self,
                 max_equity: int = 3,
                 max_fno: int = 2,
                 min_confidence: float = 0.70):
        self.max_equity      = max_equity
        self.max_fno         = max_fno
        self.min_confidence  = min_confidence
        self.positions       = {}   # {scrip_code: position_dict}
        self.win_count       = 0
        self.loss_count      = 0

    # ── Slot checks ───────────────────────────────────────────

    def equity_slots_free(self) -> int:
        open_eq = sum(
            1 for p in self.positions.values()
            if p["segment"] == "EQUITY"
        )
        return max(0, self.max_equity - open_eq)

    def fno_slots_free(self) -> int:
        open_fno = sum(
            1 for p in self.positions.values()
            if p["segment"] == "DERIVATIVE"
        )
        return max(0, self.max_fno - open_fno)

    def can_enter(self, segment: str, confidence: float) -> tuple:
        """
        Returns (can_enter: bool, reason: str)
        Checks confidence threshold and slot availability.
        """
        if confidence < self.min_confidence:
            return False, f"Confidence {confidence:.1%} below {self.min_confidence:.1%} threshold"

        if segment == "EQUITY" and self.equity_slots_free() == 0:
            return False, f"Max equity positions ({self.max_equity}) reached"

        if segment == "DERIVATIVE" and self.fno_slots_free() == 0:
            return False, f"Max FNO positions ({self.max_fno}) reached"

        return True, "OK"

    def has_position(self, scrip_code: str) -> bool:
        return scrip_code in self.positions

    # ── Position sizing ───────────────────────────────────────

    def position_size(self, balance: float, price: float,
                      segment: str, confidence: float) -> int:
        """
        Dynamic position sizing based on:
        - Available capital split across max slots
        - Confidence score (higher confidence = larger size)
        - Segment type (FNO gets smaller allocation)
        """
        # Base allocation per slot
        if segment == "EQUITY":
            slot_allocation = balance / max(self.max_equity, 1)
        else:
            slot_allocation = balance / max(self.max_fno * 2, 1)

        # Scale by confidence: 70% conf = 70% of slot, 100% = 100%
        confidence_factor = min(confidence, 1.0)
        allocated = slot_allocation * confidence_factor

        qty = int(allocated / price)
        return max(1, qty)

    # ── Open / close ──────────────────────────────────────────

    def open_position(self, scrip_code: str, name: str,
                      segment: str, qty: int,
                      entry_price: float, order_id: str,
                      signal_meta: dict):
        self.positions[scrip_code] = {
            "scrip_code":  scrip_code,
            "name":        name,
            "segment":     segment,
            "qty":         qty,
            "entry":       entry_price,
            "order_id":    order_id,
            "signal_meta": signal_meta,
            "opened_at":   datetime.now().isoformat()
        }
        logger.info(
            f"📂 Position opened: {name} | "
            f"{qty} @ ₹{entry_price:,.2f} | "
            f"Equity slots free: {self.equity_slots_free()} | "
            f"FNO slots free: {self.fno_slots_free()}"
        )

    def close_position(self, scrip_code: str,
                       exit_price: float) -> dict:
        pos = self.positions.pop(scrip_code, None)
        if pos:
            pnl = (exit_price - pos["entry"]) * pos["qty"]
            if pnl >= 0:
                self.win_count += 1
            else:
                self.loss_count += 1
            logger.info(
                f"📁 Position closed: {pos['name']} | "
                f"PnL ₹{pnl:+,.2f}"
            )
            return {**pos, "exit": exit_price, "pnl": pnl}
        return {}

    # ── Signal ranking ────────────────────────────────────────

    def rank_signals(self, signals: list) -> list:
        """
        Takes list of signal dicts, returns sorted by confidence.
        Only returns actionable BUY signals above threshold.
        signals = [{"cfg": ..., "result": ..., "ltp": ...}, ...]
        """
        actionable = [
            s for s in signals
            if s["result"]["signal"] == "BUY"
            and s["result"]["confidence"] >= self.min_confidence
            and not self.has_position(s["cfg"]["scrip_code"])
        ]
        actionable.sort(
            key=lambda x: x["result"]["confidence"],
            reverse=True
        )
        return actionable

    # ── Status ────────────────────────────────────────────────

    def status(self) -> dict:
        return {
            "open_positions":   len(self.positions),
            "equity_open":      sum(1 for p in self.positions.values() if p["segment"] == "EQUITY"),
            "fno_open":         sum(1 for p in self.positions.values() if p["segment"] == "DERIVATIVE"),
            "equity_slots_free": self.equity_slots_free(),
            "fno_slots_free":    self.fno_slots_free(),
            "win_count":        self.win_count,
            "loss_count":       self.loss_count,
            "win_rate":         self.win_count / max(self.win_count + self.loss_count, 1) * 100,
            "positions":        self.positions
        }