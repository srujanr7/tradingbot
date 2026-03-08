import logging

logger = logging.getLogger(__name__)


class RewardEngine:
    """
    Converts trade outcomes into RL reward signals.

    Rules:
      Base reward    = PnL %
      Quick win      × 1.5  (efficient capital use)
      Long loser     × 2.0  (punish holding losses)
      High drawdown  − 5.0  (punish risky wins)
      Volatile win   × 1.3  (bonus for hard conditions)
      Clean win      × 1.2  (low drawdown + profitable)
    """

    def calculate(self,
                  entry:            float,
                  exit_price:       float,
                  held_candles:     int,
                  signal:           dict,
                  max_drawdown_pct: float = 0.0) -> float:

        if entry <= 0:
            return 0.0

        pnl_pct = (exit_price - entry) / entry * 100
        reward  = pnl_pct

        # Quick wins are efficient — reward them
        if pnl_pct > 0 and held_candles <= 3:
            reward *= 1.5

        # Held a loser too long — double the pain
        if pnl_pct < 0 and held_candles > 10:
            reward *= 2.0

        # High drawdown even on a winner — penalise
        if max_drawdown_pct > 2.0:
            reward -= 5.0

        # Won in volatile conditions — bonus
        if (signal.get("regime") == "VOLATILE"
                and pnl_pct > 0):
            reward *= 1.3

        # Clean win: profitable AND low drawdown
        if pnl_pct > 0 and max_drawdown_pct < 0.5:
            reward *= 1.2

        logger.debug(
            f"RewardEngine: pnl={pnl_pct:+.2f}% "
            f"candles={held_candles} "
            f"drawdown={max_drawdown_pct:.2f}% "
            f"→ reward={reward:.3f}"
        )
        return round(reward, 4)
