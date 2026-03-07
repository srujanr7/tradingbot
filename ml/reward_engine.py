import numpy as np
import logging

logger = logging.getLogger(__name__)


class RewardEngine:
    """
    Calculates rewards for RL training.
    Rewards wins heavily, penalizes losses harder,
    encourages risk-adjusted returns.
    """

    def __init__(self):
        self.win_streak  = 0
        self.loss_streak = 0

    def calculate(self, pnl: float, pnl_pct: float,
                  hold_minutes: int, confidence: float,
                  sentiment: float) -> float:
        """
        Multi-factor reward calculation.
        Returns reward score used to train RL models.
        """
        reward = 0.0

        # ── Base reward from PnL ──────────────────────────
        if pnl_pct > 0:
            # Win — reward scales with size of win
            reward += pnl_pct * 100
            self.win_streak  += 1
            self.loss_streak  = 0

            # Bonus for win streak
            if self.win_streak >= 3:
                reward *= 1.2
                logger.debug(f"Win streak bonus: {self.win_streak}")

        else:
            # Loss — penalize 2x to make agent risk-averse
            reward += pnl_pct * 200
            self.loss_streak += 1
            self.win_streak   = 0

            # Extra penalty for loss streak
            if self.loss_streak >= 3:
                reward *= 1.5
                logger.debug(f"Loss streak penalty: {self.loss_streak}")

        # ── Hold time penalty ─────────────────────────────
        # Penalize holding too long (> 4 hours for intraday)
        if hold_minutes > 240:
            reward -= 0.5

        # ── Confidence alignment bonus ────────────────────
        # Reward when high confidence trade wins
        if confidence > 0.8 and pnl_pct > 0:
            reward += 1.0
        # Penalize when high confidence trade loses
        elif confidence > 0.8 and pnl_pct < 0:
            reward -= 1.5

        # ── Sentiment alignment bonus ─────────────────────
        if sentiment > 0.3 and pnl_pct > 0:
            reward += 0.3   # sentiment agreed, reward
        elif sentiment < -0.3 and pnl_pct > 0:
            reward += 0.5   # traded against sentiment and won — big reward
        elif sentiment < -0.3 and pnl_pct < 0:
            reward -= 0.3   # ignored negative sentiment warning

        return round(reward, 4)

    def get_streaks(self) -> dict:
        return {
            "win_streak":  self.win_streak,
            "loss_streak": self.loss_streak
        }
