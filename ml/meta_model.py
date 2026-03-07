import logging
import numpy as np

logger = logging.getLogger(__name__)


class MetaModel:
    """
    Combines XGBoost + RL ensemble + Sentiment
    into one final signal with confidence score.
    Learns which models perform best over time.
    """

    def __init__(self):
        # Model weights — update based on performance
        self.weights = {
            "xgb":       0.50,   # XGBoost starts with most weight
            "ppo":       0.25,   # PPO
            "a2c":       0.15,   # A2C
            "sentiment": 0.10    # Sentiment
        }
        self.history = []   # track predictions vs outcomes

    def predict(self, xgb_signal: str, xgb_conf: float,
                rl_result: dict, sentiment: float) -> dict:
        """Combine all signals into final decision."""

        votes = {"BUY": 0.0, "SELL": 0.0, "HOLD": 0.0}

        # XGBoost vote
        votes[xgb_signal] += self.weights["xgb"] * xgb_conf

        # RL votes
        rl_signal = rl_result.get("signal", "HOLD")
        rl_conf   = rl_result.get("confidence", 0.5)
        votes[rl_signal] += (
            self.weights["ppo"] + self.weights["a2c"]
        ) * rl_conf

        # Sentiment vote
        if sentiment >= 0.3:
            votes["BUY"]  += self.weights["sentiment"]
        elif sentiment <= -0.3:
            votes["SELL"] += self.weights["sentiment"]
        else:
            votes["HOLD"] += self.weights["sentiment"]

        # Final signal
        final      = max(votes, key=votes.get)
        total      = sum(votes.values())
        confidence = votes[final] / total if total > 0 else 0.0

        # Hard blocks
        # Block BUY if very negative sentiment
        if final == "BUY" and sentiment <= -0.6:
            logger.info(f"BUY blocked — extreme negative sentiment {sentiment}")
            final      = "HOLD"
            confidence = 0.0

        # Block SELL if very positive sentiment
        if final == "SELL" and sentiment >= 0.6:
            logger.info(f"SELL blocked — extreme positive sentiment {sentiment}")
            final      = "HOLD"
            confidence = 0.0

        return {
            "signal":     final,
            "confidence": round(confidence, 3),
            "votes":      votes,
            "weights":    self.weights.copy()
        }

    def update_weights(self, xgb_correct: bool, rl_correct: bool,
                       sentiment_correct: bool):
        """
        Adjust model weights based on which was right.
        Models that predict correctly get more weight.
        """
        alpha = 0.02   # small adjustment per trade

        if xgb_correct:
            self.weights["xgb"] = min(0.7, self.weights["xgb"] + alpha)
        else:
            self.weights["xgb"] = max(0.1, self.weights["xgb"] - alpha)

        if rl_correct:
            rl_boost = alpha / 2
            self.weights["ppo"] = min(0.4, self.weights["ppo"] + rl_boost)
            self.weights["a2c"] = min(0.4, self.weights["a2c"] + rl_boost)
        else:
            self.weights["ppo"] = max(0.05, self.weights["ppo"] - alpha / 2)
            self.weights["a2c"] = max(0.05, self.weights["a2c"] - alpha / 2)

        if sentiment_correct:
            self.weights["sentiment"] = min(
                0.3, self.weights["sentiment"] + alpha / 2
            )
        else:
            self.weights["sentiment"] = max(
                0.05, self.weights["sentiment"] - alpha / 2
            )

        # Normalize so all weights sum to 1
        total = sum(self.weights.values())
        for k in self.weights:
            self.weights[k] = round(self.weights[k] / total, 3)

        logger.debug(f"Meta weights updated: {self.weights}")
