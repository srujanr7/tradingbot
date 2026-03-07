import numpy as np
import pandas as pd
import logging
import os
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import gymnasium as gym
from gymnasium import spaces

logger = logging.getLogger(__name__)


class TradingEnv(gym.Env):
    """
    Custom trading environment for RL agent.
    State  : technical indicators + sentiment score
    Actions: 0=HOLD, 1=BUY, 2=SELL
    Reward : risk-adjusted PnL (penalizes losses harder)
    """

    def __init__(self, df: pd.DataFrame, sentiment: float = 0.0):
        super().__init__()
        self.df        = df.reset_index(drop=True)
        self.sentiment = sentiment
        self.current   = 0
        self.position  = 0      # 0=flat, 1=long
        self.entry     = 0.0
        self.total_pnl = 0.0

        # Actions: 0=HOLD, 1=BUY, 2=SELL
        self.action_space = spaces.Discrete(3)

        # Observation: 15 features
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(15,), dtype=np.float32
        )

    def _get_obs(self):
        row = self.df.iloc[self.current]
        return np.array([
            row.get("rsi", 50) / 100,
            row.get("macd", 0) / 100,
            row.get("macd_signal", 0) / 100,
            row.get("bb_upper", 0) / row.get("close", 1),
            row.get("bb_lower", 0) / row.get("close", 1),
            row.get("ema_20", 0) / row.get("close", 1),
            row.get("ema_50", 0) / row.get("close", 1),
            row.get("volume_ratio", 1),
            row.get("atr", 0) / row.get("close", 1),
            row.get("adx", 25) / 100,
            row.get("stoch_k", 50) / 100,
            row.get("stoch_d", 50) / 100,
            row.get("cci", 0) / 200,
            self.position,
            float(self.sentiment)
        ], dtype=np.float32)

    def reset(self, seed=None, options=None):
        self.current   = 0
        self.position  = 0
        self.entry     = 0.0
        self.total_pnl = 0.0
        return self._get_obs(), {}

    def step(self, action):
        row   = self.df.iloc[self.current]
        price = row["close"]
        reward = 0.0

        # BUY
        if action == 1 and self.position == 0:
            self.position = 1
            self.entry    = price
            reward        = -0.001   # small cost to enter

        # SELL
        elif action == 2 and self.position == 1:
            pnl            = (price - self.entry) / self.entry
            reward         = pnl * 100
            # Penalize losses 2x to make agent risk-averse
            if pnl < 0:
                reward *= 2
            self.total_pnl += pnl
            self.position   = 0
            self.entry      = 0.0

        # HOLD with open position — small reward for holding winner
        elif action == 0 and self.position == 1:
            unrealized = (price - self.entry) / self.entry
            reward     = unrealized * 0.1

        # Sentiment bonus — reward aligns with sentiment
        if self.sentiment > 0.3 and action == 1:
            reward += 0.05
        elif self.sentiment < -0.3 and action == 2:
            reward += 0.05

        self.current += 1
        done = self.current >= len(self.df) - 1

        return self._get_obs(), reward, done, False, {}


class RLAgent:
    """PPO-based RL trading agent."""

    MODEL_PATH = "ml/models/ppo_trader"

    def __init__(self, scrip_code: str):
        self.scrip_code = scrip_code
        self.model      = None
        self.trained    = False
        self._load()

    def _load(self):
        path = f"{self.MODEL_PATH}_{self.scrip_code}.zip"
        if os.path.exists(path):
            try:
                self.model   = PPO.load(path)
                self.trained = True
                logger.info(f"✅ RL model loaded for {self.scrip_code}")
            except Exception as e:
                logger.warning(f"Could not load RL model: {e}")

    def train(self, df: pd.DataFrame, sentiment: float = 0.0,
              timesteps: int = 10000):
        if len(df) < 50:
            logger.warning(f"Not enough data to train RL for {self.scrip_code}")
            return

        try:
            env        = DummyVecEnv([lambda: TradingEnv(df, sentiment)])
            self.model = PPO(
                "MlpPolicy", env,
                learning_rate=3e-4,
                n_steps=256,
                batch_size=64,
                n_epochs=10,
                gamma=0.99,
                verbose=0
            )
            self.model.learn(total_timesteps=timesteps)

            # Save model
            os.makedirs("ml/models", exist_ok=True)
            self.model.save(f"{self.MODEL_PATH}_{self.scrip_code}")
            self.trained = True
            logger.info(f"✅ RL model trained for {self.scrip_code}")

        except Exception as e:
            logger.error(f"RL training error: {e}")

    def predict(self, df: pd.DataFrame,
                sentiment: float = 0.0) -> dict:
        if not self.trained or self.model is None:
            return {"signal": "HOLD", "confidence": 0.0}

        try:
            env = TradingEnv(df, sentiment)
            obs, _ = env.reset()

            # Run through recent candles to get current state
            for i in range(min(len(df) - 1, 20)):
                action, _ = self.model.predict(obs, deterministic=True)
                obs, _, done, _, _ = env.step(int(action))
                if done:
                    break

            # Final prediction
            action, _ = self.model.predict(obs, deterministic=True)
            action     = int(action)

            signal_map = {0: "HOLD", 1: "BUY", 2: "SELL"}
            signal     = signal_map.get(action, "HOLD")

            return {
                "signal":     signal,
                "confidence": 0.75 if signal != "HOLD" else 0.0,
                "action_id":  action
            }

        except Exception as e:
            logger.error(f"RL predict error: {e}")
            return {"signal": "HOLD", "confidence": 0.0}
