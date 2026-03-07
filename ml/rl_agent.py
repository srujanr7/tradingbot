import numpy as np
import pandas as pd
import logging
import os
from stable_baselines3 import PPO, A2C
from stable_baselines3.common.vec_env import DummyVecEnv
import gymnasium as gym
from gymnasium import spaces
from ml.reward_engine import RewardEngine

logger    = logging.getLogger(__name__)
_reward_e = RewardEngine()

MODELS_DIR = "ml/models"


class TradingEnv(gym.Env):
    """
    Custom Gymnasium environment for RL trading.
    Learns from both price data AND past trade outcomes.
    """

    def __init__(self, df: pd.DataFrame,
                 sentiment: float = 0.0,
                 trade_history: pd.DataFrame = None):
        super().__init__()
        self.df            = df.reset_index(drop=True)
        self.sentiment     = sentiment
        self.trade_history = trade_history
        self.current       = 0
        self.position      = 0
        self.entry         = 0.0
        self.entry_step    = 0
        self.total_pnl     = 0.0

        self.action_space = spaces.Discrete(3)  # 0=HOLD 1=BUY 2=SELL

        # 15 price features + 5 memory features = 20 total
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(20,), dtype=np.float32
        )

    def _get_memory_features(self) -> list:
        """Extract learnings from past trades for this instrument."""
        if self.trade_history is None or len(self.trade_history) == 0:
            return [0.0, 0.0, 0.5, 0.0, 0.0]

        th = self.trade_history
        wins     = len(th[th["outcome"] == "WIN"])
        total    = len(th)
        win_rate = wins / total if total > 0 else 0.5
        avg_pnl  = th["pnl_pct"].mean() if total > 0 else 0.0
        avg_hold = th["hold_minutes"].mean() / 240 if total > 0 else 0.5
        best     = th["pnl_pct"].max() if total > 0 else 0.0
        worst    = th["pnl_pct"].min() if total > 0 else 0.0

        return [win_rate, avg_pnl, avg_hold, best / 10, worst / 10]

    def _get_obs(self) -> np.ndarray:
        row = self.df.iloc[self.current]
        mem = self._get_memory_features()
        return np.array([
            row.get("rsi", 50) / 100,
            row.get("macd", 0) / 100,
            row.get("macd_signal", 0) / 100,
            row.get("bb_pct", 0.5),
            row.get("ema_20", 1) / row.get("close", 1),
            row.get("ema_50", 1) / row.get("close", 1),
            row.get("volume_ratio", 1) / 3,
            row.get("atr_pct", 0),
            row.get("adx", 25) / 100,
            row.get("stoch_k", 50) / 100,
            row.get("stoch_d", 50) / 100,
            row.get("cci", 0) / 200,
            row.get("roc", 0) / 10,
            self.position,
            float(self.sentiment),
            *mem   # 5 memory features
        ], dtype=np.float32)

    def reset(self, seed=None, options=None):
        self.current    = 0
        self.position   = 0
        self.entry      = 0.0
        self.entry_step = 0
        self.total_pnl  = 0.0
        return self._get_obs(), {}

    def step(self, action):
        row    = self.df.iloc[self.current]
        price  = row["close"]
        reward = 0.0

        if action == 1 and self.position == 0:
            # BUY
            self.position  = 1
            self.entry     = price
            self.entry_step = self.current
            reward          = -0.001

        elif action == 2 and self.position == 1:
            # SELL — calculate reward
            pnl_pct      = (price - self.entry) / self.entry
            hold_minutes = (self.current - self.entry_step) * 15
            reward       = _reward_e.calculate(
                pnl      = (price - self.entry),
                pnl_pct  = pnl_pct,
                hold_minutes = hold_minutes,
                confidence   = 0.7,
                sentiment    = self.sentiment
            )
            self.total_pnl += pnl_pct
            self.position   = 0
            self.entry      = 0.0

        elif action == 0 and self.position == 1:
            # HOLD — small unrealized reward
            unrealized = (price - self.entry) / self.entry
            reward     = unrealized * 0.05

        self.current += 1
        done = self.current >= len(self.df) - 1
        return self._get_obs(), reward, done, False, {}


class RLEnsemble:
    """
    Ensemble of PPO + A2C agents.
    Both train continuously and vote on signals.
    The better-performing model gets more weight.
    """

    def __init__(self, scrip_code: str):
        self.scrip_code = scrip_code
        self.ppo        = None
        self.a2c        = None
        self.ppo_score  = 0.5   # running win rate
        self.a2c_score  = 0.5
        os.makedirs(MODELS_DIR, exist_ok=True)
        self._load()

    def _path(self, algo: str) -> str:
        return f"{MODELS_DIR}/{algo}_{self.scrip_code}"

    def _load(self):
        """Load saved models if they exist."""
        try:
            if os.path.exists(f"{self._path('ppo')}.zip"):
                self.ppo = PPO.load(self._path("ppo"))
                logger.info(f"✅ PPO loaded for {self.scrip_code}")
        except Exception as e:
            logger.warning(f"PPO load error: {e}")

        try:
            if os.path.exists(f"{self._path('a2c')}.zip"):
                self.a2c = A2C.load(self._path("a2c"))
                logger.info(f"✅ A2C loaded for {self.scrip_code}")
        except Exception as e:
            logger.warning(f"A2C load error: {e}")

    def train(self, df: pd.DataFrame,
              sentiment: float = 0.0,
              trade_history: pd.DataFrame = None,
              timesteps: int = 10000):
        """Train both PPO and A2C. Called after market close."""
        if len(df) < 50:
            logger.warning(f"Not enough data for RL training: {self.scrip_code}")
            return

        logger.info(f"🧠 Training RL ensemble for {self.scrip_code}...")

        def make_env():
            return TradingEnv(df, sentiment, trade_history)

        # Train PPO
        try:
            env = DummyVecEnv([make_env])
            if self.ppo is None:
                self.ppo = PPO(
                    "MlpPolicy", env,
                    learning_rate=3e-4,
                    n_steps=256,
                    batch_size=64,
                    n_epochs=10,
                    gamma=0.99,
                    verbose=0
                )
            else:
                self.ppo.set_env(env)
            self.ppo.learn(total_timesteps=timesteps)
            self.ppo.save(self._path("ppo"))
            logger.info(f"✅ PPO trained & saved for {self.scrip_code}")
        except Exception as e:
            logger.error(f"PPO training error: {e}")

        # Train A2C
        try:
            env = DummyVecEnv([make_env])
            if self.a2c is None:
                self.a2c = A2C(
                    "MlpPolicy", env,
                    learning_rate=7e-4,
                    n_steps=128,
                    gamma=0.99,
                    verbose=0
                )
            else:
                self.a2c.set_env(env)
            self.a2c.learn(total_timesteps=timesteps)
            self.a2c.save(self._path("a2c"))
            logger.info(f"✅ A2C trained & saved for {self.scrip_code}")
        except Exception as e:
            logger.error(f"A2C training error: {e}")

    def predict(self, df: pd.DataFrame,
                sentiment: float = 0.0,
                trade_history: pd.DataFrame = None) -> dict:
        """Ensemble vote from PPO + A2C."""
        signal_map = {0: "HOLD", 1: "BUY", 2: "SELL"}
        votes      = {"BUY": 0.0, "SELL": 0.0, "HOLD": 0.0}

        env = TradingEnv(df, sentiment, trade_history)

        def run_model(model, weight: float):
            if model is None:
                return
            try:
                obs, _ = env.reset()
                for _ in range(min(len(df) - 1, 30)):
                    action, _ = model.predict(obs, deterministic=True)
                    obs, _, done, _, _ = env.step(int(action))
                    if done:
                        break
                action, _ = model.predict(obs, deterministic=True)
                signal     = signal_map.get(int(action), "HOLD")
                votes[signal] += weight
            except Exception as e:
                logger.error(f"RL predict error: {e}")

        run_model(self.ppo, self.ppo_score)
        run_model(self.a2c, self.a2c_score)

        # Pick signal with most weight
        final  = max(votes, key=votes.get)
        total  = sum(votes.values())
        conf   = votes[final] / total if total > 0 else 0.0

        return {
            "signal":     final,
            "confidence": round(conf, 3),
            "ppo_vote":   signal_map.get(0, "HOLD"),
            "a2c_vote":   signal_map.get(0, "HOLD"),
            "votes":      votes
        }

    def update_scores(self, pnl_pct: float, ppo_was_right: bool,
                      a2c_was_right: bool):
        """
        Update model scores based on trade outcome.
        Better performing model gets more weight next time.
        """
        alpha = 0.1   # learning rate for score update

        if ppo_was_right:
            self.ppo_score = min(1.0, self.ppo_score + alpha)
        else:
            self.ppo_score = max(0.1, self.ppo_score - alpha)

        if a2c_was_right:
            self.a2c_score = min(1.0, self.a2c_score + alpha)
        else:
            self.a2c_score = max(0.1, self.a2c_score - alpha)

        logger.debug(
            f"Model scores — PPO: {self.ppo_score:.2f} | "
            f"A2C: {self.a2c_score:.2f}"
        )
