import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces

def compute_rsi(close, n=14):
    r = close.diff()
    up = r.clip(lower=0)
    dn = -r.clip(upper=0)
    rs = up.rolling(n).mean() / dn.rolling(n).mean()
    return 100 - (100 / (1 + rs))

class MRDiscreteEnv(gym.Env):
    """
    Discrete buy/hold/sell env with volatility targeting and costs.

    Action space: 
        0 -> SELL (-1)
        1 -> HOLD (0)
        2 -> BUY (+1)

    Uses Close to close returns. Volatility targeting maps discrete action to final
    weight via daily budget / vol20. The start index skips indicator warmup
    (20-day windows + RSI).
    """
    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        df: pd.DataFrame,
        entry_window: int = 20,
        vol_window: int = 20,
        rsi_window: int = 14,
        budget_annual_vol: float = 0.10,
        lev_cap: float = 3.0,
        cost_bps_one_way: float = 5, 
        reward_mode: str = "return",
        dd_lambda: float = 0.5,
        rendor_mode: str | None = None,
    ):
        super().__init__()

        self.df = df.copy()
        if not isinstance(self.df.index, pd.DatetimeIndex):
            # If column Date exists, set it as index; otherwisse create one
            if "Date" in self.df.columns:
                self.df["Date"] = pd.to_datetime(self.df["Date"])
                self.df.set_index("Date", inplace=True)
            else:
                self.df.index = pd.to_datetime(self.df.index)

        # Features
        mp = (self.df['High'] + self.df['Low'] + self.df['Close']) / 3.0
        ma = mp.rolling(entry_window).mean()
        std = mp.rolling(entry_window).std().replace(0, np.nan)
        self.df['Z'] = (mp - ma) / std

        self.df['ret1']  = self.df['Close'].pct_change().fillna(0)
        self.df['mom5']  = self.df['Close'].pct_change(5)
        self.df['mom20'] = self.df['Close'].pct_change(20)
        self.df['vol20'] = self.df['ret1'].rolling(vol_window).std()
        self.df['rsi14'] = compute_rsi(self.df['Close'], 14)

        # Risk budget per day
        self.budget = budget_annual_vol / np.sqrt(252.0)
        self.lev_cap = float(lev_cap)
        self.cost = float(cost_bps_one_way) / 1e4 

        # Reward Setting
        assert reward_mode in {"return", "sharpe", "dd_penalized"}
        self.reward_mode = reward_mode
        self.dd_lambda = float(dd_lambda)

        # Observation space
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32)
       
        # Action space
        self.action_space = spaces.Discrete(3)

        # Episode state
        # start after max warmup of indicators
        self.warmup = max(entry_window, vol_window, rsi_window, 20)
        self.i0 = self.warmup
        self.i = self.i0

        self.weight = 0.0     
        self.equity = 1.0
        self.max_equity = 1.0

        # Sharpe-like running variance
        self._running_var = 1e-4

        self.rendor_mode = rendor_mode

    # helpers
    def _obs(self, i: int) -> np.ndarray:
        z    = self.df['Z'].iloc[i]
        m5   = self.df['mom5'].iloc[i]
        m20  = self.df['mom20'].iloc[i]
        vol  = self.df['vol20'].iloc[i]
        rsi  = self.df['rsi14'].iloc[i]
        dd   = 1.0 - (self.equity / max(self.max_equity, 1e-12))
        return np.array([z, m5, m20, dd, vol, rsi], dtype=np.float32)

    def _vol_target_scale(self, i):
        vol = self.df['vol20'].iloc[i]
        if not np.isfinite(vol) or vol <= 0:
            return 0.0
        scale = self.budget / vol
        return float(np.clip(scale, 0.0, self.lev_cap))
    
    def _action_to_direction(self, a: int) -> float:
        # 0 -> -1, 1 -> 0, 2 -> 1
        if a == 0:
            return -1.0
        if a == 1:
            return 0.0
        if a == 2:
            return +1.0
        raise ValueError("Action must be 0/1/2")

    # Gym methods
    def reset(self, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self.i = self.i0
        self.weight = 0.0
        self.equity = 1.0
        self.max_equity = 1.0
        self._running_var = 1e-4
        return self._obs(self.i), {}

    def step(self, action: int):
        # Map discrete action to directional target in {-1, 0, 1}
        direction = self._action_to_direction(int(action))

        # Vol targeting scal uses info up to t-1
        scale = self._vol_target_scale(self.i - 1)  # avoid look-ahead
        target_weight = direction * scale

        # Turnover cost on change in weight
        turnover = abs(target_weight - self.weight)
        cost = self.cost * turnover

        # Apply P&L over the next return
        ret = float(self.df['ret1'].iloc[self.i])  # r_t
        pnl = target_weight * ret
        
        # Rewards
        if self.reward_mode == "return":
            reward = pnl - cost

        elif self.reward_mode == "sharpe":
            alpha = 0.01
            self._running_var = (1-alpha) * self._running_var + alpha * (pnl ** 2)
            sharpe_like = pnl / (np.sqrt(self._running_var) + 1e-8)
            reward = sharpe_like - cost

        # drawdown penalized
        else:
            prev_dd = 1.0 - (self.equity / max(self.max_equity, 1e-12))
            reward = pnl - cost
            tmp_equity = self.equity * (1.0 + reward)
            tmp_max_eq = max(self.max_equity, tmp_equity)
            new_dd = 1.0 - (tmp_equity / tmp_max_eq)
            dd_penalty = self.dd_lambda * max(0.0, new_dd - prev_dd)
            reward -= dd_penalty

        # Commit equity updates
        self.equity *= (1.0 + reward)
        self.max_equity = max(self.max_equity, self.equity)
        self.weight = target_weight

        # Move time forward
        self.i += 1
        terminated = self.i >= (len(self.df) - 1)
        truncated = False
        
        obs = self._obs(min(self.i, len(self.df) - 1))
        info = {
            "equity": self.equity,
            "weight": self.weight,
            "turnover": turnover,
            "scale": scale,
        }

        if terminated:
            info["final_equity"] = self.equity

        if self.rendor_mode == "human":
            print(f"t={self.i} eq={self.equity:.4f} w={self.weight:.3f} a={action} r={reward:.6f}")

        return obs, reward, terminated, truncated, info

    def render(self):
        print(f"[MRDiscreteTradingEnv] t={self.i} equity={self.equity:.4f} weight={self.weight:.3f}")