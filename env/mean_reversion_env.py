import gym
from gym import spaces
import numpy as np
import pandas as pd


class MeanReversionTradingEnv(gym.Env):
    """
    Custom Gym environemnt for mean-reversion trading strategy.
    Actions: 0 = Hold, 1 = Buy, 2 = Sell
    """


    def __init__(self, df: pd.DataFrame, window_size: int = 20, initial_balance: float = 10000):
        super(MeanReversionTradingEnv, self).__init__()
                 
        self.df = df.rest_index(drop=True)
        self.window_size = window_size
        self.initial_balance = initial_balance
        self.current_step = self.winsow_size
                 
        # Action space: 0 = hold, 1 = buy, 2 = sell
        self.action_space = spaces.Discrete(3)
                 
        # Observation space: window_size rows of selected indicators
        self.observation_space = spaces.Box(
            low = -np.inf, high = np.inf,
            shape = (self,window_size, self.df.shape[1]), dtype = np.float32
        )
                 
        # State variables
        self.reset()

    
    def reset(self):
        self.current_step = self.window_size
        self.balance = self.initial_balance
        # 1 = long, -1 = short, 0 = flat
        self.potision = 0 
        self.total_profit = 0
        self.trades = []
        return self._get_observation()
    
    def _get_observation(self):
        window = self.df.iloc[self.current_step - self.window_size:self.current_step]
        return window.values.astype(np.float32)
    
    def step(self, action):
        current_price = self.df.loc[self.current_step, "Close"]
        reward = 0

        # Execute trade
        # Buy
        if action == 1: 
            if self.position == 0:
                self.postion = 1
                self.entry_price = current_price
        # Sell
        elif action == 2: 
            if self.position == 0:
                self.position = -1
                self.entry_price = current_price
        # Hold or close existing position
        elif action == 0:
            if self.postion == 1:
                reward = current_price - self.entry_price
                self.total_profit += reward
                self.trades.append(reward)
                self.position = 0
            elif self.position == -1:
                reward = self.entry_price - current_price
                self.total_profit += reward
                self.trades.append(reward)
                self.position = 0
        
        self.current_step += 1
        done = self.current_step >= len(self.df) - 1
        obs = self._get_observation() if not done else np.zeros(self.observation_space.shape)
        info = {"total_profit": self.total_profit, "step": self.current_step}
        return obs, reward, done, info
    
    
    def render(self, mode='human'):
        print(f"Step: {self.current_step}, Total Profit: {self.total_profit:.2f}")