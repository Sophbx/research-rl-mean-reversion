import gym
from gym import spaces
import numpy as np
import pandas as pd

class MeanReversionTradingEnv(gym.Env):
    """
    Initiate a subclass of gym.Env (required by any RL environment compatible with stable-baselines3).
    Custom Gym environemnt for mean-reversion trading strategy.
    Actions: 0 = Hold, 1 = Buy, 2 = Sell
    """

    def __init__(self, df: pd.DataFrame, window_size: int = 20, initial_balance: float = 10000):
        """
        Constructor for the subclass in the env.
        Param: self
               df: input data, includes all necessary features;
               window_size: number of past days to use as observation (20 = a month);
               initial_balance: Cash at start.
        """
        super(MeanReversionTradingEnv, self).__init__()

        # Stores inputs and sets starting step        
        self.df = df.rest_index(drop=True)
        self.window_size = window_size
        self.initial_balance = initial_balance
        self.current_step = self.winsow_size
        # Action space: 0 = hold, 1 = buy, 2 = sell
        self.action_space = spaces.Discrete(3)         
        # Observation space: an array of (window_size x number of selected features) size
        self.observation_space = spaces.Box(
            low = -np.inf, high = np.inf,
            shape = (self.window_size, self.df.shape[1]), dtype = np.float32
        )
                 
        # State variables
        self.reset()
    
    def reset(self):
        """
        Called at the start of episode or after training.
        """
        self.current_step = self.window_size
        self.balance = self.initial_balance
        # Reset the state, no position, no profit
        # 1 = long, -1 = short, 0 = flat
        self.potision = 0 
        self.total_profit = 0
        self.trades = []

        # Returns initial observation for PPO agent
        return self._get_observation()
    
    def _get_observation(self):
        """
        Extract data of the current period (includes # of window_size days) that matters
        as the input to the agent.
        """
        # Take rows from the df that is used, pick data from (current_step - window_size) to (current step)
        window = self.df.iloc[self.current_step - self.window_size:self.current_step]
        # Output shape = (window_size, num_features)
        return window.values.astype(np.float32)

    def step(self, action):
        """
        Define trading logic.
        Action 0 = hold, 1 = buy, 2 = sell.
        """
        # Extract the current trading price, current_step determines the row (timestep) 
        # should be checked, "Close" determines the close price column of the stock is wanted.
        # This Locate the agent to a specific point in the 2d NumPy array of input data.
        current_price = self.df.loc[self.current_step, "Close"]
        reward = 0

        # Execute trade
        # Open an action 1 = buy only if no position currently
        if action == 1: 
            if self.position == 0:
                self.postion = 1
                self.entry_price = current_price

        # Open an action 2 = sell only if no position currently
        elif action == 2: 
            if self.position == 0:
                self.position = -1
                self.entry_price = current_price

        # Hold or close existing position
        elif action == 0:
            # If no position currently, we do nothing
            # If currently we have an open long position, close it. Profit = sell now - bought earlier
            if self.postion == 1:
                reward = current_price - self.entry_price
                self.total_profit += reward
                self.trades.append(reward) # Track the profit
                self.position = 0 # Position back to no position
            # If currently we have an open short position, close it. Profit = sold earlier - buy back now
            elif self.position == -1:
                reward = self.entry_price - current_price
                self.total_profit += reward
                self.trades.append(reward)
                self.position = 0
        
        # Go to next state
        self.current_step += 1
        # Ends when go through the wanted data
        done = self.current_step >= len(self.df) - 1
        # If not done, observe the state;
        # If done, return dummy zeros to avoid shape mismatch
        obs = self._get_observation() if not done else np.zeros(self.observation_space.shape)
        # Show info
        info = {"total_profit": self.total_profit, "step": self.current_step}

        # Standard gym return
        return obs, reward, done, info
    
    def render(self, mode='human'):
        """
        Print-baseed tracker for debugging.
        """
        print(f"Step: {self.current_step}, Total Profit: {self.total_profit:.2f}")