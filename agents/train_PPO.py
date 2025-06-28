import pandas as pd
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from env.mean_reversion_env import MeanReversionTradingEnv

# === Step 1: Load your preprocessed data ===
df = pd.read_csv("data/Combined_data.csv")

# Ensure the DataFrame includes z-score and Close columns
required_cols = ["Close", "z_score"]
assert all(col in df.columns for col in required_cols), "Missing required features in the dataset."

# Optionally drop rows with NaNs (e.g., from rolling z-score calc)
df.dropna(inplace=True)

# === Step 2: Create the Environment ===
window_size = 20  # must match env config

env = DummyVecEnv([lambda: MeanReversionTradingEnv(df=df, window_size=window_size)])

# === Step 3: Initialize PPO Agent ===
model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="log/ppo")

# === Step 4: Train the Model ===
timesteps = 100_000
model.learn(total_timesteps=timesteps)

# === Step 5: Save the Trained Model ===
model.save("models/ppo_mean_reversion")
print("\n✅ PPO model trained and saved to models/ppo_mean_reversion.zip")