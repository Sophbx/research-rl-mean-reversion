from data.get_SPY_day import fetch_daily_OHLCV
from analysis.metrics import calculate_sharpe_and_drawdown
from analysis.plot_returns import plot_cumulative_returns
from analysis.logger import log_to_csv

import importlib
import os

# Baseline strategies
strategy_modules = [
    'models.mean_reversion_20d_z1'
]

# RL imports 
from agents.train_mr_discrete import train as train_rl, split_df
from analysis.eval_policy_to_df import eval_policy_to_df
from stable_baselines3 import PPO

# fetch data 
df = fetch_daily_OHLCV('SPY', '2010-01-01', '2025-08-22')  # train/val/test need a bit more history
results = []

# run baseline strategies
for module_name in strategy_modules:
    module = importlib.import_module(module_name)
    strategy_func = getattr(module, 'mean_reversion_20d_z1')
    strategy_name = getattr(module, 'get_name')()

    df_strategy = strategy_func(df.copy())
    sharpe, drawdown = calculate_sharpe_and_drawdown(df_strategy)
    plot_cumulative_returns(df_strategy, title=f"{strategy_name}\nSharpe={sharpe:.2f}, Drawdown={drawdown:.2%}")

    results.append({
        "Strategy": strategy_name,
        "Sharpe": round(sharpe, 4),
        "Drawdown": round(drawdown, 4)
    })

# train OR load RL agent (Discrete buy/hold/sell)
model_path = "models/ppo_mr_discrete_buy_hold_sell.zip"
if os.path.exists(model_path):
    model = PPO.load(model_path)
else:
    model = train_rl(df)  # trains, saves to model_path inside the trainer

# evaluate RL on held-out test set & plot 
_, _, df_test = split_df(df)
df_rl = eval_policy_to_df(model, df_test, reward_mode="return")  # same reward as training by default

sharpe_rl, drawdown_rl = calculate_sharpe_and_drawdown(df_rl)
plot_cumulative_returns(df_rl, title=f"mr_discrete_rl\nSharpe={sharpe_rl:.2f}, Drawdown={drawdown_rl:.2%}")

results.append({
    "Strategy": "mr_discrete_rl",
    "Sharpe": round(sharpe_rl, 4),
    "Drawdown": round(drawdown_rl, 4)  
})

# Save results
log_to_csv(results, "strategy_comparison.csv")
print("Saved metrics to strategy_comparison.csv")
