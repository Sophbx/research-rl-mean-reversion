from data.get_SPY_day import fetch_daily_OHLCV
from analysis.metrics import calculate_sharpe_and_drawdown
from analysis.plot_returns import plot_cumulative_returns
from analysis.logger import log_to_csv
from analysis.quick_stats import buy_and_hold_sharpe, subsample_sharpe, quick_stats
from analysis.robustness_harness import evaluate_policy_robustness
# import your discrete env class name
from env.mr_discrete_env import MRDiscreteEnv 

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
plot_cumulative_returns(df_rl, title=f"mr_discrete_rl_return\nSharpe={sharpe_rl:.2f}, Drawdown={drawdown_rl:.2%}")

results.append({
    "Strategy": "mr_discrete_rl",
    "Sharpe": round(sharpe_rl, 4),
    "Drawdown": round(drawdown_rl, 4)  
})

# Print quick stats for the RL run you just plotted
print("=== Quick stats (mr_discrete_rl, as-evaluated) ===")
print(quick_stats(df_rl))

# Robustness grid (no retraining): vary costs & lev caps
robust_table = evaluate_policy_robustness(
    model,
    MRDiscreteEnv,       # match your class name
    df_test,
    costs=(5, 10, 20),
    lev_caps=(1.0, 2.0, 3.0),
    reward_mode="return",
    save_csv_path="robustness_grid.csv"
)
print("\n=== Robustness grid ===")
print(robust_table)

print("\n=== Buy & Hold Sharpe on test period ===")
bh_stats = buy_and_hold_sharpe(df_test)
print(bh_stats)

print("\n=== Sub-sample Sharpe (by year) for RL agent ===")
sharpe_by_year = subsample_sharpe(df_rl, ret_col="Strategy", by_year=True)
print(sharpe_by_year)

print("\n=== Sub-sample Sharpe (rolling 252-day windows) for RL agent ===")
rolling_sharpes = subsample_sharpe(df_rl, ret_col="Strategy", window=252)
for period, s in rolling_sharpes.items():
    print(f"{period}: {s:.2f}")

# Save results
log_to_csv(results, "strategy_comparison.csv")
print("Saved metrics to strategy_comparison.csv")
