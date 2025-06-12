from data.get_SPY_day import fetch_daily_yahoo_data
from analysis.metrics import calculate_sharpe_and_drawdown
from analysis.plot_returns import plot_cumulative_returns
from analysis.logger import log_to_csv

import importlib

strategy_modules = [
    'models.mean_reversion_20d_z1'
]

df = fetch_daily_yahoo_data()
results = []

for module_name in strategy_modules:
    module = importlib.import_module(module_name)
    strategy_func = getattr(module, 'mean_reversion_20d_z1')
    strategy_name = getattr(module, 'get_name')()
    
    df_strategy = strategy_func(df)
    sharpe, drawdown = calculate_sharpe_and_drawdown(df_strategy)
    plot_cumulative_returns(df_strategy, title=f"{strategy_name}\nSharpe={sharpe:.2f}, Drawdown={drawdown:.2%}")

    results.append({
        "Strategy": strategy_name,
        "Sharpe": round(sharpe, 4),
        "Drawdown": round(drawdown, 4)
    })

# Save results
log_to_csv(results, "strategy_comparison.csv")