from data.get_SPY_day import fetch_daily_yahoo_data
from analysis.metrics import calculate_sharpe_and_drawdown
from analysis.plot_returns import plot_cumulative_returns

import importlib

strategy_modules = [
    'models.mean_reversion_20d_z1'
]

df = fetch_daily_yahoo_data()

for module_name in strategy_modules:
    module = importlib.import_module(module_name)
    strategy_func = getattr(module, 'mean_reversion_20d_z1')
    strategy_name = getattr(module, 'get_name')()
    
    df_strategy = strategy_func(df)
    sharpe, drawdown = calculate_sharpe_and_drawdown(df_strategy)
    plot_cumulative_returns(df_strategy, title=f"{strategy_name}\nSharpe={sharpe:.2f}, Drawdown={drawdown:.2%}")