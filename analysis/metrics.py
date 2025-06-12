import numpy as np

def calculate_sharpe_and_drawdown(df):
    sharpe = np.sqrt(252) * df['Strategy'].mean() / df['Strategy'].std()
    drawdown = (df['Strategy'].cumsum() - df['Strategy'].cumsum().cummax()).min()
    return sharpe, drawdown
