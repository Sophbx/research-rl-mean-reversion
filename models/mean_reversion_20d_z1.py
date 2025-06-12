import pandas as pd
import numpy as np

def mean_reversion_20d_z1(data, window=20, threshold=1):
    """
    Implements a simple mean-reversion strategy:
    Buy when price is significantly below the rolling mean, sell when significantly above.
    
    Parameters:
    - data: DataFrame with at least ['High', 'Low', 'Close'] columns
    - window: Moving average window length of 20
    - threshold: Z-score threshold 1 for entering positions
    
    Returns:
    - DataFrame with strategy signals and performance
    """
    df = data.copy()
    df['MeanPrice'] = (df['High'] + df['Low'] + df['Close']) / 3
    df['MA'] = df['MeanPrice'].rolling(window=window).mean()
    df['STD'] = df['MeanPrice'].rolling(window=window).std()
    df['ZScore'] = (df['MeanPrice'] - df['MA']) / df['STD']
    
    # Signal generation
    df['Position'] = 0
    df.loc[df['ZScore'] > threshold, 'Position'] = -1  # Short
    df.loc[df['ZScore'] < -threshold, 'Position'] = 1  # Long
    df['Position'] = df['Position'].shift(1)  # Trade on next day

    # Strategy returns
    df['Returns'] = df['Close'].pct_change()
    df['Strategy'] = df['Position'] * df['Returns']
    df.dropna(inplace=True)

    return df


def get_name():
    return "MeanReversion_20d_z1"
