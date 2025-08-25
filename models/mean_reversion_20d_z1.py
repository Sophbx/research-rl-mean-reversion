import pandas as pd
import numpy as np

def mean_reversion_20d_z1(data, window=20, entry=1.0, exit=0.25, target_vol_annual=0.10, cost_bps_one_way=5):
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
    df['MA']  = df['MeanPrice'].rolling(window=window).mean()
    df['STD'] = df['MeanPrice'].rolling(window=window).std().replace(0, np.nan)
    df['Z']   = (df['MeanPrice'] - df['MA']) / df['STD']

    # Entry/exit band -> discrete signal {-1, 0, +1}
    sig = pd.Series(0, index=df.index)
    sig[df['Z'] <= -entry] = 1
    sig[df['Z'] >=  entry] = -1
    # Hold until exit toward the center
    hold = sig.replace(0, np.nan).ffill().fillna(0)
    hold[(hold == 1) & (df['Z'] > -exit)] = 0
    hold[(hold == -1) & (df['Z'] <  exit)] = 0
    df['Signal'] = hold

    # Trade next day
    df['Position'] = df['Signal'].shift(1).fillna(0)

    # Vol targeting (use 20d realized vol on Close)
    daily_vol = df['Close'].pct_change().rolling(20).std()
    budget = target_vol_annual / np.sqrt(252)
    target_w  = (budget / daily_vol.replace(0, np.nan)).clip(upper=3).reindex(df.index).fillna(0) 
    df['Weight'] = df['Position'].mul(target_w, fill_value=0)

    # P&L with costs (bps per one-way trade)
    ret = df['Close'].pct_change().fillna(0)
    turnover = df['Weight'].diff().abs().fillna(df['Weight'].abs())
    cost = (cost_bps_one_way/1e4) * turnover
    df['Strategy'] = df['Weight'].shift(0) * ret - cost

    return df.dropna()


def get_name():
    return "MeanReversion_20d_z1"
