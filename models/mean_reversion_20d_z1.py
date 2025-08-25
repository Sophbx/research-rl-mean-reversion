import pandas as pd
import numpy as np

def _ensure_series(x, name=None, index=None):
    """Coerce 1-col DataFrame/ndarray/list to a pandas Series."""
    if isinstance(x, pd.DataFrame):
        x = x.iloc[:, 0]
    elif not isinstance(x, pd.Series):
        x = pd.Series(x, index=index)
    if name is not None:
        x.name = name
    return x

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
    ret_d = _ensure_series(df['Close'].pct_change(), name='ret_d', index=df.index)
    daily_vol = _ensure_series(ret_d.rolling(20).std(), name='daily_vol', index=df.index)
    budget = float(target_vol_annual) / np.sqrt(252)
    target_w  = budget / daily_vol.replace(0, np.nan)
    target_w = _ensure_series(target_w, name='target_w', index=df.index).clip(upper=3).fillna(0)

    # sanity checks 
    assert isinstance(target_w, pd.Series) and target_w.ndim == 1
    assert isinstance(df['Position'], pd.Series) and df['Position'].ndim == 1

    df['Weight'] = df['Position'].mul(target_w, fill_value=0)

    # P&L with costs (bps per one-way trade)
    ret = _ensure_series(df['Close'].pct_change(), index=df.index).fillna(0)
    turnover = _ensure_series(df['Weight'].diff().abs(), index=df.index).fillna(df['Weight'].abs())
    cost = (cost_bps_one_way/1e4) * turnover
    df['Strategy'] = df['Weight'] * ret - cost
    df['Returns'] = ret
    if 'Date' not in df.columns:
        df['Date'] = df.index

    return df.dropna()

def get_name():
    return "MeanReversion_20d_z1"
