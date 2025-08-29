# analysis/quick_stats.py
import numpy as np

def quick_stats(df, ret_col="Strategy"):
    r = df[ret_col].astype(float)
    mu = r.mean()
    sd = r.std()
    ann_ret = (1 + r).prod()**(252/len(r)) - 1
    ann_vol = sd * np.sqrt(252)
    sharpe = (mu / (sd + 1e-12)) * np.sqrt(252)
    # max drawdown on the equity curve
    curve = (1 + r).cumprod()
    dd = (1 - (curve / curve.cummax())).max()
    return {
        "len": len(r),
        "mean": float(mu),
        "std": float(sd),
        "ann_ret": float(ann_ret),
        "ann_vol": float(ann_vol),
        "Sharpe": float(sharpe),
        "MaxDD": float(dd),
    }

def buy_and_hold_sharpe(df, price_col="Close", freq=252):
    """
    Compute annualized Sharpe of simple buy & hold from given price series.
    Uses daily log returns by default.
    """
    r = df[price_col].pct_change().dropna()
    mu, sd = r.mean(), r.std()
    sharpe = (mu / (sd + 1e-12)) * np.sqrt(freq)
    ann_ret = (1 + r).prod()**(freq/len(r)) - 1
    ann_vol = sd * np.sqrt(freq)
    return {
        "Sharpe": float(sharpe),
        "ann_ret": float(ann_ret),
        "ann_vol": float(ann_vol),
        "n_days": len(r),
    }

def subsample_sharpe(df, ret_col="Strategy", window=252, freq=252, by_year=False):
    """
    Compute Sharpe over rolling windows or calendar years.

    Args:
        df : DataFrame with returns
        ret_col : which column to use for returns
        window : number of days for rolling window (ignored if by_year=True)
        freq : trading days per year
        by_year : if True, compute Sharpe per calendar year

    Returns:
        dict of {period: sharpe}
    """
    r = df[ret_col].astype(float).dropna()
    out = {}

    if by_year:
        grouped = r.groupby(df["Date"].dt.year)
        for year, vals in grouped:
            mu, sd = vals.mean(), vals.std()
            sharpe = (mu / (sd + 1e-12)) * np.sqrt(freq)
            out[year] = float(sharpe)
    else:
        for start in range(0, len(r) - window + 1, window):
            slice_r = r.iloc[start:start+window]
            if len(slice_r) < window: break
            mu, sd = slice_r.mean(), slice_r.std()
            sharpe = (mu / (sd + 1e-12)) * np.sqrt(freq)
            period = f"{df['Date'].iloc[start].date()}–{df['Date'].iloc[start+window-1].date()}"
            out[period] = float(sharpe)
    return out

