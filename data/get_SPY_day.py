import yfinance as yf
import pandas as pd

# Download raw daily prices (NOT adjusted)
def fetch_daily_yahoo_data(symbol='SPY', start='2020-01-01', end='2025-06-10'):
    df = yf.download(symbol, start=start, end=end, auto_adjust=False)
    df.reset_index(inplace=True)
    df.to_csv(f"{symbol}_day.csv", index=False)
    return df

df = fetch_daily_yahoo_data('SPY', '2020-01-01', '2025-06-10')
print(df.head())