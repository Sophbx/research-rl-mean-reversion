import yfinance as yf
import pandas as pd

# Download raw daily prices (NOT adjusted)
def fetch_daily_OHLCV(ticker, start='2020-01-01', end='2025-08-22'):
    """
    Download OHLCV data daily for a given ticker.
    """
    df = yf.download(ticker, start=start, end=end, auto_adjust=False)
    df = df[['Open', 'High', 'Low', 'Close', 'Volume']].copy()

    return df

if __name__ == '__main__':
    # fetch the basic data
    data = fetch_daily_OHLCV('SPY', '2020-01-01', '2025-08-22')

    # save
    data.to_csv('data/raw_data/SPY_day.csv', index=True)
    print("OHLCV for SPY saved.")
