import matplotlib.pyplot as plt

def plot_cumulative_returns(df, title="Strategy Performance"):
    df['Cumulative Strategy'] = (1 + df['Strategy']).cumprod()
    df['Cumulative Market'] = (1 + df['Returns']).cumprod()

    plt.figure(figsize=(10, 5))
    plt.plot(df['Date'], df['Cumulative Strategy'], label='Strategy')
    plt.plot(df['Date'], df['Cumulative Market'], label='Market')
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Cumulative Return")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
