# Reinforcement Learning for Long-Term Robustness of Mean-Reversion Strategies

This project investigates whether standard reinforcement learning agents (e.g., PPO) can improve the long-term performance of mean-reversion-based trading strategies over volatile market regimes. The performance is mainly measured by Sharpe ratio and drawdown.

**Abstract (1)**

Using daily SPY (S\&P 500 ETF) data as an initial case study, a baseline Z-score–based mean reversion model is implemented to establish a reference point. While such rule-based strategies are intuitive, their profitability is limited and sensitive to regime shifts.

To address this, a structured research pipeline is developed comprising: (i) data acquisition and preprocessing; (ii) indicator construction, including moving-average Z-scores, momentum, realized volatility, relative strength index, and drawdown; (iii) design and evaluation of the baseline strategy; (iv) development of a custom Gymnasium environment incorporating realistic frictions such as transaction costs, volatility targeting, and leverage limits; and (v) training of an RL agent (PPO with discrete buy, hold, and sell actions) to make trading decisions from state features.

Empirical evaluation demonstrates that the RL agent consistently outperforms both the baseline strategy and a buy-and-hold benchmark over the 2022–2025 test horizon, achieving annualized Sharpe ratios exceeding 3.5 while maintaining low drawdowns across a range of transaction cost and leverage settings. These findings suggest that RL can adaptively exploit regime-dependent opportunities and manage risk more effectively than static heuristics.

This work represents an initial step in a broader research program. Future experiments will apply the same RL framework to a variety of mean-reversion–based strategies, such as greed index models, as well as to multiple asset classes and tickers. The overarching goal is to assess whether RL methods can serve as a general mechanism for enhancing the long-term robustness of systematic trading strategies.