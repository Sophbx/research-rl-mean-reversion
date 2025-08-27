# analysis/eval_policy_to_df.py
import pandas as pd
import numpy as np
from env.mr_discrete_env import MRDiscreteEnv

def eval_policy_to_df(model, df_test: pd.DataFrame, **env_kwargs) -> pd.DataFrame:
    env = MRDiscreteEnv(df_test, **env_kwargs)
    obs, _ = env.reset()
    rewards = []
    dates = []
    closes = []

    # pull aligned date/close (skip warmup)
    price = env.df["Close"].reset_index()
    price.columns = ["Date", "Close"]

    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(int(action))
        rewards.append(float(reward))
        if terminated or truncated:
            break

    strat = pd.Series(rewards, name="Strategy")
    # Align dates to rewards length
    d = price["Date"].tail(len(strat)).reset_index(drop=True)
    c = price["Close"].tail(len(strat)+1).reset_index(drop=True)
    mkt = c.pct_change().fillna(0).tail(len(strat)).reset_index(drop=True)

    out = pd.DataFrame({"Date": d, "Strategy": strat, "Returns": mkt, "Close": c.tail(len(strat)).values})
    return out
