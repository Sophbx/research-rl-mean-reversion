# analysis/robustness_harness.py
import pandas as pd
import numpy as np
from analysis.quick_stats import quick_stats

# single-env evaluation to avoid VecEnv unpacking gotchas
def eval_policy_single_env(model, env_cls, df, **env_kwargs):
    env = env_cls(df, **env_kwargs)
    obs, _ = env.reset()
    rewards = []
    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(int(np.asarray(action).item()))
        rewards.append(float(reward))
        if terminated or truncated:
            break

    # build output DF: Date, Strategy (agent), Returns (market), Close
    price = env.df["Close"].reset_index()
    price.columns = ["Date", "Close"]
    strat = pd.Series(rewards, name="Strategy")
    dates = price["Date"].tail(len(strat)).reset_index(drop=True)
    closes = price["Close"].tail(len(strat) + 1).reset_index(drop=True)
    mkt = closes.pct_change().fillna(0).tail(len(strat)).reset_index(drop=True)
    out = pd.DataFrame({"Date": dates, "Strategy": strat, "Returns": mkt, "Close": closes.tail(len(strat)).values})
    return out

def evaluate_policy_robustness(
    model,
    env_cls,           # e.g., MRDiscreteEnv
    df_test,
    costs=(5, 10, 20), # bps one-way
    lev_caps=(1.0, 2.0, 3.0),
    reward_mode="return",
    save_csv_path=None
):
    rows = []
    for cost in costs:
        for lev in lev_caps:
            df_eval = eval_policy_single_env(
                model,
                env_cls,
                df_test,
                reward_mode=reward_mode,
                cost_bps_one_way=cost,
                lev_cap=lev,
            )
            s = quick_stats(df_eval, ret_col="Strategy")
            rows.append({
                "cost_bps": cost,
                "lev_cap": lev,
                "Sharpe": round(s["Sharpe"], 3),
                "MaxDD": round(s["MaxDD"], 4),
                "ann_ret": round(s["ann_ret"], 4),
                "ann_vol": round(s["ann_vol"], 4),
                "n_days": s["len"],
            })
    table = pd.DataFrame(rows).sort_values(["cost_bps", "lev_cap"]).reset_index(drop=True)
    if save_csv_path:
        table.to_csv(save_csv_path, index=False)
    return table
