# agents/train_mr_discrete.py
import pandas as pd
from gymnasium.wrappers import TimeLimit
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from env.mr_discrete_env import MRDiscreteEnv

def split_df(df, train=("2010-01-01","2018-12-31"),
                 val=("2019-01-01","2021-12-31"),
                 test=("2022-01-01","2025-08-21")):
    df = df.copy()
    df.index = pd.to_datetime(df.index)
    return df.loc[train[0]:train[1]], df.loc[val[0]:val[1]], df.loc[test[0]:test[1]]

def make_env(df, **kwargs):
    # Optional: cap each episode length; helps PPO’s on-policy updates
    return TimeLimit(MRDiscreteEnv(df, **kwargs), max_episode_steps=len(df)-1)

def train(df):
    df_train, df_val, df_test = split_df(df)

    env_train = DummyVecEnv([lambda: make_env(df_train, reward_mode="return")])
    env_val   = DummyVecEnv([lambda: make_env(df_val,   reward_mode="return")])

    model = PPO(
        "MlpPolicy",
        env_train,
        verbose=1,
        n_steps=2048,
        batch_size=256,
        gamma=0.99,
        gae_lambda=0.95,
        ent_coef=0.0,
        learning_rate=3e-4,
        n_epochs=10,
        clip_range=0.2,
        seed=42,
    )

    model.learn(total_timesteps=500_000)
    model.save("models/ppo_mr_discrete_buy_hold_sell.zip")

    # Quick sanity rollout on val
    _ = evaluate(model, env_val, tag="VAL")

    # Test rollout
    env_test = DummyVecEnv([lambda: make_env(df_test, reward_mode="return")])
    _ = evaluate(model, env_test, tag="TEST")
    return model

def evaluate(model, vec_env, tag="EVAL"):
    obs = vec_env.reset()
    rewards_hist = []
    final_equity = None

    while True:
        action, _ = model.predict(obs, deterministic=True)
        # VecEnv returns 4-tuple
        obs, rewards, dones, infos = vec_env.step(action)

        r = float(rewards[0])     # rewards is vectorized
        done = bool(dones[0])
        info = infos[0]

        rewards_hist.append(r)

        if done:
            final_equity = info.get("final_equity", info.get("equity"))
            print(f"[{tag}] steps={len(rewards_hist)} "
                  f"mean_r={sum(rewards_hist)/len(rewards_hist):.6f} "
                  f"final_equity={final_equity:.4f}")
            break

    return rewards_hist
