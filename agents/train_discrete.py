# agents/train_discrete.py
import pandas as pd
from stable_baselines3 import PPO, DQN
from stable_baselines3.common.vec_env import DummyVecEnv
from env.mr_discrete_env import MeanReversionTradingEnv  # your env
from env.discrete_wrapper import DiscreteActionWrapper

def make_env(df, budget_annual_vol=0.10, cost_bps_one_way=5, start=None, end=None):
    base = MeanReversionTradingEnv(
        df.loc[start:end],
        budget_annual_vol=budget_annual_vol,
        cost_bps_one_way=cost_bps_one_way
    )
    return DiscreteActionWrapper(base)

def split_df(df, train=("2010-01-01","2018-12-31"),
                 val=("2019-01-01","2021-12-31"),
                 test=("2022-01-01","2025-08-21")):
    df = df.copy()
    df.index = pd.to_datetime(df.index)
    return (
        df.loc[train[0]:train[1]],
        df.loc[val[0]:val[1]],
        df.loc[test[0]:test[1]],
    )

def train(df):
    df_train, df_val, df_test = split_df(df)

    # Vec envs
    env_train = DummyVecEnv([lambda: make_env(df_train)])
    env_val   = DummyVecEnv([lambda: make_env(df_val)])

    # Algorithm pick
    # PPO (discrete)
    model = PPO("MlpPolicy", env_train, verbose=1, n_steps=2048, batch_size=256,
                gamma=0.99, gae_lambda=0.95, ent_coef=0.00, learning_rate=3e-4)

    # (Alternative) DQN
    # model = DQN("MlpPolicy", env_train, verbose=1, learning_rate=1e-4,
    #             buffer_size=200_000, learning_starts=10_000,
    #             batch_size=256, gamma=0.99, target_update_interval=2000)

    model.learn(total_timesteps=500_000)

    # Save
    model.save("models/ppo_discrete_buy_hold_sell.zip")

    # Evaluate on validation (quick roll-out)
    val_rewards, val_equity = rollout(model, env_val)
    print(f"[VAL] steps={len(val_rewards)} mean_r={pd.Series(val_rewards).mean():.6f} final_equity={val_equity:.4f}")

    # Final test roll-out (no learning)
    env_test = DummyVecEnv([lambda: make_env(df_test)])
    test_rewards, test_equity = rollout(model, env_test)
    print(f"[TEST] steps={len(test_rewards)} mean_r={pd.Series(test_rewards).mean():.6f} final_equity={test_equity:.4f}")

    return model

def rollout(model, vec_env):
    obs = vec_env.reset()
    rewards = []
    final_equity = None
    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, trunc, infos = vec_env.step(action)
        rewards.append(float(reward[0]))
        if done[0] or trunc[0]:
            # pull equity from info if your env exposes it
            info = infos[0]
            final_equity = info.get("final_equity", info.get("equity", None))
            break
    return rewards, final_equity
