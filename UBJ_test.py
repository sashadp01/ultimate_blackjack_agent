

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
import torch
from BlackJack_Simulator.BlackJack import UltimateBlackjackRoundEnv
from time import sleep
from statistics import mean

monitor_kwargs = {"info_keywords": ("illegal", "won", "hit", "stand", "double", "surrender", "split","return")}
env_kwargs = {"card_counting": False}

# Perform evaluation on a single environment
model = PPO.load("ppo_UBJ_10M_no_count")

# Perform evaluation on a single environment
env = UltimateBlackjackRoundEnv(**env_kwargs)
env = Monitor(env, filename=None, **monitor_kwargs)
returns = []
infos = []
actions = []
for i in range(100000):
    obs = env.reset()[0]  # obs, info = env.reset()` (Gym) vs `obs = vec_env.reset()
    done = False
    reward = None
    while not done:
        action, _states = model.predict(obs,deterministic=True)
        obs, reward, done, _, info = env.step(action) 
        actions.append(action)
    returns.append(reward)
    infos.append(info)

print(mean(returns))

#number of illegal moves in 50000 games
#info is a dict with "illegal": True/False
print(f"""Number of illegal moves / episode in {len(infos)} episode: {mean([i["episode"]["illegal"] for i in infos])}""")
print(f"""Win rate: {mean([i["episode"]["won"] for i in infos])}""")
#use actions list (0-4)
print(f"""Number of hits / episode: {mean([i["episode"]["hit"] for i in infos])}""")
print(f"""Number of stands / episode: {mean([i["episode"]["stand"] for i in infos])}""")
print(f"""Number of doubles / episode: {mean([i["episode"]["double"] for i in infos])}""")
print(f"""Number of surrenders / episode: {mean([i["episode"]["surrender"] for i in infos])}""")
print(f"""Number of splits / episode: {mean([i["episode"]["split"] for i in infos])}""")
print(f"""Average returns from env: {mean([i["episode"]["return"] for i in infos])}""")
print(f"""Average returns from monitor: {mean([i["episode"]["r"] for i in infos])}""")

