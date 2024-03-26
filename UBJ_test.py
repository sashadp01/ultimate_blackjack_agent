

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
import torch
from BlackJack_Simulator.BlackJack import UltimateBlackjackRoundEnv
from time import sleep

model = PPO.load("ppo_UBJ")

# Perform evaluation on a single environment
env = UltimateBlackjackRoundEnv(render_mode='console')

for i in range(5):
    print(" ")
    print(f"Round {i+1}")
    obs = env.reset()[0]  # obs, info = env.reset()` (Gym) vs `obs = vec_env.reset()
    done = False
    reward = None
    while not done:
        env.render()
        action, _states = model.predict(obs)
        print(f"Action: {action}")
        obs, reward, done, _, info = env.step(action)
    env.render()
    print(f"Reward: {reward}")
    sleep(1)


