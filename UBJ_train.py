

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO,A2C
from stable_baselines3.common.env_util import make_vec_env
import torch
from BlackJack_Simulator.BlackJack import UltimateBlackjackRoundEnv

# Define the parallel environments
vec_env = make_vec_env(UltimateBlackjackRoundEnv, n_envs=4)
# Train the agent
model = PPO("MlpPolicy", vec_env, verbose=1,device='cpu')
model.learn(total_timesteps=250000,progress_bar=True,log_interval=20)
model.save("ppo_UBJ")

del model  # remove to demonstrate saving and loading

model = PPO.load("ppo_UBJ")

# Perform evaluation on a single environment
env = UltimateBlackjackRoundEnv()
returns = []
for i in range(25000):
    obs = env.reset()[0]  # obs, info = env.reset()` (Gym) vs `obs = vec_env.reset()
    done = False
    reward = None
    while not done:
        action, _states = model.predict(obs)
        obs, reward, done, _, info = env.step(action)
    returns.append(reward)

print(np.mean(returns))
