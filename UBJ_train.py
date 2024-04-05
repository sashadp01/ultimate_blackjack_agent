

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO,A2C
from stable_baselines3.common.env_util import make_vec_env
import torch
from BlackJack_Simulator.BlackJack import UltimateBlackjackRoundEnv, UltimateBlackjackRoundEnvNoCount

# Define the parallel environments
vec_env = make_vec_env(UltimateBlackjackRoundEnvNoCount, n_envs=4)
# Train the agent
model = PPO("MlpPolicy", vec_env, verbose=1,device='cpu',learning_rate=1e-4)
model.learn(total_timesteps=1000000,progress_bar=True,log_interval=200)
model.save("ppo_UBJ_no_count")

del model  # remove to demonstrate saving and loading

model = PPO.load("ppo_UBJ_no_count")

# Perform evaluation on a single environment
env = UltimateBlackjackRoundEnvNoCount()
returns = []
infos = []
actions = []
for i in range(25000):
    obs = env.reset()[0]  # obs, info = env.reset()` (Gym) vs `obs = vec_env.reset()
    done = False
    reward = None
    while not done:
        action, _states = model.predict(obs,deterministic=True)
        obs, reward, done, _, info = env.step(action)
        infos.append(info)
        actions.append(action)
    returns.append(reward)

print(np.mean(returns))

#number of illegal moves in 50000 games
#info is a dict with "illegal": True/False
print(f"Number of illegal moves in {len(infos)} actions: {sum([1 for i in infos if i['illegal']])}")
print(f"Number of games: {sum([1 for i in infos if i['status']!=None])}")
print(f"Number of wins: {sum([1 for i in infos if i['status']=='WON'])}")
#use actions list (0-4)
print(f"Number of hits: {sum([1 for i in actions if i==0])}")
print(f"Number of stands: {sum([1 for i in actions if i==1])}")
print(f"Number of doubles: {sum([1 for i in actions if i==2])}")
print(f"Number of surrender: {sum([1 for i in actions if i==3])}")
print(f"Number of splits: {sum([1 for i in actions if i==4])}")


