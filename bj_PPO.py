import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

from custom_bj_env import BlackjackModified

# Define the parallel environments
vec_env = make_vec_env(BlackjackModified, n_envs=4)

# Train the agent
model = PPO("MlpPolicy", vec_env, verbose=1)
model.learn(total_timesteps=25000)
model.save("ppo_cartpole")

del model  # remove to demonstrate saving and loading

model = PPO.load("ppo_cartpole")

# Perform evaluation on a single environment
env = BlackjackModified()
returns = []
for i in range(50000):
    obs = env.reset()[0]  # obs, info = env.reset()` (Gym) vs `obs = vec_env.reset()
    done = False
    reward = None
    while not done:
        action, _states = model.predict(obs)
        obs, reward, done, _, info = env.step(action)
    returns.append(reward)

print(np.mean(returns))
