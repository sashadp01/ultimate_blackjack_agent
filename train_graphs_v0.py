import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO, A2C
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecMonitor
import torch
from BlackJack_Simulator.BlackJack import UltimateBlackjackRoundEnv

testing = False


def learning_rate_schedule(progress_remaining):
    start_lr = 0.0003
    end_lr = 0.000001
    return start_lr - progress_remaining * (start_lr - end_lr)


# Define the parallel environments
vec_env = make_vec_env(UltimateBlackjackRoundEnv, n_envs=4)

# Monitor the environment
log_dir = "logs/"
vec_env_monitor = VecMonitor(vec_env, log_dir)

# Train the agent
model = PPO(
    "MlpPolicy",
    vec_env_monitor,
    verbose=1,
    device="cpu",
    # learning_rate=learning_rate_schedule, # To be fine tuned
    tensorboard_log="./ppo_UBJ_tensorboard/",
)
model.learn(total_timesteps=250000, progress_bar=True, log_interval=1)
model.save("ppo_UBJ")

del model  # remove to demonstrate saving and loading

model = PPO.load("ppo_UBJ")

# Perform evaluation on a single environment
env = UltimateBlackjackRoundEnv()
returns = []
if testing:
    for i in range(25000):
        obs = env.reset()[0]  # obs, info = env.reset()` (Gym) vs `obs = vec_env.reset()
        done = False
        reward = None
        while not done:
            action, _states = model.predict(obs)
            obs, reward, done, _, info = env.step(action)
        returns.append(reward)

    print(np.mean(returns))
