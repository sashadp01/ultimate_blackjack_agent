import gymnasium as gym

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

# Define the parallel environments
vec_env = make_vec_env("CartPole-v1", n_envs=4)

# Train the agent
model = PPO("MlpPolicy", vec_env, verbose=1,device='cuda')
model.learn(total_timesteps=25000)
model.save("ppo_cartpole")

del model  # remove to demonstrate saving and loading

model = PPO.load("ppo_cartpole")

# Perform evaluation
obs = vec_env.reset()
while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = vec_env.step(action)
    vec_env.render("human")
