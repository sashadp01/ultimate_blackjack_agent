import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO, A2C
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecMonitor
from stable_baselines3.common.monitor import Monitor
import torch
from BlackJack_Simulator.BlackJack import UltimateBlackjackRoundEnv
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from statistics import mean

testing = False


# def learning_rate_schedule(progress_remaining):
#     start_lr = 0.0003
#     end_lr = 0.000001
#     return start_lr - ((1-progress_remaining) * (start_lr - end_lr))
def learning_rate_schedule(progress_remaining):
    start_lr = 0.0003
    decay_rate = 1000
    return start_lr / (1 + decay_rate * (1-progress_remaining))

monitor_kwargs = {"info_keywords": ("illegal", "won", "hit", "stand", "double", "surrender", "split","return")}
env_kwargs = {"card_counting": False}


# Define the parallel environments
vec_env = make_vec_env(UltimateBlackjackRoundEnv, n_envs=4, monitor_kwargs=monitor_kwargs,env_kwargs=env_kwargs)


# Train the agent
model = PPO(
    "MlpPolicy",
    vec_env,
    verbose=0,
    device="cpu",
    learning_rate=learning_rate_schedule, # To be fine tuned
    tensorboard_log="./ppo_UBJ_tensorboard/",
)


class TensorboardCallback(BaseCallback):
    """
    Custom callback for plotting additional values in tensorboard.
    """

    def __init__(self, verbose=0):
        super().__init__(verbose)

    def _on_step(self) -> bool:
        # Log scalar value (here a random variable)
        illegal = []
        won = []
        hit = []
        stand = []
        double = []
        surrender = []
        split = []
        return_ = []
        
        for idx, info in enumerate(self.locals["infos"]):
            if info.get("episode") is not None:
                episode = info["episode"]
                illegal.append(episode["illegal"])
                won.append(episode["won"])
                hit.append(episode["hit"])
                stand.append(episode["stand"])
                double.append(episode["double"])
                surrender.append(episode["surrender"])
                split.append(episode["split"])
                return_.append(episode["return"])
                
        
        if len(illegal) > 0:
            self.logger.record("action_log/illegal", mean(illegal))
            self.logger.record("action_log/won", mean(won))
            self.logger.record("action_log/hit", mean(hit))
            self.logger.record("action_log/stand", mean(stand))
            self.logger.record("action_log/double", mean(double))
            self.logger.record("action_log/surrender", mean(surrender))
            self.logger.record("action_log/split", mean(split))
            self.logger.record("action_log/return", mean(return_))
             
        return True


eval_env = UltimateBlackjackRoundEnv(**env_kwargs)
eval_env = Monitor(eval_env, filename=None, **monitor_kwargs)
eval_callback = EvalCallback(
    eval_env,
    log_path="./ppo_UBJ_tensorboard/",
    eval_freq=25000,
    deterministic=True,
    render=False,
    n_eval_episodes=10000,
)

model.learn(total_timesteps=10000000, progress_bar=True, log_interval=1, callback=[TensorboardCallback(), eval_callback])
model.save("ppo_UBJ")
