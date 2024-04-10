from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from BlackJack_Simulator.BlackJack import UltimateBlackjackRoundEnv
from stable_baselines3.common.callbacks import EvalCallback
from utils.utils import *

EPISODE_LENGTH = 2
N_ENVS = 1

monitor_kwargs = {
    "info_keywords": (
        "illegal",
        "won",
        "hit",
        "stand",
        "double",
        "surrender",
        "split",
        "return",
    )
}
env_kwargs = {"card_counting": True}
model_name = "test_lr_linear_n_env_1"


def lr_fixed_schedule(progress_remaining):
    return 0.0003


def lr_step_schedule(progress_remaining):
    """
    Learning rate step schedule as explained in the paper
    http://cs230.stanford.edu/projects_fall_2021/reports/103066753.pdf
    """
    if 1 - progress_remaining < (1e4 / (1e4 + 1e6 + 1e7)):
        return 0.0001
    elif 1 - progress_remaining < (1e6 / (1e4 + 1e6 + 1e7)):
        return 0.00001
    else:
        return 0.000001


def lr_linear_schedule(progress_remaining):
    start_lr = 0.0003
    end_lr = 0.000001
    return start_lr - ((1 - progress_remaining) * (start_lr - end_lr))


def lr_decay_schedule(progress_remaining):
    start_lr = 0.0003
    decay_rate = 1000
    return start_lr / (1 + decay_rate * (1 - progress_remaining))


# Define the parallel environments
vec_env = make_vec_env(
    UltimateBlackjackRoundEnv,
    n_envs=N_ENVS,
    monitor_kwargs=monitor_kwargs,
    env_kwargs=env_kwargs,
)


# Train the agent
model = PPO(
    "MlpPolicy",
    vec_env,
    verbose=0,
    device="cpu",
    learning_rate=lr_linear_schedule,  # To be fine tuned
    # batch_size=32,
    # n_steps=8192,
    # n_epochs=5,
    tensorboard_log="./ppo_UBJ_tensorboard/",
)


eval_env = UltimateBlackjackRoundEnv(**env_kwargs)
eval_env = Monitor(eval_env, filename=None, **monitor_kwargs)
eval_callback = EvalCallback(
    eval_env,
    log_path="./ppo_UBJ_tensorboard/",
    eval_freq=25000,
    deterministic=True,
    render=False,
    n_eval_episodes=10000,
    verbose=0,  # For no output
)

model.learn(
    # Would multiplying by N_ENVS * EPISODE_LENGTH be the "best" way to compare trainig with the paper?
    # Training time explodes if we do that
    total_timesteps=(1e4 + 1e6 + 1e7),
    # total_timesteps=(1e6),
    progress_bar=True,
    log_interval=1,
    callback=[TensorboardCallback(), eval_callback],
    tb_log_name=model_name,
)

model.save(model_name)

# Try CnnPolicy
# Try
