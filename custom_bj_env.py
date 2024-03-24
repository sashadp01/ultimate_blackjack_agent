import gymnasium as gym
from gymnasium.envs.toy_text.blackjack import BlackjackEnv
from gymnasium.spaces import MultiDiscrete


class BlackjackModified(BlackjackEnv):
    def __init__(self, render_mode=None):
        super(BlackjackModified, self).__init__()
        self.observation_space = MultiDiscrete(
            [32, 11, 2]
        )  # [Player sum: 0-31, Dealer's upcard: 1-10, Usable ace: 0 or 1]
        self.render_mode = render_mode

    def step(self, action):
        observation, reward, terminated, _, info = super().step(action)
        return self._convert_obs(observation), reward, terminated, False, {}

    def reset(self, seed=None):
        obs, info = super().reset(seed)
        return self._convert_obs(obs), info

    def _convert_obs(self, obs):
        player_sum, dealer_card, usable_ace = obs
        return [player_sum, dealer_card, int(usable_ace)]


# Create the modified Blackjack environment
# custom_env = BlackjackModified()
# original_env = gym.make("Blackjack-v1")

# print("Custom Reset: ", custom_env.reset(seed=42))
# print("Original Reset: ", original_env.reset(seed=42))
