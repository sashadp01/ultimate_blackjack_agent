from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from BlackJack_Simulator.BlackJack import UltimateBlackjackRoundEnv
from statistics import mean
from tqdm import tqdm
import pandas as pd
from catboost import CatBoostRegressor
import numpy as np
import json

#change the model name to the model you want to test
MODEL_NAME = "f_PPO_counting_1_deck_linear_lr.zip"
CLASSIFIER_NAME = "f_PPO_counting_1_deck_linear_lr_catboost_model.cbm"
no_count = False  # set to true when using a model trained without card counting

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
env_kwargs = {
    "card_counting": True,
    "n_decks": 1,
}
# Load the model and classifier
model = PPO.load("data/models/" + MODEL_NAME)
classifier = CatBoostRegressor().load_model("data/classifiers/" + CLASSIFIER_NAME)
threshold = np.load(
    "data/classifiers/" + CLASSIFIER_NAME.replace("catboost_model.cbm", "threshold.npy")
)
env = UltimateBlackjackRoundEnv(**env_kwargs)
env = Monitor(env, filename=None, **monitor_kwargs)

#setup results dictionary
results_dict = {MODEL_NAME: {}}
results_dict[MODEL_NAME] = {
    "avg_return": [],
    "avg_return_bet_high": [],
    "avg_return_bet_low": [],
    "avg_return_bet_dynamic": [],
    "avg_bet": [],
    "avg_edge": [],
    "min_edge": None,
    "max_edge": None,
}

for j in range(5):  # 5 independent trials
    infos = []
    ep_returns = []
    bet_decision = []
    for i in tqdm(range(100000)):
        obs, info = env.reset()
        bet_state = list(obs[-10:])
        # Decrement the card count, to account for the cards in the player's hand
        for card in info["cards"]:
            if card.name == "Ace":
                bet_state[0] -= 1
            else:
                bet_state[card.value - 1] -= 1

        # make a bet using the classifier
        bet = classifier.predict([bet_state])[0]
        if bet < threshold:
            bet = 0
        else:
            bet = 1

        done = False
        reward = None
        while not done:
            if no_count:
                # truncate the observation to remove the card count
                obs = obs[:-10]
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, _, info = env.step(action)
        ep_return = info["episode"]["return"]
        infos.append(info)
        ep_returns.append(ep_return)
        bet_decision.append(bet)

    # convert to numpy arrays
    ep_returns = np.array(ep_returns)
    bet_decision = np.array(bet_decision)

    avg_return = np.mean(ep_returns)
    print(f"Average return constant bet: {avg_return}")
    avg_return_bet_high = np.mean(ep_returns[bet_decision == 1])
    print(
        f"Average return bet high: {avg_return_bet_high}, for {sum(bet_decision)} bets"
    )
    avg_return_bet_low = np.mean(ep_returns[bet_decision == 0])
    print(
        f"Average return bet low: {avg_return_bet_low}, for {len(bet_decision) - sum(bet_decision)} bets"
    )
    # calculate the average edge with dynamic betting
    odds = 20.0
    avg_return_bet_dynamic = np.mean(
        np.where(bet_decision == 0, ep_returns, ep_returns * odds)
    )
    avg_bet = np.mean(np.where(bet_decision == 0, 1, odds))
    avg_edge = avg_return_bet_dynamic / avg_bet
    print(f"Average return bet dynamic: {avg_return_bet_dynamic}")
    print(f"Average bet: {avg_bet}")
    print(f"average edge: {avg_edge}")

    results_dict[MODEL_NAME]["avg_return"].append(avg_return)
    results_dict[MODEL_NAME]["avg_return_bet_high"].append(avg_return_bet_high)
    results_dict[MODEL_NAME]["avg_return_bet_low"].append(avg_return_bet_low)
    results_dict[MODEL_NAME]["avg_return_bet_dynamic"].append(avg_return_bet_dynamic)
    results_dict[MODEL_NAME]["avg_bet"].append(avg_bet)
    results_dict[MODEL_NAME]["avg_edge"].append(avg_edge)
    results_dict[MODEL_NAME]["min_edge"] = min(results_dict[MODEL_NAME]["avg_edge"])
    results_dict[MODEL_NAME]["max_edge"] = max(results_dict[MODEL_NAME]["avg_edge"])


# save dict
with open("data/results/" + MODEL_NAME.replace(".zip", "_results.json"), "w") as f:
    json.dump(results_dict, f)



