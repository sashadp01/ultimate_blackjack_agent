from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from BlackJack_Simulator.BlackJack import UltimateBlackjackRoundEnv
from statistics import mean
from tqdm import tqdm
import pandas as pd
from catboost import CatBoostRegressor
import numpy as np

MODEL_NAME = "test_lr_linear_schedule"

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

model = PPO.load(MODEL_NAME)
classifier = CatBoostRegressor().load_model("catboost_model.cbm")
threshold = np.load("threshold.npy")
env = UltimateBlackjackRoundEnv(**env_kwargs)
env = Monitor(env, filename=None, **monitor_kwargs)
infos = []
ep_returns = []
bet_decision = []

for i in tqdm(range(100000)):
    obs, info = env.reset()
    bet_state = list(obs[-10:])
    # Decrement the card count
    for card in info["cards"]:
        if card.name == "Ace":
            bet_state[0] -= 1
        else:
            bet_state[card.value - 1] -= 1
    # print([str(card) for card in info["cards"]])
    # print(obs)
    # print(bet_state)
    
    #make a bet using the classifier
    bet = classifier.predict([bet_state])[0]
    if bet < threshold:
        bet = 0
    else:
        bet = 1
    
    done = False
    reward = None
    while not done:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, _, info = env.step(action)
    ep_return = info["episode"]["return"]
    infos.append(info)
    ep_returns.append(ep_return)
    bet_decision.append(bet)


#convert to numpy arrays
ep_returns = np.array(ep_returns)
bet_decision = np.array(bet_decision)

avg_return = np.mean(ep_returns)
print(f"Average return constant bet: {avg_return}")
avg_return_bet_high = np.mean(ep_returns[bet_decision == 1])
print(f"Average return bet high: {avg_return_bet_high}, for {sum(bet_decision)} bets")
avg_return_bet_low = np.mean(ep_returns[bet_decision == 0])
print(f"Average return bet low: {avg_return_bet_low}, for {len(bet_decision) - sum(bet_decision)} bets")

odds = 10.0
avg_return_bet_dynamic = np.mean(np.where(bet_decision == 0, ep_returns, ep_returns * odds))
print(f"Average return bet dynamic: {avg_return_bet_dynamic}")

print(
    f"""Number of illegal moves / episode in {len(infos)} episode: {mean([i["episode"]["illegal"] for i in infos])}"""
)
print(f"""Win rate: {mean([i["episode"]["won"] for i in infos])}""")
print(f"""Number of hits / episode: {mean([i["episode"]["hit"] for i in infos])}""")
print(f"""Number of stands / episode: {mean([i["episode"]["stand"] for i in infos])}""")
print(
    f"""Number of doubles / episode: {mean([i["episode"]["double"] for i in infos])}"""
)
print(
    f"""Number of surrenders / episode: {mean([i["episode"]["surrender"] for i in infos])}"""
)
print(f"""Number of splits / episode: {mean([i["episode"]["split"] for i in infos])}""")
print(f"""Average returns from env: {mean([i["episode"]["return"] for i in infos])}""")
print(f"""Average returns from monitor: {mean([i["episode"]["r"] for i in infos])}""")


