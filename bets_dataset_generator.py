from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from BlackJack_Simulator.BlackJack import UltimateBlackjackRoundEnv
from statistics import mean
from tqdm import tqdm
import pandas as pd

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
env = UltimateBlackjackRoundEnv(**env_kwargs)
env = Monitor(env, filename=None, **monitor_kwargs)
infos = []
samples = []

for i in tqdm(range(10000000)):
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
    done = False
    reward = None
    while not done:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, _, info = env.step(action)
    infos.append(info)
    samples.append((bet_state, reward))

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
print(f"""Average returns from samples: {mean([s[1] for s in samples])}""")

# Convert the samples to a Dataframe
samples = [s[0] + [s[1]] for s in samples]
data = pd.DataFrame(
    samples,
    columns=[
        "Ace",
        "Two",
        "Three",
        "Four",
        "Five",
        "Six",
        "Seven",
        "Eight",
        "Nine",
        "Ten",
        "Reward",
    ],
)

# print(data)
data.to_csv("bets_dataset.csv", index=False)
