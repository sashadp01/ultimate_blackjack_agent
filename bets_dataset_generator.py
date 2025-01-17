from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from BlackJack_Simulator.BlackJack import UltimateBlackjackRoundEnv
from statistics import mean
from tqdm import tqdm
import pandas as pd

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
# Params
env_kwargs = {"card_counting": True, "n_decks": 4}
model_name = "f_PPO_no_counting_1_deck_linear_lr"
no_count = True  # set to true when using a model trained without card counting
model_dir = "data/models/"
output_dir = "data/datasets/"

model = PPO.load(model_dir + model_name)
env = UltimateBlackjackRoundEnv(**env_kwargs)
env = Monitor(env, filename=None, **monitor_kwargs)
infos = []
samples = []

#generate 10,000,000 samples
for i in tqdm(range(10000000)):
    obs, info = env.reset()
    bet_state = list(obs[-10:])
    # Decrement the card count, to account for the cards in the player's hand
    for card in info["cards"]:
        if card.name == "Ace":
            bet_state[0] -= 1
        else:
            bet_state[card.value - 1] -= 1

    done = False
    reward = None
    #play episode
    while not done:
        if no_count:
            # truncate the observation to remove the card count
            obs = obs[:-10]
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
data.to_csv(output_dir + model_name + "4_decks_dataset.csv", index=False)
