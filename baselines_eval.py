import json
from BlackJack_Simulator.BlackJack import omega_II_baseline_eval

basic_strategy_filename = "./BlackJack_Simulator/strategy/BasicStrategy.csv"
eval_output_filename = "data/results/baselines_performance.json"
perf = {}

for bet_spread in [1, 20]:
    for n_decks in [1, 2, 4]:
        perf[str((bet_spread, n_decks))] = []
        print(f"bet_spread: {bet_spread}, n_decks: {n_decks}")
        for trial in range(5):
            perf[str((bet_spread, n_decks))].append(
                omega_II_baseline_eval(
                    basic_strategy_filename,
                    bet_spread=bet_spread,
                    n_decks=n_decks,
                )
            )

with open(eval_output_filename, "w") as f:
    json.dump(perf, f)

print(perf)
