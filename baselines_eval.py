import json
from BlackJack_Simulator.BlackJack import omega_II_baseline_eval

#initialize results dictionary
basic_strategy_filename = "./BlackJack_Simulator/strategy/BasicStrategy.csv"
eval_output_filename = "data/results/baselines_performance.json"
results_dict = {"baseline_1_no_count": {}, "baseline_2_no_count": {}, "baseline_4_no_count": {}, "baseline_1_count": {}, "baseline_2_count": {}, "baseline_4_count": {}}
for baseline in results_dict.keys():
    results_dict[baseline] = {
            "avg_edge": [],
            "min_edge": None,
            "max_edge": None,
    }

#loop through different bet_spreads and n_decks
for bet_spread in [1, 20]:
    for n_decks in [1, 2, 4]:
        count = "no_count" if bet_spread == 1 else "count"
        print(f"bet_spread: {bet_spread}, n_decks: {n_decks}")
        for trial in range(5):
            results_dict[f"baseline_{n_decks}_{count}"]["avg_edge"].append(
                #run omega_II_baseline_eval function, for 1000000 rounds
                omega_II_baseline_eval(
                    basic_strategy_filename,
                    bet_spread=bet_spread,
                    n_decks=n_decks,
                )
            )
            results_dict[f"baseline_{n_decks}_{count}"]["min_edge"] = min(results_dict[f"baseline_{n_decks}_{count}"]["avg_edge"])
            results_dict[f"baseline_{n_decks}_{count}"]["max_edge"] = max(results_dict[f"baseline_{n_decks}_{count}"]["avg_edge"])

#save results to json file
with open(eval_output_filename, "w") as f:
    json.dump(results_dict, f)

print(results_dict)
