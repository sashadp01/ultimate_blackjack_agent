# Ultimate Blackjack Agent

## Overview

This repository consists of training a RL agent to play Blackjack. The Proximal Policy Optimization (PPO) algorithm is used to learn optimal Blackjack actions given the state of the deck, i.e. the cards that were previously played. A regression algorithm, CatBoost, is trained for dynamic betting upon the state of the deck before starting a round. The objective is to compare the performance of the PPO agent plus the dynamic betting component compared to the basic strategy, an optimal strategy for Blackjack actions, with Omega II card counting for dynamic betting.

## Installation

Install the required Python packages: `pip install -r requirements.txt`

## Folder Structure

- `BlackJack_Simulator`: This folder contains the original implementation of the [BlackJack_Simulator](https://github.com/seblau/BlackJack-Simulator/tree/master) as well as our custom implementation of the gym environment interface wrapping around the Blackjack simulator and adding features such as the state of the deck.

- `data`: This folder contains all the relevant data for the project:

  - `models`: model weights after training
  - `ppo_UBJ_tensorboard`: tensorboard visualization data
  - `datasets`: training data for the dynamic betting algorithms
  - `classifiers`: Weights and threshold for the dynamic betting algorithms
  - `results`: Final results from combined algorithms (PPO + CatBoost regressor)

- `utils`: This folder contains utility functions and reusable code for the project.

## Usage

- `training.py`: This script trains the PPO agent on the Blackjack environment. It contains different learning rate schedules and allows to save the performance under `data/ppo_UBJ_tensorboard/` and weights of the trained models under `data/models/`. This file was also used for hyperparameter tuning.

- `bets_dataset_generator.py`: This script generates a dataset for training the dynamic betting algorithm from a particular PPO model under `data/models/`. For each sample, the state of the deck before the round and the outcome of the round are saved. 10M samples are generated and saved into a file under `data/datasets/`. (The generated csv file might need to be compressed into a zip file.)

- `return_analysis.ipynb`: This notebooks analizes the datasets under `data/datasets/` with graphs.

- `bet_classifier_nb.ipynb`: This file takes the dataset under `data/datasets/` and trains a CatBoost regression algorithm to learn when to bet high and optimize the mean reward for each round. The model weights and thresholds are saved under `data/classifiers/`.

- `final_test.py`: This script tests the performance of the PPO agent combined with the CatBoost regressor and saves the performances metrics under `data/results/`.

- `baselines_eval.py`: This script tests the performance of the baselines for different high bet amount, 1 and 20, and for different number of decks, 1, 2, and 4. The baseline consists of the basic strategy for action selection and Omega II card counting for dynamic betting (normal bet/ high bet). Performances metrics are saved under `data/results/`.

- `results_plotting.ipynb`: This notebook takes the final performances metrics under `data/results/` to produce graphs.
