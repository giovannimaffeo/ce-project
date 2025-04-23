import itertools
import pandas as pd
import numpy as np

from evolve_structure import ea_search
from fixed_controllers import alternating_gait
import utils

def basic_test():
  params = {
    # ---- PARAMETERS ----
    "NUM_GENERATIONS": 2,  # Number of generations to evolve
    "MIN_GRID_SIZE": (5, 5),  # Minimum size of the robot grid
    "MAX_GRID_SIZE": (5, 5),  # Maximum size of the robot grid
    "STEPS": 500,
    "SCENARIO": "Walker-v0",  # "BridgeWalkerv0"
    "POP_SIZE": 5,  # 15
    "CROSSOVER_RATE": 0.95,
    "MUTATION_RATE": 0.03,
    "SURVIVORS_COUNT": 1,
    "PARENT_SELECTION_COUNT": 3,  # 4
    # ---- VOXEL TYPES ----
    "VOXEL_TYPES": [0, 1, 2, 3, 4],  # Empty, Rigid, Soft, Active (+/-)
    "CONTROLLER": alternating_gait,
    "SEED": None
  }

  best_robot, best_fitness, fitness_history = ea_search(**params)
  print("Best robot structure found:")
  print(best_robot)
  print("Best fitness score:")
  print(best_fitness)

  # generate results
  fitness_history_df = pd.DataFrame(fitness_history)
  utils.generate_results(fitness_history_df, best_robot, params)

def hiperparams_fatorial_search():
  # rodar os basic_tests como assumptions e offline testing para encontrar os melhores 3/4 valores para as listas
  NUM_GENERATIONS_LIST = [2, 5, 10]
  POP_SIZE_LIST = [5, 10, 15]
  CROSSOVER_RATE_LIST = [0.6, 0.75, 0.9]
  MUTATION_RATE_LIST = [0.01, 0.05, 0.1]
  SURVIVORS_COUNT_LIST = [1, 2]
  PARENT_SELECTION_COUNT_LIST = [2, 3, 4]
  SEEDS = [3223, 19676, 85960, 12577, 62400]

  results = []

  all_combinations = list(itertools.product(
    NUM_GENERATIONS_LIST,
    POP_SIZE_LIST,
    CROSSOVER_RATE_LIST,
    MUTATION_RATE_LIST,
    SURVIVORS_COUNT_LIST,
    PARENT_SELECTION_COUNT_LIST
  ))

  for comb in all_combinations:
    NUM_GENERATIONS, POP_SIZE, CROSSOVER_RATE, MUTATION_RATE, SURVIVORS_COUNT, PARENT_SELECTION_COUNT = comb
    fitnesses = []

    for seed in SEEDS:
      result = ea_search(
        NUM_GENERATIONS=NUM_GENERATIONS,
        MIN_GRID_SIZE=(5, 5),
        MAX_GRID_SIZE=(5, 5),
        STEPS=500,
        SCENARIO="Walker-v0",
        POP_SIZE=POP_SIZE,
        CROSSOVER_RATE=CROSSOVER_RATE,
        MUTATION_RATE=MUTATION_RATE,
        SURVIVORS_COUNT=SURVIVORS_COUNT,
        PARENT_SELECTION_COUNT=PARENT_SELECTION_COUNT,
        VOXEL_TYPES=[0, 1, 2, 3, 4],
        CONTROLLER=None,
        SEED=seed
      )
      _, fitness, _ = result
      fitnesses.append(fitness)

    # pensar em uma forma melhor de guardar esses resultados
    avg_fitness = np.mean(fitnesses)

    results.append({
      "NUM_GENERATIONS": NUM_GENERATIONS,
      "POP_SIZE": POP_SIZE,
      "CROSSOVER_RATE": CROSSOVER_RATE,
      "MUTATION_RATE": MUTATION_RATE,
      "SURVIVORS_COUNT": SURVIVORS_COUNT,
      "PARENT_SELECTION_COUNT": PARENT_SELECTION_COUNT,
      "avg_fitness": avg_fitness
    })

  return results

basic_test()