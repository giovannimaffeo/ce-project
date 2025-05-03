import itertools
import os
import pandas as pd

from evolve_structure import uniform_crossover
from fixed_controllers import alternating_gait
import utils

def load_combinations_results_from_disk(base_output_dir):
  combinations_results = []
  i = 1
  while True:
    combination_dir = os.path.join(base_output_dir, f"combination{i}")
    if not os.path.exists(combination_dir):
      break  # terminou as combinações
    try:
      combination_results_path = os.path.join(combination_dir, "combination_results.csv")
      fitness_history_path = os.path.join(combination_dir, "combination_fitness_history.csv")

      combination_results_df = pd.read_csv(combination_results_path)
      combination_fitness_history_df = pd.read_csv(fitness_history_path)

      combinations_results.append((None, combination_results_df, combination_fitness_history_df))
    except Exception as e:
      print(f"Erro ao carregar combinação {i}: {e}")
    i += 1

  return combinations_results

combinations_results = load_combinations_results_from_disk("outputs/evolve_structure/ea_search/hiperparams_fatorial_tests/reference_run")
print(combinations_results)

fixed_params = {
  "NUM_GENERATIONS": 100,
  "MIN_GRID_SIZE": (5, 5),
  "MAX_GRID_SIZE": (5, 5),
  "STEPS": 500,
  "SCENARIO": "Walker-v0",
  "POP_SIZE": 50,
  "CROSSOVER_TYPE": uniform_crossover,
  "VOXEL_TYPES": [0, 1, 2, 3, 4],
  "CONTROLLER": alternating_gait
}

variable_params_grid = {
  "MUTATION_RATE": [0.03, 0.05],
  "CROSSOVER_RATE": [0.9, 0.95],
  "SURVIVORS_COUNT": [3, 5],
  "PARENT_SELECTION_COUNT": [3, 4]
}

output_dir = f"outputs/evolve_structure/ea_search/hiperparams_fatorial_tests/reference_run"
variable_param_keys = list(variable_params_grid.keys())
all_combinations = list(itertools.product(*variable_params_grid.values()))
utils.generate_hiperparams_fatorial_test_results(
  fixed_params,
  variable_params_grid,
  combinations_results,
  output_dir
)