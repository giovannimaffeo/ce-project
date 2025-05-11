import itertools
import os
import pandas as pd

from ea_structure import evaluate_fitness, uniform_crossover
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

combinations_results = load_combinations_results_from_disk(f"outputs/evolve_both/evolve_both/{utils.test_types[1]}/reference_run")
print(combinations_results)

fixed_params = {
  "STRUCTURE_NUM_GENERATIONS": 7,
  "MIN_GRID_SIZE": (5, 5),
  "MAX_GRID_SIZE": (5, 5),
  "STEPS": 500,
  "STRUCTURE_POP_SIZE": 5,
  "CROSSOVER_RATE": 0.9,
  "CROSSOVER_TYPE": uniform_crossover,
  "STRUCTURE_MUTATION_RATE": 0.3,
  "SURVIVORS_COUNT": 3,
  "PARENT_SELECTION_COUNT": 2,
  "VOXEL_TYPES": [0, 1, 2, 3, 4],
  "CONTROLLER_NUM_GENERATIONS": 30,
  "CONTROLLER_POP_SIZE": 30,
  "CONTROLLER_MUTATION_RATE": 0.5,
  "SIGMA": 0.5,
  "NUM_OFFSPRINGS": 5,
  "LOG_FILE": None,
  "evaluate_fitness_fn": evaluate_fitness#evaluate_fitness3
}

variable_params_grid = {
  "SCENARIO": ["GapJumper-v0", "CaveCrawler-v0"]
}

output_dir = f"outputs/evolve_both/evolve_both/{utils.test_types[1]}/reference_run"
variable_param_keys = list(variable_params_grid.keys())
all_combinations = list(itertools.product(*variable_params_grid.values()))
utils.generate_param_combinations_results(
  fixed_params,
  variable_params_grid,
  combinations_results,
  output_dir,
  utils.test_types[1]
)