import itertools
from multiprocessing import Pool, cpu_count
import os
import pandas as pd
from datetime import datetime

from evolve_structure import ea_search, one_point_crossover, two_point_crossover, two_point_crossover2, uniform_crossover
from fixed_controllers import alternating_gait
import utils

def basic_test(params, output_dir=None):
  if output_dir is None:
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = f"outputs/evolve_structure/ea_search/basic_tests/{timestamp}"

  best_robot, best_fitness, fitness_history = ea_search(**params)
  print("Best robot structure found:")
  print(best_robot)
  print("Best fitness score:")
  print(best_fitness)

  # generate results
  fitness_history_df = pd.DataFrame(fitness_history)
  utils.generate_results(fitness_history_df, best_robot, params, output_dir)
  return best_robot, best_fitness, fitness_history

def run_combination(args):
  i, combination, variable_param_keys, fixed_params, SEEDS, timestamp = args
  combination_output_dir = f"outputs/evolve_structure/ea_search/hiperparams_fatorial_tests/{timestamp}/combination{i+1}"
  os.makedirs(combination_output_dir, exist_ok=True)

  combination_variable_params = dict(zip(variable_param_keys, combination))
  best_fitnesses = []
  fitness_historics = []

  for j, seed in enumerate(SEEDS):
    params = {
      **fixed_params,
      **combination_variable_params,
      "SEED": seed        
    }
    _, best_fitness, fitness_history = basic_test(params, f"{combination_output_dir}/run{j+1}")
    best_fitnesses.append(best_fitness)
    fitness_historics.append(fitness_history)

  result = utils.generate_combination_results(
    combination_variable_params, 
    best_fitnesses, 
    fitness_historics, 
    combination, 
    combination_output_dir
  )
  return list(result)

def hiperparams_fatorial_test():
  fixed_params = {
    "NUM_GENERATIONS": 2,
    "MIN_GRID_SIZE": (5, 5),
    "MAX_GRID_SIZE": (5, 5),
    "STEPS": 500,
    "SCENARIO": "Walker-v0",
    "POP_SIZE": 50,
    "CROSSOVER_TYPE": uniform_crossover,
    "VOXEL_TYPES": [0, 1, 2, 3, 4],
    "CONTROLLER": alternating_gait,
  }

  variable_params_grid = {
    "MUTATION_RATE": [0.03, 0.05],
    "CROSSOVER_RATE": [0.9, 0.95],
    "SURVIVORS_COUNT": [3, 5],
    "PARENT_SELECTION_COUNT": [3, 4]
  }

  SEEDS = [3223, 19676, 85960, 12577, 62400]
  timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
  output_dir = f"outputs/evolve_structure/ea_search/hiperparams_fatorial_tests/{timestamp}"

  variable_param_keys = list(variable_params_grid.keys())
  all_combinations = list(itertools.product(*variable_params_grid.values()))

  # Cria args para cada combinação
  args_list = [
    (i, combination, variable_param_keys, fixed_params, SEEDS, timestamp)
    for i, combination in enumerate(all_combinations)
  ]

  with Pool(processes=min(cpu_count(), len(args_list))) as pool:
    combinations_results = pool.map(run_combination, args_list)

  utils.generate_hiperparams_fatorial_test_results(
    fixed_params, 
    variable_params_grid, 
    combinations_results, 
    output_dir
  )

seeds = [3223, 19676, 85960, 12577, 62400]
params = {
  # ---- PARAMETERS ----
  "NUM_GENERATIONS": 100,  # Number of generations to evolve
  "MIN_GRID_SIZE": (5, 5),  # Minimum size of the robot grid
  "MAX_GRID_SIZE": (5, 5),  # Maximum size of the robot grid
  "STEPS": 500,
  "SCENARIO": "Walker-v0",  # "BridgeWalkerv0"
  "POP_SIZE": 30,  # 15
  "CROSSOVER_RATE": 0.95,
  "CROSSOVER_TYPE": two_point_crossover2,
  "MUTATION_RATE": 0.03,
  "SURVIVORS_COUNT": 5,
  "PARENT_SELECTION_COUNT": 4,  # 4
  # ---- VOXEL TYPES ----
  "VOXEL_TYPES": [0, 1, 2, 3, 4],  # Empty, Rigid, Soft, Active (+/-)
  "CONTROLLER": alternating_gait,
  "SEED": 3223
}
# basic_test(params)
hiperparams_fatorial_test()
