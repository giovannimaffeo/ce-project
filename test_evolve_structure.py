import itertools
from multiprocessing import Pool, cpu_count
import os
import pandas as pd
from datetime import datetime
import gc

from ea_structure import ea_search, one_point_crossover, two_point_crossover, two_point_crossover2, uniform_crossover
from fixed_controllers import alternating_gait, hopping_motion, sinusoidal_wave
from random_structure import random_search
import utils

import tracemalloc
tracemalloc.start()

def basic_test(params, algorithm, output_dir=None, should_create_gif=True):
  if output_dir is None:
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = f"outputs/evolve_structure/{algorithm.__name__}/basic_test/{timestamp}"

  best_robot, best_fitness, fitness_history = algorithm(**params)
  print("Best robot structure found:")
  print(best_robot)
  print("Best fitness score:")
  print(best_fitness)

  # generate results
  fitness_history_df = pd.DataFrame(fitness_history)
  utils.generate_results(fitness_history_df, best_robot, params, output_dir, should_create_gif)
  return best_robot, best_fitness, fitness_history

def run_combination(args):
  i, combination, variable_param_keys, fixed_params, SEEDS, algorithm, output_dir = args
  combination_output_dir = os.path.join(output_dir, f"combination{i+1}")
  os.makedirs(combination_output_dir, exist_ok=True)

  combination_variable_params = dict(zip(variable_param_keys, combination))
  best_fitnesses = []
  fitness_historic_paths = []

  for j, seed in enumerate(SEEDS):
    params = {
      **fixed_params,
      **combination_variable_params,
      "SEED": seed,
      "LOG_FILE": os.path.join(combination_output_dir, f"log_combination{i + 1}_pid{os.getpid()}.txt")        
    }
    run_output_dir = os.path.join(combination_output_dir, f"run{j+1}")
    _, best_fitness, fitness_history = basic_test(params, algorithm, run_output_dir, False)
    best_fitnesses.append(best_fitness)

    current, peak = tracemalloc.get_traced_memory()
    print(f"[PID {os.getpid()}] Current memory usage: {current / 1024**2:.2f} MB; Peak: {peak / 1024**2:.2f} MB")
    del fitness_history
    gc.collect()
    current, peak = tracemalloc.get_traced_memory()
    print(f"[PID {os.getpid()}] Current memory usage: {current / 1024**2:.2f} MB; Peak: {peak / 1024**2:.2f} MB")
    fitness_historic_paths.append(os.path.join(run_output_dir, "fitness_history.csv"))

  result = utils.generate_combination_results(
    combination_variable_params, 
    best_fitnesses, 
    fitness_historic_paths, 
    combination_output_dir
  )
  return list(result)

def run_param_combinations(fixed_params, variable_params_grid, algorithm, test_type):
  SEEDS = [3223, 19676, 85960, 12577, 62400]
  timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
  output_dir = f"outputs/evolve_structure/{algorithm.__name__}/{test_type}/{timestamp}"

  variable_param_keys = list(variable_params_grid.keys())
  all_combinations = list(itertools.product(*variable_params_grid.values()))

  # Cria args para cada combinação
  args_list = [
    (i, combination, variable_param_keys, fixed_params, SEEDS, algorithm, output_dir)
    for i, combination in enumerate(all_combinations)
  ]

  with Pool(processes=min(cpu_count(), len(args_list))) as pool:
    combinations_results = pool.map(run_combination, args_list)

  utils.generate_param_combinations_results(
    fixed_params, 
    variable_params_grid, 
    combinations_results, 
    output_dir,
    test_type
  )

test_types = ["hiperparams_fatorial_test", "controller_scenario_test"]
def ea_search_basic_test():
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
  basic_test(params, ea_search)

def random_search_basic_test():
  # ---- PARAMETERS ----
  params = {
    "NUM_GENERATIONS": 2,             # 250  # Number of generations to evolve
    "MIN_GRID_SIZE": (5, 5),          # Minimum size of the robot grid
    "MAX_GRID_SIZE": (5, 5),          # Maximum size of the robot grid
    "STEPS": 500,
    "SCENARIO": "Walker-v0",
    # ---- VOXEL TYPES ----
    "VOXEL_TYPES": [0, 1, 2, 3, 4],   # Empty, Rigid, Soft, Active (+/-)
    "CONTROLLER": alternating_gait
  }
  basic_test(params, random_search)

def ea_search_hiperparams_fatorial_test():
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
  run_param_combinations(fixed_params, variable_params_grid, ea_search, test_types[0])

def ea_search_controller_scenario_test():
  fixed_params = {
    "NUM_GENERATIONS": 100,
    "MIN_GRID_SIZE": (5, 5),
    "MAX_GRID_SIZE": (5, 5),
    "STEPS": 500,
    "POP_SIZE": 50,
    "CROSSOVER_TYPE": uniform_crossover,
    "VOXEL_TYPES": [0, 1, 2, 3, 4],
    "MUTATION_RATE": 0.05,
    "CROSSOVER_RATE": 0.95,
    "SURVIVORS_COUNT": 5,
    "PARENT_SELECTION_COUNT": 4
  }
  variable_params_grid = {
    "SCENARIO": ["BridgeWalker-v0", "Walker-v0"],
    "CONTROLLER": [sinusoidal_wave, alternating_gait, hopping_motion]
  }
  run_param_combinations(fixed_params, variable_params_grid, ea_search, test_types[1])

def random_search_controller_scenario_test():
  fixed_params = {
    "NUM_GENERATIONS": 100,
    "MIN_GRID_SIZE": (5, 5),
    "MAX_GRID_SIZE": (5, 5),
    "STEPS": 500,
    "VOXEL_TYPES": [0, 1, 2, 3, 4]
  }
  variable_params_grid = {
    "SCENARIO": ["BridgeWalker-v0", "Walker-v0"],
    "CONTROLLER": [sinusoidal_wave, alternating_gait, hopping_motion]
  }
  run_param_combinations(fixed_params, variable_params_grid, random_search, test_types[1])

# ea_search_basic_test()
# random_search_basic_test()
# ea_search_hiperparams_fatorial_test()
# ea_search_controller_scenario_test()
random_search_controller_scenario_test()