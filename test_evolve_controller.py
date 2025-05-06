import itertools
from multiprocessing import Pool, cpu_count
import os
import pandas as pd
from datetime import datetime
import gc

from es_controller import es_search
from fixed_controllers import alternating_gait, hopping_motion, sinusoidal_wave
from random_controller import random_search
import utils

import tracemalloc
tracemalloc.start()

def basic_test(params, algorithm, output_dir=None, should_create_gif=True):
  if output_dir is None:
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = f"outputs/evolve_controller/{algorithm.__name__}/basic_test/{timestamp}"

  best_weights, best_fitness, fitness_history = algorithm(**params)
  print("Best weights found:")
  print(best_weights)
  print("Best fitness score:")
  print(best_fitness)

  # generate results
  fitness_history_df = pd.DataFrame(fitness_history)
  utils.generate_results(fitness_history_df, best_weights, params, output_dir, should_create_gif, "evolve_controller")
  return best_weights, best_fitness, fitness_history

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
  output_dir = f"outputs/evolve_controller/{algorithm.__name__}/{test_type}/{timestamp}"

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

def es_search_basic_test():
  params = {
    "NUM_GENERATIONS": 2,
    "STEPS": 500,
    "SCENARIO": "DownStepper-v0",
    "POP_SIZE": 2,
    "MUTATION_RATE": 0.1,
    "SIGMA": 0.3,
    "NUM_OFFSPRINGS": 1,
    "SEED": 3223,
    "LOG_FILE": None
  }
  basic_test(params, es_search, None, False)

def random_search_basic_test():
  params = {
    "NUM_GENERATIONS": 5,
    "STEPS": 500,
    "SCENARIO": "DownStepper-v0",
    "SEED": 3223,
    "LOG_FILE": None
  }
  basic_test(params, random_search, None, False)

def es_search_hiperparams_fatorial_test():
  fixed_params = {
    "NUM_GENERATIONS": 100,
    "STEPS": 500,
    "SCENARIO": "DownStepper-v0",
    "POP_SIZE": 30
  }
  variable_params_grid = {
    "MUTATION_RATE": [0.1, 0.3],
    "SIGMA": [0.3, 0.5],
    "NUM_OFFSPRINGS": [2, 3]
  }
  run_param_combinations(fixed_params, variable_params_grid, es_search, utils.test_types[0])

def es_search_scenario_test():
  fixed_params = {
    "NUM_GENERATIONS": 100,
    "STEPS": 500,
    "POP_SIZE": 30, 
    "MUTATION_RATE": 0.3,
    "SIGMA": 0.5,
    "NUM_OFFSPRINGS": 5,
  }
  variable_params_grid = {
    "SCENARIO": ["DownStepper-v0", "ObstacleTraverser-v0"]
  }
  run_param_combinations(fixed_params, variable_params_grid, es_search, utils.test_types[1])

def random_search_scenario_test():
  fixed_params = {
    "NUM_GENERATIONS": 100,
    "STEPS": 500
  }
  variable_params_grid = {
    "SCENARIO": ["DownStepper-v0", "ObstacleTraverser-v0"]
  }
  run_param_combinations(fixed_params, variable_params_grid, random_search, utils.test_types[1])

# es_search_basic_test()
# random_search_basic_test()
es_search_hiperparams_fatorial_test()
# es_search_scenario_test()
# random_search_scenario_test()