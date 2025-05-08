import gc
import itertools
import os
import pandas as pd
from datetime import datetime

from ea_structure import uniform_crossover
from es_controller import evaluate_fitness3
from evolve_both import evolve_both
import utils

import tracemalloc
tracemalloc.start()

def basic_test(params, algorithm, output_dir=None, should_create_gif=True):
  if output_dir is None:
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = f"outputs/evolve_both/{algorithm.__name__}/basic_test/{timestamp}"

  best_individual, fitness_history = algorithm(**params)
  print("Best structure found:")
  print(best_individual.structure)
  print("Best weights found:")
  print(best_individual.weights)
  print("Best fitness score:")
  print(best_individual.fitness)

  # generate results
  fitness_history_df = pd.DataFrame(fitness_history)
  utils.generate_results(fitness_history_df, best_individual, params, output_dir, should_create_gif, "evolve_both")
  return best_individual.structure, best_individual.weights, best_individual.fitness, fitness_history

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
  output_dir = f"outputs/evolve_both/{algorithm.__name__}/{test_type}/{timestamp}"

  variable_param_keys = list(variable_params_grid.keys())
  all_combinations = list(itertools.product(*variable_params_grid.values()))

  # Cria args para cada combinação
  args_list = [
    (i, combination, variable_param_keys, fixed_params, SEEDS, algorithm, output_dir)
    for i, combination in enumerate(all_combinations)
  ]
 
  combinations_results = []
  for args in args_list:
    result = run_combination(args)
    combinations_results.append(result)

  utils.generate_param_combinations_results(
    fixed_params, 
    variable_params_grid, 
    combinations_results, 
    output_dir,
    test_type
  )

def evolve_both_basic_test():
  params = {
    "STRUCTURE_NUM_GENERATIONS": 1,
    "MIN_GRID_SIZE": (5, 5),
    "MAX_GRID_SIZE": (5, 5),
    "STEPS": 500,
    "SCENARIO": "GapJumper-v0",
    "STRUCTURE_POP_SIZE": 50,
    "CROSSOVER_RATE": 0.9,
    "CROSSOVER_TYPE": uniform_crossover,
    "STRUCTURE_MUTATION_RATE": 0.3,
    "SURVIVORS_COUNT": 10,
    "PARENT_SELECTION_COUNT": 10,
    "VOXEL_TYPES": [0, 1, 2, 3, 4],
    "CONTROLLER_NUM_GENERATIONS": 30,
    "CONTROLLER_POP_SIZE": 30,
    "CONTROLLER_MUTATION_RATE": 0.3,
    "SIGMA": 0.5,
    "NUM_OFFSPRINGS": 5,
    "SEED": 42,
    "LOG_FILE": None,
    "evaluate_fitness_fn": evaluate_fitness3
  }  
  basic_test(params, evolve_both, None, False)

def evolve_both_hiperparams_fatorial_test():
  fixed_params = {
    "STRUCTURE_NUM_GENERATIONS": 100,
    "MIN_GRID_SIZE": (5, 5),
    "MAX_GRID_SIZE": (5, 5),
    "STEPS": 500,
    "SCENARIO": "GapJumper-v0",
    "STRUCTURE_POP_SIZE": 5,
    "CROSSOVER_RATE": 0.95,
    "CROSSOVER_TYPE": uniform_crossover,
    "SURVIVORS_COUNT": 3,
    "PARENT_SELECTION_COUNT": 2,
    "VOXEL_TYPES": [0, 1, 2, 3, 4],
    "CONTROLLER_NUM_GENERATIONS": 30,
    "CONTROLLER_POP_SIZE": 30,
    "CONTROLLER_MUTATION_RATE": 0.3,
    "NUM_OFFSPRINGS": 5
  }  
  variable_params_grid = {
    "STRUCTURE_MUTATION_RATE": [0.05, 0.1],
    "SIGMA": [0.5, 0.7]
  }
  run_param_combinations(fixed_params, variable_params_grid, evolve_both, utils.test_types[0])

def evolve_both_scenario_test():
  fixed_params = {
    "STRUCTURE_NUM_GENERATIONS": 100,
    "MIN_GRID_SIZE": (5, 5),
    "MAX_GRID_SIZE": (5, 5),
    "STEPS": 500,
    "SCENARIO": "GapJumper-v0",
    "STRUCTURE_POP_SIZE": 5,
    "CROSSOVER_RATE": 0.95,
    "CROSSOVER_TYPE": uniform_crossover,
    "STRUCTURE_MUTATION_RATE": 0.3, # complete
    "SURVIVORS_COUNT": 3,
    "PARENT_SELECTION_COUNT": 2,
    "VOXEL_TYPES": [0, 1, 2, 3, 4],
    "CONTROLLER_NUM_GENERATIONS": 30,
    "CONTROLLER_POP_SIZE": 30,
    "CONTROLLER_MUTATION_RATE": 0.3,
    "SIGMA": 0.7, # complete
    "NUM_OFFSPRINGS": 5
  }  
  variable_params_grid = {
    "SCENARIO": ["GapJumper-v0", "CaveCrawler-v0"]
  }
  run_param_combinations(fixed_params, variable_params_grid, evolve_both, utils.test_types[1])

evolve_both_basic_test()