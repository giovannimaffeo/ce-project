import os
import numpy as np
import pandas as pd
from itertools import combinations

from stat_alunos import test_normal_sw, kruskal_wallis, mann_whitney

def log_stats_msg(log_file, msg, reset=False):
  print(msg)
  mode = "w" if reset else "a"
  with open(log_file, mode) as f:
    f.write(msg + "\n")

def check_combinations_normality(evolve_type, algorithm, test_type):
  reference_run_dir = f"outputs/{evolve_type}/{algorithm.__name__}/{test_type}/reference_run"
  log_file = os.path.join(reference_run_dir, "statistical_test_result.txt")
  log_stats_msg(log_file, f"1. Shapiro-Wilk normality test results for algorithm '{algorithm.__name__}' and test '{test_type}':\n", True)
  
  combination_dirs = sorted([os.path.join(reference_run_dir, d) for d in os.listdir(reference_run_dir) if d.startswith("combination")])
  all_parametric = True
  for i, combination_dir in enumerate(combination_dirs, start=1):
    combination_fitness_values = []

    for run_index in range(1, 6):
      run_path = os.path.join(combination_dir, f"run{run_index}", "best_result.csv")
      best_result_df = pd.read_csv(run_path)
      combination_fitness_values.append(best_result_df["best_fitness"].iloc[-1])

    _, p = test_normal_sw(combination_fitness_values)
    log_stats_msg(log_file, f"Combination {i}: p = {p:.4f} → {'Normal' if p > 0.05 else 'Not normal'}")

    if p <= 0.05:
      all_parametric = False

  log_stats_msg(log_file, "\nResult: All combinations are parametric." if all_parametric else "\nResult: At least one combination is not parametric.")
  return all_parametric

def check_non_parametric_combinations_difference(evolve_type, algorithm, test_type):
  reference_run_dir = f"outputs/{evolve_type}/{algorithm.__name__}/{test_type}/reference_run"
  log_file = os.path.join(reference_run_dir, "statistical_test_result.txt")
  log_stats_msg(log_file, f"\n2. Due to the normality test indicating non-parametric data, performing Kruskal-Wallis test for differences between parameter combinations of algorithm '{algorithm.__name__}' and test '{test_type}':")

  combination_dirs = sorted([os.path.join(reference_run_dir, d) for d in os.listdir(reference_run_dir) if d.startswith("combination")])
  all_combination_fitnesses = []
  for combination_dir in combination_dirs:
    combination_fitness_values = []

    for run_index in range(1, 6):
      run_path = os.path.join(combination_dir, f"run{run_index}", "best_result.csv")
      best_result_df = pd.read_csv(run_path)
      combination_fitness_values.append(best_result_df["best_fitness"].iloc[-1])

    all_combination_fitnesses.append(combination_fitness_values)

  _, p = kruskal_wallis(all_combination_fitnesses)

  log_stats_msg(log_file, f"Result: p = {p:.4f}")
  log_stats_msg(log_file, "Conclusion: No significant difference between combinations." if p > 0.05 else "Conclusion: At least one combination differs significantly from the others.")
  return p <= 0.05

def check_pairwise_non_parametric_comparisons(evolve_type, algorithm, test_type):
  reference_run_dir = f"outputs/{evolve_type}/{algorithm.__name__}/{test_type}/reference_run"
  log_file = os.path.join(reference_run_dir, "statistical_test_result.txt")

  log_stats_msg(log_file, f"\n3. Pairwise Mann-Whitney U tests for combinations of algorithm '{algorithm.__name__}' and test '{test_type}':")

  combination_dirs = sorted([os.path.join(reference_run_dir, d) for d in os.listdir(reference_run_dir) if d.startswith("combination")])
  all_combination_fitnesses = []

  for combination_dir in combination_dirs:
    combination_fitness_values = []

    for run_index in range(1, 6):
      run_path = os.path.join(combination_dir, f"run{run_index}", "best_result.csv")
      best_result_df = pd.read_csv(run_path)
      combination_fitness_values.append(best_result_df["best_fitness"].iloc[-1])

    all_combination_fitnesses.append(combination_fitness_values)

  n = len(all_combination_fitnesses)
  wins = [0] * n

  for (i, data1), (j, data2) in combinations(enumerate(all_combination_fitnesses), 2):
    _, p = mann_whitney(data1, data2)
    mean1 = np.mean(data1)
    mean2 = np.mean(data2)

    if p <= 0.05:
      if mean1 > mean2:
        wins[i] += 1
        result = f"Combination {i+1} beats Combination {j+1} (p = {p:.4f})"
      else:
        wins[j] += 1
        result = f"Combination {j+1} beats Combination {i+1} (p = {p:.4f})"
    else:
      result = f"Combination {i+1} vs Combination {j+1}: Not significant (p = {p:.4f})"

    log_stats_msg(log_file, result)

  # Generate ranking
  ranked = sorted(enumerate(wins), key=lambda x: x[1], reverse=True)
  log_stats_msg(log_file, "\nRanking based on number of pairwise wins:")
  for rank, (idx, win_count) in enumerate(ranked, start=1):
    log_stats_msg(log_file, f"{rank}. Combination {idx+1} — {win_count} wins")

def base_stat_test(evolve_type, algorithm, test_type):
  is_parametric = check_combinations_normality(evolve_type, algorithm, test_type)
  if not is_parametric:
    is_different = check_non_parametric_combinations_difference(evolve_type, algorithm, test_type) 
    if is_different:
      check_pairwise_non_parametric_comparisons(evolve_type, algorithm, test_type)