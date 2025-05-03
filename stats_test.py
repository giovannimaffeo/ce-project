import random

from ea_structure import ea_search
from random_structure import random_search


def stats_test():
  seeds = [3223, 19676, 85960, 12577, 62400]
  results = {
    "RS": [],
    "ES": []
  }
  for seed in seeds:
    _, best_fitness = random_search(seed)
    results["RS"].append(best_fitness)
    _, best_fitness, _ = ea_search(seed)
    results["ES"].append(best_fitness)

  print(results)
  # test_normal_sw(data) saber se é paramétrico
  # se sim: Dep. t-test
  # se não: Wilcoxon

  
stats_test()
