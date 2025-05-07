from evolve_both import evolve_both
from stats_test import base_stat_test
import utils

evolve_type = utils.evolve_types["evolve_both"]["label"]

def stats_evolve_both_hiperparams_fatorial_test():
  algorithm = evolve_both
  test_type = utils.test_types[0]
  base_stat_test(evolve_type, algorithm, test_type)

def stats_evolve_both_scenario_test():
  algorithm = evolve_both
  test_type = utils.test_types[1]
  base_stat_test(evolve_type, algorithm, test_type, True)

stats_evolve_both_hiperparams_fatorial_test()
# stats_evolve_both_scenario_test()