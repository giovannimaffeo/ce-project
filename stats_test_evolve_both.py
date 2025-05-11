from evolve_both import evolve_both
from random_both import random_evolve_both
from stats_test import base_stat_test, check_pairwise_parametric_comparisons
import utils

evolve_type = utils.evolve_types["evolve_both"]["label"]

def stats_evolve_both_hiperparams_fatorial_test():
  algorithm = evolve_both
  test_type = utils.test_types[0]
  base_stat_test(evolve_type, algorithm, test_type, True)

def stats_evolve_both_scenario_test():
  algorithm = evolve_both
  test_type = utils.test_types[1]
  base_stat_test(evolve_type, algorithm, test_type, True)

def stats_random_both_scenario_test():
  algorithm = random_evolve_both
  test_type = utils.test_types[1]
  base_stat_test(evolve_type, algorithm, test_type, True)

def stats_random_both_vs_evolve_both_scenario_test():
  check_pairwise_parametric_comparisons(
    evolve_type="-",
    algorithm=stats_random_both_vs_evolve_both_scenario_test,
    test_type="random_search_vs_evolve_both_test",
    only_two=True,
    combination_dirs=[
      "outputs/evolve_both/random_evolve_both/controller_scenario_test/reference_run/combination2",
      "outputs/evolve_both/evolve_both/controller_scenario_test/reference_run/combination2"
    ],
    reference_run_dir="outputs/evolve_both"
  )

stats_random_both_vs_evolve_both_scenario_test()
# stats_evolve_both_hiperparams_fatorial_test()
# stats_evolve_both_scenario_test()
#stats_random_both_scenario_test()