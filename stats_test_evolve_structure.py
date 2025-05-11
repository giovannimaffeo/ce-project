from ea_structure import ea_search
from random_structure import random_search
from stats_test import base_stat_test, check_pairwise_non_parametric_comparisons
import utils

evolve_type = utils.evolve_types["evolve_structure"]["label"]

def stats_ea_search_hiperparams_fatorial_test():
  algorithm = ea_search
  test_type = utils.test_types[0]
  base_stat_test(evolve_type, algorithm, test_type)

def stats_random_search_controller_scenario_test():
  algorithm = random_search
  test_type = utils.test_types[1]
  base_stat_test(evolve_type, algorithm, test_type)

def stats_ea_search_controller_scenario_test():
  algorithm = ea_search
  test_type = utils.test_types[1]
  base_stat_test(evolve_type, algorithm, test_type)

def stats_random_search_vs_ea_search_controller_scenario_test():
  check_pairwise_non_parametric_comparisons(
    evolve_type="-",
    algorithm=stats_random_search_vs_ea_search_controller_scenario_test,
    test_type="random_search_vs_ea_search_test",
    only_two=True,
    combination_dirs=[
      "outputs/evolve_structure/random_search/controller_scenario_test/reference_run/combination6",
      "outputs/evolve_structure/ea_search/controller_scenario_test/reference_run/combination6"
    ],
    reference_run_dir="outputs/evolve_structure"
  )

# stats_ea_search_hiperparams_fatorial_test()
# stats_random_search_controller_scenario_test()
stats_ea_search_controller_scenario_test()