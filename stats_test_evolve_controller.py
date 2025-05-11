from es_controller import es_search
from random_controller import random_search
from stats_test import base_stat_test, check_pairwise_parametric_comparisons
import utils

evolve_type = utils.evolve_types["evolve_controller"]["label"]

def stats_es_search_hiperparams_fatorial_test():
  algorithm = es_search
  test_type = utils.test_types[0]
  base_stat_test(evolve_type, algorithm, test_type)

def stats_random_search_scenario_test():
  algorithm = random_search
  test_type = utils.test_types[1]
  base_stat_test(evolve_type, algorithm, test_type, True)

def stats_es_search_scenario_test():
  algorithm = es_search
  test_type = utils.test_types[1]
  base_stat_test(evolve_type, algorithm, test_type, True)

def stats_random_search_vs_es_search_controller_scenario_test():
  check_pairwise_parametric_comparisons(
    evolve_type="-",
    algorithm=stats_random_search_vs_es_search_controller_scenario_test,
    test_type="random_search_vs_es_search_test",
    only_two=True,
    combination_dirs=[
      "outputs/evolve_controller/random_search/controller_scenario_test/reference_run/combination1",
      "outputs/evolve_controller/es_search/controller_scenario_test/reference_run/combination1"
    ],
    reference_run_dir="outputs/evolve_controller"
  )

# stats_es_search_hiperparams_fatorial_test()
# stats_random_search_scenario_test()
stats_es_search_scenario_test()