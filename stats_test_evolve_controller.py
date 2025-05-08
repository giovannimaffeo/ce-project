from es_controller import es_search
from random_controller import random_search
from stats_test import base_stat_test
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

stats_es_search_hiperparams_fatorial_test()
# stats_random_search_scenario_test()
# stats_es_search_scenario_test()