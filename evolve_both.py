from multiprocessing import Pool, current_process
from ea_structure import create_random_robot, mutate, parent_selection, survivor_selection, uniform_crossover
import numpy as np
import random
import gymnasium as gym
from evogym.envs import *
from evogym import EvoViewer, get_full_connectivity, EvoSim, EvoWorld
from fixed_controllers import *
from utils import log
from neural_controller import *
from es_controller import es_search, evaluate_fitness, evaluate_fitness3

class Individual():  
  def __init__(self, scenario, steps, min_grid_size, max_grid_size, structure=None, weights=None, fitness=None, reward=None):
    self.reward = reward
    self.fitness = fitness
    
    # structure
    self.structure = structure if structure is not None else create_random_robot(min_grid_size, max_grid_size)
    
    # controller
    connectivity = get_full_connectivity(self.structure)
    env = gym.make(scenario, max_episode_steps=steps, body=self.structure, connections=connectivity)
    input_size = env.observation_space.shape[0]
    output_size = env.action_space.shape[0]
    self.brain = NeuralController(input_size, output_size)
    self.weights = weights if weights is not None else [np.random.randn(*param.shape) for param in self.brain.parameters()]

def evaluate_fitness_parallel(population, scenario, steps, evaluate_fitness_fn):
  indexes_to_evaluate = [i for i, ind in enumerate(population) if ind.reward is None]
  
  args_list = [
    (
      population[i].weights,
      population[i].brain,
      scenario,
      steps,
      population[i].structure,
      get_full_connectivity(population[i].structure),
      True
    ) for i in indexes_to_evaluate
  ]
  if current_process().daemon:
    evaluations = [evaluate_fitness_fn(*args) for args in args_list]
  else:
    with Pool() as pool:
      evaluations = pool.starmap(evaluate_fitness_fn, args_list)

  for i, evaluation in zip(indexes_to_evaluate, evaluations):
    population[i].fitness = evaluation[0]
    population[i].reward = evaluation[1]

  return population

def evolve_both(
  STRUCTURE_NUM_GENERATIONS,
  MIN_GRID_SIZE,
  MAX_GRID_SIZE,
  STEPS,
  SCENARIO,
  STRUCTURE_POP_SIZE,
  CROSSOVER_RATE,
  CROSSOVER_TYPE,
  STRUCTURE_MUTATION_RATE,
  SURVIVORS_COUNT,
  PARENT_SELECTION_COUNT,
  VOXEL_TYPES,
  CONTROLLER_NUM_GENERATIONS,
  CONTROLLER_POP_SIZE,
  CONTROLLER_MUTATION_RATE,
  SIGMA,
  NUM_OFFSPRINGS,
  SEED,
  LOG_FILE=None,
  evaluate_fitness_fn=evaluate_fitness
): 
  if SEED is not None:
    random.seed(SEED)
    np.random.seed(SEED)

  best_individual = None
  best_reward = -np.inf
  population = [Individual(SCENARIO, STEPS, MIN_GRID_SIZE, MAX_GRID_SIZE) for _ in range(STRUCTURE_POP_SIZE)]
  fitness_history = []

  for it in range(STRUCTURE_NUM_GENERATIONS):
    log("starting evaluation structure population", LOG_FILE)
    population = evaluate_fitness_parallel(population, SCENARIO, STEPS, evaluate_fitness_fn)
    population = sorted(population, key=lambda x: x.fitness, reverse=True)

    best_current_reward = population[0].reward
    if best_current_reward > best_reward:
      best_reward = best_current_reward
      best_individual = population[0]
    
    mean_reward = sum(ind.reward for ind in population) / len(population)
    fitness_history.append({
      "generation": it+1,
      "best_fitness": best_reward,
      "mean_fitness": mean_reward
    })

    log("starting gen of new population", LOG_FILE)
    new_population = []
    for _ in range(STRUCTURE_POP_SIZE - SURVIVORS_COUNT):
      # evolve structure
      evolve_structure_population = [
        [individual.structure, individual.fitness] for individual in population
      ]
      p1, p2 = parent_selection(evolve_structure_population, PARENT_SELECTION_COUNT)
      [structure, fitness] = CROSSOVER_TYPE(p1, p2, CROSSOVER_RATE)
      [structure, fitness] = mutate([structure, fitness], STRUCTURE_MUTATION_RATE, VOXEL_TYPES)

      # evolve controller
      best_weights, best_fitness, _ = es_search(
        CONTROLLER_NUM_GENERATIONS,
        STEPS,
        SCENARIO,
        CONTROLLER_POP_SIZE,
        CONTROLLER_MUTATION_RATE,
        SIGMA,
        NUM_OFFSPRINGS,
        SEED,
        LOG_FILE,
        robot_structure=structure,
        evaluate_fitness_fn=evaluate_fitness_fn
      )

      # add new individual
      child = Individual(
        SCENARIO, 
        STEPS, 
        MIN_GRID_SIZE, 
        MAX_GRID_SIZE,
        structure,
        best_weights,
        best_fitness,
        None
      )
      set_weights(child.brain, best_weights)
      new_population.append(child)

    log("starting survivor_selection", LOG_FILE)
    population = survivor_selection(population, new_population, SURVIVORS_COUNT)
    log(f"structure generation {it + 1}/{STRUCTURE_NUM_GENERATIONS}, Best current fitness: {best_current_reward}, Best global fitness: {best_reward}, Avg fitness: {mean_reward}", LOG_FILE)

  return best_individual, fitness_history