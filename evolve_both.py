import gc
import os
from ea_structure import create_random_robot, mutate, parent_selection, survivor_selection, uniform_crossover
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
import gymnasium as gym
from evogym.envs import *
import tracemalloc
from evogym import EvoWorld, EvoSim, EvoViewer, sample_robot, get_full_connectivity, is_connected
from evogym import EvoViewer, get_full_connectivity, EvoSim, EvoWorld
from fixed_controllers import *
from utils import log
from neural_controller import *
from multiprocessing import Pool
import copy
from es_controller import es_search, evaluate_fitness

class Individual():  
  def __init__(self, scenario, steps, min_grid_size, max_grid_size, structure=None, weights=None, fitness=None):
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
  LOG_FILE=None
): 
  if SEED is not None:
    random.seed(SEED)
    np.random.seed(SEED)

  best_individual = None
  best_fitness = -np.inf
  population = [Individual(SCENARIO, STEPS, MIN_GRID_SIZE, MAX_GRID_SIZE) for _ in range(STRUCTURE_POP_SIZE)]
  fitness_history = []

  for it in range(STRUCTURE_NUM_GENERATIONS):
    for individual in population:
      if individual.fitness is None:
        connectivity = get_full_connectivity(individual.structure)
        individual.fitness = evaluate_fitness(
          individual.weights, 
          individual.brain,
          SCENARIO, 
          STEPS, 
          individual.structure,
          connectivity
        )
    
    population = sorted(population, key=lambda x: x.fitness, reverse=True)

    best_current_fitness = population[0].fitness
    if best_current_fitness > best_fitness:
      best_fitness = best_current_fitness
      best_individual = population[0]
    
    mean_fitness = sum(ind.fitness for ind in population) / len(population)
    fitness_history.append({
      "generation": it+1,
      "best_fitness": best_fitness,
      "mean_fitness": mean_fitness
    })
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
        robot_structure=structure
      )

      # add new individual
      child = Individual(
        SCENARIO, 
        STEPS, 
        MIN_GRID_SIZE, 
        MAX_GRID_SIZE,
        structure,
        best_weights,
        best_fitness
      )
      set_weights(child.brain, best_weights)
      new_population.append(child)

      population = survivor_selection(population, new_population, SURVIVORS_COUNT)
      print(f"Structure generation {it + 1}/{STRUCTURE_NUM_GENERATIONS}, Best current fitness: {best_current_fitness}, Best global fitness: {best_fitness}, Avg fitness: {mean_fitness}")

  return best_individual, best_fitness, fitness_history

evolve_both(
  STRUCTURE_NUM_GENERATIONS=100,
  MIN_GRID_SIZE=(5, 5),
  MAX_GRID_SIZE=(5, 5),
  STEPS=500,
  SCENARIO="Walker-v0",
  STRUCTURE_POP_SIZE=5,
  CROSSOVER_RATE=0.9,
  CROSSOVER_TYPE=uniform_crossover,
  STRUCTURE_MUTATION_RATE=0.3,
  SURVIVORS_COUNT=3,
  PARENT_SELECTION_COUNT=2,
  VOXEL_TYPES=[0, 1, 2, 3, 4],
  CONTROLLER_NUM_GENERATIONS=2, #10,
  CONTROLLER_POP_SIZE=10,
  CONTROLLER_MUTATION_RATE=0.5,
  SIGMA=0.5,
  NUM_OFFSPRINGS=3,
  SEED=42,
  LOG_FILE=None
)
