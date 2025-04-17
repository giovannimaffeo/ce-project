from datetime import datetime
import os
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
import gymnasium as gym
from evogym.envs import *

from evogym import EvoWorld, EvoSim, EvoViewer, sample_robot, get_full_connectivity, is_connected
import utils
from fixed_controllers import *


# ---- PARAMETERS ----
NUM_GENERATIONS = 100  # Number of generations to evolve
MIN_GRID_SIZE = (5, 5)  # Minimum size of the robot grid
MAX_GRID_SIZE = (5, 5)  # Maximum size of the robot grid
STEPS = 500
SCENARIO = "Walker-v0" #"BridgeWalkerv0"
POP_SIZE = 15
CROSSOVER_RATE = 0.95
MUTATION_RATE = 0.03
SURVIVORS_COUNT = 1
PARENT_SELECTION_COUNT = 4
# ---- VOXEL TYPES ----
VOXEL_TYPES = [0, 1, 2, 3, 4]  # Empty, Rigid, Soft, Active (+/-)

CONTROLLER = alternating_gait

def evaluate_fitness(robot_structure, view=False):    
  try:
    connectivity = get_full_connectivity(robot_structure)

    env = gym.make(SCENARIO, max_episode_steps=STEPS, body=robot_structure, connections=connectivity)
    env.reset()
    sim = env.sim
    viewer = EvoViewer(sim)
    viewer.track_objects("robot")
    t_reward = 0
    action_size = sim.get_dim_action_space("robot")  # Get correct action size
    initial_position = sim.object_pos_at_time(sim.get_time(), 'robot')
    
    for t in range(STEPS):  
      # Update actuation before stepping
      actuation = CONTROLLER(action_size, t)
      if view:
        viewer.render("screen") 
      ob, reward, terminated, truncated, info = env.step(actuation)
      #t_reward += reward
      current_position = sim.object_pos_at_time(sim.get_time(), 'robot')
      current_velocity = np.max(sim.object_vel_at_time(sim.get_time(), 'robot'))
      #print(current_velocity)
      t_reward += reward
      t_reward += 0.1 * current_velocity
      #check backward movement
      if np.max(np.subtract(current_position, initial_position)) < 0:
          t_reward = -5
      
      initial_position = current_position
      if terminated or truncated:
        env.reset()
        break

    viewer.close()
    env.close()
    return t_reward
  except (ValueError, IndexError) as e:
    return 0.0

# def evaluate_fitness(robot_structure, view=False):    
#   try:
#     connectivity = get_full_connectivity(robot_structure)

#     env = gym.make(SCENARIO, max_episode_steps=STEPS, body=robot_structure, connections=connectivity)
#     env.reset()
#     sim = env.sim
#     viewer = EvoViewer(sim)
#     viewer.track_objects("robot")
#     t_reward = 0
#     action_size = sim.get_dim_action_space("robot")  # Get correct action size
#     for t in range(STEPS):  
#       # Update actuation before stepping
#       actuation = CONTROLLER(action_size, t)
#       if view:
#         viewer.render("screen") 
#       ob, reward, terminated, truncated, info = env.step(actuation)
#       t_reward += reward

#       if terminated or truncated:
#         env.reset()
#         break

#     viewer.close()
#     env.close()
#     return t_reward
#   except (ValueError, IndexError) as e:
#     return 0.0

def create_random_robot():
  """Generate a valid random robot structure."""
  grid_size = (random.randint(MIN_GRID_SIZE[0], MAX_GRID_SIZE[0]), random.randint(MIN_GRID_SIZE[1], MAX_GRID_SIZE[1]))
  random_robot, _ = sample_robot(grid_size)
  return random_robot

def is_connected(robot):
  rows, cols = robot.shape
  # create visited matrix
  visited = np.zeros_like(robot, dtype=bool)

  # get first node different from zero
  s = None
  for i in range(rows):
    for j in range(cols):
      if robot[i][j] > 0:
        s = (i, j)
        break
    if s: break
  
  # if no index is different from zero, return false
  if s is None:
    return False

  # initialize a queue
  q = [s]
  # mark node "s" as visited
  visited[s] = True
  while len(q) != 0:
    # explore first node from the queue
    v = q.pop(0)
    
    # get (i,j) from node v
    x, y = v

    # explore neighbors from node "v"
    for movement in [[-1, 0], [1, 0], [0, -1], [0, 1]]:
      # get indexes from neighbor
      w = [x + movement[0], y + movement[1]]
      # verify if index is valid 
      if 0 <= w[0] < rows and 0 <= w[1] < cols:
        # if neighbor "w" is a node and not visited 
        if robot[w[0]][w[1]] > 0 and not visited[w[0]][w[1]]:
          # mark neighbor "w" as visited and add to queue
          visited[w[0]][w[1]] = True
          q.append(w)
    
  return np.all((robot == 0) | visited)

def parent_selection(population, t):
  # select a subset of "t" random individuals
  selected_parents = random.sample(population, t)
  # sort the set by individual fitness
  selected_parents = sorted(selected_parents, key=lambda x: x[1], reverse=True)

  # return the two with the best fitness
  return selected_parents[0], selected_parents[1]

def two_point_crossover(p1, p2, crossover_rate):
  # skip crossover with probability (1 - crossover_rate) and return parent 1
  if random.random() > crossover_rate:
    return p1
  
  # flat the parents
  p1 = p1[0].flatten()
  p2 = p2[0].flatten()
  individual_length = len(p1)

  max_retries = 100
  for _ in range(max_retries):
    #to avoid having cuts in the end of the sequence
    crossover_point_1 = random.randint(0, int(individual_length/2))
    #we are doing this to start a little bit ahead from the first cut
    crossover_point_2 = random.randint(crossover_point_1, individual_length)
    
    offspring_part1 = p1[:crossover_point_1]
    offspring_part2 = p2[crossover_point_1:crossover_point_2]
    offspring_part3 = p1[crossover_point_2:]
    # generate offspring by concatenating parts
    offspring_flat = np.concatenate([offspring_part1, offspring_part2, offspring_part3])
    offspring = offspring_flat.reshape((5, 5))

    # return the offspring if it is connected
    if is_connected(offspring):
      # return offspring with no fitness calculated
      return [offspring, None]
    return p1
  
def crossover(p1, p2, crossover_rate):
  # skip crossover with probability (1 - crossover_rate) and return parent 1
  if random.random() > crossover_rate:
    return p1
  
  # flat the parents
  p1 = p1[0].flatten()
  p2 = p2[0].flatten()
  individual_length = len(p1)

  # keep trying until a connected offspring is generated
  while True:
    # choose a crossover point randomly
    crossover_point = random.randint(0, individual_length)
    # get first part from parent1 and second part from parent2
    offspring_part1 = p1[:crossover_point]
    offspring_part2 = p2[crossover_point:]

    # generate offspring by concatenating parts
    offspring_flat = np.concatenate([offspring_part1, offspring_part2])
    offspring = offspring_flat.reshape((5, 5))

    # return the offspring if it is connected
    if is_connected(offspring):
      # return offspring with no fitness calculated
      return [offspring, None]

def mutate(child, mutation_rate, mutation_possibilities):
  # set variable child as the robot
  child = child[0]

  # iterate through each gene in robot (or element on matrix 5x5)
  for i, chromosome in enumerate(child):
    for j, gene in enumerate(chromosome):
      # skip mutation of gene with probability (1 - mutation_rate)
      if random.random() <= mutation_rate:
        # generate possible mutations for gene
        possibilities = [x for x in mutation_possibilities if x != gene]
        # shuffle possibilities to choose randomly
        random.shuffle(possibilities)
        
        # apply all possible mutations until a connected robot is generated
        # if is not possible, skip mutation 
        new_child = child.copy()
        for value in possibilities:
          new_child[i][j] = value
          if is_connected(new_child):
            child[i][j] = value 
            break
        
  # return mutated child with no fitness calculated
  return [child, None] 
   
def survivor_selection(population, new_population, t):
  # tira t elementos melhores da antiga, e randoms da nova
  survivors = population[:t]
  new_population = [*new_population, *survivors]

  return new_population

def ea_search():
  best_robot = None
  best_fitness = -float("inf")
  # generate initial population randomly 
  # population is a list of individuals as [robot, fitness]
  population = [[create_random_robot(), None] for _ in range(POP_SIZE)]
  # store fitness history
  fitness_history = []

  for it in range(NUM_GENERATIONS):
    # get individuals fitness for sorting population
    for i, individual in enumerate(population):
      # calculate the fitness only for the new individuals (fitness is None)
      if (individual[1] is None):
        fitness = evaluate_fitness(individual[0])
        population[i][1] = fitness
    # sort population by individual fitness
    population = sorted(population, key=lambda x: x[1], reverse=True)

    # update best_fitness and best_robot
    best_current_fitness = population[0][1]
    if best_current_fitness > best_fitness:
      best_fitness = best_current_fitness
      best_robot = population[0][0]
    print(f"Iteration {it + 1}: Fitness = {best_fitness}")
    # store best and mean fitness in the history
    mean_fitness = sum(ind[1] for ind in population) / len(population)
    fitness_history.append({
      "generation": it,
      "best_fitness": best_current_fitness,
      "mean_fitness": mean_fitness
    })

    new_population = []
    # generate new population with length of (POP_SIZE - SURVIVORS_COUNT)
    for i in range(POP_SIZE - SURVIVORS_COUNT):
      p1, p2 = parent_selection(population, PARENT_SELECTION_COUNT)
      child = two_point_crossover(p1,p2,CROSSOVER_RATE)#crossover(p1, p2, CROSSOVER_RATE) 
      #child = mutate(child, MUTATION_RATE, VOXEL_TYPES)
      new_population.append(child)

    # set population as new population plus best individuals of previous population
    population = survivor_selection(population, new_population, SURVIVORS_COUNT)

  # return best individual regarding all iterations
  return best_robot, best_fitness, fitness_history

best_robot, best_fitness, fitness_history = ea_search()
print("Best robot structure found:")
print(best_robot)
print("Best fitness score:")
print(best_fitness)

# generate results
fitness_history_df = pd.DataFrame(fitness_history)
params = {
  "NUM_GENERATIONS": NUM_GENERATIONS,
  "MIN_GRID_SIZE": MIN_GRID_SIZE,
  "MAX_GRID_SIZE": MAX_GRID_SIZE,
  "STEPS": STEPS,
  "SCENARIO": SCENARIO,
  "POP_SIZE": POP_SIZE,
  "CROSSOVER_RATE": CROSSOVER_RATE,
  "MUTATION_RATE": MUTATION_RATE,
  "SURVIVORS_COUNT": SURVIVORS_COUNT,
  "PARENT_SELECTION_COUNT": PARENT_SELECTION_COUNT,
  "VOXEL_TYPES": str(VOXEL_TYPES),
  "CONTROLLER": CONTROLLER
}
#utils.generate_results(fitness_history_df, best_robot, params)
