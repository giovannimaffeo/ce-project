import numpy as np
import random
import copy
import gymnasium as gym
from evogym.envs import *

from evogym import EvoWorld, EvoSim, EvoViewer, sample_robot, get_full_connectivity, is_connected
import utils
from fixed_controllers import *


# ---- PARAMETERS ----
NUM_GENERATIONS = 5  # Number of generations to evolve
MIN_GRID_SIZE = (5, 5)  # Minimum size of the robot grid
MAX_GRID_SIZE = (5, 5)  # Maximum size of the robot grid
STEPS = 500
SCENARIO = 'Walker-v0' #'BridgeWalkerv0'
# ---- VOXEL TYPES ----
VOXEL_TYPES = [0, 1, 2, 3, 4]  # Empty, Rigid, Soft, Active (+/-)
POP_SIZE = 5
CROSSOVER_RATE = 0.1
MUTATION_RATE = 0.05

CONTROLLER = alternating_gait

def evaluate_fitness(robot_structure, view=False):    
  try:
    connectivity = get_full_connectivity(robot_structure)

    env = gym.make(SCENARIO, max_episode_steps=STEPS, body=robot_structure, connections=connectivity)
    env.reset()
    sim = env.sim
    viewer = EvoViewer(sim)
    viewer.track_objects('robot')
    t_reward = 0
    action_size = sim.get_dim_action_space('robot')  # Get correct action size
    for t in range(STEPS):  
      # Update actuation before stepping
      actuation = CONTROLLER(action_size, t)
      if view:
        viewer.render('screen') 
      ob, reward, terminated, truncated, info = env.step(actuation)
      t_reward += reward

      if terminated or truncated:
        env.reset()
        break

    viewer.close()
    env.close()
    return t_reward
  except (ValueError, IndexError) as e:
    return 0.0


def create_random_robot():
  """Generate a valid random robot structure."""
  grid_size = (random.randint(MIN_GRID_SIZE[0], MAX_GRID_SIZE[0]), random.randint(MIN_GRID_SIZE[1], MAX_GRID_SIZE[1]))
  random_robot, _ = sample_robot(grid_size)
  return random_robot

def parent_selection(population, t=3):
  # select a subset of "t" random individuals
  selected_parents = random.sample(population, t)
  selected_parents = sorted(selected_parents, key=evaluate_fitness, reverse=True)

  # return the two with the best fitness
  return selected_parents[0], selected_parents[1]

def crossover(p1, p2):
  #one point crossover
  """
  robot_structure = np.array([ 
    [1,3,1,0,0],
    [4,1,3,2,2],
    [3,4,4,4,4],
    [3,0,0,3,2],
    [0,0,0,0,2]
    ])
  """  
  offsprings = [
    np.array([*p1[0:2], *p2[2:]]), 
    np.array([*p2[0:2], *p1[2:]])
  ]
  
  offsprings = sorted(offsprings, key=evaluate_fitness, reverse=True)
  return offsprings[0]  


def mutate(child, mutation_rate):
  for i, chromosome in enumerate(child):
    for j, gene in enumerate(chromosome):
      if random.random() < mutation_rate:
        #to generate a random number that is different from the current gene
        possibilites = [x for x in range(0, 5) if x != gene]
        child[i][j] = random.choice(possibilites)
  return child  
   
def survivor_selection(population, new_population, t=2):
  new_population = sorted(new_population, key=evaluate_fitness, reverse=True)
  best_individuals = population[:t]
  new_population[-t:] = best_individuals
  
  return new_population

def ea_search():
  best_robot = None
  best_fitness = -float('inf')

  population = [create_random_robot() for _ in range(POP_SIZE)]
  for it in range(NUM_GENERATIONS):
    population = sorted(population, key=evaluate_fitness, reverse=True)
    
    best_current_fitness = evaluate_fitness(population[0])
    if best_current_fitness > best_fitness:
      best_fitness = best_current_fitness
      best_robot = population[0]
    # if evaluate_fitness(population[0]) == STEPS:
    #  break

    new_population = []
    while len(new_population) < POP_SIZE:
      p1, p2 = parent_selection(population)
      
      #if random.random() < CROSSOVER_RATE: perguntar se é preciso ter uma prob de fzr ou uma coisa ou outra
      child = crossover(p1, p2) # cuidado com as alterações, utilizar np.copy ou equivalente
      child = mutate(child, MUTATION_RATE)
      new_population.append(child)
    population = survivor_selection(population, new_population)

  return best_robot, best_fitness


def random_search():
  """Perform a random search to find the best robot structure."""
  best_robot = None
  best_fitness = -float('inf')
  
  for it in range(NUM_GENERATIONS):
    robot = create_random_robot() 
    fitness_score = evaluate_fitness(robot)
    
    if fitness_score > best_fitness:
      best_fitness = fitness_score
      best_robot = robot
    
    print(f"Iteration {it + 1}: Fitness = {fitness_score}")
  
  return best_robot, best_fitness

# Example usage
best_robot, best_fitness = ea_search()#random_search()
print("Best robot structure found:")
print(best_robot)
print("Best fitness score:")
print(best_fitness)

i = 0
while i < 10:
  utils.simulate_best_robot(best_robot, scenario=SCENARIO, steps=STEPS)
  i += 1

utils.create_gif(best_robot, filename='random_search.gif', scenario=SCENARIO, steps=STEPS, controller=CONTROLLER)
