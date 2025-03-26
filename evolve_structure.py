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

def is_valid_structure(robot):
  for i in range(len(robot)):
    for j in range(len(robot[i])):
      if robot[i][j] != 0:
        return (
          # up
          robot[i - 1][j] != 0 and 
          # down
          robot[i + 1][j] != 0 and
          # right
          robot[i][j + 1] != 0 and
          # left
          robot[i][j - 1] != 0
        )

def parent_selection(population, t=3):
  # select a subset of "t" random individuals
  selected_parents = random.sample(population, t)
  #n sera necessario dar evaluate again, pq pode dar valores diferentes
  selected_parents = sorted(selected_parents, key=lambda x: x[1], reverse=True)

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
  #poderemos fzr isto mas seria melhor fazer uma coisa mais soft, dar flat ao array, fzr crossover e voltar a montar
  # offsprings = [
  #   np.array([*p1[0:2], *p2[2:]]), 
  #   np.array([*p2[0:2], *p1[2:]])
  # ]

  #flat the parents
  p1 = p1[0].flatten()
  p2 = p2[0].flatten()
  
  # get first half of p1 and second half of p2 to generate offspring
  half_index = int(len(p1) / 2)
  p1_genes = p1[:half_index]
  p2_genes = p2[half_index:]
  offspring_flat = np.concatenate([p1_genes, p2_genes])
  offspring = offspring_flat.reshape((5, 5))  
  
  # return offspring tuple: (robot, fitness)
  return [offspring, None]

def generate_possible_locations(child):
  robot = child
  possible_locations = np.zeros_like(robot)
  #directions to search the neighborhood
  directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
  
  rows, cols = robot.shape
  
  for i in range(rows):
    for j in range(cols):
      if robot[i, j] != 0:
        possible_locations[i, j] = 1
        continue
            
      for di, dj in directions:
        ni, nj = i + di, j + dj
        if 0 <= ni < rows and 0 <= nj < cols:
          if robot[ni, nj] != 0:
            possible_locations[i, j] = 1
            break

  return possible_locations

def mutate(previousChild, mutation_rate):
  child = previousChild[0]
  for i, chromosome in enumerate(child):
    for j, gene in enumerate(chromosome):
      if random.random() < mutation_rate:
        #to generate a random number that is different from the current gene
        #podiamos ver aqui se gerar algo diferente de 1 ver na vizinhança se ta tudo a 0
        #dps tentar again mas cuidado pra n ficar ciclo infinito.
        #se n conseguir, retorna o parent
        #vamos gerar a matriz de possibilidades, temos que gerar sempre uma nova apos cada alteracao
        possible_positions = generate_possible_locations(child)
        if possible_positions[i][j] == 1: 
          possibilites = [x for x in range(0, 5) if x != gene]
          child[i][j] = random.choice(possibilites)
        else:
          continue
        
  # return offspring tuple: (robot, fitness)
  return [child, None] 
   
def survivor_selection(population, new_population, t=2):
  #tbm tirar aqui o evaluate, e sempre com base na fitness anterior (da population anterior!). so ha uma fase de fitness
  #tira t elementos melhores da antiga, e randoms da nova
  # new_population = sorted(new_population, key=evaluate_fitness, reverse=True)
  # best_individuals = population[:t]
  # new_population[-t:] = best_individuals
  

  #n sera necessario dar evaluate again, pq pode dar valores diferentes
  sorted_previous_population = sorted(population, key=lambda x: x[1], reverse=True)
  best_individuals = sorted_previous_population[:t]
  sorted_previous_population[-t:] = best_individuals

  return new_population

def ea_search():
  best_robot = None
  best_fitness = -float('inf')
  #list with tuples <robot, fitness>
  population = [[create_random_robot(), None] for _ in range(POP_SIZE)]

  for it in range(NUM_GENERATIONS):
    # get individuals fitness
    for i, individual in enumerate(population):
      fitness = evaluate_fitness(individual[0])
      population[i][1] = fitness
    # sort individuals by fitness
    population = sorted(population, key=lambda x: x[1], reverse=True)

    # update best_fitness and best_robot
    best_current_fitness = population[0][1]
    if best_current_fitness > best_fitness:
      best_fitness = best_current_fitness
      best_robot = population[0][0]

    new_population = []
    while len(new_population) < POP_SIZE:
      p1, p2 = parent_selection(population)
      
      #podemos ter uma mutation rate se vai haver ou noa e dps outra por cada gene, mas n e necessario
      #if random.random() < CROSSOVER_RATE: perguntar se é preciso ter uma prob de fzr ou uma coisa ou outra
      # if random.random() < CROSSOVER_RATE:
      child = crossover(p1, p2) 
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
