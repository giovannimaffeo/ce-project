import numpy as np
import random
import copy
import gymnasium as gym
from evogym.envs import *

from evogym import EvoWorld, EvoSim, EvoViewer, sample_robot, get_full_connectivity, is_connected
import utils
from fixed_controllers import *


# ---- PARAMETERS ----
NUM_GENERATIONS = 50  # Number of generations to evolve
MIN_GRID_SIZE = (5, 5)  # Minimum size of the robot grid
MAX_GRID_SIZE = (5, 5)  # Maximum size of the robot grid
STEPS = 500
SCENARIO = 'Walker-v0' #'BridgeWalkerv0'
# ---- VOXEL TYPES ----
VOXEL_TYPES = [0, 1, 2, 3, 4]  # Empty, Rigid, Soft, Active (+/-)
POP_SIZE = 20
CROSSOVER_RATE = 0.1
MUTATION_RATE = 0.05
SURVIVORS_COUNT = 2

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
  initial_p1 = p1
  p1 = p1[0].flatten()
  p2 = p2[0].flatten()
  
  """
  robot_structure = np.array([ 
  [1,3,1,0,0],
  [4,1,3,2,2],
  [3,4,4,4,4],
  [3,0,0,3,2],
  [0,0,0,0,2]
  ])

  [1,1,1,1,1],
  [1,1,1,1,1],
  [1,1,1,1,1],
  [1,1,1,1,1],
  [1,0,0,0,1]
  ])
  """

  # get first half of p1 and second half of p2 to generate offspring 1
  half_index = int(len(p1) / 2)
  p1_genes = p1[:half_index]
  p2_genes = p2[half_index:]
  offspring_flat1 = np.concatenate([p1_genes, p2_genes])
  offspring1 = offspring_flat1.reshape((5, 5))  

  # get first half of p2 and second half of p1 to generate offspring 2
  p2_genes = p2[:half_index]
  p1_genes = p1[half_index:]
  offspring_flat2 = np.concatenate([p2_genes, p1_genes])
  offspring2 = offspring_flat2.reshape((5, 5)) 

  if (is_connected(offspring1)):
    return [offspring1, None]
  elif (is_connected(offspring2)):
    return [offspring2, None]
  # if none is a valid structered then we return the first original parent
  else:
    return initial_p1
  
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

def mutate(child, mutation_rate):
  child = child[0]
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
          possibilites = [x for x in range(1, 5) if x != gene]
          child[i][j] = random.choice(possibilites)
        else:
          continue
        
  # return offspring tuple: (robot, fitness)
  return [child, None] 
   
def survivor_selection(population, new_population, t=SURVIVORS_COUNT):
  #tbm tirar aqui o evaluate, e sempre com base na fitness anterior (da population anterior!). so ha uma fase de fitness
  #tira t elementos melhores da antiga, e randoms da nova
  survivors = population[:t]
  new_population = [*new_population, *survivors]

  return new_population

def ea_search():
  best_robot = None
  best_fitness = -float('inf')
  #list with tuples <robot, fitness>
  population = [[create_random_robot(), None] for _ in range(POP_SIZE)]

  for it in range(NUM_GENERATIONS):
    # get individuals fitness
    for i, individual in enumerate(population):
      #calculate the fitness only for the new individuals (not survivors)
      if (individual[1] is None):
        fitness = evaluate_fitness(individual[0])
        population[i][1] = fitness
    # sort individuals by fitness
    population = sorted(population, key=lambda x: x[1], reverse=True)

    # update best_fitness and best_robot
    best_current_fitness = population[0][1]
    if best_current_fitness > best_fitness:
      best_fitness = best_current_fitness
      best_robot = population[0][0]
      print("Generation: ", it)
      print("Structure of the best robot: ")
      print(best_robot)
      print("Best fitness: " + str(best_fitness))
      utils.simulate_best_robot(best_robot, scenario=SCENARIO, steps=STEPS)

      utils.create_gif(best_robot, filename='evolve_structure_cena.gif', scenario=SCENARIO, steps=STEPS, controller=CONTROLLER)


    new_population = []
    while len(new_population) < (POP_SIZE - SURVIVORS_COUNT):
      p1, p2 = parent_selection(population)
      
      #podemos ter uma mutation rate se vai haver ou noa e dps outra por cada gene, mas n e necessario
      #if random.random() < CROSSOVER_RATE: perguntar se é preciso ter uma prob de fzr ou uma coisa ou outra
      # if random.random() < CROSSOVER_RATE:
      child = crossover(p1, p2) 
      if (not is_connected(child[0])):
        print(child)
        print("não conectado CROSSOVER")
      child = mutate(child, MUTATION_RATE)
      if (not is_connected(child[0])):
        print(child)
        print("não conectado MUTATION")
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

utils.create_gif(best_robot, filename='evolve_structure.gif', scenario=SCENARIO, steps=STEPS, controller=CONTROLLER)
