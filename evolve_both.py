from multiprocessing import Pool, current_process
from ea_structure import create_random_robot, mutate, parent_selection, survivor_selection, uniform_crossover
import numpy as np
import random
import gymnasium as gym
from evogym.envs import *
from evogym import get_full_connectivity
from fixed_controllers import *
from utils import log
from neural_controller import *
from es_controller import es_search, evaluate_fitness


from evogym import EvoViewer
import imageio
import torch

def create_gif(weights, brain, scenario, steps, robot_structure, filename='best_robot.gif', duration=0.066, view=False):
  try:
    set_weights(brain, weights)  # Load weights into the network
    connectivity = get_full_connectivity(robot_structure)
    env = gym.make(scenario, max_episode_steps=steps, body=robot_structure, connections=connectivity)
    env.reset()
    sim = env.sim
    viewer = EvoViewer(sim)
    viewer.track_objects('robot')
    state = env.reset()[0]  # Get initial state
    t_reward = 0
    
    frames = []
    for t in range(steps):  
      # Update actuation before stepping
      state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)  # Convert to tensor
      action = brain(state_tensor).detach().numpy().flatten() # Get action
      if view:
        viewer.render('screen') 
      state, reward, terminated, truncated, info = env.step(action)
      t_reward += reward
      if terminated or truncated:
        env.reset()
        break
      frame = viewer.render('rgb_array')
      frames.append(frame)

    viewer.close()
    imageio.mimsave(filename, frames, duration=duration, optimize=True)
  except Exception as e:
    print("Exception while creating the gif: " + str(e))

def create_gif_2(robot_structure, filename='best_robot.gif', duration=0.066, scenario=None, steps=500, controller=alternating_gait):
  try:
    """Create a smooth GIF of the robot simulation at 30fps."""
    connectivity = get_full_connectivity(robot_structure)
    env = gym.make(scenario, max_episode_steps=steps, body=robot_structure, connections=connectivity)
    env.reset()
    sim = env.sim
    viewer = EvoViewer(sim)
    viewer.track_objects('robot')

    action_size = sim.get_dim_action_space('robot')  # Get correct action size
    t_reward = 0

    frames = []
    for t in range(steps):
      actuation = controller(action_size,t)
      ob, reward, terminated, truncated, info = env.step(actuation)
      t_reward += reward
      if terminated or truncated:
        env.reset()
        break
      frame = viewer.render('rgb_array')
      frames.append(frame)

    viewer.close()
    imageio.mimsave(filename, frames, duration=duration, optimize=True)
  except ValueError as e:
    print('Invalid')

# def create_gif(robot_structure, filename='best_robot.gif', duration=0.066, scenario=None, steps=500, controller=alternating_gait):
#     print(f'body: {robot_structure}')
#     """Create a GIF of the robot simulation, handling both controller types"""
#     try:
#         connectivity = get_full_connectivity(robot_structure)
#         env = gym.make(scenario, max_episode_steps=steps, body=robot_structure, connections=connectivity)
#         state = env.reset()[0]
#         sim = env.sim
#         viewer = EvoViewer(sim)
#         viewer.track_objects('robot')

#         if hasattr(controller, 'forward'):
#             print('===>>> Using neural network controller')
#         else:
#             print('===>>> Using fixed controller')

#         frames = []
#         for t in range(steps):
#             # Handle different controller types
#             if hasattr(controller, 'forward'):  # Neural network controller
#                 state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
#                 actuation = controller(state_tensor).detach().numpy().flatten()
#             else:  # Fixed controller
#                 action_size = sim.get_dim_action_space('robot')
#                 actuation = controller(action_size, t)

#             state, reward, terminated, truncated, _ = env.step(actuation)
#             if terminated or truncated:
#                 break
#             frame = viewer.render('rgb_array')
#             frames.append(frame)

#         viewer.close()
#         env.close()
#         imageio.mimsave(filename, frames, duration=duration, optimize=True)
#     except ValueError as e:
#         print(f'Error creating GIF: {e}')

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

def evaluate_fitness_parallel(population, scenario, steps):
  indexes_to_evaluate = [i for i, ind in enumerate(population) if ind.fitness is None]
  
  args_list = [
    (
      population[i].weights,
      population[i].brain,
      scenario,
      steps,
      population[i].structure,
      get_full_connectivity(population[i].structure)
    ) for i in indexes_to_evaluate
  ]
  if current_process().daemon:
    fitnesses = [evaluate_fitness(*args) for args in args_list]
  else:
    with Pool() as pool:
      fitnesses = pool.starmap(evaluate_fitness, args_list)

  for i, fit in zip(indexes_to_evaluate, fitnesses):
    population[i].fitness = fit

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
    log("starting evaluation structure population", LOG_FILE)
    population = evaluate_fitness_parallel(population, SCENARIO, STEPS)
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

    log("starting survivor_selection", LOG_FILE)
    population = survivor_selection(population, new_population, SURVIVORS_COUNT)
    log(f"structure generation {it + 1}/{STRUCTURE_NUM_GENERATIONS}, Best current fitness: {best_current_fitness}, Best global fitness: {best_fitness}, Avg fitness: {mean_fitness}", LOG_FILE)
    create_gif(best_weights, population[0].brain, SCENARIO, STEPS, population[0].structure)
  return best_individual, best_fitness, fitness_history

evolve_both(
  STRUCTURE_NUM_GENERATIONS=100,
  MIN_GRID_SIZE=(5, 5),
  MAX_GRID_SIZE=(5, 5),
  STEPS=500,
  SCENARIO="GapJumper-v0",
  STRUCTURE_POP_SIZE=5,
  CROSSOVER_RATE=0.9,
  CROSSOVER_TYPE=uniform_crossover,
  STRUCTURE_MUTATION_RATE=0.3,
  SURVIVORS_COUNT=3,
  PARENT_SELECTION_COUNT=2,
  VOXEL_TYPES=[0, 1, 2, 3, 4],
  CONTROLLER_NUM_GENERATIONS=10, #10,
  CONTROLLER_POP_SIZE=5,#30,
  CONTROLLER_MUTATION_RATE=0.5,
  SIGMA=0.7,
  NUM_OFFSPRINGS=1,
  SEED=42,
  LOG_FILE=None
)
