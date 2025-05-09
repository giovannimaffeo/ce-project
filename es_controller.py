import csv
import os
import numpy as np
import random
import gymnasium as gym
from evogym.envs import *
from evogym import EvoViewer, get_full_connectivity, EvoSim, EvoWorld
from neural_controller import *
from multiprocessing import Pool, current_process
import copy

from utils import log

def evaluate_fitness4(weights, brain, scenario, steps, robot_structure, connectivity, return_reward=False, view=False):
  set_weights(brain, weights)  # Load weights into the network
  env = gym.make(scenario, max_episode_steps=steps, body=robot_structure, connections=connectivity)
  sim = env
  viewer = EvoViewer(sim)
  viewer.track_objects('robot')

  state = env.reset()[0]
  t_reward = 0
  actuation_history = []
  initial_x = env.get_robot_com()[0]  # Posição x inicial do centro de massa

  for t in range(steps):
    state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
    action = brain(state_tensor).detach().numpy().flatten()
    actuation_history.append(action)

    if view:
      viewer.render('screen')

    state, reward, terminated, truncated, info = env.step(action)
    t_reward += reward

    if terminated or truncated:
      break

  final_x = env.get_robot_com()[0]  # Posição x final do centro de massa
  delta_x = final_x - initial_x

  # Detecta se o robô caiu (ex: Y muito baixo ou reward negativo)
  robot_fell = info.get("robot_fell", False)
  if not robot_fell:
    y_pos = env.get_robot_com()[1]
    robot_fell = y_pos < 1.0  # Ajuste esse valor de limiar conforme necessário

  # Penalidade por queda
  penalty = 5.0 if robot_fell else 0.0

  # Penalidade por movimentos bruscos
  total_actuation_change = sum(
    np.abs(actuation_history[i+1] - actuation_history[i]).sum()
    for i in range(len(actuation_history) - 1)
  )
  smoothness_bonus = -0.01 * total_actuation_change

  fitness = delta_x - penalty + smoothness_bonus

  viewer.close()
  env.close()
  return [fitness, reward] if return_reward else fitness

def evaluate_fitness3(weights, brain, scenario, steps, robot_structure, connectivity, return_reward=False, view=False):
  set_weights(brain, weights)
  env = gym.make(scenario, max_episode_steps=steps, body=robot_structure, connections=connectivity)
  sim = env.sim
  viewer = EvoViewer(sim)
  viewer.track_objects('robot')

  state = env.reset()[0]
  t_reward = 0

  initial_pos = np.array(sim.object_pos_at_time(sim.get_time(), "robot"))
  initial_x = np.max(initial_pos[:, 0])  # maior x
  initial_y = np.max(initial_pos[:, 1])  # maior y (altura)
  max_height = initial_y

  for t in range(steps):
    state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
    action = brain(state_tensor).detach().numpy().flatten()
    if view:
      viewer.render('screen')
    state, reward, terminated, truncated, info = env.step(action)
    t_reward += reward

    current_pos = np.array(sim.object_pos_at_time(sim.get_time(), "robot"))
    current_y = np.max(current_pos[:, 1])
    max_height = max(max_height, current_y)

    if terminated or truncated:
      break

  final_pos = np.array(sim.object_pos_at_time(sim.get_time(), "robot"))
  final_vel = np.array(sim.object_vel_at_time(sim.get_time(), "robot"))

  final_x = np.max(final_pos[:, 0])
  final_vx = np.mean(final_vel[:, 0])

  # Métricas de interesse
  horizontal_progress = final_x - initial_x
  jump_height = np.tanh(max_height - initial_y)   # normalização
  horizontal_velocity = final_vx

  # Bonificação por atravessar gap
  gap_threshold = 5.0  # pode ajustar conforme tamanho médio do gap
  crossed_gap = horizontal_progress > gap_threshold
  success_bonus = 10.0 if crossed_gap else 0.0

  # Penalidade se caiu abruptamente
  final_y = np.max(final_pos[:, 1])
  fall_penalty = -5.0 if final_y < initial_y - 1.0 else 0.0  # queda brusca

  fitness = (
    1.0 * horizontal_velocity +           # foco em se mover, mesmo que não atravesse
    0.7 * jump_height                       # salto eficiente
    # success_bonus +                     # atravessou o gap
    # fall_penalty                        # caiu feio? penaliza
  )

  # --- LOG EM CSV ---
  log_path = "fitness_reward_log.csv"
  file_exists = os.path.isfile(log_path)
  with open(log_path, mode='a', newline='') as file:
    writer = csv.writer(file)
    if not file_exists:
      writer.writerow(["fitness", "reward"])
    writer.writerow([fitness, t_reward])

  viewer.close()
  env.close()

  return [fitness, reward] if return_reward else fitness

# ---- FITNESS FUNCTION ----
def evaluate_fitness_2(args, view=False):
  weights = args["population_to_evaluate"]
  brain = args["brain"]
  connectivity = args["connectivity"]
  robot_structure = args["robot_structure"]
  scenario = args["scenario"]
  steps = args["steps"]
  weights = copy.deepcopy(weights[0])
  set_weights(brain, weights)  # Load weights into the network
  env = gym.make(scenario, max_episode_steps=steps, body=robot_structure, connections=connectivity)
  sim = env
  viewer = EvoViewer(sim)
  viewer.track_objects('robot')
  state = env.reset()[0]  # Get initial state
  t_reward = 0
  original_reward = 0
  initial_position = sim.object_pos_at_time(sim.get_time(), 'robot')
  for t in range(steps):  
    # Update actuation before stepping
    state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)  # Convert to tensor
    action = brain(state_tensor).detach().numpy().flatten() # Get action
    if view:
      viewer.render('screen') 
    state, reward, terminated, truncated, info = env.step(action)
    
    current_position = sim.object_pos_at_time(sim.get_time(), 'robot')
    current_velocity = np.max(sim.object_vel_at_time(sim.get_time(), 'robot'))
    #print(current_velocity)
    original_reward += reward
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
  #t_reward = our custom fitness
  return [weights, t_reward, original_reward]

# ---- FITNESS FUNCTION ----
def evaluate_fitness(weights, brain, scenario, steps, robot_structure, connectivity, view=False):
  set_weights(brain, weights)  # Load weights into the network
  env = gym.make(scenario, max_episode_steps=steps, body=robot_structure, connections=connectivity)
  sim = env
  viewer = EvoViewer(sim)
  viewer.track_objects('robot')
  state = env.reset()[0]  # Get initial state
  t_reward = 0
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

  viewer.close()
  env.close()
  return t_reward 

def evaluate_fitness_parallel(all_candidates, brain, connectivity, robot_structure, scenario, steps, evaluate_fitnes_fn):
  indexes_to_evaluate = [i for i, ind in enumerate(all_candidates) if ind[1] is None]
  
  args_list = [
    (
      all_candidates[i][0],
      brain,
      scenario,
      steps,
      robot_structure,
      connectivity
    ) for i in indexes_to_evaluate
  ]
  if current_process().daemon:
    fitnesses = [evaluate_fitnes_fn(*args) for args in args_list]
  else:
    with Pool() as pool:
      fitnesses = pool.starmap(evaluate_fitnes_fn, args_list)

  for i, fit in zip(indexes_to_evaluate, fitnesses):
    all_candidates[i][1] = fit

  return all_candidates

default_robot_structure = np.array([ 
  [1,3,1,0,0],
  [4,1,3,2,2],
  [3,4,4,4,4],
  [3,0,0,3,2],
  [0,0,0,0,2]
])
def es_search(
  NUM_GENERATIONS,
  STEPS,
  SCENARIO,
  POP_SIZE,
  MUTATION_RATE,
  SIGMA,
  NUM_OFFSPRINGS,
  SEED,
  LOG_FILE=None,
  robot_structure=default_robot_structure,
  evaluate_fitness_fn=evaluate_fitness
):
  if SEED is not None:
    random.seed(SEED)
    np.random.seed(SEED)

  connectivity = get_full_connectivity(robot_structure)
  env = gym.make(SCENARIO, max_episode_steps=STEPS, body=robot_structure, connections=connectivity)
  sim = env.sim
  input_size = env.observation_space.shape[0]  # Observation size
  output_size = env.action_space.shape[0]  # Action size
  
  brain = NeuralController(input_size, output_size)

  best_fitness = -np.inf
  fitness_history = []
  population = [([[np.random.randn(*param.shape) for param in brain.parameters()], None]) for i in range(POP_SIZE)]

  for it in range(NUM_GENERATIONS):
    log("starting gen offsprings", LOG_FILE)
    offsprings = []
    # selects mu parents (the previous population)
    for parent in population:
      # each parent generates lambda offsprings
      for _ in range(NUM_OFFSPRINGS):
        offspring = []
        for param_vector in parent[0]:
          shape = param_vector.shape
          mutation_mask = np.random.rand(*shape) < MUTATION_RATE
          noise = SIGMA * np.random.randn(*shape) * mutation_mask
          offspring.append(param_vector + noise)
        offsprings.append([offspring, None])

    all_candidates = population + offsprings
    log("starting evaluation individuals", LOG_FILE)
    all_candidates = evaluate_fitness_parallel(all_candidates, brain, connectivity, robot_structure, SCENARIO, STEPS, evaluate_fitness_fn)
    all_candidates = sorted(all_candidates, key=lambda x: x[1], reverse=True)
    # keep best individuals as population
    population = all_candidates[:POP_SIZE]

    if population[0][1] > best_fitness:
      best_fitness = population[0][1]
      best_weights = population[0][0]
    
    fitnesses = [individual[1] for individual in population]
    mean_fitness = np.mean(fitnesses)
    fitness_history.append({
      "generation": it+1,
      "best_fitness": best_fitness,
      "mean_fitness": mean_fitness
    })
    
    log(f"Iteration {it + 1}/{NUM_GENERATIONS}, Best current fitness: {population[0][1]}, Best global fitness: {best_fitness}, Avg fitness: {mean_fitness}", LOG_FILE)
  
  return best_weights, best_fitness, fitness_history