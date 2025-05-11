import numpy as np
import random
import gymnasium as gym
from evogym.envs import *
from evogym import EvoViewer, get_full_connectivity
from neural_controller import *
import torch
from utils import log

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

default_robot_structure = np.array([ 
    [1,3,1,0,0],
    [4,1,3,2,2],
    [3,4,4,4,4],
    [3,0,0,3,2],
    [0,0,0,0,2]
])
# ---- RANDOM SEARCH ALGORITHM ----
def random_search(
    NUM_GENERATIONS, 
    STEPS, 
    SCENARIO, 
    SEED, 
    LOG_FILE=None,
    robot_structure=default_robot_structure
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
    best_weights = None
    fitness_history = []

    for it in range(NUM_GENERATIONS):
        # Generate random weights for the neural network
        random_weights = [np.random.randn(*param.shape) for param in brain.parameters()]
        
        # Evaluate the fitness of the current weights
        fitness = evaluate_fitness(random_weights, brain, SCENARIO, STEPS, robot_structure, connectivity)
        
        # Check if the current weights are the best so far
        if fitness > best_fitness:
            best_fitness = fitness
            best_weights = random_weights
        
        fitness_history.append({
            "generation": it + 1,
            "best_fitness": best_fitness,
            "mean_fitness": best_fitness
        })
        
        log(f"Iteration {it + 1}/{NUM_GENERATIONS}, Fitness: {best_fitness}", LOG_FILE)

    return best_weights, best_fitness, fitness_history

# ---- VISUALIZATION ----
def visualize_policy(weights, brain, scenario, steps, robot_structure, connectivity):
    set_weights(brain, weights)  # Load weights into the network
    env = gym.make(scenario, max_episode_steps=steps, body=robot_structure, connections=connectivity)
    sim = env.sim
    viewer = EvoViewer(sim)
    viewer.track_objects('robot')
    state = env.reset()[0]  # Get initial state
    for t in range(steps):  
        # Update actuation before stepping
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)  # Convert to tensor
        action = brain(state_tensor).detach().numpy().flatten() # Get action
        viewer.render('screen') 
        state, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            env.reset()
            break

    viewer.close()
    env.close()
