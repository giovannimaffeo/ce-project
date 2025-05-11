import numpy as np
import random
import copy
import gymnasium as gym
from evogym.envs import *

from evogym import EvoWorld, EvoSim, EvoViewer, sample_robot, get_full_connectivity, is_connected
from utils import log
from fixed_controllers import *

def evaluate_fitness(robot_structure, scenario, steps, controller, view=False):    
    try:
        connectivity = get_full_connectivity(robot_structure)
  
        env = gym.make(scenario, max_episode_steps=steps, body=robot_structure, connections=connectivity)
        env.reset()
        sim = env.sim
        viewer = EvoViewer(sim)
        viewer.track_objects('robot')
        t_reward = 0
        action_size = sim.get_dim_action_space('robot')  # Get correct action size
        for t in range(steps):  
            # Update actuation before stepping
            actuation = controller(action_size, t)
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

def create_random_robot(min_grid_size, max_grid_size):
    """Generate a valid random robot structure."""
    
    grid_size = (random.randint(min_grid_size[0], max_grid_size[0]), random.randint(min_grid_size[1], max_grid_size[1]))
    random_robot, _ = sample_robot(grid_size)
    return random_robot

def random_search_structure(
    NUM_GENERATIONS,
    MIN_GRID_SIZE,
    MAX_GRID_SIZE,
    STEPS,
    SCENARIO,
    VOXEL_TYPES,
    CONTROLLER,
    SEED=None,
    LOG_FILE=None
):
    if SEED is not None:
        random.seed(SEED)
        np.random.seed(SEED)

    best_robot = None
    best_fitness = -float("inf")
    fitness_history = []

    for it in range(NUM_GENERATIONS):
        robot = create_random_robot(MIN_GRID_SIZE, MAX_GRID_SIZE)
        fitness_score = evaluate_fitness(robot, SCENARIO, STEPS, CONTROLLER)

        if fitness_score > best_fitness:
            best_fitness = fitness_score
            best_robot = robot

        fitness_history.append({
            "generation": it + 1,
            "best_fitness": best_fitness,
            "mean_fitness": best_fitness
        })

        log(f"Iteration {it + 1}: Fitness = {best_fitness}", LOG_FILE)

    return best_robot, best_fitness, fitness_history

# Example usage
"""best_robot, best_fitness = random_search()
print("Best robot structure found:")
print(best_robot)
print("Best fitness score:")
print(best_fitness)
i = 0
while i < 10:
    utils.simulate_best_robot(best_robot, scenario=SCENARIO, steps=STEPS)
    i += 1
utils.create_gif(best_robot, filename='random_search.gif', scenario=SCENARIO, steps=STEPS, controller=CONTROLLER)"""
