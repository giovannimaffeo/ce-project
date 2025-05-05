import gc
import os
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

class Individual():
    robot_structure = None
    neural_controller = None
    fitness = None
    scenario = None
    steps = None
    input_size = None
    output_size = None

    def __init__(self, scenario, steps, robot_structure=None):
        self.robot_structure = robot_structure
        self.fitness = None
        self.scenario = scenario
        self.steps = steps
        connectivity = get_full_connectivity(robot_structure)
        env = gym.make(scenario, max_episode_steps=self.steps, body=self.robot_structure, connections=connectivity)
        sim = env.sim
        self.input_size = env.observation_space.shape[0]  # Observation size
        self.output_size = env.action_space.shape[0]  # Action size
        self.neural_controller = NeuralController(self.input_size, self.output_size)

def evaluate_fitness_controller(args, view=False):
    try:
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
        return [weights, t_reward] 
    except Exception as e:
        print("Exception: " + str(e))

def evaluate_fitness_structure(robot_structure, SCENARIO, STEPS, CONTROLLER, view=False):    
    try:
        connectivity = get_full_connectivity(robot_structure)
        env = gym.make(SCENARIO, max_episode_steps=STEPS, body=robot_structure, connections=connectivity)
        env.reset()
        sim = env.sim
        viewer = EvoViewer(sim)
        viewer.track_objects("robot")
        t_reward = 0
        action_size = sim.get_dim_action_space("robot")  # Get correct action size
        for t in range(STEPS):  
            # Update actuation before stepping
            actuation = CONTROLLER(action_size, t)
            if view:
                viewer.render("screen") 
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


def evalaute_fitness(child: Individual, view=False):
    try:
        connectivity = get_full_connectivity(child.robot_structure)
        env = gym.make(child.scenario, max_episode_steps=child.steps, body=child.robot_structure, connections=connectivity)
        env.reset()
        sim = env.sim
        viewer = EvoViewer(sim)
        viewer.track_objects("robot")
        t_reward = 0
        #action_size = sim.get_dim_action_space("robot")  # Get correct action size
        state = env.reset()[0]  # Get initial state
        for t in range(child.steps):  
            # Update actuation before stepping
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)  # Convert to tensor
            action = child.neural_controller(state_tensor).detach().numpy().flatten() # Get action
            if view:
                viewer.render("screen") 
            state, reward, terminated, truncated, info = env.step(action)
            t_reward += reward
            if terminated or truncated:
                env.reset()
                break
        viewer.close()
        env.close()
        return t_reward
    except (ValueError, IndexError) as e:
        return [0.0, None]

def create_random_robot(MIN_GRID_SIZE, MAX_GRID_SIZE):
    """Generate a valid random robot structure."""
    grid_size = (random.randint(MIN_GRID_SIZE[0], MAX_GRID_SIZE[0]), random.randint(MIN_GRID_SIZE[1], MAX_GRID_SIZE[1]))
    random_robot, _ = sample_robot(grid_size)
    return random_robot

def parent_selection(population, PARENT_SELECTION_COUNT):
    pass

def crossover(p1: Individual, p2: Individual, CROSSOVER_RATE:float):
    pass

def mutate(child: Individual, MUTATION_RATE: float, VOXEL_TYPES: list):
    pass

def evolve_controller(child: Individual, MUTATION_RATE: float, SIGMA: float, NUM_OFFSPRINGS: int):
    #começar por fazer um parent só gera um indivíduo novo se nao nunca mais saimos daqui
    #for i in range(NUM_OFFSPRINGS):
    previous_controller = copy.deepcopy(child.neural_controller)
    
    connectivity = get_full_connectivity(child.robot_structure)
    env = gym.make(child.scenario, max_episode_steps=child.steps, body=child.robot_structure, connections=connectivity)
    sim = env.sim
    input_size = env.observation_space.shape[0]  # Observation size
    output_size = env.action_space.shape[0]  # Action size
    new_neural_controller = NeuralController(input_size, output_size)
    
    #the robot is bigger now, increment the vector
    if input_size > child.input_size:
        for i, param_vector in enumerate(new_neural_controller):
            #copy the parameters from the previous model to the new
            for j in range(len(previous_controller[i][0])):
                for k in range(len(previous_controller[i][0][j])):
                    new_neural_controller[i][0][j][k] = previous_controller[i][0][j][k]

            for j in range(len(previous_controller[i][1])):
                new_neural_controller[i][1][j] = previous_controller[i][1][j]
                
    #the robot is smaller, cut the vector
    elif input_size < child.input_size:
        for i, param_vector in enumerate(new_neural_controller):
            #copy the parameters from the previous model to the new
            for j in range(len(new_neural_controller[i][0])):
                for k in range(len(new_neural_controller[i][0][j])):
                    new_neural_controller[i][0][j][k] = previous_controller[i][0][j][k]

            for j in range(len(new_neural_controller[i][1])):
                new_neural_controller[i][1][j] = previous_controller[i][1][j]

    for j, param_vector in enumerate(new_neural_controller):
        shape = param_vector.shape
        mutation_mask = (np.random.rand(*shape) < MUTATION_RATE).astype(float)
        noise = SIGMA * np.random.randn(*shape) * mutation_mask
        new_neural_controller[j] = param_vector + noise
    child.neural_controller = new_neural_controller

    return child 

def survivor_selection(population, new_population, SURVIVORS_COUNT):
    pass

def evolve_both(
    NUM_GENERATIONS,
    MIN_GRID_SIZE,
    MAX_GRID_SIZE,
    STEPS,
    SCENARIO,
    POP_SIZE,
    CROSSOVER_RATE,
    CROSSOVER_TYPE,
    MUTATION_RATE,
    SURVIVORS_COUNT,
    PARENT_SELECTION_COUNT,
    VOXEL_TYPES,
    NUM_OFFSPRINGS,
    SIGMA,
    SEED,
    LOG_FILE=None
    ):

    best_fitness = -np.inf
    best_robot = None
    best_controller = None
    fitness_history = []
    population = [Individual(SCENARIO, STEPS, robot_structure=create_random_robot(MIN_GRID_SIZE, MAX_GRID_SIZE)) for _ in range(POP_SIZE)]

    for it in range(NUM_GENERATIONS):
        for i, individual in enumerate(population):
            if(individual.fitness == None):
                individual.fitness = evalaute_fitness(individual)
        
        population = sorted()
        best_current_fitness = population[0].fitness
        if best_current_fitness > best_fitness:
            best_fitness = best_current_fitness
            best_robot = population[0].robot_structure
            best_controller = population[0].neural_controller
        
        mean_fitness = sum(ind[1] for ind in population) / len(population)
        fitness_history.append({
            "generation": it+1,
            "best_fitness": best_current_fitness,
            "mean_fitness": mean_fitness
        })
        new_population = []
        for i in range(POP_SIZE - SURVIVORS_COUNT):
            #evolve structure
            p1, p2 = parent_selection(population, PARENT_SELECTION_COUNT)
            child = crossover(p1, p2, CROSSOVER_RATE)
            child = mutate(child, MUTATION_RATE, VOXEL_TYPES)

            #evolve controller
            child = evolve_controller(child, MUTATION_RATE, SIGMA, NUM_OFFSPRINGS)
            new_population.append(child)

        population = survivor_selection(population, new_population, SURVIVORS_COUNT)