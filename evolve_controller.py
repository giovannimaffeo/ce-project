import numpy as np
import random
import gymnasium as gym
from evogym.envs import *
from evogym import EvoViewer, get_full_connectivity, EvoSim, EvoWorld
from neural_controller import *
from multiprocessing import Pool

NUM_GENERATIONS = 100  # Number of generations to evolve
POPULATION_SIZE = 20
STEPS = 500
SCENARIO = 'DownStepper-v0'
SEED = 42
np.random.seed(SEED)
random.seed(SEED)


robot_structure = np.array([ 
[1,3,1,0,0],
[4,1,3,2,2],
[3,4,4,4,4],
[3,0,0,3,2],
[0,0,0,0,2]
])



connectivity = get_full_connectivity(robot_structure)
env = gym.make(SCENARIO, max_episode_steps=STEPS, body=robot_structure, connections=connectivity)
sim = env.sim
input_size = env.observation_space.shape[0]  # Observation size
output_size = env.action_space.shape[0]  # Action size

brain = NeuralController(input_size, output_size)

# ---- FITNESS FUNCTION ----
def evaluate_fitness(weights, view=False):
        set_weights(brain, weights)  # Load weights into the network
        env = gym.make(SCENARIO, max_episode_steps=STEPS, body=robot_structure, connections=connectivity)
        sim = env
        viewer = EvoViewer(sim)
        viewer.track_objects('robot')
        state = env.reset()[0]  # Get initial state
        t_reward = 0
        initial_position = sim.object_pos_at_time(sim.get_time(), 'robot')
        for t in range(STEPS):  
            # Update actuation before stepping
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)  # Convert to tensor
            action = brain(state_tensor).detach().numpy().flatten() # Get action
            if view:
                viewer.render('screen') 
            state, reward, terminated, truncated, info = env.step(action)
            
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
# # ---- FITNESS FUNCTION ----
# def evaluate_fitness(weights, view=False):
#         set_weights(brain, weights)  # Load weights into the network
#         env = gym.make(SCENARIO, max_episode_steps=STEPS, body=robot_structure, connections=connectivity)
#         sim = env
#         viewer = EvoViewer(sim)
#         viewer.track_objects('robot')
#         state = env.reset()[0]  # Get initial state
#         t_reward = 0
#         for t in range(STEPS):  
#             # Update actuation before stepping
#             state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)  # Convert to tensor
#             action = brain(state_tensor).detach().numpy().flatten() # Get action
#             if view:
#                 viewer.render('screen') 
#             state, reward, terminated, truncated, info = env.step(action)
#             t_reward += reward
#             if terminated or truncated:
#                 env.reset()
#                 break

#         viewer.close()
#         env.close()
#         return t_reward 


def evaluate_fitness_parallel(population):
    with Pool() as pool:  # Creates a pool of worker processes (defaults to number of CPU cores)
        fitnesses = np.array(pool.map(evaluate_fitness, population))
    return fitnesses

def es_search(brain, population_size=200, generations=20, alpha=0.01, sigma=0.1):
    best_fitness = -np.inf
    param_vector = get_weights(brain)
    best_weights = param_vector
    num_params = len(best_weights)
    population = []
    num_parents = 5
    mutation_rate = 0.07

    for generation in range(NUM_GENERATIONS):
        for i in range(population_size):
            for j in range(num_parents):
                param_vector = population[0] if generation != 0 else param_vector
                mutation_mask = np.random.rand(num_params) < mutation_rate
                noise = sigma * np.random.randn(num_params) * mutation_mask
                ind = param_vector + noise
                population.append(ind)
        fitnesses = evaluate_fitness_parallel(population)
        #fitnesses = np.array([evaluate_fitness(individual) for individual in population])
        sorted_indices = np.argsort(fitnesses)[::-1]
        sorted_population = [population[i] for i in sorted_indices]
        sorted_rewards = fitnesses[sorted_indices]
        population = sorted_population[:num_parents]

        if sorted_rewards[0] > best_fitness :
            best_fitness = sorted_rewards[0]
            best_weights = sorted_population[0] 
        print(f"Generation {generation + 1}/{NUM_GENERATIONS}, Best current fitness: {sorted_rewards[0]}, Best global fitness: {best_fitness}")

    # Set the best weights found
    set_weights(brain, best_weights)
    print(f"Best Fitness: {best_fitness}")
    return best_weights
    
def random_search(brain):
    # ---- RANDOM SEARCH ALGORITHM ----
    best_fitness = -np.inf
    best_weights = None

    for generation in range(NUM_GENERATIONS):
        # Generate random weights for the neural network
        random_weights = [np.random.randn(*param.shape) for param in brain.parameters()]
        
        # Evaluate the fitness of the current weights
        fitness = evaluate_fitness(random_weights)
        
        # Check if the current weights are the best so far
        if fitness > best_fitness:
            best_fitness = fitness
            best_weights = random_weights
        
        print(f"Generation {generation + 1}/{NUM_GENERATIONS}, Fitness: {fitness}")

    # Set the best weights found
    set_weights(brain, best_weights)
    print(f"Best Fitness: {best_fitness}")
    return best_weights

#best_weights = random_search(brain)
best_weights = es_search(brain, POPULATION_SIZE)

# ---- VISUALIZATION ----
def visualize_policy(weights):
    set_weights(brain, weights)  # Load weights into the network
    env = gym.make(SCENARIO, max_episode_steps=STEPS, body=robot_structure, connections=connectivity)
    sim = env.sim
    viewer = EvoViewer(sim)
    viewer.track_objects('robot')
    state = env.reset()[0]  # Get initial state
    for t in range(STEPS):  
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
i = 0
while i < 10:
    visualize_policy(best_weights)
    i += 1