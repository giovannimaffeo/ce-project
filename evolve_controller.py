import numpy as np
import random
import gymnasium as gym
from evogym.envs import *
from evogym import EvoViewer, get_full_connectivity, EvoSim, EvoWorld
from neural_controller import *
from multiprocessing import Pool
import copy


# ---- FITNESS FUNCTION ----
def evaluate_fitness_2(weights, brain, connectivity, view=False):
        weights = args["population_to_evaluate"]
        brain = args["brain"]
        connectivity = args["connectivity"]
        robot_structure = args["robot_structure"]
        weights = copy.deepcopy(weights[0])
        set_weights(brain, weights)  # Load weights into the network
        env = gym.make(SCENARIO, max_episode_steps=STEPS, body=robot_structure, connections=connectivity)
        sim = env
        viewer = EvoViewer(sim)
        viewer.track_objects('robot')
        state = env.reset()[0]  # Get initial state
        t_reward = 0
        original_reward = 0
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
def evaluate_fitness(args, view=False):
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

def evaluate_fitness_parallel(population, brain, connectivity, robot_structure, scenario, steps):
    population_to_evaluate = [ind for ind in population if ind[1] is None]
    # args = {
    #     "population_to_evaluate": population_to_evaluate,
    #     "brain": brain, 
    #     "connectivity": connectivity
    # }
    args_list = [{
        "population_to_evaluate": ind,
        "brain": brain,
        "connectivity": connectivity,
        "robot_structure": robot_structure,
        "scenario": scenario,
        "steps": steps
    } for ind in population_to_evaluate]
    with Pool() as pool:
        population = np.array(pool.map(evaluate_fitness, args_list))
    return population


def es_search(
        STEPS,
        SCENARIO,
        MUTATION_RATE,
        SEED,
        NUM_OFFSPRINGS = 3, 
        POP_SIZE = 50,#30, 
        NUM_GENERATIONS = 100, 
        sigma = 0.01):

    if SEED is not None:
        random.seed(SEED)
        np.random.seed(SEED)

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
    print(brain)
    best_fitness = -np.inf
    param_vector = get_weights(brain)
    best_weights = param_vector
    num_params = len(best_weights)
    fitness_history = []

    population = [([[np.random.randn(*param.shape) for param in brain.parameters()], None]) for i in range(POP_SIZE)]

    for generation in range(NUM_GENERATIONS):
        #selects u parents (the previous population)
        for i in range(POP_SIZE):
            #each parent generates lambda offsprings
            for j in range(NUM_OFFSPRINGS):
                mutation_mask = np.random.rand(num_params) < MUTATION_RATE
                noise = sigma * np.random.randn(num_params) * mutation_mask
                ind = population[i][0] + noise
                population.append([ind, None])
                
        population = evaluate_fitness_parallel(population, brain, connectivity, robot_structure, SCENARIO, STEPS)
        population = sorted(population, key=lambda x: x[1], reverse=True)
        
        if population[0][1] > best_fitness:
            best_fitness = population[0][1]
            best_weights = population[0][0]
        
        fitnesses = [fintess[1] for fintess in population]
        mean_fitness = np.mean(fitnesses)
        std_fitness = np.std(fitnesses)
        fitness_history.append({
            "generation": generation+1,
            "best_fitness": best_fitness,
            "mean_fitness": mean_fitness,
            "std": std_fitness
        })

        print(f"Generation {generation + 1}/{NUM_GENERATIONS}, Best current fitness: {population[0][1]}, Best global fitness: {best_fitness}, Avg fitness: {mean_fitness}")
    # Set the best weights found
    set_weights(brain, best_weights)
    print(f"Best Fitness: {best_fitness}")
    return best_weights
es_search(STEPS=500, SCENARIO="Walker-v0", MUTATION_RATE=0.05, SEED=42)
    
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
