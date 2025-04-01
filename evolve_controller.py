import numpy as np
import random
import gymnasium as gym
from evogym.envs import *
from evogym import EvoViewer, get_full_connectivity
from neural_controller import *
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda import gpuarray
from pycuda.compiler import SourceModule
import torch

# Constants
NUM_GENERATIONS = 100
POPULATION_SIZE = 20
STEPS = 500
SCENARIO = 'DownStepper-v0'
SEED = 42
np.random.seed(SEED)
random.seed(SEED)

# Initialize robot structure
robot_structure = np.array([
    [1,3,1,0,0],
    [4,1,3,2,2],
    [3,4,4,4,4],
    [3,0,0,3,2],
    [0,0,0,0,2]
])

# CUDA kernel for mutation
mutation_kernel = """
__global__ void mutate_population(float* population, float* parent, 
                                float* noise, float* mask, 
                                float sigma, int param_size) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < param_size) {
        population[idx] = parent[idx] + sigma * noise[idx] * mask[idx];
    }
}
"""

# Initialize environment and controller
connectivity = get_full_connectivity(robot_structure)
env = gym.make(SCENARIO, max_episode_steps=STEPS, body=robot_structure, connections=connectivity)
input_size = env.observation_space.shape[0]
output_size = env.action_space.shape[0]
brain = NeuralController(input_size, output_size)

def get_weights(brain):
    """Extract weights from neural network as a flat numpy array"""
    weights = []
    for param in brain.parameters():
        weights.append(param.data.cpu().numpy().flatten())
    return np.concatenate(weights)

def set_weights(brain, weights):
    """Set weights from flat numpy array to neural network"""
    ptr = 0
    for param in brain.parameters():
        shape = param.data.shape
        size = np.prod(shape)
        param.data = torch.from_numpy(weights[ptr:ptr+size].reshape(shape)).float()
        ptr += size

def evaluate_fitness(weights, view=False):
    """Evaluate fitness of an individual"""
    set_weights(brain, weights)
    env = gym.make(SCENARIO, max_episode_steps=STEPS, body=robot_structure, connections=connectivity)
    viewer = EvoViewer(env.sim) if view else None
    if view:
        viewer.track_objects('robot')
    
    state = env.reset()[0]
    t_reward = 0
    
    for t in range(STEPS):
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        action = brain(state_tensor).detach().numpy().flatten()
        
        if view and viewer:
            viewer.render('screen')
            
        state, reward, terminated, truncated, info = env.step(action)
        t_reward += reward
        
        if terminated or truncated:
            break
    
    if view and viewer:
        viewer.close()
    env.close()
    return t_reward

def es_search_gpu(brain, population_size=200, generations=20, alpha=0.01, sigma=0.1):
    """Evolution Strategy search with GPU acceleration"""
    # Get initial parameters
    param_vector = get_weights(brain)
    num_params = len(param_vector)
    best_fitness = -np.inf
    best_weights = param_vector.copy()
    
    # Compile CUDA kernel
    mod = SourceModule(mutation_kernel)
    mutate_population = mod.get_function("mutate_population")
    
    # Allocate GPU memory for parent
    parent_gpu = gpuarray.to_gpu(param_vector.astype(np.float32))
    
    for generation in range(NUM_GENERATIONS):
        # Generate population on CPU
        population = np.zeros((population_size, num_params), dtype=np.float32)
        noise = np.random.randn(population_size, num_params).astype(np.float32)
        mask = (np.random.rand(population_size, num_params) < 0.07).astype(np.float32)
        
        # Transfer to GPU
        noise_gpu = gpuarray.to_gpu(noise)
        mask_gpu = gpuarray.to_gpu(mask)
        population_gpu = gpuarray.to_gpu(population)
        
        # Set block and grid sizes
        block_size = 256
        grid_size = (num_params + block_size - 1) // block_size
        
        # Mutate population on GPU
        for i in range(population_size):
            mutate_population(
                population_gpu[i:i+1,:].reshape(-1),  # Current individual
                parent_gpu,                          # Parent
                noise_gpu[i:i+1,:].reshape(-1),      # Noise for this individual
                mask_gpu[i:i+1,:].reshape(-1),       # Mask for this individual
                np.float32(sigma),                   # Sigma
                np.int32(num_params),                # Parameter size
                block=(block_size, 1, 1), 
                grid=(grid_size, 1)
            )
        
        # Evaluate fitness
        population_cpu = population_gpu.get()
        fitness_scores = np.zeros(population_size)
        for i in range(population_size):
            fitness_scores[i] = evaluate_fitness(population_cpu[i])
        
        best_idx = np.argmax(fitness_scores)
        current_best = fitness_scores[best_idx]
        
        if current_best > best_fitness:
            best_fitness = current_best
            best_weights = population_cpu[best_idx]
            parent_gpu = gpuarray.to_gpu(best_weights)
        
        print(f"Generation {generation + 1}/{NUM_GENERATIONS}, Best fitness: {best_fitness}")
    
    # Set the best weights found
    set_weights(brain, best_weights)
    return best_weights

# Run the GPU-accelerated ES search
best_weights = es_search_gpu(brain, POPULATION_SIZE)

# Visualization function
def visualize_policy(weights):
    set_weights(brain, weights)
    env = gym.make(SCENARIO, max_episode_steps=STEPS, body=robot_structure, connections=connectivity)
    viewer = EvoViewer(env.sim)
    viewer.track_objects('robot')
    state = env.reset()[0]
    
    for t in range(STEPS):
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        action = brain(state_tensor).detach().numpy().flatten()
        viewer.render('screen')
        state, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            break
    
    viewer.close()
    env.close()

# Visualize the best policy
for i in range(10):
    visualize_policy(best_weights)