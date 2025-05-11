from multiprocessing import Pool, current_process
from ea_structure import create_random_robot, mutate, parent_selection, survivor_selection, uniform_crossover
import numpy as np
import pandas as pd
import ast
import random
import gymnasium as gym
from evogym.envs import *
from evogym import EvoViewer, get_full_connectivity, EvoSim, EvoWorld
from fixed_controllers import *
from utils import log
from neural_controller import *
from es_controller import es_search, evaluate_fitness, evaluate_fitness3
import imageio

def create_gif(brain, scenario, steps, robot_structure, filename='best_robot.gif', duration=0.066, view=False):
    try:
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
        #if not isinstance(connectivity, np.ndarray):
        #    connectivity = np.zeros(robot_structure.shape, dtype=int)
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

def load_experiment_data_1(folder_path):
    best_result_path = os.path.join(folder_path, 'best_result.csv')
    best_df = pd.read_csv(best_result_path)
    best_robot = ast.literal_eval(best_df.loc[0, 'best_robot'])

    params_path = os.path.join(folder_path, 'parameters_info.csv')
    params_df = pd.read_csv(params_path)
    controller = params_df.loc[0, 'CONTROLLER']
    scenario = params_df.loc[0, 'SCENARIO']

    return best_robot, controller, scenario

def load_experiment_data_2(folder_path):
    best_result_path = os.path.join(folder_path, 'best_result.csv')
    best_df = pd.read_csv(best_result_path)
    
    best_robot = np.array([ 
        [1,3,1,0,0],
        [4,1,3,2,2],
        [3,4,4,4,4],
        [3,0,0,3,2],
        [0,0,0,0,2]
    ])

    best_weights = ast.literal_eval(best_df.loc[0, 'best_weights'])

    params_path = os.path.join(folder_path, 'parameters_info.csv')
    params_df = pd.read_csv(params_path)
    
    scenario = params_df.loc[0, 'SCENARIO']

    return best_robot, best_weights, scenario

def load_experiment_data_3(folder_path):
    best_result_path = os.path.join(folder_path, 'best_result.csv')
    best_df = pd.read_csv(best_result_path)
    
    # Parse robot structure and weights from string to Python objects
    best_robot = ast.literal_eval(best_df.loc[0, 'best_robot'])
    best_weights = ast.literal_eval(best_df.loc[0, 'best_weights'])

    params_path = os.path.join(folder_path, 'parameters_info.csv')
    params_df = pd.read_csv(params_path)
    
    scenario = params_df.loc[0, 'SCENARIO']

    return best_robot, best_weights, scenario

#3.1
def gif_3_1_scenario1():
    folder = "outputs/evolve_structure/ea_search/hiperparams_fatorial_test/reference_run/combination16/run3"
    robot, ctrl, scen = load_experiment_data_1(folder)
    controllers = {
        "alternating_gait": alternating_gait
    }

    ctrl = controllers[ctrl]
    robot = np.array(robot)
    create_gif_2(robot_structure=robot, scenario=scen, controller=ctrl, filename="best_robot_3_1_scenario1.gif")

def gif_3_1_scenario2():
    folder = "outputs/evolve_structure/ea_search/controller_scenario_test/reference_run/combination3/run4"
    
    robot, ctrl, scen = load_experiment_data_1(folder)
    controllers = {
        "hopping_motion": hopping_motion
    }

    ctrl = controllers[ctrl]
    robot = np.array(robot)
    create_gif_2(robot_structure=robot, scenario=scen, controller=ctrl, filename="best_robot_3_1_scenario2.gif")

def gif_3_3_scenario1():
    folder = "outputs/evolve_both/evolve_both/hiperparams_fatorial_test/reference_run/combination2/run2"
    robot, ctrl, scen = load_experiment_data_3(folder)

    robot = np.array(robot)
    ctrl = np.array(ctrl)
    
    connectivity = get_full_connectivity(robot)
    env = gym.make(scen, max_episode_steps=500, body=robot, connections=connectivity)
    input_size = env.observation_space.shape[0]
    output_size = env.action_space.shape[0]
    brain = NeuralController(input_size, output_size)
    set_weights(brain, ctrl)

    create_gif(brain, scen, 500, robot, "best_robot_3_3_scenario1.gif")


def gif_3_3_scenario2():
    folder = "outputs/evolve_both/evolve_both/hiperparams_fatorial_test/reference_run1/combination1/run4"
    robot, ctrl, scen = load_experiment_data_3(folder)

    robot = np.array(robot)
    ctrl = np.array(ctrl)
    
    connectivity = get_full_connectivity(robot)
    env = gym.make(scen, max_episode_steps=500, body=robot, connections=connectivity)
    input_size = env.observation_space.shape[0]
    output_size = env.action_space.shape[0]
    brain = NeuralController(input_size, output_size)
    set_weights(brain, ctrl)

    create_gif(brain, scen, 500, robot, "best_robot_3_3_scenario2.gif")


def gif_3_2_scenario1():
    folder = "outputs/evolve_controller/es_search/hiperparams_fatorial_test/reference_run/combination8/run5"
    robot, ctrl, scen = load_experiment_data_2(folder)

    robot = np.array(robot)
    ctrl = np.array(ctrl)
    
    connectivity = get_full_connectivity(robot)
    env = gym.make(scen, max_episode_steps=500, body=robot, connections=connectivity)
    input_size = env.observation_space.shape[0]
    output_size = env.action_space.shape[0]
    brain = NeuralController(input_size, output_size)
    set_weights(brain, ctrl)

    create_gif(brain, scen, 500, robot, "best_robot_3_2_scenario1.gif")

def gif_3_2_scenario2():
    folder = "outputs/evolve_controller/es_search/controller_scenario_test/reference_run/combination2/run5"
    robot, ctrl, scen = load_experiment_data_2(folder)

    robot = np.array(robot)
    ctrl = np.array(ctrl)
    
    connectivity = get_full_connectivity(robot)
    env = gym.make(scen, max_episode_steps=500, body=robot, connections=connectivity)
    input_size = env.observation_space.shape[0]
    output_size = env.action_space.shape[0]
    brain = NeuralController(input_size, output_size)
    set_weights(brain, ctrl)

    create_gif(brain, scen, 500, robot, "best_robot_3_2_scenario2.gif")

gif_3_1_scenario2()