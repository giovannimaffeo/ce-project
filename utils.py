import numpy as np
import json
import os
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import gymnasium as gym
from evogym import EvoViewer, get_full_connectivity
import imageio
from fixed_controllers import *

# ---- SIMULATE BEST ROBOT ----
def simulate_best_robot(robot_structure, scenario=None, steps=500, controller = alternating_gait):
    
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
    
    for t in range(200):  # Simulate for 200 timesteps
        # Update actuation before stepping
        actuation = controller(action_size,t)

        ob, reward, terminated, truncated, info = env.step(actuation)
        t_reward += reward
        if terminated or truncated:
            env.reset()
            break
        viewer.render('screen')
    viewer.close()
    env.close()

    return t_reward #(max_height - initial_height) #-  abs(np.mean(positions[0, :])) # Max height gained is jump performance


def create_gif(robot_structure, filename='best_robot.gif', duration=0.066, scenario=None, steps=500, controller=alternating_gait):
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
        for t in range(200):
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

def generate_results(fitness_history_df, best_robot, params):
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = f"outputs/evolve_structure/{timestamp}"
    os.makedirs(output_dir, exist_ok=True)

    # save best result (fitness + robot matrix) as csv
    best_result_path = os.path.join(output_dir, "best_result.csv")
    robot_str = json.dumps(best_robot.tolist()) 
    best_result_df = pd.DataFrame([{
        "best_fitness": fitness_history_df["best_fitness"].max(),
        "best_robot": robot_str
    }])
    best_result_df.to_csv(best_result_path, index=False)

    # save fitness history df as csv
    fitness_csv_path = os.path.join(output_dir, "fitness_history.csv")
    fitness_history_df.to_csv(fitness_csv_path, index=False)

    # generate and save plot for fitness history
    plt.figure()
    plt.plot(fitness_history_df["generation"], fitness_history_df["best_fitness"], "b-", label="Best Fitness")
    plt.plot(fitness_history_df["generation"], fitness_history_df["mean_fitness"], "r-", label="Mean Fitness")
    plt.xlabel("Generation")
    plt.ylabel("Fitness")
    plt.title(f"Generation: {fitness_history_df['generation'].iloc[-1]} | Best Fitness: {fitness_history_df['best_fitness'].iloc[-1]:.2f}")
    plt.legend()
    plt.grid(True)
    plot_path = os.path.join(output_dir, "fitness_plot.png")
    plt.savefig(plot_path)
    plt.close()

    # save global parameters of execution as csv
    params_df = pd.DataFrame([params])
    params_csv_path = os.path.join(output_dir, "parameters_info.csv")
    params_df.to_csv(params_csv_path, index=False)

    # generate and save gif of best robot
    gif_path = os.path.join(output_dir, "evolve_structure.gif")
    create_gif(best_robot, filename=gif_path, scenario=params["SCENARIO"], steps=params["STEPS"], controller=params["CONTROLLER"])
        


    