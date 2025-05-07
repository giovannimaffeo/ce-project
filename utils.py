from datetime import datetime
import gc
import numpy as np
import json
import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import gymnasium as gym
from evogym import EvoViewer, get_full_connectivity
import imageio
from fixed_controllers import *

import tracemalloc

test_types = ["hiperparams_fatorial_test", "controller_scenario_test"]

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
    
    for t in range(steps):  # Simulate for 200 timesteps
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

evolve_types = {
    "evolve_structure": {
        "entity": "robot",
        "label": "evolve_structure"
    },
    "evolve_controller": {
        "entity": "weights",
        "label": "evolve_controller"
    }
}

def generate_results(fitness_history_df, best_entity, params, output_dir, should_create_gif=True, evolve_type="evolve_structure"):
    os.makedirs(output_dir, exist_ok=True)

    best_entity_name = f"best_{evolve_types[evolve_type]['entity']}"
    best_result_path = os.path.join(output_dir, "best_result.csv")
    if evolve_type == "evolve_structure":
        best_entity_str = json.dumps(best_entity.tolist())
    else:
        best_entity_str = json.dumps([w.tolist() for w in best_entity])
    best_result_df = pd.DataFrame([{
        "best_fitness": fitness_history_df["best_fitness"].max(),
        best_entity_name: best_entity_str
    }])
    best_result_df.to_csv(best_result_path, index=False)

    fitness_csv_path = os.path.join(output_dir, "fitness_history.csv")
    fitness_history_df.to_csv(fitness_csv_path, index=False)

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

    params_df = pd.DataFrame([{
        k: (v.__name__ if callable(v) else v)
        for k, v in params.items()
    }])
    params_csv_path = os.path.join(output_dir, "parameters_info.csv")
    params_df.to_csv(params_csv_path, index=False)

    if should_create_gif:
        gif_path = os.path.join(output_dir, f"{evolve_type}.gif")
        create_gif(best_entity, filename=gif_path, scenario=params["SCENARIO"], steps=params["STEPS"], controller=params["CONTROLLER"])

def generate_combination_results(combination_variable_params, best_fitnesses, fitness_historic_paths, combination_output_dir):
    combination_variable_parameters_info_df = pd.DataFrame([list(combination_variable_params.values())], columns=list(combination_variable_params.keys()))
    combination_variable_parameters_info_path = os.path.join(combination_output_dir, "combination_variable_parameters_info.csv")
    combination_variable_parameters_info_df.to_csv(combination_variable_parameters_info_path, index=False)

    combination_best_fitness = max(best_fitnesses)
    combination_avg_best_fitness = np.mean(best_fitnesses)
    combination_std_best_fitness = np.std(best_fitnesses)    

    combination_results_df = pd.DataFrame([{
        "combination_avg_best_fitness": combination_avg_best_fitness,
        "combination_std_best_fitness": combination_std_best_fitness,
        "combination_best_fitness": combination_best_fitness
    }])
    combination_results_file_path = os.path.join(combination_output_dir, "combination_results.csv")
    combination_results_df.to_csv(combination_results_file_path, index=False)

    fitness_historics = [pd.read_csv(path) for path in fitness_historic_paths]
    n_generations = len(fitness_historics[0])
    combination_fitness_history_df = pd.DataFrame({
        "generation": list(range(1, n_generations + 1)),
        "combination_avg_best_fitness": [
            np.mean([fitness_history.iloc[i]["best_fitness"] for fitness_history in fitness_historics])
            for i in range(n_generations)
        ],
        "combination_avg_mean_fitness": [
            np.mean([fitness_history.iloc[i]["mean_fitness"] for fitness_history in fitness_historics])
            for i in range(n_generations)
        ],
        "combination_std_best_fitness": [
            np.std([fitness_history.iloc[i]["best_fitness"] for fitness_history in fitness_historics])
            for i in range(n_generations)
        ],
        "combination_std_mean_fitness": [
            np.std([fitness_history.iloc[i]["mean_fitness"] for fitness_history in fitness_historics])
            for i in range(n_generations)
        ],
        "combination_best_fitness": [
            max([fitness_history.iloc[i]["best_fitness"] for fitness_history in fitness_historics])
            for i in range(n_generations)
        ]
    })
    combination_fitness_history_file_path = os.path.join(combination_output_dir, "combination_fitness_history.csv")
    combination_fitness_history_df.to_csv(combination_fitness_history_file_path, index=False)   

    plt.figure()
    for i, fitness_history in enumerate(fitness_historics):
        plt.plot(
            fitness_history["generation"],
            fitness_history["best_fitness"],
            label=f"Run {i+1}"
        )
    plt.plot(
        combination_fitness_history_df["generation"],
        combination_fitness_history_df["combination_avg_best_fitness"],
        "k--",
        label="Avg"
    )
    avg = combination_fitness_history_df["combination_avg_best_fitness"]
    std = combination_fitness_history_df["combination_std_best_fitness"]
    plt.fill_between(
        combination_fitness_history_df["generation"],
        avg - std,
        avg + std,
        color="gray",
        alpha=0.3,
        label="± Std Dev"
    )
    plt.xlabel("Generation")
    plt.ylabel("Best Fitness")
    plt.title(f"Generation {combination_fitness_history_df['generation'].iloc[-1]} | Combination Avg Best Fitness: {np.max(combination_fitness_history_df['combination_avg_best_fitness']):.2f}")
    plt.legend()
    plt.grid(True)
    combination_fitness_plot_path = os.path.join(combination_output_dir, "combination_fitness_plot.png")
    plt.savefig(combination_fitness_plot_path)
    plt.close()

    current, peak = tracemalloc.get_traced_memory()
    print(f"[PID {os.getpid()}] Current memory usage: {current / 1024**2:.2f} MB; Peak: {peak / 1024**2:.2f} MB")
    del fitness_historics
    gc.collect()
    current, peak = tracemalloc.get_traced_memory()
    print(f"[PID {os.getpid()}] Current memory usage: {current / 1024**2:.2f} MB; Peak: {peak / 1024**2:.2f} MB")

    return combination_variable_parameters_info_df, combination_results_df, combination_fitness_history_df

def generate_param_combinations_results(fixed_params, variable_params_grid, combinations_results, output_dir, test_type):
    param_combinations_fixed_parameters_df = pd.DataFrame([fixed_params])
    fixed_parameters_file_path = os.path.join(output_dir, f"{test_type}_fixed_parameters.csv")
    param_combinations_fixed_parameters_df.to_csv(fixed_parameters_file_path, index=False)

    param_combinations_variable_parameters_df = pd.DataFrame([list(variable_params_grid.keys())])
    variable_parameters_file_path = os.path.join(output_dir, f"{test_type}_variable_parameters.csv")
    param_combinations_variable_parameters_df.to_csv(variable_parameters_file_path, index=False, header=False)

    param_combinations_fitnesses_rows = []
    for i, (_, combination_results_df, _) in enumerate(combinations_results):
        row_dict = {
            "combination": i + 1,
            "combination_avg_best_fitness": combination_results_df["combination_avg_best_fitness"].iloc[0],
            "combination_std_best_fitness": combination_results_df["combination_std_best_fitness"].iloc[0],
            "combination_best_fitness": combination_results_df["combination_best_fitness"].iloc[0]
        }
        param_combinations_fitnesses_rows.append(row_dict)
    param_combinations_fitnesses_df = pd.DataFrame(param_combinations_fitnesses_rows)
    fitnesses_file_path = os.path.join(output_dir, f"{test_type}_fitnesses.csv")
    param_combinations_fitnesses_df.to_csv(fitnesses_file_path, index=False)

    best_combination_index = int(param_combinations_fitnesses_df["combination_avg_best_fitness"].idxmax())
    best_combination_row = param_combinations_fitnesses_df.loc[best_combination_index]
    param_combinations_best_result_df = pd.DataFrame([{
        "combination": best_combination_row["combination"],
        "best_combination_avg_best_fitness": best_combination_row["combination_avg_best_fitness"],
        "best_combination_std_best_fitness": best_combination_row["combination_std_best_fitness"],
        "best_combination_best_fitness": best_combination_row["combination_best_fitness"],
    }])
    best_result_file_path = os.path.join(output_dir, f"{test_type}_best_result.csv")
    param_combinations_best_result_df.to_csv(best_result_file_path, index=False)

    plt.figure(figsize=(10, 8))
    for i, (_, _, combination_fitness_history_df) in enumerate(combinations_results):
        plt.plot(
            combination_fitness_history_df["generation"],
            combination_fitness_history_df["combination_avg_best_fitness"],
            label=f"Combination {i+1}"
        )
    plt.xlabel("Generation")
    plt.ylabel("Best Fitness")
    plt.title("Best Fitness Evolution for All Combinations")
    plt.grid(True)
    plt.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.35),
        ncol=3,
        fontsize="small"
    )
    plt.tight_layout(rect=[0, 0.2, 1, 1])
    plot_path = os.path.join(output_dir, f"{test_type}_plot.png")
    plt.savefig(plot_path)
    plt.close()

def log(msg, LOG_FILE):
    if LOG_FILE:
        with open(LOG_FILE, "a") as f:
            f.write(f"[{datetime.now()}] [PID {os.getpid()}] {msg}\n")
    else:
            print(f"[{datetime.now()}] [PID {os.getpid()}] {msg}")