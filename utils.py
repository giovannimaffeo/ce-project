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

def generate_results(fitness_history_df, best_robot, params, output_dir):
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

def generate_combination_results(combination_variable_params, best_fitnesses, fitness_historics, combination, combination_output_dir):
    combination_variable_parameters_info_df = pd.DataFrame([list(combination_variable_params.values())], columns=list(combination_variable_params.keys()))
    combination_variable_parameters_info_path = os.path.join(combination_output_dir, "combination_variable_parameters_info.csv")
    combination_variable_parameters_info_df.to_csv(combination_variable_parameters_info_path, index=False)

    combination_best_result = np.mean(best_fitnesses)
    combination_best_result_df = pd.DataFrame([{
        "best_fitness": combination_best_result,
    }])
    combination_best_result_file_path = os.path.join(combination_output_dir, "combination_best_result.csv")
    combination_best_result_df.to_csv(combination_best_result_file_path, index=False)

    n_generations = len(fitness_historics[0])
    combination_fitness_history_df = pd.DataFrame({
        "generation": list(range(1, n_generations + 1)),
        "best_fitness": [
            np.mean([fitness_history[i]["best_fitness"] for fitness_history in fitness_historics])
            for i in range(n_generations)
        ],
        "mean_fitness": [
            np.mean([fitness_history[i]["mean_fitness"] for fitness_history in fitness_historics])
            for i in range(n_generations)
        ]
    }) 
    combination_fitness_history_file_path = os.path.join(combination_output_dir, "combination_fitness_history.csv")
    combination_fitness_history_df.to_csv(combination_fitness_history_file_path, index=False)   

    plt.figure()
    for i, fitness_history in enumerate(fitness_historics):
        plt.plot(
            [generation["generation"] for generation in fitness_history], 
            [generation["best_fitness"] for generation in fitness_history], 
            label=f"Run {i+1}"
        )
    plt.plot(
        combination_fitness_history_df["generation"],
        combination_fitness_history_df["best_fitness"],
        "k--",
        label="Avg"
    )
    plt.xlabel("Generation")
    plt.ylabel("Best Fitness")
    plt.title(f"Generation: {combination_fitness_history_df['generation'].iloc[-1]} | Combination Best Fitness: {combination_fitness_history_df['best_fitness'].iloc[-1]:.2f}")
    plt.legend()
    plt.grid(True)
    combination_fitness_plot_path = os.path.join(combination_output_dir, "combination_fitness_plot.png")
    plt.savefig(combination_fitness_plot_path)
    plt.close()

    return combination_variable_parameters_info_df, combination_best_result_df, combination_fitness_history_df

def generate_hiperparams_fatorial_test_results(fixed_params, variable_params_grid, combinations_results, hiperparams_fatorial_test_output_dir):
    hiperparams_fatorial_test_fixed_parameters_df = pd.DataFrame([fixed_params])
    hiperparams_fatorial_test_fixed_parameters_file_path = os.path.join(hiperparams_fatorial_test_output_dir, "hiperparams_fatorial_test_fixed_parameters.csv")
    hiperparams_fatorial_test_fixed_parameters_df.to_csv(hiperparams_fatorial_test_fixed_parameters_file_path, index=False)

    hiperparams_fatorial_test_variable_parameters_df = pd.DataFrame([list(variable_params_grid.keys())])
    hiperparams_fatorial_test_variable_parameters_file_path = os.path.join(hiperparams_fatorial_test_output_dir, "hiperparams_fatorial_test_variable_parameters.csv")
    hiperparams_fatorial_test_variable_parameters_df.to_csv(hiperparams_fatorial_test_variable_parameters_file_path, index=False, header=False)    

    variable_params_df = pd.DataFrame([list(variable_params_grid.keys())])
    variable_params_file_path = os.path.join(hiperparams_fatorial_test_output_dir, "hiperparams_fatorial_test_variable_parameters.csv")
    variable_params_df.to_csv(variable_params_file_path, index=False, header=False)   
    
    hiperparams_fatorial_test_fitnesses_rows = []
    for i, (_, combination_best_result_df, _) in enumerate(combinations_results):
        row_dict = {
            "combination": i + 1,
            "combination_fitness": combination_best_result_df["best_fitness"].iloc[0]
        }
        hiperparams_fatorial_test_fitnesses_rows.append(row_dict)
    hiperparams_fatorial_test_fitnesses_df = pd.DataFrame(hiperparams_fatorial_test_fitnesses_rows)
    hiperparams_fatorial_test_fitnesses_file_path = os.path.join(hiperparams_fatorial_test_output_dir, "hiperparams_fatorial_test_fitnesses.csv")
    hiperparams_fatorial_test_fitnesses_df.to_csv(hiperparams_fatorial_test_fitnesses_file_path, index=False)

    best_fitnesses = [combination_best_result_df["best_fitness"].iloc[0] for (_, combination_best_result_df, _) in combinations_results]
    best_combination_fitness_index = int(np.argmax(best_fitnesses))
    best_combination_fitness = best_fitnesses[best_combination_fitness_index]
    hiperparams_fatorial_test_best_result_df = pd.DataFrame([{
        "combination": best_combination_fitness_index + 1,
        "best_combination_fitness": best_combination_fitness
    }])
    hiperparams_fatorial_test_best_result_file_path = os.path.join(hiperparams_fatorial_test_output_dir, "hiperparams_fatorial_test_best_result.csv")
    hiperparams_fatorial_test_best_result_df.to_csv(hiperparams_fatorial_test_best_result_file_path, index=False)

    plt.figure()
    for i, (_, _, combination_fitness_history_df) in enumerate(combinations_results):
        plt.plot(
            combination_fitness_history_df["generation"],
            combination_fitness_history_df["best_fitness"],
            label=f"Combination {i+1}"
        )
    plt.xlabel("Generation")
    plt.ylabel("Best Fitness")
    plt.title("Best Fitness Evolution for All Combinations")
    plt.grid(True)
    plt.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.2),
        ncol=3,
        fontsize="small"
    )
    plt.tight_layout(rect=[0, 0.2, 1, 1])
    hiperparams_fatorial_test_plot_path = os.path.join(hiperparams_fatorial_test_output_dir, "hiperparams_fatorial_test_plot.png")
    plt.savefig(hiperparams_fatorial_test_plot_path)
    plt.close()