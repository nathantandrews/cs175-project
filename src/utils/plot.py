import matplotlib.pyplot as plt
import numpy as np

def plot_cumulative_curve(steps_history, reward_history, label="Agent", color='red', output_filepath="figures/cumulative_graph.png"):
    """
    Plots the cumulative reward curve.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(steps_history, reward_history, label=label, color=color)
    plt.title(f"{label}: Cumulative Reward")
    plt.xlabel("Simulation Steps")
    plt.ylabel("Cumulative Reward")
    plt.grid(True)
    plt.legend()
    
    # Ensure directory exists
    import os
    os.makedirs(os.path.dirname(output_filepath), exist_ok=True)
    
    plt.savefig(output_filepath)
    plt.close()

def plot_learning_curve(step_rewards, label="Agent", color="orange", window_size=500, output_filepath="figures/plot.png"):
    """
    Plots the rolling average reward curve.
    """
    if len(step_rewards) < window_size:
        print(f"Warning: Not enough data points ({len(step_rewards)}) for window size {window_size}. Plotting raw rewards instead.")
        rolling_avg = step_rewards
    else:
        rolling_avg = np.convolve(step_rewards, np.ones(window_size)/window_size, mode='valid')

    plt.figure(figsize=(10, 6))
    plt.plot(rolling_avg, label=f"{label} (Rolling Avg: {window_size} steps)" if len(step_rewards) >= window_size else label, color=color)
    plt.title(f"{label}: Average Reward Over Time")
    plt.xlabel(f"Simulation Steps (after initial {window_size} steps)" if len(step_rewards) >= window_size else "Simulation Steps")
    plt.ylabel("Average Reward per Step")
    plt.grid(True)
    plt.legend()
    
    # Ensure directory exists
    import os
    os.makedirs(os.path.dirname(output_filepath), exist_ok=True)
    
    plt.savefig(output_filepath)
    plt.close()
