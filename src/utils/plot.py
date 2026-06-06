import matplotlib.pyplot as plt
import numpy as np
import os

def plot_cumulative_curve(steps_history, reward_history, label="Agent", color='red', output_filepath="figures/cumulative_graph.png"):
    """
    Plots the cumulative reward curve.
    """
    sorted_indices = np.argsort(steps_history)
    clean_steps = np.array(steps_history)[sorted_indices]
    clean_rewards = np.array(reward_history)[sorted_indices]
    plt.figure(figsize=(10, 6))
    plt.plot(clean_steps, clean_rewards, label=label, color=color)
    plt.title(f"{label}: Cumulative Reward")
    plt.xlabel("Simulation Steps")
    plt.ylabel("Cumulative Reward")
    plt.grid(True)
    plt.legend()
    

    os.makedirs(os.path.dirname(output_filepath), exist_ok=True)
    save_path = os.path.join(os.path.dirname(output_filepath), f"{label.replace(' ', '_')}_cumulative_rewards.png")
    
    plt.savefig(save_path)
    plt.close()


def plot_learning_curve(reward_history, label="Agent", color="orange", window_size=500, output_filepath="figures/plot.png"):
    """
    Plots the learning curve directly from individual step rewards using pure NumPy.
    To make it less jagged without EMA, increase window_size (e.g., 1500 or 2000).
    """
    step_rewards = np.array(reward_history)
    
    if len(step_rewards) == 0:
        print("Error: The provided reward history is empty.")
        return

    # 1. Compute the moving/rolling average (No np.diff needed!)
    if len(step_rewards) < window_size:
        print(f"Warning: Not enough data points ({len(step_rewards)}) for window size {window_size}. Plotting raw rewards instead.")
        rolling_avg = step_rewards
        x_axis = np.arange(len(rolling_avg))
        xlabel = "Simulation Steps"
    else:
        # A larger uniform window dilutes isolated penalties, smoothing out the curve
        rolling_avg = np.convolve(step_rewards, np.ones(window_size)/window_size, mode='valid')
        x_axis = np.arange(window_size - 1, window_size - 1 + len(rolling_avg))
        xlabel = f"Simulation Steps (Rolling Window: {window_size} steps)"
        
    # 2. Plotting using subplots
    fig, ax = plt.subplots(figsize=(10, 6))
    
    label_text = f"{label} ({window_size}-step MA)" if len(step_rewards) >= window_size else label
    ax.plot(x_axis, rolling_avg, label=label_text, color=color)
    
    ax.set_title(f"{label}: Average Reward per Step")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Average Reward per Step")
    ax.grid(True, linestyle="--", linewidth=0.5)
    ax.legend()
    
    # 3. Save file infrastructure
    os.makedirs(os.path.dirname(output_filepath), exist_ok=True)
    save_path = os.path.join(os.path.dirname(output_filepath), f"{label.replace(' ', '_')}_learning_curve.png")
    
    plt.savefig(save_path)
    plt.close(fig)
    
    
def plot_action_distribution(actions, label="Agent", output_filepath="figures/action_distribution.png"):
    """
    Plots the distribution of actions taken by the agent during evaluation.
    Works for both discrete and continuous action spaces.
    """
    actions = np.array(actions)
    
    plt.figure(figsize=(8, 5))
    
    # Check if actions are discrete (integers or 1D categorical)
    if np.issubdtype(actions.dtype, np.integer) or len(actions.shape) == 1 and len(np.unique(actions)) < 30:
        # Discrete Action Space: Count frequencies of each unique action
        unique, counts = np.unique(actions, return_counts=True)
        # Handle potential multi-dimensional discrete actions by flattening
        plt.bar(unique, counts, color='skyblue', edgecolor='black', alpha=0.8)
        plt.xlabel('Action ID')
        plt.ylabel('Frequency / Count')
        plt.xticks(unique)  # Show every action ID clearly
    else:
        # Continuous Action Space: Use a histogram
        if len(actions.shape) > 1:
            # If multi-dimensional continuous actions (e.g., [steering, throttle]), plot each dim
            for i in range(actions.shape[1]):
                plt.hist(actions[:, i], bins=30, alpha=0.5, label=f'Dim {i}')
            plt.legend()
        else:
            plt.hist(actions, bins=30, color='salmon', edgecolor='black', alpha=0.8)
        
        plt.xlabel('Action Value')
        plt.ylabel('Frequency')

    plt.title(f'Action Distribution - {label}')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Save the plot
    os.makedirs(os.path.dirname(output_filepath), exist_ok=True)
    save_path = os.path.join(os.path.dirname(output_filepath), f"{label.replace(' ', '_')}_action_distribution.png")
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()
    
    print(f"Action distribution plot saved to: {save_path}")
