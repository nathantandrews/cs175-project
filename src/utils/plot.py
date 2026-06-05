import matplotlib.pyplot as plt
import numpy as np
import os

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
    
def plot_action_distribution(actions, output_dir, agent_name):
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

    plt.title(f'Action Distribution - {agent_name}')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Save the plot
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, f"{agent_name}_action_distribution.png")
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()
    
    print(f"Action distribution plot saved to: {save_path}")
