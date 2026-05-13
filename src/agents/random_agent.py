import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import gymnasium as gym
import numpy as np
import security_gym
import utils.constants as const
import matplotlib.pyplot as plt

env = gym.make("SecurityLogStream-v1", db_path=const.BRUTE_7D_FILEPATH)
obs, info = env.reset()

print("Initializing Random Agent...")
cumulative_reward = 0.0
step_count = 0

# Trackers for both of our graphs
step_rewards = [] 
steps_history = []
reward_history = []

while True:
    action = env.action_space.sample()
    print('action', action, type(action))
    obs, reward, terminated, truncated, info = env.step(action)
    
    cumulative_reward += reward
    step_count += 1
    
    # Save the reward for the Rolling Average graph
    step_rewards.append(reward)

    # Save data for the Cumulative graph every 100 steps
    if step_count % 100 == 0:
        steps_history.append(step_count)
        reward_history.append(cumulative_reward)

    if truncated or terminated:
        # Catch the final step for the Cumulative graph
        steps_history.append(step_count)
        reward_history.append(cumulative_reward)
        
        print("\n--- Simulation Finished ---")
        print(f"Total Steps: {step_count}")
        print(f"Final Cumulative Reward: {cumulative_reward:.2f}")
        break

env.close()

# --- GRAPH GENERATION ---
print("\nGenerating performance graphs...")

# 1. Generate the Cumulative Reward Graph
plt.figure(figsize=(10, 6))
plt.plot(steps_history, reward_history, label="Random Agent", color='red')
plt.title("Random Agent Baseline: Cumulative Reward")
plt.xlabel("Simulation Steps")
plt.ylabel("Cumulative Reward")
plt.grid(True)
plt.legend()
plt.savefig("figures/random_agent_cumulative_graph.png")
plt.close()

# 2. Generate the Rolling Average Graph
window_size = 500
rolling_avg = np.convolve(step_rewards, np.ones(window_size)/window_size, mode='valid')

plt.figure(figsize=(10, 6))
plt.plot(rolling_avg, label=f"Random Agent (Rolling Avg: {window_size} steps)", color='orange')
plt.title("Random Agent Baseline: Average Reward Over Time")
plt.xlabel(f"Simulation Steps (after initial {window_size} steps)")
plt.ylabel("Average Reward per Step")
plt.grid(True)
plt.legend()
plt.savefig("figures/random_agent_avg_graph.png")
plt.close()

print("Graphs saved. Named 'random_agent_cumulative_graph.png' and 'random_agent_avg_graph.png'.")