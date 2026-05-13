import gymnasium as gym
import numpy as np
import security_gym
import tqdm
import os
import copy

import utils.constants as const
# from agents.dueling_dqn_agent import DQNAgent
from agents.heuristic_agent import HeuristicAgent
from utils.argparse import parse_args

def train(env, agent : DQNAgent, num_episodes=10000):
  pbar = tqdm.tqdm(range(num_episodes), desc="Training...")
  for episode in pbar:
      state, _ = env.reset()

      done = False
      while not done:
          action = agent.get_action(state)
          next_state, reward, done, _, _ = env.step(action)
          agent.update(state, action, reward, next_state, done)
          state = next_state
      agent.epsilon_decay()
      agent.rewards.append(reward)
      if (episode+1) % 100 == 0:
          pbar.set_description(f"Episode {episode+1}/{num_episodes}, avg reward (last 100)={np.mean(agent.rewards[-100:]):.2f}")
  return agent.Q, agent.rewards

def main():
  args = parse_args()
  os.makedirs(args.output, exist_ok=True)
  env = gym.make("SecurityLogStream-v1", db_path=const.BRUTE_7D_FILEPATH)
  obs, _ = env.reset()
  agent = DQNAgent(
      env,
      gamma=args.gamma,
      alpha=args.alpha,
      epsilon=args.epsilon,
      decay_rate=args.decay_rate,
      min_eps=args.min_eps
  )
  Q, rewards = train(env, agent, num_episodes=args.num_episodes)
  plot_learning_curve(rewards, os.path.join(args.output, "plot.png"))
  
def run_test():
    # Initialize the environment
    env = gym.make("SecurityLogStream-v1", db_path=const.BRUTE_7D_FILEPATH) # Ensure the env name matches your local install
    agent = HeuristicAgent(env)
    
    obs, _ = env.reset()
    print(type(obs))
    done = False
    total_reward = 0
    step = 0

    print(f"{'Step':<5} | {'Action':<10} | {'Risk':<5} | {'Reward':<8} | {'Event Type'}")
    print("-" * 60)

    while not done:
        # Get action from our heuristic
        agent_output = agent.get_action(obs)
        
        # Step the environment
        obs, reward, done, truncated, info = env.step(agent_output)
        
        total_reward += reward
        step += 1

        # Print progress
        action_name = ["PASS", "ALERT", "THROTTLE", "BLOCK", "UNBLOCK", "ISOLATE"][agent_output["action"]]
        event = info.get('event_type', 'benign')
        
        print(f"{step:<5} | {action_name:<10} | {agent_output['risk_score'][0]:<5.1f} | {reward:<8.2f} | {event}")

        if truncated:
          print("\n--- Simulation Truncated ---")
          break

    print("-" * 60)
    print(f"Test Finished. Total Reward: {total_reward:.2f}")

if __name__ == "__main__":
    run_test()

# if __name__ == "__main__":
#   main()
