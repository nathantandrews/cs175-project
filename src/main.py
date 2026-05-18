import gymnasium as gym
import numpy as np
import security_gym
import tqdm
import os
import copy

import utils.constants as const

# from agents.dueling_dqn_agent import DQNAgent
from agents.agent import Agent
from agents.heuristic_agent import HeuristicAgent
# from utils.argparse import parse_args
from utils.plotting import plot_learning_curve


def train(env, agent: Agent, num_episodes=10000):
    """Train the agent in the environment. Works on any agent inheriting from the base Agent class."""
    pbar = tqdm.tqdm(range(num_episodes), desc="Training...")
    for episode in pbar:
      state, _ = env.reset()

      terminated = False
      truncated = False
      while not terminated and not truncated:
        action = agent.get_action(state)
        obs, reward, terminated, truncated, info = env.step(action)
        agent.update(state, action, reward, obs, terminated or truncated)
        state = obs
      agent.epsilon_decay()
      agent.rewards.append(reward)
      if (episode + 1) % 100 == 0:
          pbar.set_description(
            f"Episode {episode+1}/{num_episodes}, avg reward (last 100)={np.mean(agent.rewards[-100:]):.2f}"
          )
    return agent.rewards


def main():
    # args = parse_args()
    output = "output"
    num_episodes = 200
    os.makedirs(output, exist_ok=True)
    env = gym.make("SecurityLogStream-v1", db_path=const.BRUTE_7D_FILEPATH)
    obs, _ = env.reset()
    # agent = DQNAgent(
    #     env,
    #     gamma=args.gamma,
    #     alpha=args.alpha,
    #     epsilon=args.epsilon,
    #     decay_rate=args.decay_rate,
    #     min_eps=args.min_eps,
    # )
    agent = HeuristicAgent(env)
    rewards = train(env, agent, num_episodes=num_episodes)
    plot_learning_curve(rewards, os.path.join(output, "plot.png"))
    print(f"Training complete. Plot saved to {os.path.join(output, 'plot.png')}")
    print("Cumulated reward:", sum(rewards))



if __name__ == "__main__":
  main()
