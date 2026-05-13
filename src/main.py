import gymnasium as gym
import numpy as np
import security_gym
import tqdm
import os
import copy

import utils.constants as const
from agents.dueling_dqn_agent import DQNAgent
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
  obs, info = env.reset()
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

if __name__ == "__main__":
  main()
