import gymnasium as gym
import security_gym
import os

import utils.constants as const
from agents.random_agent import RandomAgent
from agents.heuristic_agent import HeuristicAgent
from agents.dueling_dqn_agent import DQNAgent
from utils.argparse import parse_args
from utils.plot import plot_learning_curve


def build_agent(agent_name, env, args):
  if agent_name == "random":
    return RandomAgent(env)
  elif agent_name == "heuristic":
    return HeuristicAgent(env)
  elif agent_name == "dqn":
    return DQNAgent(
        env,
        gamma=args.gamma,
        alpha=args.alpha,
        epsilon=args.epsilon,
        decay_rate=args.decay_rate,
        min_eps=args.min_eps,
    )
  else:
    raise ValueError(f"Unknown agent: {agent_name}")


def main():
  args = parse_args()
  os.makedirs(args.output_dir, exist_ok=True)

  db_path = const.DATASETS[args.dataset]
  env = gym.make(args.env, db_path=db_path, disable_env_checker=True)

  agent = build_agent(args.agent, env, args)

  if args.mode == "train":
    print(f"Training {agent.name} on {args.dataset}...")
    Q, rewards = agent.train(env, num_episodes=args.num_episodes)

    if rewards:
      plot_learning_curve(
          rewards,
          label=agent.name,
          output_filepath=os.path.join(args.output_dir, f"{args.agent}_plot.png"),
      )

    if hasattr(agent, "save"):
      model_filepath = args.model_path or os.path.join(args.output_dir, f"{args.agent}_model.npz")
      agent.save(model_filepath)

  elif args.mode == "test":
    if hasattr(agent, "load"):
      model_filepath = args.model_path or os.path.join(args.output_dir, f"{args.agent}_model.npz")
      if not os.path.exists(model_filepath):
        raise FileNotFoundError(
            f"Model weights not found at {model_filepath}. "
            "Please train the model first or specify the correct --model-path."
        )
      obs, info = env.reset()
      agent.load(model_filepath, obs)

    print(f"\nEvaluating {agent.name} on {args.dataset}...")
    test_reward, test_steps = agent.evaluate(env)
    print(f"Test Result | Steps: {test_steps} | Total Reward: {test_reward:.2f}")

if __name__ == "__main__":
  main()
