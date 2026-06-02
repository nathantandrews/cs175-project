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
        hidden_dim=args.hidden_dim,
        batch_size=args.batch_size,
        target_update=args.target_update,
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
          output_filepath=os.path.join(args.output_dir, f"{args.agent}_learning_curve.png"),
      )

    if hasattr(agent, "save"):
      model_filepath = args.model_path or os.path.join(args.output_dir, f"{args.agent}_model.pt")
      agent.save(model_filepath)

  elif args.mode == "test":
    if hasattr(agent, "load"):
      model_filepath = args.model_path or os.path.join(args.output_dir, f"{args.agent}_model.pt")
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

  elif args.mode == "grid_search":
    import itertools
    import json
    
    if args.agent != "dqn":
        print("Grid search currently only supports the DQN agent.")
        return
        
    # Select the hyperparameter search grid based on --grid-type
    if getattr(args, "grid_type", "standard") == "fast":
        grid = {
            "alpha": [3e-4, 1e-3],
            "gamma": [0.99],
            "hidden_dim": [128, 256]
        }
    elif getattr(args, "grid_type", "standard") == "comprehensive":
        grid = {
            "alpha": [1e-4, 3e-4, 1e-3],
            "gamma": [0.95, 0.99],
            "hidden_dim": [128, 256],
            "decay_rate": [0.999, 0.9999],
            "target_update": [1000, 2000]
        }
    elif getattr(args, "grid_type", "standard") == "custom" and getattr(args, "custom_grid", None):
        try:
            grid = json.loads(args.custom_grid)
            if not isinstance(grid, dict):
                raise ValueError("Custom grid must be a JSON dictionary.")
        except Exception as e:
            print(f"Error parsing custom JSON grid: {e}. Falling back to standard grid.")
            grid = {
                "alpha": [1e-4, 3e-4, 1e-3],
                "gamma": [0.95, 0.99],
                "hidden_dim": [128, 256]
            }
    else:  # standard (default)
        grid = {
            "alpha": [1e-4, 3e-4, 1e-3],
            "gamma": [0.95, 0.99],
            "hidden_dim": [128, 256]
        }
    
    keys, values = zip(*grid.items())
    permutations = [dict(zip(keys, v)) for v in itertools.product(*values)]
    
    best_reward = -float('inf')
    best_params = None
    
    print(f"Starting Grid Search with {len(permutations)} configurations on {args.dataset}...")
    
    for i, params in enumerate(permutations):
        print(f"\n=== Configuration {i+1}/{len(permutations)} ===")
        print(f"Parameters: {params}")
        
        for k, v in params.items():
            setattr(args, k, v)
            
        env = gym.make(args.env, db_path=db_path, disable_env_checker=True)
        agent = build_agent(args.agent, env, args)
        
        _, rewards = agent.train(env, num_episodes=args.num_episodes)
        test_reward, test_steps = agent.evaluate(env)
        
        print(f"Result -> Test Reward: {test_reward:.2f}")
        if test_reward > best_reward:
            best_reward = test_reward
            best_params = params
            print("New best configuration found! Saving model...")
            model_filepath = os.path.join(args.output_dir, f"{args.agent}_best_grid_model.pt")
            if hasattr(agent, "save"):
                agent.save(model_filepath)
                
    print("\n===============================")
    print(f"Grid Search Complete!")
    print(f"Best Parameters: {json.dumps(best_params, indent=2)}")
    print(f"Best Test Reward: {best_reward:.2f}")
    print("===============================")

if __name__ == "__main__":
  main()
