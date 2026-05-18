from argparse import ArgumentParser

def parse_args():
    parser = ArgumentParser()
    parser.add_argument("-a", "--agent", type=str, choices=["random", "heuristic", "dqn"], default="dqn", help="Agent to run (random, heuristic, or dqn)")
    parser.add_argument("-d", "--dataset", type=str, choices=["benign_v3", "campaigns_v2", "exp01_90d", "exp_30d_heavy", "exp_365d_realistic", "exp_7d_brute"], default="exp_7d_brute", help="Dataset to use")
    parser.add_argument("-o", "--output-dir", type=str, default="out", help="Output directory")
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument("--alpha", type=float, default=0.8)
    parser.add_argument("--epsilon", type=float, default=1.0)
    parser.add_argument("--decay-rate", type=float, default=0.9999)
    parser.add_argument("--min-eps", type=float, default=1e-4)
    parser.add_argument("--env", type=str, default="FrozenLake-v1")
    parser.add_argument("--num-episodes", type=int, default=100000)
    return parser.parse_args()
