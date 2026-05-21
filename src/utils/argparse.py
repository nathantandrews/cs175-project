from argparse import ArgumentParser

def parse_args():
    parser = ArgumentParser()
    parser.add_argument("-a", "--agent", type=str, choices=["random", "heuristic", "dqn"], default="dqn", help="Agent to run (random, heuristic, or dqn)")
    parser.add_argument("-d", "--dataset", type=str, choices=["benign_v3", "campaigns_v2", "exp01_90d", "exp_30d_heavy", "exp_365d_realistic", "exp_7d_brute"], default="exp_7d_brute", help="Dataset to use")
    parser.add_argument("-o", "--output-dir", type=str, default="out", help="Output directory")
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument("--alpha", type=float, default=0.8)
    parser.add_argument("--epsilon", type=float, default=1.0)
    parser.add_argument("--decay-rate", type=float, default=0.95)
    parser.add_argument("--min-eps", type=float, default=1e-4)
    parser.add_argument("-env", "--env", type=str, default="SecurityLogStream-v1")
    parser.add_argument("--num-episodes", type=int, default=100)
    parser.add_argument("--mode", type=str, choices=["train", "test", "grid_search"], default="train", help="Mode to run (train, test, or grid_search)")
    parser.add_argument("--model-path", type=str, default=None, help="Path to load/save model weights (.npz)")
    return parser.parse_args()
