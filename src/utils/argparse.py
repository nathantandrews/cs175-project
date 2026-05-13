from argparse import ArgumentParser

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--output", type=str, default="submission", help="Output directory")
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument("--alpha", type=float, default=0.8)
    parser.add_argument("--epsilon", type=float, default=1.0)
    parser.add_argument("--decay-rate", type=float, default=0.9999)
    parser.add_argument("--min-eps", type=float, default=1e-4)
    parser.add_argument("--env", type=str, default="FrozenLake-v1")
    parser.add_argument("--num-episodes", type=int, default=100000)
    return parser.parse_args()
