# Dueling DQN for log-stream defense

UCI **CS 175** (Project in AI). An agent sits on a stream of authentication and web logs and chooses a response at each event: pass, alert, throttle, block, unblock, or isolate the host.

The learned policy is a **Dueling Double DQN** in PyTorch. It is scored against a random policy and a keyword / failed-login **heuristic**, using [security-gym](https://github.com/j-klawson/security-gym) `SecurityLogStream-v1`.

## Why this exists

Signature rules catch known strings (`jndi:ldap`, brute-force `failed` counts) and miss everything else. The DQN learns a policy from reward on the same event stream, including when to undo a bad block.

This is a class RL project, not a production IDS.

## Method

| Piece | What the code actually does |
|---|---|
| Environment | Gymnasium + `security-gym` log databases (`benign_v3`, `campaigns_v2`, `exp_7d_brute`, …) |
| Network | Two-layer MLP, then split **value** and **advantage** heads (`DuelingMLP`) |
| Learning | **Double DQN**: online net picks the next action, target net evaluates it. Uniform replay buffer (50k). Adam, grad clip 1.0, MSE TD loss |
| Exploration | ε-greedy, multiplicative decay |
| Logging | TensorBoard: episode reward, eval reward/step, TD loss |
| Baselines | `RandomAgent`; `HeuristicAgent` (critical web signatures + failed-auth thresholds) |

Hyperparameters (`--alpha`, `--gamma`, `--hidden-dim`, `--decay-rate`, `--target-update`) can be swept with `--mode grid_search`.

Default `--alpha 0.8` in the CLI is a leftover tabular rate. For DQN pass something like `--alpha 1e-3` (that is what `DQNAgent` was written for).

## Layout

```
src/main.py                 train / test / grid_search
src/agents/dueling_dqn_agent.py
src/agents/heuristic_agent.py
src/agents/random_agent.py
src/utils/constants.py      action dicts + dataset paths
scripts/setup-docker.sh
scripts/download-datasets.sh
train.sh  test.sh  grid-search.sh
```

## Setup

Datasets (needs `zstd` and `curl` or `wget`):

```bash
./scripts/download-datasets.sh
```

**Docker (recommended)**

```bash
./scripts/setup-docker.sh    # builds image `cs175`, starts container `model`
```

**Local**

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export PYTHONPATH=src
```

`requirements.txt` pins CUDA wheels. On CPU-only machines install `torch` from the [CPU index](https://pytorch.org/get-started/locally/) first, then the rest without the `nvidia-*` packages.

## Run

Inside the container, or with `PYTHONPATH=src`:

```bash
# train (writes weights + learning curve under ./out)
./train.sh --dataset exp_7d_brute --num-episodes 100 --alpha 1e-3 --gamma 0.95

# evaluate a checkpoint
./test.sh --dataset exp_7d_brute --model-path out/dqn_model.pt --alpha 1e-3

# baselines
./test.sh -a heuristic --dataset exp_7d_brute
./test.sh -a random --dataset exp_7d_brute

# hyperparameter sweep
./grid-search.sh --dataset exp_7d_brute --grid-type standard --num-episodes 50 --alpha 1e-3
```

TensorBoard (Docker maps `6006:6006`):

```bash
tensorboard --logdir runs/security_dqn --bind_all --port 6006
```

`./test.sh` writes action-distribution and cumulative-reward plots under `figures/`.

## Results

Put the numbers you actually measured here (test return vs heuristic vs random on a named dataset, plus seed). Do not cite a multiplier you cannot regenerate from `./test.sh`.

## Stack

Python, PyTorch, Gymnasium, security-gym, TensorBoard, Docker.
