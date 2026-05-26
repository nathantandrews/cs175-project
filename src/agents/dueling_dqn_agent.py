import numpy as np
from collections import deque
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from agents.agent import Agent
import utils.constants as const


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, capacity=50000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards, dtype=np.float32),
            np.array(next_states),
            np.array(dones, dtype=np.float32),
        )

    def __len__(self):
        return len(self.buffer)


def _flatten_obs(obs):
    """
    Your original flattener that gave +19k rewards. 
    Kept exactly the same to preserve feature dimensions.
    """
    if isinstance(obs, np.ndarray):
        return obs.astype(np.float32).flatten()
    if isinstance(obs, dict):
        parts = []
        for key in sorted(obs.keys()):
            val = obs[key]
            if isinstance(val, np.ndarray):
                parts.append(val.astype(np.float32).flatten())
            elif isinstance(val, (int, float)):
                parts.append(np.array([float(val)], dtype=np.float32))
        if parts:
            return np.concatenate(parts)
        return np.zeros(1, dtype=np.float32)
    return np.array(obs, dtype=np.float32).flatten()


# ---------------------------------------------------------------------------
# Dueling DQN with PyTorch
# ---------------------------------------------------------------------------

class DuelingMLP(nn.Module):
    """Your original high-performing network architecture."""

    def __init__(self, input_dim, hidden_dim, num_actions):
        super(DuelingMLP, self).__init__()
        self.num_actions = num_actions
        
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)

        self.value_stream = nn.Linear(hidden_dim, 1)
        self.advantage_stream = nn.Linear(hidden_dim, num_actions)

    def forward(self, x):
        h1 = F.relu(self.fc1(x))
        h2 = F.relu(self.fc2(h1))

        value = self.value_stream(h2)
        advantage = self.advantage_stream(h2)

        q = value + advantage - advantage.mean(dim=-1, keepdim=True)
        return q


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class DQNAgent(Agent):
    """
    Dueling DQN agent matching your original performance, upgraded with safe 
    TensorBoard KPI tracking for rewards, steps, and TD Loss.
    """

    name = "Dueling DQN Agent"

    def __init__(
        self,
        env,
        gamma=0.95,
        alpha=1e-3,
        epsilon=1.0,
        decay_rate=0.9999,
        min_eps=1e-4,
        hidden_dim=128,
        buffer_capacity=50000,
        batch_size=64,
        target_update=1000,
        log_dir="runs/security_dqn"
    ):
        super().__init__(env)
        self.gamma = gamma
        self.alpha = alpha
        self.epsilon = epsilon
        self.decay_rate = decay_rate
        self.min_eps = min_eps
        self.batch_size = batch_size
        self.target_update = target_update
        self.hidden_dim = hidden_dim

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"DQNAgent initialized using device: {self.device}")

        # Safe TensorBoard logger
        self.writer = SummaryWriter(log_dir=log_dir)

        self._input_dim = None
        self._q_net = None
        self._target_net = None
        self.optimizer = None
        self.criterion = None

        self.replay_buffer = ReplayBuffer(capacity=buffer_capacity)

    def _ensure_networks(self, obs):
        if self._q_net is not None:
            return
        state = _flatten_obs(obs)
        self._input_dim = state.shape[0]
        
        self._q_net = DuelingMLP(self._input_dim, self.hidden_dim, const.NUM_DISCRETE_ACTIONS).to(self.device)
        self._target_net = DuelingMLP(self._input_dim, self.hidden_dim, const.NUM_DISCRETE_ACTIONS).to(self.device)
        self._target_net.load_state_dict(self._q_net.state_dict())
        self._target_net.eval()
        
        self.optimizer = optim.Adam(self._q_net.parameters(), lr=self.alpha)
        self.criterion = nn.MSELoss()

    def get_action(self, state, deterministic=False):
        """Your exact action selection logic, returning pristine env constants."""
        self._ensure_networks(state)
        flat = _flatten_obs(state)

        if not deterministic and np.random.random() < self.epsilon:
            action_idx = np.random.randint(const.NUM_DISCRETE_ACTIONS)
        else:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(flat).unsqueeze(0).to(self.device)
                q_values = self._q_net(state_tensor)
                action_idx = int(torch.argmax(q_values).item())
        
        # Pull the pristine dictionary mapping (e.g. BLOCK_ACTION) directly
        return const.ACTION_MAP.get(action_idx, const.PASS_ACTION), action_idx

    def epsilon_decay(self):
        self.epsilon = max(self.min_eps, self.epsilon * self.decay_rate)

    def evaluate_test_epoch(self, env, test_episodes=2):
        """Runs an isolated deterministic test epoch to track reward-per-step KPIs."""
        total_reward = 0.0
        total_steps = 0

        for _ in range(test_episodes):
            obs, info = env.reset()
            done = False
            while not done:
                env_action, _ = self.get_action(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(env_action)
                total_reward += reward
                total_steps += 1
                done = terminated or truncated

        avg_reward_per_step = total_reward / max(1, total_steps)
        return total_reward / test_episodes, avg_reward_per_step

    def train(self, env, num_episodes=100, patience=3, eval_every=5):
        all_rewards = []
        global_step = 0
        best_reward = -float('inf')
        patience_counter = 0
        best_weights = None

        try:
            for episode in range(num_episodes):
                obs, info = env.reset()
                self._ensure_networks(obs)
                episode_reward = 0.0
                step = 0

                while True:
                    env_action, action_idx = self.get_action(obs)
                    next_obs, reward, terminated, truncated, info = env.step(env_action)

                    flat_state = _flatten_obs(obs)
                    flat_next = _flatten_obs(next_obs)
                    done = terminated or truncated
                    self.replay_buffer.push(flat_state, action_idx, reward, flat_next, float(done))

                    episode_reward += reward
                    step += 1
                    global_step += 1

                    if len(self.replay_buffer) >= self.batch_size:
                        loss_val = self._learn()
                        # KPI 3: Track TD Loss Error per optimization step
                        self.writer.add_scalar("Loss/TD_Loss_Error", loss_val, global_step)

                    if global_step % self.target_update == 0:
                        self._target_net.load_state_dict(self._q_net.state_dict())

                    self.epsilon_decay()
                    obs = next_obs

                    if done:
                        break

                # KPI 1: Cumulative reward per epoch
                self.writer.add_scalar("Reward/Train_Cumulative_Per_Epoch", episode_reward, episode)
                all_rewards.append(episode_reward)

                print(f"Episode {episode + 1}/{num_episodes} | Steps: {step} | Reward: {episode_reward:.2f} | Epsilon: {self.epsilon:.4f}")
                
                # Periodic KPI Evaluation
                if (episode + 1) % eval_every == 0:
                    eval_cum, eval_per_step = self.evaluate_test_epoch(env)
                    # KPI 2: Average reward per step in test epoch
                    self.writer.add_scalar("Reward/Test_Epoch_Cumulative", eval_cum, episode)
                    self.writer.add_scalar("Reward/Test_Epoch_Avg_Reward_Per_Step", eval_per_step, episode)
                    print(f" ---> [TEST] Cumulative: {eval_cum:.2f} | Avg Reward Per Step: {eval_per_step:.4f}")

                # Early stopping check
                if episode_reward > best_reward:
                    best_reward = episode_reward
                    patience_counter = 0
                    best_weights = {k: v.cpu().clone() for k, v in self._q_net.state_dict().items()}
                else:
                    patience_counter += 1
                    print(f"  -> No improvement. Patience: {patience_counter}/{patience}")
                    if patience_counter >= patience:
                        print(f"\nEarly stopping triggered! Restoring best weights with reward: {best_reward:.2f}")
                        if best_weights is not None:
                            self._q_net.load_state_dict(best_weights)
                            self._target_net.load_state_dict(best_weights)
                        break

        except KeyboardInterrupt:
            print("\nTraining interrupted by user. Saving progress...")
            if best_weights is not None and best_reward > episode_reward:
                self._q_net.load_state_dict(best_weights)
                self._target_net.load_state_dict(best_weights)

        self.writer.close()
        return self._q_net, all_rewards

    def save(self, filepath):
        if self._q_net is None:
            raise ValueError("Cannot save weights; network not initialized.")
        torch.save(self._q_net.state_dict(), filepath)
        print(f"Agent weights saved to {filepath}")

    def load(self, filepath, obs_sample):
        self._ensure_networks(obs_sample)
        self._q_net.load_state_dict(torch.load(filepath, map_location=self.device, weights_only=True))
        self._target_net.load_state_dict(self._q_net.state_dict())
        print(f"Agent weights loaded from {filepath}")

    def _learn(self):
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        with torch.no_grad():
            q_next_online = self._q_net(next_states)
            best_actions = torch.argmax(q_next_online, dim=1)

            q_next_target = self._target_net(next_states)
            q_target_vals = q_next_target[torch.arange(self.batch_size), best_actions]

            targets = rewards + self.gamma * q_target_vals * (1.0 - dones)

        q_values = self._q_net(states)
        q_pred = q_values[torch.arange(self.batch_size), actions]

        loss = self.criterion(q_pred, targets)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self._q_net.parameters(), 1.0)
        self.optimizer.step()

        return float(loss.item())