import numpy as np
from collections import deque
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

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
    Flatten a SecurityLogStream observation into a fixed-size numeric vector.

    The observation is a Dict that may contain text channels (strings) and
    numeric arrays.  We keep only the numeric parts so the Q-network can
    consume them.  If the observation is already a flat array we return it
    directly.
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
            # skip text / string channels
        if parts:
            return np.concatenate(parts)
        # fallback: return a zero vector (will be resized on first call)
        return np.zeros(1, dtype=np.float32)
    # tuple / list – try to convert
    return np.array(obs, dtype=np.float32).flatten()


# ---------------------------------------------------------------------------
# Dueling DQN with PyTorch
# ---------------------------------------------------------------------------

class DuelingMLP(nn.Module):
    """
    A small multi-layer perceptron that outputs Q-values via the dueling
    architecture:  Q(s,a) = V(s) + A(s,a) - mean(A(s,:))
    """

    def __init__(self, input_dim, hidden_dim, num_actions):
        super(DuelingMLP, self).__init__()
        self.num_actions = num_actions
        
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)

        # Value stream  (hidden → 1)
        self.value_stream = nn.Linear(hidden_dim, 1)

        # Advantage stream  (hidden → num_actions)
        self.advantage_stream = nn.Linear(hidden_dim, num_actions)

    def forward(self, x):
        h1 = F.relu(self.fc1(x))
        h2 = F.relu(self.fc2(h1))

        value = self.value_stream(h2)
        advantage = self.advantage_stream(h2)

        # Dueling combination
        q = value + advantage - advantage.mean(dim=-1, keepdim=True)
        return q


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class DQNAgent(Agent):
    """
    Dueling DQN agent for SecurityLogStream-v1.

    Uses a PyTorch MLP with:
      - experience replay
      - target network (hard-copy every ``target_update`` steps)
      - epsilon-greedy exploration with exponential decay
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

        # Determine device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"DQNAgent initialized using device: {self.device}")

        # Lazily initialised on the first observation
        self._input_dim = None
        self._q_net = None
        self._target_net = None
        self.optimizer = None
        self.criterion = None

        self.replay_buffer = ReplayBuffer(capacity=buffer_capacity)

    # ---- lazy init --------------------------------------------------------

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

    # ---- Agent interface --------------------------------------------------

    def get_action(self, state):
        """
        Epsilon-greedy action selection.

        Returns the Dict action expected by SecurityLogStream-v1.
        """
        self._ensure_networks(state)
        flat = _flatten_obs(state)

        if np.random.random() < self.epsilon:
            action_idx = np.random.randint(const.NUM_DISCRETE_ACTIONS)
        else:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(flat).unsqueeze(0).to(self.device)
                q_values = self._q_net(state_tensor)
                action_idx = int(torch.argmax(q_values).item())
        return const.ACTION_MAP.get(action_idx, const.PASS_ACTION), action_idx

    def epsilon_decay(self):
        """Exponentially decay epsilon towards min_eps."""
        self.epsilon = max(self.min_eps, self.epsilon * self.decay_rate)

    def train(self, env, num_episodes=1, patience=3):
        """
        Train the agent on the SecurityLogStream environment.
        """
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
                    # Select action
                    env_action, action_idx = self.get_action(obs)
                    next_obs, reward, terminated, truncated, info = env.step(env_action)

                    # Store transition
                    flat_state = _flatten_obs(obs)
                    flat_next = _flatten_obs(next_obs)
                    done = terminated or truncated
                    self.replay_buffer.push(flat_state, action_idx, reward, flat_next, float(done))

                    # Record reward
                    all_rewards.append(reward)
                    episode_reward += reward
                    step += 1
                    global_step += 1

                    # Learn from replay buffer
                    if len(self.replay_buffer) >= self.batch_size:
                        self._learn()

                    # Sync target network
                    if global_step % self.target_update == 0:
                        self._target_net.load_state_dict(self._q_net.state_dict())

                    # Decay exploration
                    self.epsilon_decay()

                    obs = next_obs

                    if done:
                        break

                print(
                    f"Episode {episode + 1}/{num_episodes} | "
                    f"Steps: {step} | "
                    f"Reward: {episode_reward:.2f} | "
                    f"Epsilon: {self.epsilon:.4f}"
                )
                
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
                print(f"Restoring best weights so far (Reward: {best_reward:.2f}).")
                self._q_net.load_state_dict(best_weights)
                self._target_net.load_state_dict(best_weights)

        return self._q_net, all_rewards

    def save(self, filepath):
        """Save the Q-network weights to a file."""
        if self._q_net is None:
            raise ValueError("Cannot save weights; network not initialized.")
        torch.save(self._q_net.state_dict(), filepath)
        print(f"Agent weights saved to {filepath}")

    def load(self, filepath, obs_sample):
        """Load Q-network weights and initialize the networks."""
        self._ensure_networks(obs_sample)  # Build network structure first
        self._q_net.load_state_dict(torch.load(filepath, map_location=self.device, weights_only=True))
        self._target_net.load_state_dict(self._q_net.state_dict())
        print(f"Agent weights loaded from {filepath}")

    # ---- internal ---------------------------------------------------------

    def _learn(self):
        """Sample a mini-batch and do one gradient step."""
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(
            self.batch_size
        )
        
        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        # Double DQN target: use online net to pick action, target net to evaluate
        with torch.no_grad():
            q_next_online = self._q_net(next_states)
            best_actions = torch.argmax(q_next_online, dim=1)

            q_next_target = self._target_net(next_states)
            q_target_vals = q_next_target[torch.arange(self.batch_size), best_actions]

            targets = rewards + self.gamma * q_target_vals * (1.0 - dones)

        # Current Q values
        q_values = self._q_net(states)
        q_pred = q_values[torch.arange(self.batch_size), actions]

        # Compute loss
        loss = self.criterion(q_pred, targets)

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self._q_net.parameters(), 1.0)
        
        self.optimizer.step()
