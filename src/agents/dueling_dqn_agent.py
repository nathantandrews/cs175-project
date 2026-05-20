import numpy as np
from collections import deque
import random

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
# Dueling DQN with a simple NumPy MLP (no PyTorch / TensorFlow dependency)
# ---------------------------------------------------------------------------

class DuelingMLP:
    """
    A small multi-layer perceptron that outputs Q-values via the dueling
    architecture:  Q(s,a) = V(s) + A(s,a) - mean(A(s,:))

    Layers
    ------
    input  → hidden (ReLU) → hidden (ReLU)
                                ├─ value stream  → V  (scalar)
                                └─ advantage stream → A  (num_actions)

    All weights are updated with vanilla SGD.
    """

    def __init__(self, input_dim, hidden_dim, num_actions, lr=1e-3):
        self.num_actions = num_actions
        self.lr = lr

        # Xavier-style initialisation
        def _init(rows, cols):
            scale = np.sqrt(2.0 / (rows + cols))
            return np.random.randn(rows, cols).astype(np.float32) * scale

        # Shared layers
        self.W1 = _init(input_dim, hidden_dim)
        self.b1 = np.zeros(hidden_dim, dtype=np.float32)
        self.W2 = _init(hidden_dim, hidden_dim)
        self.b2 = np.zeros(hidden_dim, dtype=np.float32)

        # Value stream  (hidden → 1)
        self.Wv = _init(hidden_dim, 1)
        self.bv = np.zeros(1, dtype=np.float32)

        # Advantage stream  (hidden → num_actions)
        self.Wa = _init(hidden_dim, num_actions)
        self.ba = np.zeros(num_actions, dtype=np.float32)

    # ----- forward pass ----------------------------------------------------

    @staticmethod
    def _relu(x):
        return np.maximum(0, x)

    def forward(self, x):
        """Return Q-values for a single state vector *x* (1-D)."""
        h1 = self._relu(x @ self.W1 + self.b1)
        h2 = self._relu(h1 @ self.W2 + self.b2)

        value = h2 @ self.Wv + self.bv          # (1,)
        advantage = h2 @ self.Wa + self.ba       # (num_actions,)

        # Dueling combination
        q = value + advantage - advantage.mean()
        return q

    def forward_batch(self, X):
        """Return Q-values for a batch of states *X* (2-D)."""
        H1 = self._relu(X @ self.W1 + self.b1)
        H2 = self._relu(H1 @ self.W2 + self.b2)

        V = H2 @ self.Wv + self.bv               # (batch, 1)
        A = H2 @ self.Wa + self.ba                # (batch, num_actions)

        Q = V + A - A.mean(axis=1, keepdims=True)
        return Q, H1, H2, V, A

    # ----- backward pass (vanilla SGD) ------------------------------------

    def update(self, states, actions, targets):
        """
        One gradient-descent step.

        Parameters
        ----------
        states  : (batch, input_dim)
        actions : (batch,)  int indices
        targets : (batch,)  TD-target values for the chosen actions
        """
        batch_size = states.shape[0]

        # --- forward ---
        H1 = self._relu(states @ self.W1 + self.b1)
        H2 = self._relu(H1 @ self.W2 + self.b2)
        V = H2 @ self.Wv + self.bv
        A = H2 @ self.Wa + self.ba
        Q = V + A - A.mean(axis=1, keepdims=True)

        # --- loss = 0.5 * (Q[a] - target)^2 ---
        q_pred = Q[np.arange(batch_size), actions.astype(int)]
        error = q_pred - targets                  # (batch,)

        # --- gradient of Q w.r.t. V and A ---
        dQ = np.zeros_like(Q)
        dQ[np.arange(batch_size), actions.astype(int)] = error / batch_size

        # Dueling backprop:  Q = V + A - mean(A)
        dA = dQ - dQ.mean(axis=1, keepdims=True)
        dV = dQ.sum(axis=1, keepdims=True)

        # Advantage head
        dWa = H2.T @ dA
        dba = dA.sum(axis=0)

        # Value head
        dWv = H2.T @ dV
        dbv = dV.sum(axis=0)

        # Shared layer 2
        dH2 = dA @ self.Wa.T + dV @ self.Wv.T
        dH2 = dH2 * (H2 > 0)                     # ReLU grad

        dW2 = H1.T @ dH2
        db2 = dH2.sum(axis=0)

        # Shared layer 1
        dH1 = dH2 @ self.W2.T
        dH1 = dH1 * (H1 > 0)

        dW1 = states.T @ dH1
        db1 = dH1.sum(axis=0)

        # Gradient clipping (max-norm)
        max_norm = 1.0
        for g in [dW1, db1, dW2, db2, dWv, dbv, dWa, dba]:
            norm = np.linalg.norm(g)
            if norm > max_norm:
                g *= max_norm / norm

        # SGD step
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2
        self.Wv -= self.lr * dWv
        self.bv -= self.lr * dbv
        self.Wa -= self.lr * dWa
        self.ba -= self.lr * dba

    def copy_weights_from(self, other):
        """Hard-copy weights from *other* network (for target network)."""
        self.W1 = other.W1.copy()
        self.b1 = other.b1.copy()
        self.W2 = other.W2.copy()
        self.b2 = other.b2.copy()
        self.Wv = other.Wv.copy()
        self.bv = other.bv.copy()
        self.Wa = other.Wa.copy()
        self.ba = other.ba.copy()

# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class DQNAgent(Agent):
    """
    Dueling DQN agent for SecurityLogStream-v1.

    Uses a lightweight NumPy-only MLP with:
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

        # Lazily initialised on the first observation
        self._input_dim = None
        self._q_net = None
        self._target_net = None

        self.hidden_dim = hidden_dim
        self.replay_buffer = ReplayBuffer(capacity=buffer_capacity)

    # ---- lazy init --------------------------------------------------------

    def _ensure_networks(self, obs):
        if self._q_net is not None:
            return
        state = _flatten_obs(obs)
        self._input_dim = state.shape[0]
        self._q_net = DuelingMLP(
            self._input_dim, self.hidden_dim, const.NUM_DISCRETE_ACTIONS, lr=self.alpha
        )
        self._target_net = DuelingMLP(
            self._input_dim, self.hidden_dim, const.NUM_DISCRETE_ACTIONS, lr=self.alpha
        )
        self._target_net.copy_weights_from(self._q_net)

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
            q_values = self._q_net.forward(flat)
            action_idx = int(np.argmax(q_values))
        return const.ACTION_MAP.get(action_idx, const.PASS_ACTION), action_idx

    def epsilon_decay(self):
        """Exponentially decay epsilon towards min_eps."""
        self.epsilon = max(self.min_eps, self.epsilon * self.decay_rate)

    def train(self, env, num_episodes=1):
        """
        Train the agent on the SecurityLogStream environment.

        Because SecurityLogStream is a continuous stream that runs until the
        data is exhausted (``terminated`` is always False, ``truncated`` fires
        once at the end), each "episode" is a full pass through the dataset.

        Parameters
        ----------
        env : gymnasium.Env
        num_episodes : int
            Number of full episodes (environment resets) to train over.

        Returns
        -------
        Q : DuelingMLP
            The trained Q-network.
        all_rewards : list[float]
            Per-step rewards across all episodes (for plotting).
        """
        all_rewards = []
        global_step = 0

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
                    self._target_net.copy_weights_from(self._q_net)

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

        return self._q_net, all_rewards

    def save(self, filepath):
        """Save the Q-network weights to a file."""
        if self._q_net is None:
            raise ValueError("Cannot save weights; network not initialized.")
        weights = {
            "W1": self._q_net.W1, "b1": self._q_net.b1,
            "W2": self._q_net.W2, "b2": self._q_net.b2,
            "Wv": self._q_net.Wv, "bv": self._q_net.bv,
            "Wa": self._q_net.Wa, "ba": self._q_net.ba
        }
        np.savez(filepath, **weights)
        print(f"Agent weights saved to {filepath}")

    def load(self, filepath, obs_sample):
        """Load Q-network weights and initialize the networks."""
        self._ensure_networks(obs_sample)  # Build network structure first
        data = np.load(filepath)
        self._q_net.W1 = data["W1"]
        self._q_net.b1 = data["b1"]
        self._q_net.W2 = data["W2"]
        self._q_net.b2 = data["b2"]
        self._q_net.Wv = data["Wv"]
        self._q_net.bv = data["bv"]
        self._q_net.Wa = data["Wa"]
        self._q_net.ba = data["ba"]
        self._target_net.copy_weights_from(self._q_net)
        print(f"Agent weights loaded from {filepath}")

    # ---- internal ---------------------------------------------------------

    def _learn(self):
        """Sample a mini-batch and do one gradient step."""
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(
            self.batch_size
        )

        # Double DQN target: use online net to pick action, target net to evaluate
        q_next_online = self._q_net.forward_batch(next_states)[0]
        best_actions = np.argmax(q_next_online, axis=1)

        q_next_target = self._target_net.forward_batch(next_states)[0]
        q_target_vals = q_next_target[np.arange(self.batch_size), best_actions]

        targets = rewards + self.gamma * q_target_vals * (1.0 - dones)

        self._q_net.update(states, actions, targets)
