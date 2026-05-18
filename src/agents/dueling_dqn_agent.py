import numpy as np
import random
from agents.agent import Agent

class DQNAgent(Agent):
    """
    Dueling DQN Agent implemented in NumPy.
    Inherits from Agent base class.
    """
    name = "DuelingDQNAgent"

    def __init__(self, env, gamma=0.99, alpha=0.001, epsilon=1.0, decay_rate=0.995, min_eps=0.01):
        super().__init__(env)
        self.gamma = gamma
        self.alpha = alpha
        self.epsilon = epsilon
        self.decay_rate = decay_rate
        self.min_eps = min_eps
        
        self.rewards = []
        
        # State dimension: 3 (system_stats) + 6 (lengths of text fields) = 9
        self.state_dim = 9
        self.action_dim = 6
        self.hidden_dim = 32
        
        # Initialize weights for Dueling architecture
        # Feature layer
        self.W1 = np.random.randn(self.hidden_dim, self.state_dim) * 0.1
        self.b1 = np.zeros((self.hidden_dim, 1))
        
        # Value stream
        self.W_v = np.random.randn(1, self.hidden_dim) * 0.1
        self.b_v = np.zeros((1, 1))
        
        # Advantage stream
        self.W_a = np.random.randn(self.action_dim, self.hidden_dim) * 0.1
        self.b_a = np.zeros((self.action_dim, 1))
        
        # Target network weights
        self.target_W1 = self.W1.copy()
        self.target_b1 = self.b1.copy()
        self.target_W_v = self.W_v.copy()
        self.target_b_v = self.b_v.copy()
        self.target_W_a = self.W_a.copy()
        self.target_b_a = self.b_a.copy()
        
        # Expose weights as Q for main.py
        self.Q = {
            'W1': self.W1, 'b1': self.b1,
            'W_v': self.W_v, 'b_v': self.b_v,
            'W_a': self.W_a, 'b_a': self.b_a
        }
        
        self.replay_buffer = []
        self.buffer_size = 1000
        self.batch_size = 32
        self.update_target_freq = 100
        self.steps = 0
        
        # Mapping from discrete action to risk score based on constants.py
        self.risk_scores = {
            0: 0.0,  # Assume NOOP or similar
            1: 7.0,  # ALERT
            2: 5.0,  # THROTTLE
            3: 8.0,  # BLOCK
            4: 1.0,  # UNBLOCK
            5: 10.0  # ISOLATE
        }

    def _preprocess_state(self, state):
        """Convert the dict state to a fixed-size vector."""
        # System stats (Box(3))
        stats = state.get('system_stats', np.zeros(3))
        
        # Length of text fields as features
        l1 = len(state.get('auth_log', ''))
        l2 = len(state.get('file_events', ''))
        l3 = len(state.get('network_events', ''))
        l4 = len(state.get('process_events', ''))
        l5 = len(state.get('syslog', ''))
        l6 = len(state.get('web_log', ''))
        
        feat = np.concatenate([stats, [l1, l2, l3, l4, l5, l6]])
        return feat.reshape(-1, 1)  # Return as column vector

    def _forward(self, x, target=False):
        """Forward pass through the network."""
        W1 = self.target_W1 if target else self.W1
        b1 = self.target_b1 if target else self.b1
        W_v = self.target_W_v if target else self.W_v
        b_v = self.target_b_v if target else self.b_v
        W_a = self.target_W_a if target else self.W_a
        b_a = self.target_b_a if target else self.b_a
        
        h = np.maximum(0, np.dot(W1, x) + b1)  # ReLU
        v = np.dot(W_v, h) + b_v
        a = np.dot(W_a, h) + b_a
        
        # Dueling combination: Q = V + (A - mean(A))
        q = v + (a - np.mean(a, axis=0, keepdims=True))
        return q, h, v, a

    def get_action(self, state):
        """Return action based on epsilon-greedy policy."""
        x = self._preprocess_state(state)
        if random.random() < self.epsilon:
            action = random.randint(0, self.action_dim - 1)
        else:
            q, _, _, _ = self._forward(x)
            action = int(np.argmax(q.flatten()))
            
        return {
            "action": action,
            "risk_score": np.array([self.risk_scores.get(action, 0.0)], dtype=np.float32)
        }

    def update(self, state, action_dict, reward, next_state, done):
        """Update network weights using experience replay."""
        action = action_dict['action']
        x = self._preprocess_state(state)
        x_next = self._preprocess_state(next_state)
        
        self.replay_buffer.append((x, action, reward, x_next, done))
        if len(self.replay_buffer) > self.buffer_size:
            self.replay_buffer.pop(0)
            
        if len(self.replay_buffer) < self.batch_size:
            return
            
        batch = random.sample(self.replay_buffer, self.batch_size)
        
        # Prepare batch data
        xs = np.hstack([b[0] for b in batch])
        actions = [b[1] for b in batch]
        rewards = np.array([b[2] for b in batch]).reshape(1, -1)
        xs_next = np.hstack([b[3] for b in batch])
        dones = np.array([b[4] for b in batch]).reshape(1, -1)
        
        # Compute target Q values
        q_next, _, _, _ = self._forward(xs_next, target=True)
        max_q_next = np.max(q_next, axis=0, keepdims=True)
        target_q = rewards + self.gamma * max_q_next * (1 - dones)
        
        # Forward pass
        q, h, v, a = self._forward(xs)
        
        # Compute gradients
        dq = np.zeros_like(q)
        for i in range(self.batch_size):
            dq[actions[i], i] = q[actions[i], i] - target_q[0, i]
            
        dv = np.sum(dq, axis=0, keepdims=True)
        da = dq - np.mean(dq, axis=0, keepdims=True)
        
        # Gradients for Value stream
        dW_v = np.dot(dv, h.T)
        db_v = np.sum(dv, axis=1, keepdims=True)
        dh_v = np.dot(self.W_v.T, dv)
        
        # Gradients for Advantage stream
        dW_a = np.dot(da, h.T)
        db_a = np.sum(da, axis=1, keepdims=True)
        dh_a = np.dot(self.W_a.T, da)
        
        dh = dh_v + dh_a
        dh[h <= 0] = 0  # ReLU gradient
        
        # Gradients for Feature layer
        dW1 = np.dot(dh, xs.T)
        db1 = np.sum(dh, axis=1, keepdims=True)
        
        # Update weights
        self.W1 -= self.alpha * dW1 / self.batch_size
        self.b1 -= self.alpha * db1 / self.batch_size
        self.W_v -= self.alpha * dW_v / self.batch_size
        self.b_v -= self.alpha * db_v / self.batch_size
        self.W_a -= self.alpha * dW_a / self.batch_size
        self.b_a -= self.alpha * db_a / self.batch_size
        
        self.steps += 1
        if self.steps % self.update_target_freq == 0:
            self.target_W1 = self.W1.copy()
            self.target_b1 = self.b1.copy()
            self.target_W_v = self.W_v.copy()
            self.target_b_v = self.b_v.copy()
            self.target_W_a = self.W_a.copy()
            self.target_b_a = self.b_a.copy()

    def epsilon_decay(self):
        """Decay exploration rate."""
        self.epsilon = max(self.min_eps, self.epsilon * self.decay_rate)
