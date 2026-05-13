from abc import ABC, abstractmethod

class Agent(ABC):
    """
    Abstract base class for all agents (random, heuristic, DQN, etc.)
    in the SecurityLogStream environment.

    Subclasses must implement `get_action`.
    """

    name = "BaseAgent"

    def __init__(self, env):
        self.env = env

    @abstractmethod
    def get_action(self, state):
        """Return an action given the current environment state."""
        raise NotImplementedError

    def update(self, state, action, reward, next_state, done):
        """Learning step. No-op for non-learning agents."""
        pass

    def epsilon_decay(self):
        """Decay exploration rate. No-op for agents without epsilon."""
        pass

    def reset(self):
        """Reset any per-episode internal state."""
        pass