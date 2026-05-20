from abc import ABC, abstractmethod

class Agent(ABC):
    """
    Abstract base class for all agents (random, heuristic, DQN, etc.)
    in the SecurityLogStream environment.

    Subclasses must implement `get_action`.
    """

    name = "Default Agent"

    def __init__(self, env):
        self.env = env
        self.rewards = []

    @abstractmethod
    def get_action(self, state):
        """Return an action given the current environment state."""
        raise NotImplementedError

    def train(self, env, num_episodes):
        """Train the agent. No-op for agents without training."""
        pass

    def epsilon_decay(self):
        """Decay exploration rate. No-op for agents without epsilon."""
        pass
