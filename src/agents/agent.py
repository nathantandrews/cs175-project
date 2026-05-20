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

    def evaluate(self, env):
        """
        Run a single evaluation episode with epsilon=0 (pure exploitation).
        
        Returns the total reward and the number of steps.
        """
        # Save epsilon if the agent uses it
        old_epsilon = getattr(self, "epsilon", None)
        if old_epsilon is not None:
            self.epsilon = 0.0

        obs, info = env.reset()
        episode_reward = 0.0
        steps = 0
        done = False

        while not done:
            action_res = self.get_action(obs)
            if isinstance(action_res, tuple):
                env_action = action_res[0]
            else:
                env_action = action_res
                
            next_obs, reward, terminated, truncated, info = env.step(env_action)
            episode_reward += reward
            steps += 1
            obs = next_obs
            done = terminated or truncated

        # Restore original epsilon
        if old_epsilon is not None:
            self.epsilon = old_epsilon

        return episode_reward, steps
