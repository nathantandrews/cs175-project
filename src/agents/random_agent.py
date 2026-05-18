from agents.agent import Agent

class RandomAgent(Agent):
    """
    An agent that takes random actions.
    """
    name = "Random Agent"

    def get_action(self, state):
        """Return a random action from the environment's action space."""
        return self.env.action_space.sample()

    def train(self, env, num_episodes):
        """Random agent doesn't train. Returns dummy values."""
        print(f"Random Agent: No training required for {num_episodes} episodes.")
        return None, []