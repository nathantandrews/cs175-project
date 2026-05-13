class HeuristicAgent:
    def __init__(self, env):
        self.env = env

    def act(self, observation):
        # Implement a simple heuristic based on the environment's observation
        # For example, if the observation is a vector of features, we can use a simple rule
        # Here, we will just return a random action for demonstration purposes
        return self.env.action_space.sample()