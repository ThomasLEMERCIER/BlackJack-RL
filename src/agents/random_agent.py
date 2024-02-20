import numpy as np

from .agent import Agent

class RandomAgent(Agent):
    """ An agent that selects actions randomly. """

    def __init__(self, action_space, seed: int=None):
        """ Initialize the agent. """
        super().__init__(seed)
        self.action_space = action_space

    def step(self, state: np.ndarray, reward: float, done: bool) -> None:
        """ The method that is called at each time step. """
        pass

    def reset(self) -> None:
        """ The method that is called at the end of each episode. """
        pass

    def act(self, state: np.ndarray) -> int:
        """ The method that is called to select an action. """
        return self.generator.integers(self.action_space.n)
