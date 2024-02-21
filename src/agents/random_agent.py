import numpy as np

from .agent import Agent
from ..utils.data_struct import Transition

class RandomAgent(Agent):
    """ An agent that selects actions randomly. """

    def __init__(self, action_space, seed: int=None):
        """ Initialize the agent. """
        super().__init__(seed)
        self.action_space = action_space

    def step(self, transition: Transition) -> None:
        """ The method that is called at each time step. """
        pass

    def act(self, state: np.ndarray) -> int:
        """ The method that is called to select an action. """
        return self.rand.randrange(0, self.action_space.n)
    
    def get_best_action(self, state: np.ndarray) -> int:
        """ Returns the best action for a given state. """
        return self.rand.randrange(0, self.action_space.n)
