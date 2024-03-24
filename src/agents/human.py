import numpy as np

from .agent import Agent
from ..utils.data_struct import Transition

class HumanAgent(Agent):
    """ An agent that selects actions based on user input. """
    def __init__(self, seed: int=None) -> None:
        """ Initialize the agent. """
        super().__init__(seed)

    def step(self, transition: Transition) -> None:
        """ The method that is called at each time step. """
        pass

    def act(self, state: np.ndarray) -> int:
        """ The method that is called to select an action. """
        return int(input("Enter action (0: Stick, 1: Hit): "))

    def get_best_action(self, state: np.ndarray) -> int:
        """ Returns the best action for a given state. """
        pass
