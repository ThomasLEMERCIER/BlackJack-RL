import random
import numpy as np
from abc import ABC, abstractmethod

from ..utils.data_struct import Transition

class Agent(ABC):
    """ The base class for all agents. """
    def __init__(self, seed: int=None) -> None:
        """ Initialize the agent. """
        self.rand = random.Random(seed)

    @abstractmethod
    def step(self, transition: Transition) -> None:
        """ The method that is called at each time step. """
        pass

    @abstractmethod
    def act(self, state: np.ndarray) -> int:
        """ The method that is called to select an action. """
        pass

    @abstractmethod
    def get_best_action(self, state: np.ndarray) -> int:
        """ Returns the best action for a given state. """
        pass
