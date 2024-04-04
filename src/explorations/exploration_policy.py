import random
import numpy as np
from abc import ABC, abstractmethod

class ExplorationPolicy(ABC):
    """ The base class for all exploration strategies. """
    def __init__(self, seed: int=None) -> None:
        """ Initialize the exploration strategy. """
        self.rand = random.Random(seed)

    def argmax(q_values: list):
        """returns argmax with random choice in case of ties"""
        top=0
        for i in range(len(q_values)):
            if q_values[i] > top:
                top = q_values[i]
                ties = [i]
            elif q_values[i] == top:
                ties.append(i)
        ind = np.random.choice(ties)
        return ind

    @abstractmethod
    def __call__(self) -> int:
        """ The method that is called to select an action. """
        pass