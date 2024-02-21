import random
from abc import ABC, abstractmethod

class ExplorationPolicy(ABC):
    """ The base class for all exploration strategies. """
    def __init__(self, seed: int=None) -> None:
        """ Initialize the exploration strategy. """
        self.rand = random.Random(seed)

    @abstractmethod
    def __call__(self) -> int:
        """ The method that is called to select an action. """
        pass