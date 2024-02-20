import numpy as np
from abc import ABC, abstractmethod

class Agent(ABC):
    """ The base class for all agents. """
    def __init__(self, seed: int=None) -> None:
        """ Initialize the agent. """
        self.seed = seed
        self.generator = np.random.default_rng(seed)

    @abstractmethod
    def step(self, state: np.ndarray, reward: float, done: bool) -> None:
        """ The method that is called at each time step. """
        pass

    @abstractmethod
    def reset(self) -> None:
        """ The method that is called at the end of each episode. """
        pass

    @abstractmethod
    def act(self, state: np.ndarray) -> int:
        """ The method that is called to select an action. """
        pass
