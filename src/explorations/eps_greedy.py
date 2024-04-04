import numpy as np
from .exploration_policy import ExplorationPolicy

class EpsilonGreedy(ExplorationPolicy):
    def __init__(self, epsilon: float, decay: float=1.0, seed: int = None) -> None:
        super().__init__(seed)
        self.epsilon = epsilon
        self.decay = decay

    def __call__(self, q_values: np.ndarray, state: int) -> int:
        self.epsilon *= self.decay
        if self.rand.random() < self.epsilon:
            return self.rand.randrange(0, len(q_values))
        else:
            return np.argmax(q_values) #self.get_argmax(q_values)
