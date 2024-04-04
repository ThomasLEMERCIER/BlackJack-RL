import random
import numpy as np

from .exploration_policy import ExplorationPolicy


def argmax(q_values):
    for i in range(len(q_values)):
        if q_values[i] > top:
          top = q_values[i]
          ties = [i]
        elif q_values[i] == top:
          ties.append(i)
    ind = np.random.choice(ties)
    return ind

class UCB(ExplorationPolicy):
    def __init__(self, num_states: int, num_actions: int, c: float=1.4142135623730951, seed: int = None) -> None:
        super().__init__(seed)
        self.num_states = num_states
        self.num_actions = num_actions
        self.c = c
        self.action_counts = np.zeros((self.num_states, self.num_actions))

    def __call__(self, q_values: np.ndarray, state: int) -> int:
        ucb_values = np.zeros(self.num_actions)
        total_counts = np.sum(self.action_counts[state])
        zero_counts_mask = (self.action_counts[state] == 0)

        if np.any(zero_counts_mask):
            action_with_zero_count = argmax(zero_counts_mask)
            self.action_counts[state, action_with_zero_count] += 1
            return action_with_zero_count

        ucb_values = q_values + self.c * np.sqrt(np.log(total_counts) / self.action_counts[state])
        action = argmax(ucb_values)
        self.action_counts[state, action] += 1
        return action
