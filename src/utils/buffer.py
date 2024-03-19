from collections import deque
from .data_struct import Transition
import random


class ReplayBuffer:
    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self.memory = deque(maxlen=capacity)

    def push(self, transition: Transition) -> None:
        self.memory.append(transition)

    def sample(self, batch_size: int):
        samples = random.sample(self.memory, batch_size)
        return Transition(*zip(*samples))

    def __len__(self):
        return len(self.memory)