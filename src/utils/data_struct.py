from dataclasses import dataclass
import torch
import numpy as np
from collections import namedtuple

@dataclass
class QlearningParameters:
    num_states: int
    num_actions: int
    alpha: float = 0.1
    gamma: float = 0.9

@dataclass
class SarsaParameters:
    num_states: int
    num_actions: int
    alpha: float = 0.1
    gamma: float = 0.9

@dataclass
class MonteCarloParameters:
    num_states: int
    num_actions: int
    alpha: float = 0.1
    gamma: float = 0.9

@dataclass
class DQNParameters:
    batch_size: int = 32
    gamma: float = 0.9
    freq_target_update: int = 100
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
Transition = namedtuple("Transition", ["state", "action", "next_state", "reward", "done"])
