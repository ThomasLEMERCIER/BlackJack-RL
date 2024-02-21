from dataclasses import dataclass

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
class Transition:
    state: int
    action: int
    next_state: int
    reward: float
    done: bool
