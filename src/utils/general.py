import numpy as np
from gym import spaces

def state_to_index(state: np.ndarray, space: spaces.Box) -> int:
    """
    Converts a state to an index.

    Args:
        state (np.ndarray): State.
        space (gym.spaces): Environment.

    Returns:
        int: Index.
    """   
    state = state - space.low
    width = space.high - space.low + 1

    return np.ravel_multi_index(state, width)

def index_to_state(index: int, space: spaces.Box) -> tuple:
    """
    Converts an index to a state.

    Args:
        index (int): Index.
        space (gym.spaces): Environment.

    Returns:
        np.ndarray: State.
    """
    width = space.high - space.low + 1
    state = np.unravel_index(index, width)

    return state + space.low

def get_num_states(space: spaces.Box) -> int:
    """
    Returns the number of states.

    Args:
        space (gym.spaces): Environment.

    Returns:
        int: Number of states.
    """
    return np.prod(space.high - space.low + 1)
