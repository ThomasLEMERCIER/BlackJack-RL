import numpy as np
from gym import spaces
import torch

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


def state_to_array_encoding(state: dict, space: spaces.Dict) -> torch.Tensor:
    """
    Converts a state to a tensor.

    Args:
        state (dict): State. {"dealer": int, "player": list[int]}
        space (gym.spaces): Environment.

    Returns:
        np.ndarray: Tensor.
    """
    n_cards = (space["dealer"].high - space["dealer"].low + 1).item()
    tensor = torch.zeros(n_cards + 1, dtype=torch.float32)
    tensor[-1] = state["dealer"]
    for card in state["player"]:
        tensor[card - space["dealer"].low] += 1
    return tensor

def get_input_dim_encoding(space: spaces.Dict) -> int:
    """
    Returns the input dimension of the encoding.

    Args:
        space (gym.spaces): Environment.

    Returns:
        int: Input dimension.
    """
    return (space["dealer"].high - space["dealer"].low + 2).item()
