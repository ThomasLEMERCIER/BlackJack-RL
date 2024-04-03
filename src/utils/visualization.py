import numpy as np
from gym import spaces
import matplotlib.pyplot as plt

def plot_policy_simple_blackjack(policy: np.ndarray, space: spaces.Box, title: str="Policy") -> None:
    """Plots the policy for the simple blackjack environment.

    Args:
        policy (np.ndarray): Policy. shape (num_states, action).
        title (str, optional): Title. Defaults to "Policy".

    """
    plt.figure(figsize=(10, 5))
    shape = (space.high - space.low + 1)
    policy = policy.reshape(shape)
    plt.subplot(1, 2, 1)
    plt.imshow(policy[:, :, 0], origin="lower", cmap="RdYlGn", aspect="auto")
    plt.title(f"No aces")
    plt.xlabel("Dealer's card")
    plt.ylabel("Player's sum")
    plt.xticks(range(shape[1]), range(space.low[1], space.high[1] + 1))
    plt.yticks(range(shape[0]), range(space.low[0], space.high[0] + 1))
    # add a discrete color bar to show the mapping of colors to values and actions in a label
    cbar = plt.colorbar(ticks=[0, 1])
    cbar.ax.set_yticklabels(["Stick", "Hit"])



    plt.subplot(1, 2, 2)
    plt.imshow(policy[:, :, 1], origin="lower", cmap="RdYlGn", aspect="auto")
    plt.title(f"Aces")
    plt.xlabel("Dealer's card")
    plt.ylabel("Player's sum")
    plt.xticks(range(shape[1]), range(space.low[1], space.high[1] + 1))
    plt.yticks(range(shape[0]), range(space.low[0], space.high[0] + 1))
    # add a discrete color bar to show the mapping of colors to values and actions in a label
    cbar = plt.colorbar(ticks=[0, 1])
    cbar.ax.set_yticklabels(["Stick", "Hit"])


    plt.suptitle(title)
    plt.tight_layout()
    plt.show()
    
def plot_ucb_exploration(ucb: np.ndarray, space: spaces.Box, title: str="Policy"):
    plt.figure(figsize=(10, 5))
    shape = (space.high - space.low + 1)
    ucb_stick = ucb[:, 0].reshape(shape)
    ucb_hit = ucb[:, 1].reshape(shape)

    ucb = ucb_stick + ucb_hit

    plt.subplot(1, 2, 1)
    plt.imshow(ucb[:, :, 0], origin="lower", cmap="RdYlGn", aspect="auto")
    plt.title(f"No aces")
    plt.xlabel("Dealer's card")
    plt.ylabel("Player's sum")
    plt.xticks(range(shape[1]), range(space.low[1], space.high[1] + 1))
    plt.yticks(range(shape[0]), range(space.low[0], space.high[0] + 1))
    cbar = plt.colorbar()

    plt.subplot(1, 2, 2)
    plt.imshow(ucb[:, :, 1], origin="lower", cmap="RdYlGn", aspect="auto")
    plt.title(f"Aces")
    plt.xlabel("Dealer's card")
    plt.ylabel("Player's sum")
    plt.xticks(range(shape[1]), range(space.low[1], space.high[1] + 1))
    plt.yticks(range(shape[0]), range(space.low[0], space.high[0] + 1))
    cbar = plt.colorbar()


    plt.suptitle(title)
    plt.tight_layout()
    plt.show()
