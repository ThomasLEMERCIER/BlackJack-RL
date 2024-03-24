# https://bicyclecards.com/how-to-play/blackjack/
import gym
import random
import numpy as np
from gym import spaces

from .hand import Hand
from .deck import Deck

DECK = [2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10, 11]


class BaseBlackjack(gym.Env):
    """Blackjack environment    

    Rules:
    - The dealer is dealt one face-up card and one face-down card.
    - The player is dealt two face-up cards.

    Those states are not visible in the environment, as the agent has no action to take at that point.
    - If the player has a blackjack (an ace and a ten-value card) the player wins.
    - If the dealer has a blackjack the player loses.

    - The player can request additional cards (hit) until they decide to stop (stand) or their sum exceeds 21 (bust).
    - If the player's sum exceeds 21, the player busts and loses.

    - After the player stands, the dealer reveals their face-down card, and draws until their sum is 17 or greater.

    - If the dealer busts, the player wins.
    - If the dealer does not bust, the higher sum wins.    

    Actions:
    - 0: Stand
    - 1: Hit

    Rewards:
    - +1: Player wins
    - -1: Player loses
    - 0: Draw
    """

    def __init__(self, seed: int=42, finite_deck: bool=False, packs: int=6) -> None:
        super().__init__()
        self.game_deck = Deck(finite_deck, packs, seed)
        
        self.action_space = spaces.Discrete(2)
        self.reward_range = (-1, 1)

        self.current_state = None
        self.player_hand = Hand()
        self.dealer_hand = Hand()

    def get_current_state(self):
        """ Returns current state depending on chosen environement"""
        pass

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        """ Run one timestep of the environment's dynamics.
        Args:
            action (int): The action to be executed
        Returns:
            observation (np.ndarray): Agent's observation of the current environment
            reward (float) : Amount of reward returned after previous action
            terminated (bool): Whether the episode has ended, in which case further step() calls will return undefined results
            truncated (bool): Whether the episode was truncated
            info (dict): Contains auxiliary diagnostic information (helpful for debugging, and sometimes learning)
        """
        if self.game_deck.count() > 0 and action == 1:  # Hit
            self.player_hand.add_card(self.draw_card())
            if self.player_hand.value > 21:
                self.current_state = self.get_current_state()
                return self.current_state, -1, True, False, {}
            self.current_state = self.get_current_state()
            return self.current_state, 0, False, False, {}
        else:  # Stand
            self.dealer_play()
            self.current_state = self.get_current_state()
            reward = self.verify_winner()
            return self.current_state, reward, True, False, {}

    def draw_card(self) -> int:
        """ Draw a card from the deck """
        return self.game_deck.draw()
        
    def dealer_play(self) -> None:
        """ The dealer plays """
        while self.dealer_hand.value < 17:
            draw = self.draw_card()
            self.dealer_hand.add_card(draw)

    def verify_winner(self) -> int:
        """ Returns the reward after the dealer has played """
        if self.dealer_hand.value > 21 :
            return 1
        elif self.player_hand.value > self.dealer_hand.value:
            return 1
        elif self.player_hand.value < self.dealer_hand.value:
            return -1
        else:
            return 0

    def reset(self) -> np.ndarray:
        """ Resets the state of the environment and returns an initial observation. """
        self.player_hand.reset()
        self.dealer_hand.reset()
        self.game_deck.reset()

        self.player_hand.add_card(self.draw_card())
        self.player_hand.add_card(self.draw_card())

        self.dealer_hand.add_card(self.draw_card())
        self.dealer_hand.add_card(self.draw_card())

        if self.dealer_hand.value == 21 or self.player_hand.value == 21:
            return self.reset()

        self.current_state = self.get_current_state()
        return self.current_state

    def render(self, mode: str="ansi") -> None:
        """ Renders the environment. """
        assert mode in ["ansi", "hidden"]
        if mode == "hidden":
            print("")
            print(f"Player's hand: {self.player_hand}")
            print(f"Dealer's hand: [{self.dealer_hand.cards[0]}, ?]")
            print("")
        else:
            print("")
            print(f"Player's hand: {self.player_hand}")
            print(f"Dealer's hand: {self.dealer_hand}")
            print(f"Current state: {self.current_state}")
            print("")

class InfiniteSimpleBlackjack(BaseBlackjack):
    """ Infinite Blackjack environment

    Simplified rules:
    - Infinite deck (cards are not removed between draws and the probability of drawing a card is constant)
    - No doubling or splitting, no betting, no insurance
    - No other players  

    State space:
    - The player's current sum (2-31) The ace is counted as 11 initially. If the player's sum exceeds 21, the ace is counted as 1.
    - The dealer's face-up card (2-11)
    - If the player has a usable ace (0-1)
    """

    def __init__(self, seed: int=42) -> None:
        super().__init__(seed)
        self.observation_space = spaces.Box(low=np.array([2, 2, 0]), high=np.array([31, 11, 1]), dtype=np.int32)

    def get_current_state(self):
        return np.array([self.player_hand.value, self.dealer_hand.cards[0], self.player_hand.usable_ace], dtype=np.int32)


class SimpleBlackjack(BaseBlackjack):
    """  Simple Blackjack environment

    Simplified rules:
    - No doubling or splitting, no betting, no insurance
    - No other players  

    State space:
    - The player's current sum (2-31) The ace is counted as 11 initially. If the player's sum exceeds 21, the ace is counted as 1.
    - The dealer's face-up card (2-11)
    - If the player has a usable ace (0-1)
    - The number of cards in the player's hand (2-22)
    """

    def __init__(self, seed: int=42, packs: int=6) -> None:
        super().__init__(seed, finite_deck = True, packs=packs)
        self.observation_space = spaces.Box(low=np.array([2, 2, 0, 2]), high=np.array([31, 11, 1, 22]), dtype=np.int32)

    def get_current_state(self):
        return np.array([self.player_hand.value, self.dealer_hand.cards[0], self.player_hand.usable_ace, self.player_hand.count()], dtype=np.int32)


class Blackjack(BaseBlackjack):
    """ Blackjack environment

    State space:
    - The dealer's face-up card (2-11)
    - The player's cards represented as a list of integers (2-11)
    """

    def __init__(self, seed: int=42, packs: int=6) -> None:
        super().__init__(seed, finite_deck = True, packs = packs)

        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Dict({
            "dealer": spaces.Box(low=np.array([2]), high=np.array([11]), dtype=np.int32),
            "player": spaces.Sequence(spaces.Box(low=np.array([2]), high=np.array([11]), dtype=np.int32))
        })

    def get_current_state(self):
        return { "dealer": self.dealer_hand.cards[0], "player": self.player_hand.cards }
