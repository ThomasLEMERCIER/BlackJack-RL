# https://bicyclecards.com/how-to-play/blackjack/
import gym
import random
import numpy as np
from gym import spaces

from .hand import Hand

DECK = [2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10, 11]

class SimpleBlackjack(gym.Env):
    """ Simple Blackjack environment

    Simplified version:
    - Infinite deck (meaning cards are not removed between draws and the probability of drawing a card is constant (the same card can be drawn multiple times))
    - No doubling or splitting
    - No bet
    - No insurance
    - No other players

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

    State space:
    - The player's current sum (2-31) The ace is counted as 11 initially. If the player's sum exceeds 21, the ace is counted as 1.
    - The dealer's face-up card (2-11)
    - Number of aces in the player's hand still counting as 11 (0-1)
    """
    def __init__(self, seed: int=42) -> None:
        super().__init__()
        self.rand = random.Random(seed)

        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(low=np.array([2, 2, 0]), high=np.array([31, 11, 1]), dtype=np.int32)
        self.reward_range = (-1, 1)

        self.current_state = None
        self.player_hand = Hand()
        self.dealer_hand = Hand()

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
        if action == 1:  # Hit
            self.player_hand.add_card(self.draw_card())
            if self.player_hand.value > 21:
                return self.current_state, -1, True, False, {}
            self.current_state = np.array([self.player_hand.value, self.dealer_hand.cards[0], self.player_hand.aces], dtype=np.int32)
            return self.current_state, 0, False, False, {}
        else:  # Stand
            self.dealer_play()
            if self.dealer_hand.value > 21:
                return self.current_state, 1, True, False, {}
            if self.player_hand.value > self.dealer_hand.value:
                return self.current_state, 1, True, False, {}
            elif self.player_hand.value < self.dealer_hand.value:
                return self.current_state, -1, True, False, {}
            else:
                return self.current_state, 0, True, False, {}
                
    def reset(self) -> np.ndarray:
        """ Resets the state of the environment and returns an initial observation.
        Returns:
            observation (np.ndarray): The initial observation of the space.
        """
        self.player_hand.reset()
        self.dealer_hand.reset()
        
        self.player_hand.add_card(self.draw_card())
        self.player_hand.add_card(self.draw_card())

        self.dealer_hand.add_card(self.draw_card())
        self.dealer_hand.add_card(self.draw_card())

        self.current_state = np.array([self.player_hand.value, self.dealer_hand.cards[0], self.player_hand.aces], dtype=np.int32)
        return self.current_state
    
    def draw_card(self) -> int:
        """ Draw a card from the deck
        Returns:
            int: The value of the card
        """
        return self.rand.choice(DECK)
        
    def dealer_play(self) -> None:
        """ The dealer plays
        """
        while self.dealer_hand.value < 17:
            draw = self.draw_card()
            self.dealer_hand.add_card(draw)

    def render(self, mode: str="ansi") -> None:
        """ Renders the environment.
        Args:
            mode (str): The mode to render with. The string must be 'ansi' or 'human'
        """
        assert mode in ["ansi"]
        print(f"Player's hand: {self.player_hand}")
        print(f"Dealer's hand: {self.dealer_hand}")
        print(f"Current state: {self.current_state}")
