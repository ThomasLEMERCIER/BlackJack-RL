import random


def get_deck(packs) -> list:
    """Returns the list of card values for the number of packs."""
    deck = []
    for _ in range(4*packs):
        deck += [2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10, 11]
    return deck

class Deck:
    def __init__(self, packs: int=6) -> None:
        self.cards = get_deck(packs)
        self.packs = packs
        self.reset()

    def draw(self) -> int:
        card = self.cards[self.index]
        self.index += 1
        return card

    def shuffle(self) -> None:
        random.shuffle(self.cards)

    def __str__(self) -> str:
        return str(self.cards) + " (count: " + str(self.count) + ")"

    def __repr__(self) -> str:
        return str(self)

    def reset(self) -> None:
        self.shuffle()
        self.index = 0
