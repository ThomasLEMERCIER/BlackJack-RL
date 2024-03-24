import random

DECK = [2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10, 11]

def get_deck(packs) -> list:
    """Returns the list of card values for the number of packs."""
    deck = []
    for _ in range(4*packs):
        deck += [2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10, 11]
    return deck

class Deck:
    def __init__(self, finite_deck: bool=True, packs: int=6, seed: int=42 ) -> None:
        self.rand = random.Random(seed)
        self.packs = packs
        self.finite_deck = finite_deck
        
        if self.finite_deck:
            self.cards = get_deck(self.packs)
            self.shuffle()
        else:
            self.cards = []
        
    def draw(self) -> int:
        if self.finite_deck:
            card = self.cards.pop(0)
        else:
            card = self.rand.choice(DECK)
        return card

    def shuffle(self) -> None:
        if self.finite_deck:
            self.rand.shuffle(self.cards)
    
    def count(self) -> int:
        if self.finite_deck:
            return len(self.cards)
        else:
            return 1

    def __str__(self) -> str:
        if self.finite_deck:
            return f"{self.count()} cards in the deck : {self.cards}"
        else:
            return "Infinite deck"
        
    def __repr__(self) -> str:
        return str(self)

    def reset(self) -> None:
        if self.finite_deck:
            self.cards = get_deck(self.packs)
            self.shuffle()
