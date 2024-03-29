
class Hand:
    def __init__(self) -> None:
        self.reset()

    def add_card(self, card) -> None:
        self.cards.append(card)
        self.value += card
        if card == 11:
            self.usable_ace += 1
        self.adjust_aces()

    def adjust_aces(self):
        while self.usable_ace and self.value > 21:
            self.value -= 10
            self.usable_ace -= 1

    def count(self) -> int:
        return len(self.cards)

    def __str__(self) -> str:
        return str(self.cards) + " (value: " + str(self.value) + ")"
    
    def __repr__(self) -> str:
        return str(self)
    
    def __len__(self) -> int:
        return len(self.cards)

    def reset(self) -> None:
        self.cards = []
        self.usable_ace = 0
        self.value = 0
