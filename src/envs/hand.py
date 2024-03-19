
class Hand:
    def __init__(self) -> None:
        self.cards = []
        self.aces = 0
        self.value = 0

    def add_card(self, card) -> None:
        self.cards.append(card)
        self.value += card
        if card == 11:
            self.aces += 1
        self.adjust_for_ace()

    def adjust_for_ace(self) -> None:
        while self.aces and self.value > 21:
            self.value -= 10
            self.aces -= 1

    def __str__(self) -> str:
        return str(self.cards) + " (value: " + str(self.value) + ")"
    
    def __repr__(self) -> str:
        return str(self)
    
    def __len__(self) -> int:
        return len(self.cards)

    def reset(self) -> None:
        self.cards = []
        self.aces = 0
        self.value = 0
