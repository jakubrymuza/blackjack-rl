"""Public exposure counts; never inspect hidden ranks to construct features."""

from blackjack_rl.game.cards import Card


class Ledger:
    def __init__(self, rank_count: int):
        self.counts = [0] * rank_count
        self.exposed: set[int] = set()

    def expose(self, card: Card) -> bool:
        if card.uid in self.exposed:
            return False
        self.exposed.add(card.uid)
        self.counts[card.rank] += 1
        return True
