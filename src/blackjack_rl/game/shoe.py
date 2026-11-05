import random

from blackjack_rl.config import Config
from blackjack_rl.game.cards import Card


class Shoe:
    def __init__(self, config: Config, rng: random.Random):
        self.config = config
        self.rng = rng
        self.size = config.shoe.decks * config.shoe.suits_per_rank * len(config.shoe.ranks)
        self.reserve = 2 * (config.rules.blackjack_total // min(config.shoe.low_values) + 1)
        self.shoe_id = 0
        self.cards: list[Card] = []
        self.drawn = 0

    @property
    def penetration(self):
        return self.drawn / self.size

    def shuffle_reason(self) -> str | None:
        if not self.cards:
            return "initial" if not self.shoe_id else "reserve"
        if self.penetration >= self.config.shoe.penetration_fraction:
            return "cut"
        if len(self.cards) < self.reserve:
            return "reserve"
        return None

    def shuffle(self):
        self.shoe_id += 1
        self.cards = [Card(i, i % len(self.config.shoe.ranks)) for i in range(self.size)]
        self.rng.shuffle(self.cards)
        self.drawn = 0

    def draw(self) -> Card:
        if not self.cards:
            raise RuntimeError("Shoe exhausted mid-round; reserve invariant violated")
        self.drawn += 1
        return self.cards.pop()
