from dataclasses import dataclass

from blackjack_rl.config import Config


@dataclass(frozen=True)
class Card:
    uid: int
    rank: int


def hand_value(cards: list[Card], config: Config) -> tuple[int, bool]:
    values = config.shoe.low_values
    total = sum(values[c.rank] for c in cards)
    ace = config.shoe.ranks.index("A")
    upgrade = config.shoe.ace_high_value - values[ace]
    usable = any(c.rank == ace for c in cards) and total + upgrade <= config.rules.blackjack_total
    return total + upgrade if usable else total, usable


def natural(cards: list[Card], config: Config) -> bool:
    return (
        len(cards) == config.rules.natural_card_count
        and hand_value(cards, config)[0] == config.rules.blackjack_total
    )
