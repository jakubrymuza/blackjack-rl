from blackjack_rl.config import Config
from blackjack_rl.game.cards import Card, hand_value


def dealer_should_hit(cards: list[Card], config: Config) -> bool:
    total, soft = hand_value(cards, config)
    rules = config.rules
    return total < rules.dealer_stand_total or (
        total == rules.dealer_stand_total and soft and rules.dealer_hits_soft_17
    )
