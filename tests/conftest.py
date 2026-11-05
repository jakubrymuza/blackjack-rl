import pytest

from blackjack_rl.config import load_config
from blackjack_rl.game.cards import Card


@pytest.fixture
def config():
    return load_config("configs/experiments/smoke.yaml")


def rig(engine, ranks):
    """Set upcoming cards after reset; never used by a policy."""
    cards = [Card(i, engine.config.shoe.ranks.index(str(rank))) for i, rank in enumerate(ranks)]
    engine.shoe.cards = list(reversed(cards))
    engine.shoe.drawn = 0


def cards(config, ranks):
    return [Card(i, config.shoe.ranks.index(str(rank))) for i, rank in enumerate(ranks)]
