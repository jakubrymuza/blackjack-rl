import random

import pytest

from blackjack_rl.game.bankroll import Bankroll
from blackjack_rl.game.cards import hand_value, natural
from blackjack_rl.game.engine import Engine
from blackjack_rl.game.ledger import Ledger
from blackjack_rl.game.shoe import Shoe
from conftest import cards, rig


@pytest.mark.parametrize(
    "ranks,total,soft",
    [
        (["A", "A", "9"], 21, True),
        (["A", "6"], 17, True),
        (["A", "6", "10"], 17, False),
        (["A", "A", "9", "K"], 21, False),
        (["K", "Q", "2"], 22, False),
    ],
)
def test_aces(config, ranks, total, soft):
    assert hand_value(cards(config, ranks), config) == (total, soft)


def test_natural(config):
    assert natural(cards(config, ["A", "K"]), config)
    assert not natural(cards(config, ["A", "5", "5"]), config)


@pytest.mark.parametrize(
    "ranks,action,expected,outcome",
    [
        (["A", "9", "K", "7"], None, 11500, "natural"),
        (["A", "A", "K", "K"], None, 10000, "push"),
        (["9", "A", "8", "K"], None, 9000, "loss"),
        (["10", "10", "8", "8"], "stand", 10000, "push"),
        (["10", "10", "9", "8"], "stand", 11000, "win"),
        (["10", "10", "7", "8"], "stand", 9000, "loss"),
        (["5", "10", "6", "8", "K"], "double", 12000, "win"),
        (["5", "10", "6", "K", "5"], "double", 8000, "loss"),
        (["5", "10", "6", "7", "6"], "double", 10000, "push"),
    ],
)
def test_settlements(config, ranks, action, expected, outcome):
    config.session.rounds = 1
    e = Engine(config, 1)
    rig(e, ranks)
    e.bet(config.money.minimum_bet_minor)
    if action:
        e.play(action)
    assert e.terminal
    assert e.bank.cash == e.settled == expected
    assert e.bank.committed == 0
    assert e.last_round["outcome"] == outcome
    assert e.completed == 1


def test_double_and_all_in(config):
    e = Engine(config, 1)
    rig(e, ["5", "10", "6", "8", "K"])
    e.bet(config.money.initial_bankroll_minor)
    assert e.bank.cash == 0 and not e.terminal
    assert not e.can_double()
    with pytest.raises(ValueError):
        e.play("double")
    e.play("hit")
    assert e.settled == 20000
    assert e.last_round["double_blocked_by_cash"]


def test_no_double_after_hit(config):
    e = Engine(config, 1)
    rig(e, ["2", "10", "3", "8", "2"])
    e.bet(config.money.minimum_bet_minor)
    e.play("hit")
    assert not e.can_double()


def test_hole_not_counted_after_bust(config):
    e = Engine(config, 1)
    rig(e, ["10", "5", "8", "K", "10"])
    e.bet(config.money.minimum_bet_minor)
    hidden = e.dealer[-1]
    assert e.ledger.counts[hidden.rank] == 0
    assert sum(e.ledger.counts) == 3
    e.play("hit")
    assert hidden.uid not in e.ledger.exposed
    assert e.last_round["dealer_total"] is None


def test_soft17(config):
    config.session.rounds = 1

    def run(hit):
        config.rules.dealer_hits_soft_17 = hit
        e = Engine(config, 1)
        rig(e, ["10", "A", "8", "6", "4"])
        e.bet(config.money.minimum_bet_minor)
        assert e.peek_negative
        e.play("stand")
        return e.settled

    assert run(False) == 11000
    assert run(True) == 9000


def test_shoe_and_ledger(config):
    shoe = Shoe(config, random.Random(1))
    shoe.shuffle()
    deck = [shoe.draw() for _ in range(shoe.size)]
    assert len({c.uid for c in deck}) == shoe.size
    assert all(
        sum(c.rank == r for c in deck) == config.shoe.decks * config.shoe.suits_per_rank
        for r in range(len(config.shoe.ranks))
    )
    with pytest.raises(RuntimeError):
        shoe.draw()
    ledger = Ledger(len(config.shoe.ranks))
    assert ledger.expose(deck[0])
    assert not ledger.expose(deck[0])
    assert sum(ledger.counts) == 1


def test_shuffle_before_bet_and_count_reset(config):
    config.shoe.penetration_fraction = 0
    e = Engine(config, 1)
    rig(e, ["A", "9", "K", "7"])
    shoe_id = e.shoe.shoe_id
    e.bet(config.money.minimum_bet_minor)
    assert e.phase == "bet" and e.shoe.shoe_id == shoe_id + 1
    assert e.running_count == 0 and not any(e.ledger.counts)


def test_reserve(config):
    e = Engine(config, 1)
    e.shoe.cards = e.shoe.cards[: e.shoe.reserve - 1]
    e._prepare()
    assert e.shoe.shoe_id == 2
    assert e.events[-1]["reason"] == "reserve"


def test_cash_guard():
    bank = Bankroll(1000)
    with pytest.raises(ValueError):
        bank.commit(1001)
    bank.commit(1000)
    bank.settle(0)
    assert bank.cash == 0
