"""Phase state machine. Only public_snapshot() is used by observation encoding."""

from __future__ import annotations

import random

from blackjack_rl.config import Config
from blackjack_rl.game.bankroll import Bankroll
from blackjack_rl.game.cards import Card, hand_value, natural
from blackjack_rl.game.ledger import Ledger
from blackjack_rl.game.rules import dealer_should_hit
from blackjack_rl.game.shoe import Shoe


class Engine:
    def __init__(self, config: Config, seed: int | None = None):
        self.config = config
        self.bank = Bankroll(config.money.initial_bankroll_minor)
        self.settled = self.bank.cash
        self.completed = 0
        self.phase = "bet"
        self.player: list[Card] = []
        self.dealer: list[Card] = []
        self.peek_negative = False
        self.opening_stake = 0
        self.doubled = False
        self.initial_double_blocked = False
        self.shoe = Shoe(config, random.Random(seed))
        self.ledger = Ledger(len(config.shoe.ranks))
        self.events: list[dict] = []
        self.last_round: dict | None = None
        self._prepare()

    @property
    def terminal(self):
        return self.phase == "terminal"

    @property
    def running_count(self):
        return sum(
            n * weight for n, weight in zip(self.ledger.counts, self.config.shoe.hilo_weights)
        )

    @property
    def true_count(self):
        per_deck = len(self.config.shoe.ranks) * self.config.shoe.suits_per_rank
        decks = len(self.shoe.cards) / per_deck
        return self.running_count / max(decks, self.config.shoe.true_count_denominator_floor_decks)

    def _event(self, kind: str, **fields):
        self.events.append(
            {"kind": kind, "round": self.completed + 1, "shoe_id": self.shoe.shoe_id, **fields}
        )

    def _prepare(self):
        reason = self.shoe.shuffle_reason()
        if reason:
            penetration = self.shoe.penetration
            self.shoe.shuffle()
            self.ledger = Ledger(len(self.config.shoe.ranks))
            self._event("shuffle", reason=reason, penetration=penetration)
        self.player, self.dealer = [], []
        self.peek_negative = False
        self.phase = "bet"

    def _draw(self, hand: list[Card], visible: bool):
        card = self.shoe.draw()
        hand.append(card)
        self._event(
            "deal",
            card_id=card.uid,
            rank=card.rank if visible else None,
            visible=visible,
            recipient="player" if hand is self.player else "dealer",
        )
        if visible:
            self._expose(card)

    def _expose(self, card: Card):
        if self.ledger.expose(card):
            self._event("expose", card_id=card.uid, rank=card.rank, visible=True)

    def _reveal(self):
        for card in self.dealer:
            self._expose(card)

    def bet(self, amount: int):
        if self.phase != "bet":
            raise ValueError("Bet only during betting phase")
        m = self.config.money
        if amount < m.minimum_bet_minor or amount % m.bet_increment_minor:
            raise ValueError("Invalid minimum or increment")
        self.pre_bet_count, self.pre_bet_true_count = self.running_count, self.true_count
        self.pre_bet_depth = self.shoe.penetration
        self.round_shoe = self.shoe.shoe_id
        self.bank.commit(amount)
        self.opening_stake = amount
        self.doubled = False
        self.initial_double_blocked = False
        self._draw(self.player, True)
        self._draw(self.dealer, True)
        self._draw(self.player, True)
        self._draw(self.dealer, False)
        self.phase = "player"
        if natural(self.dealer, self.config) or natural(self.player, self.config):
            self._reveal()
            if natural(self.dealer, self.config):
                self._settle("push" if natural(self.player, self.config) else "loss")
            else:
                self._settle("natural")
            return
        self.initial_double_blocked = self.bank.cash < self.opening_stake
        up = self.config.shoe.low_values[self.dealer[0].rank]
        ace_low = self.config.shoe.low_values[self.config.shoe.ranks.index("A")]
        self.peek_negative = up in (ace_low, max(self.config.shoe.low_values))

    def can_double(self):
        return (
            self.phase == "player"
            and len(self.player) == self.config.rules.natural_card_count
            and self.bank.cash >= self.opening_stake
        )

    def play(self, action: str):
        if self.phase != "player":
            raise ValueError("Playing action outside player phase")
        if action == "double":
            if not self.can_double():
                raise ValueError("Double is not affordable or not initial hand")
            self.bank.commit(self.opening_stake)
            self.doubled = True
            self._draw(self.player, True)
        elif action == "hit":
            self._draw(self.player, True)
        elif action != "stand":
            raise ValueError(f"Unknown action {action}")
        total, _ = hand_value(self.player, self.config)
        if total > self.config.rules.blackjack_total:
            if self.config.rules.reveal_hole_after_player_bust:
                self._reveal()
            else:
                self._event("discard_unseen", visible=False, rank=None)
            self._settle("loss")
        elif action != "hit" or total == self.config.rules.blackjack_total:
            self._reveal()
            while dealer_should_hit(self.dealer, self.config):
                self._draw(self.dealer, True)
            dealer_total, _ = hand_value(self.dealer, self.config)
            outcome = (
                "win"
                if dealer_total > self.config.rules.blackjack_total or total > dealer_total
                else "push"
                if total == dealer_total
                else "loss"
            )
            self._settle(outcome)

    def _settle(self, outcome: str):
        stake = self.bank.committed
        rules = self.config.rules
        returned = {
            "loss": 0,
            "push": stake,
            "win": stake * (1 + rules.ordinary_win_profit_multiplier),
            "natural": stake
            + stake * rules.blackjack_profit_numerator // rules.blackjack_profit_denominator,
        }[outcome]
        before = self.settled
        self.bank.settle(returned)
        self.settled = self.bank.cash
        self.completed += 1
        pt, soft = hand_value(self.player, self.config)
        # Hidden dealer totals are intentionally omitted after an unrevealed bust.
        revealed = all(c.uid in self.ledger.exposed for c in self.dealer)
        self.last_round = {
            "round": self.completed,
            "shoe_id": self.round_shoe,
            "before": before,
            "after": self.settled,
            "opening_stake": self.opening_stake,
            "total_stake": stake,
            "returned": returned,
            "profit": self.settled - before,
            "outcome": outcome,
            "doubled": self.doubled,
            "player_bust": pt > rules.blackjack_total,
            "player_total": pt,
            "soft": soft,
            "dealer_total": hand_value(self.dealer, self.config)[0] if revealed else None,
            "dealer_up": self.config.shoe.low_values[self.dealer[0].rank],
            "running_count": self.pre_bet_count,
            "true_count": self.pre_bet_true_count,
            "penetration": self.pre_bet_depth,
            "natural": outcome == "natural",
            "double_blocked_by_cash": self.initial_double_blocked,
        }
        self._event("settlement", outcome=outcome, returned=returned)
        self.events[-1]["round"] = self.completed
        if (
            self.settled < self.config.money.minimum_bet_minor
            or self.completed >= self.config.session.rounds
        ):
            self.phase = "terminal"
        else:
            self._prepare()

    def public_snapshot(self) -> dict:
        player = self.player if self.phase == "player" else []
        total, soft = hand_value(player, self.config)
        return {
            "phase": self.phase,
            "player_ranks": [c.rank for c in player],
            "total": total,
            "soft": soft,
            "initial": len(player) == self.config.rules.natural_card_count,
            "dealer_rank": self.dealer[0].rank if self.phase == "player" else None,
            "peek_negative": self.peek_negative if self.phase == "player" else False,
            "settled": self.settled,
            "cash": self.bank.cash,
            "committed": self.bank.committed,
            "completed": self.completed,
            "exposed": list(self.ledger.counts),
            "running_count": self.running_count,
            "true_count": self.true_count,
            "penetration": self.shoe.penetration,
            "unseen": self.shoe.size - sum(self.ledger.counts),
            "can_double": self.can_double(),
            "shoe_id": self.shoe.shoe_id,
        }
