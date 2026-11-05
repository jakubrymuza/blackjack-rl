from bisect import bisect_right

import numpy as np

from blackjack_rl.config import Config
from blackjack_rl.env.actions import Actions


def strategy_action(public: dict, actions: Actions, config: Config) -> int:
    """Configured S17 multi-deck no-split table; other rules use the same reference."""
    total, soft = public["total"], public["soft"]
    up = config.shoe.low_values[public["dealer_rank"]]
    table = config.baseline
    double_table = table.soft_double if soft else table.hard_double
    stand_table = table.soft_stand if soft else table.hard_stand
    if public["initial"] and public["can_double"] and up in double_table.get(total, []):
        return actions.double
    stand_total = table.soft_stand_total if soft else table.stand_threshold
    # A declined soft double on soft 18 falls back to stand when appropriate.
    return (
        actions.stand if total >= stand_total or up in stand_table.get(total, []) else actions.hit
    )


class BaselineAgent:
    names = ("random", "threshold", "basic", "hilo")

    def __init__(self, name: str, config: Config):
        if name not in self.names:
            raise ValueError(f"Unknown baseline {name}")
        self.name, self.config = name, config

    def probabilities(self, observation: dict, public: dict, mask: np.ndarray, actions: Actions):
        probs = np.zeros(actions.size, dtype=float)
        if self.name == "random":
            return mask.astype(float) / mask.sum()
        if public["phase"] == "bet":
            multiple = 1
            if self.name == "hilo":
                bucket = bisect_right(self.config.baseline.count_thresholds, public["true_count"])
                multiple = self.config.baseline.count_bet_multiples[bucket]
            target = self.config.money.minimum_bet_minor * multiple
            amounts = actions.amounts(public["settled"])
            legal = np.flatnonzero(mask)
            eligible = [i for i in legal if amounts[i] <= target]
            action = max(eligible, key=lambda i: amounts[i]) if eligible else legal[0]
        elif self.name == "threshold":
            action = (
                actions.stand
                if public["total"] >= self.config.baseline.stand_threshold
                else actions.hit
            )
        else:
            action = strategy_action(public, actions, self.config)
        if not mask[action]:
            # Restricted control mode may force a configured playing action.
            legal = np.flatnonzero(mask)
            if len(legal) != 1:
                raise ValueError("Baseline selected an illegal action")
            action = legal[0]
        probs[action] = 1
        return probs
