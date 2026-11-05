"""Strict configuration composition with no hidden behavioral defaults."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, PositiveFloat, PositiveInt, root_validator


class StrictModel(BaseModel):
    class Config:
        extra = "forbid"
        validate_assignment = True
        allow_inf_nan = False


class Money(StrictModel):
    minor_units_per_dollar: PositiveInt
    initial_bankroll_minor: PositiveInt
    minimum_bet_minor: PositiveInt
    bet_increment_minor: PositiveInt


class Session(StrictModel):
    rounds: PositiveInt


class Rules(StrictModel):
    blackjack_total: PositiveInt
    dealer_stand_total: PositiveInt
    dealer_hits_soft_17: bool
    natural_card_count: PositiveInt
    blackjack_profit_numerator: PositiveInt
    blackjack_profit_denominator: PositiveInt
    ordinary_win_profit_multiplier: PositiveInt
    double_stake_multiplier: PositiveInt
    double_draw_count: PositiveInt
    double_on_initial_hand_only: bool
    dealer_peeks_for_blackjack: bool
    allow_split: bool
    allow_surrender: bool
    allow_insurance: bool
    reveal_hole_after_player_bust: bool

    @root_validator(skip_on_failure=True)
    def supported(cls, v):
        if any(v[k] for k in ("allow_split", "allow_surrender", "allow_insurance")):
            raise ValueError("Splitting, surrender, and insurance are not implemented")
        if not v["dealer_peeks_for_blackjack"] or not v["double_on_initial_hand_only"]:
            raise ValueError("Only peek and initial-hand-only doubling are supported")
        if (v["natural_card_count"], v["double_stake_multiplier"], v["double_draw_count"]) != (
            2,
            2,
            1,
        ):
            raise ValueError("Only two-card naturals and one-card, equal-stake doubling supported")
        if v["dealer_stand_total"] > v["blackjack_total"]:
            raise ValueError("Dealer stand total exceeds target")
        return v


class ShoeConfig(StrictModel):
    decks: PositiveInt
    suits_per_rank: PositiveInt
    ranks: list[str]
    low_values: list[PositiveInt]
    ace_high_value: PositiveInt
    penetration_fraction: float
    hilo_weights: list[int]
    true_count_denominator_floor_decks: PositiveFloat

    @root_validator(skip_on_failure=True)
    def valid(cls, v):
        if not 0 <= v["penetration_fraction"] < 1:
            raise ValueError("penetration_fraction must be in [0, 1)")
        if len(v["ranks"]) != len(set(v["ranks"])) or "A" not in v["ranks"]:
            raise ValueError("Ranks must be unique and include A")
        if not len(v["ranks"]) == len(v["low_values"]) == len(v["hilo_weights"]):
            raise ValueError("Rank, value, and Hi-Lo arrays must have matching lengths")
        if v["ace_high_value"] <= v["low_values"][v["ranks"].index("A")]:
            raise ValueError("High ace value must exceed low ace value")
        return v


class Betting(StrictModel):
    minimum_bet_multiples: list[PositiveInt]
    bankroll_fractions: list[PositiveFloat]

    @root_validator(skip_on_failure=True)
    def valid(cls, v):
        if 1 not in v["minimum_bet_multiples"] or 1 not in v["bankroll_fractions"]:
            raise ValueError("Bet ladder must include minimum and full-bankroll entries")
        if any(x > 1 for x in v["bankroll_fractions"]):
            raise ValueError("Bankroll fractions cannot exceed one")
        return v


class Features(StrictModel):
    counting: Literal["none", "hilo", "full"]
    running_count_scale: PositiveFloat
    true_count_scale: PositiveFloat
    control: Literal["joint", "play", "betting"]


class Reward(StrictModel):
    ruin_weight: float
    shortfall_weight: float
    negative_cash_weight: float

    @root_validator(skip_on_failure=True)
    def valid(cls, v):
        if any(x < 0 for x in v.values()):
            raise ValueError("Penalty weights must be nonnegative")
        return v


class Training(StrictModel):
    seeds: list[int]
    total_timesteps: PositiveInt
    n_envs: PositiveInt
    vectorization: Literal["dummy", "subproc"]
    n_steps: PositiveInt
    batch_size: PositiveInt
    n_epochs: PositiveInt
    learning_rate: PositiveFloat
    gamma: float
    gae_lambda: float
    clip_range: PositiveFloat
    ent_coef: float
    vf_coef: float
    max_grad_norm: PositiveFloat
    hidden_sizes: list[PositiveInt]
    eval_interval: PositiveInt
    checkpoint_interval: PositiveInt
    log_interval: PositiveInt
    torch_threads: PositiveInt
    device: str
    progress_bar: bool
    final_test: bool
    run_root: str

    @root_validator(skip_on_failure=True)
    def valid(cls, v):
        rollout = v["n_envs"] * v["n_steps"]
        if v["batch_size"] <= 1 or rollout % v["batch_size"]:
            raise ValueError("batch_size must exceed one and divide n_envs * n_steps")
        if not 0 < v["gamma"] <= 1 or not 0 <= v["gae_lambda"] <= 1:
            raise ValueError("Invalid gamma or gae_lambda")
        if not v["seeds"] or len(set(v["seeds"])) != len(v["seeds"]):
            raise ValueError("Training seeds must be nonempty and unique")
        return v


class Evaluation(StrictModel):
    sessions: PositiveInt
    validation_sessions: PositiveInt
    trace_sessions: int
    validation_trace_sessions: int
    validation_seed: int
    test_seed: int
    analysis_seed: int
    confidence: float
    bootstrap_resamples: PositiveInt
    tail_quantile: float
    quantiles: list[float]
    deterministic: bool

    @root_validator(skip_on_failure=True)
    def valid(cls, v):
        if not 0 < v["confidence"] < 1 or not 0 < v["tail_quantile"] < 1:
            raise ValueError("Confidence and tail quantile must be in (0, 1)")
        if any(not 0 <= q <= 1 for q in v["quantiles"]):
            raise ValueError("Quantiles must lie in [0, 1]")
        if min(v["trace_sessions"], v["validation_trace_sessions"]) < 0:
            raise ValueError("Trace session counts cannot be negative")
        if len({v[k] for k in ("validation_seed", "test_seed", "analysis_seed")}) != 3:
            raise ValueError("Evaluation/analysis seeds must be distinct")
        return v


class Baseline(StrictModel):
    stand_threshold: PositiveInt
    hard_stand: dict[int, list[int]]
    hard_double: dict[int, list[int]]
    soft_stand_total: PositiveInt
    soft_stand: dict[int, list[int]]
    soft_double: dict[int, list[int]]
    count_thresholds: list[float]
    count_bet_multiples: list[PositiveInt]

    @root_validator(skip_on_failure=True)
    def valid(cls, v):
        if sorted(v["count_thresholds"]) != v["count_thresholds"]:
            raise ValueError("Count thresholds must be sorted")
        if len(v["count_bet_multiples"]) != len(v["count_thresholds"]) + 1:
            raise ValueError("Count spread needs one more multiple than thresholds")
        return v


class Plots(StrictModel):
    histogram_bins: PositiveInt
    scatter_sample: PositiveInt
    minimum_cell_samples: PositiveInt
    smoothing_window: PositiveInt
    count_bin_width: PositiveFloat
    penetration_bin_width: PositiveFloat
    bankroll_bin_width: PositiveFloat
    horizon_bin_width: PositiveInt
    bet_bin_width: PositiveFloat
    figure_width: PositiveFloat
    figure_height: PositiveFloat
    dpi: PositiveInt
    static_formats: list[Literal["png", "svg", "pdf"]]


class Telemetry(StrictModel):
    schema_version: PositiveInt
    artifact_id_characters: PositiveInt
    flush_rows: PositiveInt
    training_trace_every_sessions: PositiveInt


class Config(StrictModel):
    money: Money
    session: Session
    rules: Rules
    shoe: ShoeConfig
    betting: Betting
    features: Features
    reward: Reward
    evaluation: Evaluation
    baseline: Baseline
    plots: Plots
    telemetry: Telemetry
    training: Training | None = None

    @root_validator(skip_on_failure=True)
    def valid(cls, v):
        m, r, s = v["money"], v["rules"], v["shoe"]
        evaluation = v["evaluation"]
        reserved_seeds = {
            evaluation.validation_seed,
            evaluation.test_seed,
            evaluation.analysis_seed,
        }
        if v.get("training") and reserved_seeds.intersection(v["training"].seeds):
            raise ValueError("Training seeds must differ from evaluation and analysis seeds")
        if m.initial_bankroll_minor < m.minimum_bet_minor:
            raise ValueError("Initial bankroll must afford minimum bet")
        if m.minimum_bet_minor % m.bet_increment_minor:
            raise ValueError("Minimum bet must be a multiple of the increment")
        if m.bet_increment_minor * r.blackjack_profit_numerator % r.blackjack_profit_denominator:
            raise ValueError("Bet increment must permit exact natural payouts")
        reserve = 2 * (r.blackjack_total // min(s.low_values) + 1)
        if s.decks * s.suits_per_rank * len(s.ranks) < reserve:
            raise ValueError("Shoe is smaller than conservative round reserve")
        return v

    @property
    def config_hash(self):
        return hashlib.sha256(self.json(sort_keys=True).encode()).hexdigest()


def merge(base: dict, override: dict) -> dict:
    result = dict(base)
    for key, value in override.items():
        result[key] = (
            merge(result[key], value)
            if isinstance(value, dict) and isinstance(result.get(key), dict)
            else value
        )
    return result


def read_composed(path: Path, parents: tuple[Path, ...] = ()) -> dict:
    path = path.resolve()
    if path in parents:
        raise ValueError(f"Configuration include cycle at {path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Expected YAML mapping in {path}")
    result = {}
    for include in data.pop("include", []):
        result = merge(result, read_composed(path.parent / include, (*parents, path)))
    return merge(result, data)


def load_config(path: str | Path, overrides: list[str] = ()) -> Config:
    path = Path(path)
    data = json.loads(path.read_text()) if path.suffix == ".json" else read_composed(path)
    for override in overrides:
        key, separator, raw = override.partition("=")
        if not separator:
            raise ValueError("Overrides must be dotted.key=YAML-value")
        target = data
        parts = key.split(".")
        for part in parts[:-1]:
            target = target.setdefault(part, {})
        target[parts[-1]] = yaml.safe_load(raw)
    return Config.parse_obj(data)
