"""Additional diagnostics that reuse saved decisions, including uncertainty bands."""

import numpy as np
import pandas as pd
import plotly.express as px

from blackjack_rl.reporting.charts import heatmap, histogram, line
from blackjack_rl.reporting.registry import Chart, register


@register
def research(tables, config):
    evaluations = tables["evaluations"]
    if not evaluations.empty:
        for metric, interval in (
            ("mean_reward", "mean_reward_ci"),
            ("mean_wealth", "mean_wealth_ci"),
            ("ruin_probability", "ruin_ci"),
        ):
            data = evaluations.sort_values("timestep")
            chart = line(f"validation-{metric}-uncertainty", data, "timestep", metric)
            chart.figure.add_scatter(
                x=data.timestep, y=data[interval].map(lambda x: x[0]), name="CI lower"
            )
            chart.figure.add_scatter(
                x=data.timestep,
                y=data[interval].map(lambda x: x[1]),
                name="CI upper",
                fill="tonexty",
            )
            yield chart
    decisions, rounds = tables["decisions"], tables["rounds"]
    if not decisions.empty:
        player = decisions[decisions.phase == "player"].copy()
        if "disagrees_reference" in player:
            for (soft, initial), panel in player.groupby(["soft", "initial"]):
                yield heatmap(
                    f"reference-disagreement-soft-{soft}-initial-{initial}",
                    panel,
                    "dealer_up",
                    "player_total",
                    "disagrees_reference",
                    config.plots.minimum_cell_samples,
                    "Reference is configured S17 no-split strategy; no optimality claim.",
                )
        if not player.empty and player.timestep.nunique() > 1:
            frequency = player.groupby(["timestep", "action"]).size().reset_index(name="count")
            frequency["frequency"] = frequency["count"] / frequency.groupby("timestep")[
                "count"
            ].transform("sum")
            yield line(
                "playing-action-frequency-learning", frequency, "timestep", "frequency", "action"
            )
            eligible = player[player.double_legal].copy()
            if not eligible.empty:
                eligible["chose_double"] = eligible.action == "double"
                double = eligible.groupby("timestep").chose_double.mean().reset_index()
                yield line("double-when-legal-learning", double, "timestep", "chose_double")
        bets = decisions[decisions.phase == "bet"].sort_values(["episode_id", "decision"]).copy()
        if not bets.empty:
            first = ~bets.duplicated(["episode_id", "shoe_id"])
            yield histogram(
                "count-at-first-bet-of-shoe",
                bets[first],
                "running_count",
                config,
                "Public running count should be zero after each fresh shuffle.",
            )
            bets["first_in_shoe"] = first
            bets["bet_initial_units"] = bets.bet / config.money.initial_bankroll_minor
            at_shuffle = (
                bets.groupby("first_in_shoe").bet_initial_units.agg(["mean", "count"]).reset_index()
            )
            yield Chart(
                "bet-at-shuffle", px.bar(at_shuffle, x="first_in_shoe", y="mean"), at_shuffle
            )
    if not rounds.empty:
        data = rounds.copy()
        data["bet_fraction"] = data.opening_stake / data.before
        stats = data.groupby("episode_id").bet_fraction.agg(["mean", "var"]).reset_index()
        components = pd.DataFrame(
            {
                "component": ["mean_within_session_variance", "between_session_mean_variance"],
                "variance": [stats["var"].mean(), stats["mean"].var()],
            }
        )
        yield Chart(
            "bet-variance-decomposition",
            px.bar(components, x="component", y="variance"),
            components,
            "Unweighted descriptive components; session lengths can differ.",
        )
        correlations = []
        for episode_id, group in data.groupby("episode_id"):
            varying = group.bet_fraction.nunique() > 1 and group.profit.nunique() > 1
            correlations.append(
                {
                    "episode_id": episode_id,
                    "correlation": group.bet_fraction.corr(group.profit) if varying else np.nan,
                    "covariance": group.bet_fraction.cov(group.profit)
                    if len(group) > 1
                    else np.nan,
                }
            )
        moments = pd.DataFrame(correlations)
        for metric in ("correlation", "covariance"):
            yield histogram(
                f"bet-return-session-{metric}",
                moments,
                metric,
                config,
                "Association, not a causal wager effect. Undefined statistics are missing.",
            )
        data["count_bin"] = (
            np.floor(data.true_count / config.plots.count_bin_width) * config.plots.count_bin_width
        )
        data["penetration_bin"] = (
            np.floor(data.penetration / config.plots.penetration_bin_width)
            * config.plots.penetration_bin_width
        )
        data["win"] = data.outcome.isin(["win", "natural"])
        for metric in ("profit", "win"):
            yield heatmap(
                f"count-penetration-{metric}",
                data,
                "count_bin",
                "penetration_bin",
                metric,
                config.plots.minimum_cell_samples,
            )
