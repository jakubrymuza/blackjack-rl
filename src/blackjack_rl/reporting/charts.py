from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from blackjack_rl.reporting.registry import Chart, register


def line(name, data, x, y, color=None, note=""):
    return Chart(name, px.line(data, x=x, y=y, color=color), data, note)


def histogram(name, data, x, config, note=""):
    return Chart(name, px.histogram(data, x=x, nbins=config.plots.histogram_bins), data, note)


def scatter(name, data, x, y, config, color=None, note=""):
    sample = data.sample(
        min(len(data), config.plots.scatter_sample), random_state=config.evaluation.analysis_seed
    )
    return Chart(name, px.scatter(sample, x=x, y=y, color=color), sample, note)


def heatmap(name, data, x, y, value, minimum, note=""):
    grouped = data.groupby([y, x], observed=True)[value].agg(["mean", "count"]).reset_index()
    grouped.loc[grouped["count"] < minimum, "mean"] = np.nan
    pivot = grouped.pivot(index=y, columns=x, values="mean")
    figure = go.Figure(
        go.Heatmap(
            x=pivot.columns.tolist(),
            y=pivot.index.tolist(),
            z=pivot.to_numpy(),
            colorbar={"title": value},
        )
    )
    figure.update_xaxes(title=x)
    figure.update_yaxes(title=y)
    return Chart(
        name, figure, grouped, note + f" Cells with fewer than {minimum} samples are missing."
    )


@register
def learning(tables, config):
    episodes, evaluations, optimization = (
        tables[k] for k in ("training_episodes", "evaluations", "optimization")
    )
    if not episodes.empty:
        episodes = episodes.sort_values("timestep")
        for metric in (
            "reward",
            "wealth",
            "wealth_reward",
            "ruin_penalty",
            "shortfall_penalty",
            "ruin",
            "completion",
            "rounds",
            "max_drawdown",
        ):
            data = episodes[["timestep", "seconds", metric]].copy()
            data["rolling_mean"] = (
                data[metric].rolling(config.plots.smoothing_window, min_periods=1).mean()
            )
            for axis in ("timestep", "seconds"):
                fig = go.Figure()
                fig.add_scatter(x=data[axis], y=data[metric], mode="markers", name="raw sessions")
                fig.add_scatter(x=data[axis], y=data.rolling_mean, name="rolling mean")
                fig.update_xaxes(title=axis)
                fig.update_yaxes(title=metric)
                yield Chart(
                    f"training-{metric}-{axis}",
                    fig,
                    data,
                    f"Trailing window: {config.plots.smoothing_window} completed sessions.",
                )
    if not evaluations.empty:
        evaluations = evaluations.sort_values("timestep")
        for metric in (
            "mean_reward",
            "mean_wealth",
            "ruin_probability",
            "completion_rate",
            "mean_rounds",
            "evaluation_seconds",
        ):
            yield line(f"validation-{metric}", evaluations, "timestep", metric)
    if not optimization.empty:
        for metric in optimization.select_dtypes(include="number").columns:
            if metric not in ("timestep", "seconds"):
                data = optimization[["timestep", metric]].dropna()
                if not data.empty:
                    yield line(f"optimizer-{metric.replace('/', '-')}", data, "timestep", metric)


@register
def bankroll(tables, config):
    episodes, rounds = tables["episodes"], tables["rounds"]
    if episodes.empty:
        return
    for metric in ("wealth", "reward", "max_drawdown", "rounds"):
        yield histogram(f"session-{metric}", episodes, metric, config)
    yield Chart("wealth-ecdf", px.ecdf(episodes, x="wealth"), episodes)
    yield Chart("wealth-box", px.box(episodes, x="agent", y="wealth", points=False), episodes)
    yield Chart("wealth-violin", px.violin(episodes, x="agent", y="wealth", box=True), episodes)
    ruined = episodes[episodes.ruin]
    if not ruined.empty:
        yield histogram("ruin-round", ruined, "rounds", config)
    survival = pd.DataFrame({"round": np.arange(config.session.rounds + 1)})
    survival["survival_probability"] = [
        float((~episodes.ruin | (episodes["rounds"] > r)).mean()) for r in survival["round"]
    ]
    yield line("survival", survival, "round", "survival_probability")
    if rounds.empty:
        return
    rounds = rounds.sort_values(["episode_id", "round"])
    # Include only traced complete sessions; terminal wealth is carried forward.
    ids = set(episodes.episode_id) & set(rounds.episode_id)
    trajectories = rounds.pivot(index="round", columns="episode_id", values="after").reindex(
        range(config.session.rounds + 1)
    )
    trajectories = trajectories[list(ids)]
    trajectories.loc[0] = config.money.initial_bankroll_minor
    trajectories = trajectories.ffill() / config.money.initial_bankroll_minor
    bands = pd.DataFrame({"round": trajectories.index})
    for q in config.evaluation.quantiles:
        bands[f"q{q}"] = trajectories.quantile(q, axis=1).to_numpy()
    melted = bands.melt("round", var_name="quantile", value_name="wealth")
    yield line(
        "bankroll-population-bands",
        melted,
        "round",
        "wealth",
        "quantile",
        "Traced sessions only. Terminal wealth is carried forward after ruin.",
    )
    survivors = rounds.groupby("round").after.mean().reset_index()
    survivors["wealth"] = survivors.after / config.money.initial_bankroll_minor
    yield line(
        "bankroll-active-sessions",
        survivors,
        "round",
        "wealth",
        note="Conditional on reaching this round; subject to survivorship bias.",
    )
    outcomes = (
        rounds.groupby("outcome")
        .agg(count=("outcome", "size"), profit=("profit", "sum"))
        .reset_index()
    )
    for metric in ("count", "profit"):
        yield Chart(f"outcome-{metric}", px.bar(outcomes, x="outcome", y=metric), outcomes)
    flags = pd.DataFrame(
        {
            "event": ["natural", "doubled", "player_bust"],
            "rate": [rounds[k].mean() for k in ("natural", "doubled", "player_bust")],
        }
    )
    yield Chart("hand-event-rates", px.bar(flags, x="event", y="rate"), flags)
    rounds = rounds.assign(
        return_fraction=rounds.profit / rounds.before,
        remaining_rounds=config.session.rounds - rounds["round"],
    )
    for x in ("before", "remaining_rounds"):
        yield scatter(f"round-return-vs-{x}", rounds, x, "return_fraction", config)


@register
def decisions(tables, config):
    decisions = tables["decisions"]
    if decisions.empty:
        return
    for phase, data in decisions.groupby("phase"):
        for metric in ("entropy", "normalized_entropy", "chosen_probability"):
            yield histogram(f"{phase}-{metric}", data, metric, config)
        if "timestep" in data and data.timestep.nunique() > 1:
            means = data.groupby("timestep")[["entropy", "normalized_entropy"]].mean().reset_index()
            for metric in ("entropy", "normalized_entropy"):
                yield line(f"{phase}-{metric}-learning", means, "timestep", metric)
    player = decisions[decisions.phase == "player"].copy()
    if player.empty:
        return
    actions = ("hit", "stand", "double")
    bet_count = len(config.betting.minimum_bet_multiples) + len(config.betting.bankroll_fractions)
    for offset, action in enumerate(actions):
        player[f"p_{action}"] = player.probabilities.map(lambda p: p[bet_count + offset])
        player[f"chose_{action}"] = player.action == action
    player["count_bin"] = (
        np.floor(player.true_count / config.plots.count_bin_width) * config.plots.count_bin_width
    )
    player["wealth_bin"] = (
        np.floor(player.wealth / config.plots.bankroll_bin_width) * config.plots.bankroll_bin_width
    )
    player["horizon_bin"] = (
        player.remaining_rounds // config.plots.horizon_bin_width * config.plots.horizon_bin_width
    )
    for (soft, initial), panel in player.groupby(["soft", "initial"]):
        label = f"{'soft' if soft else 'hard'}-{'initial' if initial else 'later'}"
        for action in actions:
            yield heatmap(
                f"policy-{label}-{action}",
                panel,
                "dealer_up",
                "player_total",
                f"p_{action}",
                config.plots.minimum_cell_samples,
                "Mean policy probabilities on observed reachable public states.",
            )
        visits = panel.groupby(["dealer_up", "player_total"]).size().reset_index(name="visits")
        yield heatmap(f"visitation-{label}", visits, "dealer_up", "player_total", "visits", 1)
    for axis in ("count_bin", "wealth_bin", "horizon_bin", "committed"):
        means = player.groupby(axis)[[f"p_{a}" for a in actions]].mean().reset_index()
        counts = player.groupby(axis).size()
        means = means[means[axis].map(counts) >= config.plots.minimum_cell_samples]
        long = means.melt(axis, var_name="action", value_name="probability")
        if not long.empty:
            yield line(f"action-probability-vs-{axis}", long, axis, "probability", "action")
    frequencies = player.groupby("action").size().reset_index(name="count")
    yield Chart("playing-action-frequency", px.bar(frequencies, x="action", y="count"), frequencies)
    double = player[player.double_legal]
    if not double.empty:
        data = double.groupby("player_total").chose_double.agg(["mean", "count"]).reset_index()
        yield line("double-rate-when-legal", data, "player_total", "mean")
    sequence = player.sort_values(["episode_id", "decision"]).copy()
    sequence["previous_action"] = sequence.groupby(["episode_id", "round"]).action.shift()
    sequence = sequence.dropna(subset=["previous_action"])
    if not sequence.empty:
        trans = sequence.groupby(["previous_action", "action"]).size().reset_index(name="count")
        yield heatmap("action-transitions", trans, "previous_action", "action", "count", 1)
    rounds = tables["rounds"]
    if not rounds.empty:
        joined = player.merge(
            rounds[["episode_id", "round", "profit", "outcome"]],
            on=["episode_id", "round"],
            validate="many_to_one",
        )
        outcomes = joined.groupby(["action", "outcome"]).size().reset_index(name="count")
        yield Chart(
            "action-outcomes-observational",
            px.bar(outcomes, x="action", y="count", color="outcome"),
            outcomes,
            "Observational association, not causal action value.",
        )


@register
def betting(tables, config):
    rounds, decisions = tables["rounds"], tables["decisions"]
    initial = config.money.initial_bankroll_minor
    if not decisions.empty:
        bets = decisions[decisions.phase == "bet"]
        if not bets.empty:
            distribution = bets.groupby("action_id").size().reset_index(name="count")
            yield Chart(
                "bet-action-frequency", px.bar(distribution, x="action_id", y="count"), distribution
            )
    if rounds.empty:
        return
    data = rounds.sort_values(["episode_id", "round"]).copy()
    data["bet_dollars"] = data.opening_stake / config.money.minor_units_per_dollar
    data["bet_initial_units"] = data.opening_stake / initial
    data["bet_fraction"] = data.opening_stake / data.before
    data["wealth"] = data.before / initial
    data["remaining_rounds"] = config.session.rounds - data["round"]
    peak = data.groupby("episode_id").before.cummax().clip(lower=initial)
    data["drawdown"] = (peak - data.before) / peak
    for column in ("bet_dollars", "bet_initial_units", "bet_fraction"):
        yield histogram(f"{column}-distribution", data, column, config)
    for axis in (
        "true_count",
        "running_count",
        "penetration",
        "wealth",
        "drawdown",
        "remaining_rounds",
    ):
        yield scatter(f"bet-vs-{axis}", data, axis, "bet_fraction", config)
    data["count_bin"] = (
        np.floor(data.true_count / config.plots.count_bin_width) * config.plots.count_bin_width
    )
    data["wealth_bin"] = (
        np.floor(data.wealth / config.plots.bankroll_bin_width) * config.plots.bankroll_bin_width
    )
    data["horizon_bin"] = (
        data.remaining_rounds // config.plots.horizon_bin_width * config.plots.horizon_bin_width
    )
    data["bet_bin"] = (
        np.floor(data.bet_initial_units / config.plots.bet_bin_width) * config.plots.bet_bin_width
    )
    for axis in ("wealth_bin", "horizon_bin"):
        yield heatmap(
            f"bet-count-by-{axis}",
            data,
            "count_bin",
            axis,
            "bet_fraction",
            config.plots.minimum_cell_samples,
        )
    statistics = (
        data.groupby("episode_id")
        .bet_fraction.agg(["mean", "median", "std", "var", "min", "max", "count"])
        .reset_index()
    )
    statistics["coefficient_variation"] = statistics["std"] / statistics["mean"].replace(0, np.nan)
    for metric in ("mean", "median", "std", "var", "coefficient_variation"):
        yield histogram(f"within-session-bet-{metric}", statistics, metric, config)
    data["previous_bet"] = data.groupby("episode_id").bet_fraction.shift()
    pairs = data.dropna(subset=["previous_bet"])
    if not pairs.empty:
        yield scatter("bet-lag-one", pairs, "previous_bet", "bet_fraction", config)
        pairs = pairs.assign(bet_change=pairs.bet_fraction - pairs.previous_bet)
        yield histogram("bet-changes", pairs, "bet_change", config)
    yield scatter("opening-vs-doubled-stake", data, "opening_stake", "total_stake", config)
    data["minimum_bet"] = data.opening_stake == config.money.minimum_bet_minor
    data["all_in"] = data.before - data.opening_stake < config.money.bet_increment_minor
    rates = pd.DataFrame(
        {
            "event": ["minimum_bet", "all_in", "double_blocked_by_cash"],
            "rate": [data[k].mean() for k in ("minimum_bet", "all_in", "double_blocked_by_cash")],
        }
    )
    yield Chart("bet-constraint-rates", px.bar(rates, x="event", y="rate"), rates)
    for group in ("bet_bin", "count_bin"):
        grouped = (
            data.groupby(group)
            .agg(
                count=("profit", "size"),
                profit=("profit", "sum"),
                mean_profit=("profit", "mean"),
                total_stake=("total_stake", "sum"),
            )
            .reset_index()
        )
        grouped["return_per_stake"] = grouped.profit / grouped.total_stake
        for metric in ("profit", "mean_profit", "return_per_stake"):
            yield Chart(
                f"{metric}-by-{group}",
                px.bar(grouped, x=group, y=metric),
                grouped,
                "Observed associations; count column is included in the aggregate export.",
            )
    yield scatter(
        "bet-vs-profit",
        data,
        "bet_fraction",
        "profit",
        config,
        note="Association, not causal effect of increasing wagers.",
    )
    if "timestep" in data and data.timestep.nunique() > 1:
        moments = (
            data.groupby("timestep")
            .bet_fraction.agg(["mean", "median", "std", "var"])
            .reset_index()
        )
        for metric in ("mean", "median", "std", "var"):
            yield line(f"bet-{metric}-learning", moments, "timestep", metric)


@register
def counting(tables, config):
    decisions, rounds, cards = tables["decisions"], tables["rounds"], tables["cards"]
    if not decisions.empty:
        for metric in ("running_count", "true_count", "penetration"):
            yield histogram(f"counting-{metric}", decisions, metric, config)
        sample_id = decisions.episode_id.iloc[0]
        example = decisions[decisions.episode_id == sample_id]
        for metric in ("running_count", "true_count"):
            yield line(f"example-shoe-{metric}", example, "decision", metric, "shoe_id")
    if not cards.empty:
        shuffles = cards[cards.kind == "shuffle"]
        if not shuffles.empty:
            yield histogram("shuffle-penetration", shuffles, "penetration", config)
            reasons = shuffles.groupby("reason").size().reset_index(name="count")
            yield Chart("shuffle-reasons", px.bar(reasons, x="reason", y="count"), reasons)
        if not rounds.empty:
            deals = cards[(cards.kind == "deal") & (cards.recipient == "player")].drop_duplicates(
                ["episode_id", "round"]
            )
            joined = deals[["episode_id", "round", "rank"]].merge(
                rounds[["episode_id", "round", "true_count"]], on=["episode_id", "round"]
            )
            if not joined.empty:
                joined["count_bin"] = (
                    np.floor(joined.true_count / config.plots.count_bin_width)
                    * config.plots.count_bin_width
                )
                joined["rank_name"] = joined["rank"].map(lambda r: config.shoe.ranks[int(r)])
                freq = joined.groupby(["count_bin", "rank_name"]).size().reset_index(name="count")
                freq["frequency"] = freq["count"] / freq.groupby("count_bin")["count"].transform(
                    "sum"
                )
                yield heatmap(
                    "next-player-card-vs-count",
                    freq,
                    "count_bin",
                    "rank_name",
                    "frequency",
                    1,
                    "First player card after each pre-bet count; descriptive sampled frequencies.",
                )
    if not rounds.empty:
        yield scatter("profit-vs-penetration", rounds, "penetration", "profit", config)
