import numpy as np
import pandas as pd

from blackjack_rl.config import Config
from blackjack_rl.evaluation.statistics import mean_interval, proportion_interval


def summarize(episodes: pd.DataFrame, config: Config) -> dict:
    if episodes.empty:
        raise ValueError("No completed sessions to summarize")
    rng = np.random.default_rng(config.evaluation.analysis_seed)
    wealth = episodes.wealth.to_numpy()
    tail = np.quantile(wealth, config.evaluation.tail_quantile)
    result = {
        "sessions": len(episodes),
        "mean_wealth": float(wealth.mean()),
        "median_wealth": float(np.median(wealth)),
        "mean_wealth_ci": mean_interval(wealth, config, rng),
        "mean_reward": float(episodes.reward.mean()),
        "mean_reward_ci": mean_interval(episodes.reward, config, rng),
        "ruin_probability": float(episodes.ruin.mean()),
        "ruin_ci": proportion_interval(
            int(episodes.ruin.sum()), len(episodes), config.evaluation.confidence
        ),
        "profit_probability": float((episodes.profit > 0).mean()),
        "completion_rate": float(episodes.completion.mean()),
        "mean_rounds": float(episodes["rounds"].mean()),
        "mean_max_drawdown": float(episodes.max_drawdown.mean()),
        "tail_wealth": float(wealth[wealth <= tail].mean()),
        "quantiles": {str(q): float(np.quantile(wealth, q)) for q in config.evaluation.quantiles},
        "mean_profit_per_scheduled_round": float(episodes.profit.mean() / config.session.rounds),
    }
    if "total_stake" in episodes:
        total_stake = episodes.total_stake.sum()
        result["profit_per_unit_stake"] = (
            float(episodes.profit.sum() / total_stake) if total_stake else None
        )
        result["profit_per_played_round"] = float(episodes.profit.sum() / episodes["rounds"].sum())
    return result
