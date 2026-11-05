from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd
import plotly.express as px

from blackjack_rl.config import load_config
from blackjack_rl.evaluation.metrics import summarize
from blackjack_rl.evaluation.statistics import mean_interval
from blackjack_rl.telemetry.writer import read_table, write_json


def compare_runs(runs: list[Path], output: Path):
    output.mkdir(parents=True, exist_ok=False)
    rows, groups = [], {}
    for run in runs:
        cfg = load_config(run / "config.json")
        data = run / "test" if (run / "test").exists() else run
        episodes = read_table(data, "episodes")
        metrics = summarize(episodes, cfg)
        row = {
            "run": run.name,
            "agent": episodes.agent.iloc[0],
            "config_hash": cfg.config_hash,
            **{k: v for k, v in metrics.items() if isinstance(v, (int, float))},
        }
        method_config = cfg.dict()
        if method_config["training"] is not None:
            method_config["training"].pop("seeds")
            method_config["training"].pop("run_root")
        row["method_hash"] = hashlib.sha256(
            json.dumps(method_config, sort_keys=True).encode()
        ).hexdigest()
        mode = json.loads((data / "evaluation.json").read_text())["deterministic"]
        row["policy_mode"] = "deterministic" if mode else "sampled"
        rows.append(row)
        # Same full config groups independent training seeds without mixing rule/reward variants.
        group = f"{row['agent']}:{row['method_hash']}:{row['policy_mode']}"
        groups.setdefault(group, []).append((run, episodes, cfg))
    frame = pd.DataFrame(rows)
    frame.to_csv(output / "metrics.csv", index=False)
    frame.to_parquet(output / "metrics.parquet", index=False)
    px.scatter(
        frame, x="ruin_probability", y="mean_wealth", color="agent", hover_name="run"
    ).write_html(output / "risk-growth.html")
    px.bar(frame, x="run", y="mean_reward", color="agent").write_html(output / "rewards.html")
    intervals = {}
    for group, models in groups.items():
        cfg = models[0][2]
        rng = np.random.default_rng(cfg.evaluation.analysis_seed)
        means = []
        for _ in range(cfg.evaluation.bootstrap_resamples):
            selected = rng.integers(len(models), size=len(models))
            means.append(
                np.mean(
                    [
                        rng.choice(
                            models[i][1].wealth.to_numpy(), size=len(models[i][1]), replace=True
                        ).mean()
                        for i in selected
                    ]
                )
            )
        alpha = (1 - cfg.evaluation.confidence) / 2
        intervals[group] = {
            "independent_models": len(models),
            "mean_wealth": float(np.mean([m[1].wealth.mean() for m in models])),
            "hierarchical_wealth_ci": np.quantile(means, [alpha, 1 - alpha]).tolist(),
            "warning": "One model estimates session uncertainty only" if len(models) == 1 else None,
        }
    paired = []
    for i, first in enumerate(runs):
        for second in runs[i + 1 :]:
            a = read_table(first / "test" if (first / "test").exists() else first, "episodes")
            b = read_table(second / "test" if (second / "test").exists() else second, "episodes")
            shared = a[["seed", "wealth"]].merge(
                b[["seed", "wealth"]], on="seed", suffixes=("_a", "_b"), validate="one_to_one"
            )
            if not shared.empty:
                cfg = load_config(first / "config.json")
                delta = shared.wealth_a - shared.wealth_b
                paired.append(
                    {
                        "first": first.name,
                        "second": second.name,
                        "paired_sessions": len(shared),
                        "mean_wealth_difference": float(delta.mean()),
                        "session_difference_ci": mean_interval(
                            delta, cfg, np.random.default_rng(cfg.evaluation.analysis_seed)
                        ),
                    }
                )
    write_json(
        output / "uncertainty.json",
        {
            "groups": intervals,
            "paired": paired,
            "note": "Matched seeds are not identical hand-by-hand counterfactuals. Changed configs are distinct scenarios.",
        },
    )
    (output / "index.html").write_text(
        '<h1>Blackjack comparison</h1><a href="risk-growth.html">Risk and growth</a> '
        '<a href="rewards.html">Rewards</a><p>Confidence intervals: uncertainty.json. '
        "Group intervals bootstrap models then sessions; one-model groups cannot estimate training variability.</p>"
        + frame.to_html(index=False),
        encoding="utf-8",
    )
    return output / "index.html"
