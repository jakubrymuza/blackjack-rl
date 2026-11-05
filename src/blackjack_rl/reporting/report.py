from __future__ import annotations

import html
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from plotly.offline import get_plotlyjs

from blackjack_rl.config import Config, load_config
from blackjack_rl.reporting import charts  # noqa: F401
from blackjack_rl.reporting import research  # noqa: F401
from blackjack_rl.reporting.registry import BUILDERS
from blackjack_rl.telemetry.writer import read_table, write_json


def export_static(figure, destination: Path, config: Config):
    fig, ax = plt.subplots(figsize=(config.plots.figure_width, config.plots.figure_height))
    for trace in figure.data:
        name = trace.name or ""
        if trace.type == "heatmap":
            z = np.asarray(trace.z, dtype=float)
            if not z.size:
                continue
            image = ax.imshow(z, aspect="auto", origin="lower")
            ax.set_xticks(range(len(trace.x)), trace.x, rotation="vertical")
            ax.set_yticks(range(len(trace.y)), trace.y)
            fig.colorbar(image, ax=ax)
        elif trace.type == "histogram":
            ax.hist(trace.x, bins=config.plots.histogram_bins, label=name)
        elif trace.type in ("box", "violin"):
            values = np.asarray(trace.y, dtype=float)
            values = values[np.isfinite(values)]
            if len(values):
                if trace.type == "box":
                    ax.boxplot(values)
                else:
                    ax.violinplot(values, showmedians=True)
        elif trace.type == "bar":
            ax.bar([str(x) for x in trace.x], trace.y, label=name)
        elif trace.type in ("scatter", "scattergl"):
            if trace.mode and "markers" in trace.mode and "lines" not in trace.mode:
                ax.scatter(trace.x, trace.y, label=name)
            else:
                ax.plot(trace.x, trace.y, label=name)
    ax.set_title(figure.layout.title.text or "")
    ax.set_xlabel(figure.layout.xaxis.title.text or "")
    ax.set_ylabel(figure.layout.yaxis.title.text or "")
    handles, labels = ax.get_legend_handles_labels()
    if any(labels):
        ax.legend()
    fig.tight_layout()
    for extension in config.plots.static_formats:
        fig.savefig(destination.with_suffix(f".{extension}"), dpi=config.plots.dpi)
    plt.close(fig)


def load_tables(run: Path, data: Path):
    tables = {name: read_table(data, name) for name in ("episodes", "rounds", "decisions", "cards")}
    tables["training_episodes"] = read_table(run / "training", "episodes")
    tables["evaluations"] = read_table(run / "training", "evaluations")
    tables["optimization"] = read_table(run / "training", "optimization")
    progress = run / "tensorboard" / "progress.csv"
    if progress.exists():
        csv = pd.read_csv(progress).rename(columns={"time/total_timesteps": "timestep"})
        if "timestep" in csv:
            tables["optimization"] = csv
    return tables


def build_report(
    run: Path, config: Config | None = None, data: Path | None = None, filters: list[str] = ()
) -> Path:
    run = Path(run)
    data = Path(data) if data else run / "test" if (run / "test").exists() else run
    if config is None and (data / "scenario.json").exists():
        config = Config.parse_obj(
            json.loads((data / "scenario.json").read_text())["evaluation_config"]
        )
    config = config or load_config(run / "config.json")
    output = run / "report"
    if filters:
        from hashlib import sha256

        output = (
            output
            / sha256(json.dumps(filters).encode()).hexdigest()[
                : config.telemetry.artifact_id_characters
            ]
        )
    (output / "charts").mkdir(parents=True, exist_ok=True)
    (output / "aggregates").mkdir(exist_ok=True)
    (output / "plotly.min.js").write_text(get_plotlyjs(), encoding="utf-8")
    tables = load_tables(run, data)
    # Filter only tables containing each field. Record filters prominently in the report.
    for expression in filters:
        column, separator, raw = expression.partition("=")
        if not separator:
            raise ValueError("Report filters must be column=JSON-value")
        value = json.loads(raw)
        if not any(column in table for table in tables.values()):
            raise ValueError(f"Unknown report filter column: {column}")
        for name, table in tables.items():
            if column in table:
                if isinstance(value, dict):
                    if not value or set(value) - {"min", "max"}:
                        raise ValueError("Range filters support min and max only")
                    selected = pd.Series(True, index=table.index)
                    if "min" in value:
                        selected &= table[column] >= value["min"]
                    if "max" in value:
                        selected &= table[column] <= value["max"]
                    tables[name] = table[selected].copy()
                else:
                    tables[name] = table[table[column] == value].copy()
    links, catalog = [], []
    for builder in BUILDERS:
        for chart in builder(tables, config):
            slug = re.sub(r"[^a-zA-Z0-9_-]", "-", chart.name)
            sample_count = len(chart.data)
            chart.figure.update_layout(title=f"{chart.name} (N={sample_count} exported rows)")
            chart.figure.write_html(
                output / "charts" / f"{slug}.html", include_plotlyjs="../plotly.min.js"
            )
            chart.figure.write_json(output / "aggregates" / f"{slug}.plotly.json")
            chart.data.to_csv(output / "aggregates" / f"{slug}.csv", index=False)
            chart.data.to_parquet(output / "aggregates" / f"{slug}.parquet", index=False)
            if config.plots.static_formats:
                export_static(chart.figure, output / "charts" / slug, config)
            note = html.escape(chart.note)
            links.append(
                f'<li><a target="chart" href="charts/{slug}.html">{html.escape(chart.name)}</a> {note}</li>'
            )
            catalog.append({"name": chart.name, "rows": sample_count, "note": chart.note})
    write_json(
        output / "catalog.json",
        {
            "config_hash": config.config_hash,
            "data": str(data),
            "filters": filters,
            "charts": catalog,
            "sample_sizes": {k: len(v) for k, v in tables.items()},
        },
    )
    summary_path = data / "summary.json"
    summary = (
        summary_path.read_text() if summary_path.exists() else "No evaluation summary available."
    )
    if filters:
        summary = "Whole-session metrics omitted: filters change chart populations. See the unfiltered evaluation summary.json."
    metadata = {
        "run": run.name,
        "data": str(data),
        "config_hash": config.config_hash,
        "filters": filters,
        "rules": config.rules.dict(),
        "population": "Episode metrics use all completed sessions; decision/round charts use traced sessions.",
        "interpretation": "Action and betting outcome charts are observational; no causal Q-value claims.",
    }
    first = (
        f'charts/{re.sub(r"[^a-zA-Z0-9_-]", "-", catalog[0]["name"])}.html'
        if catalog
        else "about:blank"
    )
    page = (
        '<!doctype html><html><head><meta charset="utf-8"><title>Blackjack RL research</title>'
        "<style>body{font-family:system-ui;margin:2em}iframe{width:100%;height:75vh;border:0}"
        "pre{white-space:pre-wrap}li{margin:.5em 0}input{padding:.5em;width:30em}</style></head><body>"
        "<h1>Blackjack RL research</h1><details><summary>Experiment metadata</summary><pre>"
        + html.escape(json.dumps(metadata, indent=2))
        + "</pre></details><details><summary>All-session metrics</summary><pre>"
        + html.escape(summary)
        + "</pre></details><p>Chart cells below the configured sample threshold are missing. "
        "Chart N denotes exported aggregate rows, not independent sessions. See catalog.json for population sizes.</p>"
        '<input id="search" placeholder="Filter chart names" aria-label="Filter chart names">'
        f'<iframe name="chart" title="Research chart" src="{first}"></iframe><ul id="charts">'
        + "\n".join(links)
        + '</ul><script>document.getElementById("search").oninput=function(){'
        'document.querySelectorAll("#charts li").forEach(e=>e.hidden=!e.textContent.toLowerCase().includes(this.value.toLowerCase()))}'
        "</script></body></html>"
    )
    (output / "index.html").write_text(page, encoding="utf-8")
    return output / "index.html"
