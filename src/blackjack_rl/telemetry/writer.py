from __future__ import annotations

import importlib.metadata
import json
import platform
import subprocess
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import pandas as pd

from blackjack_rl.config import Config


def write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def create_run(config: Config, label: str, root: str | Path | None = None) -> Path:
    root = Path(root or (config.training.run_root if config.training else "runs"))
    token = uuid4().hex[: config.telemetry.artifact_id_characters]
    run = root / f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-{label}-{token}"
    run.mkdir(parents=True, exist_ok=False)
    write_json(run / "config.json", json.loads(config.json()))
    versions = {}
    for name in (
        "blackjack-rl",
        "numpy",
        "gymnasium",
        "torch",
        "stable-baselines3",
        "sb3-contrib",
        "tensorboard",
        "pandas",
        "pyarrow",
        "plotly",
        "matplotlib",
    ):
        versions[name] = importlib.metadata.version(name)

    def git(*args):
        result = subprocess.run(["git", *args], capture_output=True, text=True, check=False)
        return result.stdout.strip() if result.returncode == 0 else None

    write_json(
        run / "manifest.json",
        {
            "run_id": run.name,
            "schema_version": config.telemetry.schema_version,
            "config_hash": config.config_hash,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "git_commit": git("rev-parse", "HEAD"),
            "git_status": git("status", "--porcelain"),
            "versions": versions,
            "python": sys.version,
            "platform": platform.platform(),
            "processor": platform.processor(),
            "objective": "relative wealth minus ruin and ending net-loss penalties; no credit",
            "baseline_reference": "configured multi-deck S17 no-split strategy, not an optimality claim",
        },
    )
    return run


class TableWriter:
    """Chunked tables bound memory; each chunk is committed atomically."""

    def __init__(self, root: Path, flush_rows: int):
        self.root = Path(root)
        self.flush_rows = flush_rows
        self.buffers = defaultdict(list)
        self.closed = False

    def add(self, table: str, row: dict):
        if self.closed:
            raise RuntimeError("Writer is closed")
        self.buffers[table].append(row)
        if len(self.buffers[table]) >= self.flush_rows:
            self.flush(table)

    def flush(self, table: str):
        rows = self.buffers[table]
        if not rows:
            return
        folder = self.root / table
        folder.mkdir(parents=True, exist_ok=True)
        destination = folder / f"part-{uuid4().hex}.parquet"
        temporary = destination.with_suffix(".tmp")
        pd.DataFrame(rows).to_parquet(temporary, index=False)
        temporary.replace(destination)
        rows.clear()

    def close(self):
        for table in self.buffers:
            self.flush(table)
        self.closed = True

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def read_table(root: Path, table: str) -> pd.DataFrame:
    paths = sorted((Path(root) / table).glob("*.parquet"))
    return (
        pd.concat([pd.read_parquet(path) for path in paths], ignore_index=True)
        if paths
        else pd.DataFrame()
    )
