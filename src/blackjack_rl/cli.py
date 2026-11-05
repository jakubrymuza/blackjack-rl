from __future__ import annotations

import argparse
import json
from pathlib import Path
from uuid import uuid4

from blackjack_rl.config import load_config


def parser():
    cli = argparse.ArgumentParser(description="Finite-shoe blackjack RL research")
    commands = cli.add_subparsers(dest="command", required=True)
    for name in ("validate-config", "simulate", "train", "evaluate", "report"):
        cmd = commands.add_parser(name)
        cmd.add_argument("--config", type=Path)
        cmd.add_argument("--set", action="append", default=[], metavar="SECTION.KEY=VALUE")
        if name in ("evaluate", "report"):
            cmd.add_argument("--run", type=Path, required=True)
        if name in ("simulate", "evaluate"):
            cmd.add_argument("--sessions", type=int)
            cmd.add_argument("--split", choices=("test", "validation"), default="test")
            cmd.add_argument("--sampled", action="store_true")
        if name == "simulate":
            cmd.add_argument(
                "--agent", choices=("random", "threshold", "basic", "hilo"), default="basic"
            )
        if name == "train":
            cmd.add_argument("--seed", type=int)
            cmd.add_argument("--resume", type=Path)
        if name == "evaluate":
            cmd.add_argument("--checkpoint", default="best.zip")
            cmd.add_argument("--output", type=Path)
            cmd.add_argument("--allow-changed-scenario", action="store_true")
        if name == "report":
            cmd.add_argument("--data", type=Path)
            cmd.add_argument("--filter", action="append", default=[])
        if name == "validate-config":
            cmd.add_argument("--check-env", action="store_true")
    comparison = commands.add_parser("compare")
    comparison.add_argument("--runs", nargs="+", type=Path, required=True)
    comparison.add_argument("--output", type=Path)
    return cli


def main():
    args = parser().parse_args()
    if args.command == "compare":
        from blackjack_rl.reporting.comparisons import compare_runs

        print(compare_runs(args.runs, args.output or Path("runs") / f"comparison-{uuid4().hex}"))
        return
    path = args.config
    if path is None:
        if args.command == "report" and args.data and (args.data / "config.json").exists():
            path = args.data / "config.json"
        else:
            path = args.run / "config.json" if hasattr(args, "run") else Path("configs/train.yaml")
    overrides = list(args.set)
    if getattr(args, "sessions", None) is not None:
        key = "validation_sessions" if args.split == "validation" else "sessions"
        overrides.append(f"evaluation.{key}={args.sessions}")
    if getattr(args, "seed", None) is not None:
        overrides.append(f"training.seeds=[{args.seed}]")
    config = load_config(path, overrides)
    if args.command == "validate-config":
        print(config.json(indent=2))
        print(f"Config SHA256: {config.config_hash}")
        if args.check_env:
            from stable_baselines3.common.env_checker import check_env
            from blackjack_rl.env import BlackjackSessionEnv
            from blackjack_rl.env.checking import LegalCheckAdapter

            check_env(LegalCheckAdapter(BlackjackSessionEnv(config)))
            print("SB3 interface check passed through test-only legal-action adapter")
    elif args.command == "simulate":
        from blackjack_rl.agents.baselines import BaselineAgent
        from blackjack_rl.evaluation.runner import evaluate
        from blackjack_rl.telemetry.writer import create_run

        run = create_run(config, args.agent)
        summary = evaluate(
            config,
            BaselineAgent(args.agent, config),
            run / "test",
            split=args.split,
            deterministic=not args.sampled,
        )
        print(json.dumps(summary, indent=2))
        print(f"Run: {run}")
    elif args.command == "train":
        from blackjack_rl.training.runner import train

        if config.training is None:
            raise ValueError("Training configuration required")
        if args.resume and len(config.training.seeds) != 1:
            raise ValueError("Resume requires a config with a single training seed")
        for seed in config.training.seeds:
            run = train(config, seed, args.resume)
            print(f"Run: {run}", flush=True)
            if json.loads((run / "status.json").read_text())["status"] == "interrupted":
                break
    elif args.command == "evaluate":
        from sb3_contrib import MaskablePPO
        from blackjack_rl.agents.factory import ModelAgent
        from blackjack_rl.evaluation.runner import evaluate

        original = load_config(args.run / "config.json")
        for section in ("money", "session", "rules", "shoe", "betting", "features", "reward"):
            if (
                getattr(config, section) != getattr(original, section)
                and not args.allow_changed_scenario
            ):
                raise ValueError(
                    f"Evaluation changed {section}; use an explicitly trained scenario"
                )
        output = args.output or args.run / "evaluations" / f"{args.split}-{uuid4().hex}"
        model = MaskablePPO.load(args.run / args.checkpoint, device="cpu")
        summary = evaluate(
            config, ModelAgent(model), output, split=args.split, deterministic=not args.sampled
        )
        from blackjack_rl.telemetry.writer import write_json

        write_json(
            output / "scenario.json",
            {
                "training_config_hash": original.config_hash,
                "evaluation_config": json.loads(config.json()),
                "changed_scenario_allowed": args.allow_changed_scenario,
            },
        )
        print(json.dumps(summary, indent=2))
        print(f"Evaluation: {output}")
    elif args.command == "report":
        from blackjack_rl.reporting.report import build_report

        print(build_report(args.run, config=config, data=args.data, filters=args.filter))


if __name__ == "__main__":
    main()
