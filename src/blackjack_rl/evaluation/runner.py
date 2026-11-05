from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from blackjack_rl.config import Config
from blackjack_rl.env import BlackjackSessionEnv
from blackjack_rl.evaluation.metrics import summarize
from blackjack_rl.telemetry.writer import TableWriter, write_json


def seed_manifest(config: Config, split: str, sessions: int) -> list[int]:
    root_seed = (
        config.evaluation.validation_seed if split == "validation" else config.evaluation.test_seed
    )
    # Separate namespace from training-worker seeds even for equal integer seeds.
    children = np.random.SeedSequence([root_seed, config.telemetry.schema_version]).spawn(sessions)
    return [int(child.generate_state(1, dtype=np.uint64)[0]) for child in children]


def decision_row(
    cfg, actions, observation, public, mask, probs, action, episode_id, step, agent, seed
):
    legal_probs = probs[probs > 0]
    entropy = float(-(legal_probs * np.log(legal_probs)).sum())
    n_legal = int(mask.sum())
    amounts = actions.amounts(public["settled"])
    row = {
        "episode_id": episode_id,
        "seed": seed,
        "agent": agent,
        "decision": step,
        "round": public["completed"] + 1,
        "shoe_id": public["shoe_id"],
        "phase": public["phase"],
        "action": actions.name(action),
        "action_id": action,
        "player_total": public["total"],
        "soft": public["soft"],
        "initial": public["initial"],
        "dealer_up": cfg.shoe.low_values[public["dealer_rank"]]
        if public["dealer_rank"] is not None
        else None,
        "wealth": public["settled"] / cfg.money.initial_bankroll_minor,
        "remaining_rounds": cfg.session.rounds - public["completed"],
        "committed": public["committed"],
        "running_count": public["running_count"],
        "true_count": public["true_count"],
        "penetration": public["penetration"],
        "bet": amounts[action] if action < actions.bet_count else None,
        "chosen_probability": float(probs[action]),
        "entropy": entropy,
        "normalized_entropy": entropy / np.log(n_legal) if n_legal > 1 else 0.0,
        "double_legal": bool(mask[actions.double]),
        "legal_mask": mask.tolist(),
        "probabilities": probs.tolist(),
        "observation": json.dumps({k: v.tolist() for k, v in observation.items()}),
    }
    if public["phase"] == "player":
        from blackjack_rl.agents.baselines import strategy_action

        reference = strategy_action(public, actions, cfg)
        row["reference_action"] = actions.name(reference)
        row["disagrees_reference"] = action != reference
        row["reference_probability"] = float(probs[reference])
    return row


def evaluate(
    config: Config,
    agent,
    output: Path,
    split: str = "test",
    sessions: int | None = None,
    progress: bool = True,
    timestep: int = 0,
    deterministic: bool | None = None,
):
    if split not in ("validation", "test"):
        raise ValueError("Evaluation split must be validation or test")
    sessions = sessions or (
        config.evaluation.validation_sessions
        if split == "validation"
        else config.evaluation.sessions
    )
    trace_limit = (
        config.evaluation.validation_trace_sessions
        if split == "validation"
        else config.evaluation.trace_sessions
    )
    deterministic = config.evaluation.deterministic if deterministic is None else deterministic
    seeds = seed_manifest(config, split, sessions)
    output = Path(output)
    if (output / "evaluation.json").exists():
        raise FileExistsError(f"Evaluation already exists: {output}")
    write_json(output / "config.json", json.loads(config.json()))
    write_json(output / "seeds.json", {"split": split, "seeds": seeds})
    write_json(
        output / "evaluation.json",
        {
            "agent": agent.name,
            "split": split,
            "sessions": sessions,
            "deterministic": deterministic,
            "timestep": timestep,
            "config_hash": config.config_hash,
            "traced_sessions": min(trace_limit, sessions),
        },
    )
    episodes = []
    env = BlackjackSessionEnv(config)
    with TableWriter(output, config.telemetry.flush_rows) as writer:
        for episode_id, seed in enumerate(
            tqdm(seeds, desc=f"{split} {agent.name}", disable=not progress)
        ):
            observation, reset_info = env.reset(seed=seed)
            rng = np.random.default_rng(seed)
            trace = episode_id < trace_limit
            if trace:
                for event in reset_info["events"]:
                    writer.add("cards", {"episode_id": episode_id, **event})
            step, total_stake = 0, 0
            while True:
                public = env.engine.public_snapshot()
                mask = env.action_masks()
                probs = agent.probabilities(observation, public, mask, env.actions)
                # Random baseline remains random even under deterministic model evaluation.
                action = (
                    int(np.argmax(probs))
                    if deterministic and agent.name != "random"
                    else int(rng.choice(len(probs), p=probs))
                )
                if trace:
                    row = decision_row(
                        config,
                        env.actions,
                        observation,
                        public,
                        mask,
                        probs,
                        action,
                        episode_id,
                        step,
                        agent.name,
                        seed,
                    )
                    row["timestep"] = timestep
                    writer.add("decisions", row)
                observation, reward, done, truncated, info = env.step(action)
                if "round" in info:
                    total_stake += info["round"]["total_stake"]
                    if trace:
                        writer.add(
                            "rounds",
                            {
                                "episode_id": episode_id,
                                "agent": agent.name,
                                "seed": seed,
                                **info["round"],
                            },
                        )
                if trace:
                    for event in info["events"]:
                        writer.add("cards", {"episode_id": episode_id, **event})
                step += 1
                if done or truncated:
                    row = {
                        "episode_id": episode_id,
                        "agent": agent.name,
                        "seed": seed,
                        "split": split,
                        "timestep": timestep,
                        "total_stake": total_stake,
                        **info["session"],
                    }
                    writer.add("episodes", row)
                    episodes.append(row)
                    break
    env.close()
    summary = summarize(pd.DataFrame(episodes), config)
    write_json(output / "summary.json", summary)
    return summary
