import time
from pathlib import Path

import numpy as np
import torch
from stable_baselines3.common.callbacks import BaseCallback

from blackjack_rl.agents.factory import ModelAgent
from blackjack_rl.config import Config
from blackjack_rl.env.actions import Actions
from blackjack_rl.evaluation.runner import decision_row, evaluate
from blackjack_rl.telemetry.writer import TableWriter


class ResearchCallback(BaseCallback):
    """Custom evaluator always applies masks; never uses SB3's unmasked evaluator."""

    def __init__(self, config: Config, run: Path, writer: TableWriter, worker_seeds: list[int]):
        super().__init__()
        self.config, self.run, self.writer = config, run, writer
        self.worker_seeds = worker_seeds
        self.actions = Actions(config)
        self.episodes = np.zeros(config.training.n_envs, dtype=int)
        self.steps = np.zeros(config.training.n_envs, dtype=int)
        self.stakes = np.zeros(config.training.n_envs, dtype=np.int64)
        self.last_eval = self.last_save = 0
        self.best_reward = -np.inf
        self.rounds = self.sessions = 0

    def _on_training_start(self):
        self.started = time.perf_counter()
        self.start_transitions = self.num_timesteps
        self.last_eval = self.last_save = self.num_timesteps
        for worker, info in enumerate(self.training_env.reset_infos):
            for event in info.get("events", []):
                self.writer.add("cards", {"episode_id": f"{worker}-0", "worker": worker, **event})

    def validate(self, final=False):
        started = time.perf_counter()
        summary = evaluate(
            self.config,
            ModelAgent(self.model),
            self.run
            / "validation"
            / (f"final-{self.num_timesteps}" if final else str(self.num_timesteps)),
            split="validation",
            progress=False,
            timestep=self.num_timesteps,
        )
        summary.update(
            timestep=self.num_timesteps,
            seconds=time.perf_counter() - self.started,
            evaluation_seconds=time.perf_counter() - started,
        )
        self.writer.add("evaluations", summary)
        for key in (
            "mean_reward",
            "mean_wealth",
            "ruin_probability",
            "completion_rate",
            "mean_rounds",
        ):
            self.logger.record(f"research/validation_{key}", summary[key])
        if summary["mean_reward"] > self.best_reward:
            self.best_reward = summary["mean_reward"]
            self.model.save(self.run / "best.zip")
        self.last_eval = self.num_timesteps
        return summary

    def _on_step(self):
        sampled = self.episodes % self.config.telemetry.training_trace_every_sessions == 0
        probabilities = None
        if sampled.any():
            with torch.no_grad():
                distribution = self.model.policy.get_distribution(
                    self.locals["obs_tensor"], action_masks=self.locals["action_masks"]
                )
                probabilities = distribution.distribution.probs.cpu().numpy()
        for worker, info in enumerate(self.locals["infos"]):
            episode_id = f"{worker}-{self.episodes[worker]}"
            common = {
                "episode_id": episode_id,
                "worker": worker,
                "seed": self.worker_seeds[worker],
                "agent": "ppo",
                "timestep": self.num_timesteps,
                "seconds": time.perf_counter() - self.started,
            }
            if sampled[worker]:
                public = info["decision_public"]
                obs = {k: v[worker] for k, v in self.model._last_obs.items()}
                row = decision_row(
                    self.config,
                    self.actions,
                    obs,
                    public,
                    self.locals["action_masks"][worker],
                    probabilities[worker],
                    int(self.locals["actions"][worker]),
                    episode_id,
                    int(self.steps[worker]),
                    "ppo",
                    self.worker_seeds[worker],
                )
                self.writer.add("decisions", {**row, **common})
                for event in info["events"]:
                    self.writer.add("cards", {**common, **event})
            if "round" in info:
                self.rounds += 1
                self.stakes[worker] += info["round"]["total_stake"]
                if sampled[worker]:
                    self.writer.add("rounds", {**common, **info["round"]})
            self.steps[worker] += 1
            if "session" in info:
                session = info["session"]
                self.writer.add(
                    "episodes", {**common, **session, "total_stake": int(self.stakes[worker])}
                )
                for key in (
                    "reward",
                    "wealth",
                    "ruin",
                    "max_drawdown",
                    "wealth_reward",
                    "ruin_penalty",
                    "shortfall_penalty",
                ):
                    self.logger.record_mean(f"research/session_{key}", session[key])
                self.episodes[worker] += 1
                self.steps[worker] = self.stakes[worker] = 0
                self.sessions += 1
                if self.episodes[worker] % self.config.telemetry.training_trace_every_sessions == 0:
                    for event in self.training_env.reset_infos[worker].get("events", []):
                        self.writer.add(
                            "cards",
                            {
                                "episode_id": f"{worker}-{self.episodes[worker]}",
                                "worker": worker,
                                **event,
                            },
                        )
        elapsed = time.perf_counter() - self.started
        self.logger.record("research/settled_rounds", self.rounds)
        self.logger.record("research/completed_sessions", self.sessions)
        self.logger.record(
            "research/transitions_per_second",
            (self.num_timesteps - self.start_transitions) / elapsed,
        )
        if self.num_timesteps - self.last_eval >= self.config.training.eval_interval:
            self.validate()
        if self.num_timesteps - self.last_save >= self.config.training.checkpoint_interval:
            self.model.save(self.run / f"checkpoint-{self.num_timesteps}.zip")
            self.model.save(self.run / "latest.zip")
            self.last_save = self.num_timesteps
        return True

    def _on_rollout_end(self):
        row = {"timestep": self.num_timesteps, "seconds": time.perf_counter() - self.started}
        for key, value in self.logger.name_to_value.items():
            if isinstance(value, (float, int, np.number)) and np.isfinite(value):
                row[key] = float(value)
        self.writer.add("optimization", row)
