from __future__ import annotations

import json
from functools import partial
from pathlib import Path

import numpy as np
import torch
from sb3_contrib import MaskablePPO
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from blackjack_rl.agents.factory import ModelAgent, make_model
from blackjack_rl.config import Config, load_config
from blackjack_rl.env import BlackjackSessionEnv
from blackjack_rl.evaluation.runner import evaluate
from blackjack_rl.telemetry.writer import TableWriter, create_run, write_json
from blackjack_rl.training.callbacks import ResearchCallback


def make_env(config: Config, seed: int):
    return MaskAwareMonitor(BlackjackSessionEnv(config), seed)


class MaskAwareMonitor(Monitor):
    def __init__(self, env, initial_seed):
        super().__init__(env)
        self.initial_seed = initial_seed

    def reset(self, **kwargs):
        # SB3 schedules sequential seeds; keep the independently spawned declared streams.
        if self.initial_seed is not None:
            kwargs["seed"] = self.initial_seed
            self.initial_seed = None
        return super().reset(**kwargs)

    def action_masks(self):
        return self.env.action_masks()


def train(config: Config, seed: int, resume: Path | None = None) -> Path:
    t = config.training
    if t is None:
        raise ValueError("Training section required")
    if resume:
        original = load_config(resume.parent / "config.json")
        ignored = {"training"}
        if config.dict(exclude=ignored) != original.dict(exclude=ignored):
            raise ValueError(
                "Resume requires identical game, observation, reward and evaluation configs"
            )
        saved_seed = json.loads((resume.parent / "training_seeds.json").read_text())[
            "training_seed"
        ]
        if seed != saved_seed:
            raise ValueError("Resume seed must match the original training seed")
        allowed = {"total_timesteps", "run_root", "progress_bar", "final_test", "seeds"}
        if t.dict(exclude=allowed) != original.training.dict(exclude=allowed):
            raise ValueError("Resume only permits budget, output and progress changes")
    run = create_run(config, f"ppo-seed-{seed}")
    torch.set_num_threads(t.torch_threads)
    streams = np.random.SeedSequence([seed, config.telemetry.schema_version]).spawn(t.n_envs)
    worker_seeds = [int(stream.generate_state(1)[0]) for stream in streams]
    write_json(
        run / "training_seeds.json",
        {"training_seed": seed, "workers": worker_seeds, "resume": str(resume) if resume else None},
    )
    factories = [partial(make_env, config, worker_seed) for worker_seed in worker_seeds]
    vec = (
        SubprocVecEnv(factories, start_method="spawn")
        if t.vectorization == "subproc"
        else DummyVecEnv(factories)
    )
    model = None
    status = "failed"
    try:
        model = (
            MaskablePPO.load(resume, env=vec, device=t.device)
            if resume
            else make_model(config, vec, seed, str(run / "tensorboard"))
        )
        model.set_logger(configure(str(run / "tensorboard"), ["stdout", "csv", "tensorboard"]))
        with TableWriter(run / "training", config.telemetry.flush_rows) as writer:
            callback = ResearchCallback(config, run, writer, worker_seeds)
            try:
                model.learn(
                    t.total_timesteps,
                    callback=callback,
                    progress_bar=t.progress_bar,
                    log_interval=t.log_interval,
                    reset_num_timesteps=not bool(resume),
                )
                # Final PPO update happens after the last rollout callback evaluation.
                callback.validate(final=True)
                status = "completed"
            except KeyboardInterrupt:
                status = "interrupted"
            finally:
                model.save(run / "latest.zip")
                write_json(
                    run / "status.json",
                    {
                        "status": status,
                        "timesteps": model.num_timesteps,
                        "resume_semantics": "Learning continuation; worker and rollout states restart",
                    },
                )
        if status == "completed" and t.final_test:
            selected = MaskablePPO.load(run / "best.zip", device=t.device)
            evaluate(
                config,
                ModelAgent(selected),
                run / "test",
                progress=t.progress_bar,
                timestep=model.num_timesteps,
            )
    finally:
        vec.close()
        if model is not None:
            model.logger.close()
    return run
