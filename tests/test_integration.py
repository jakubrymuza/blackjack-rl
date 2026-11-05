import json

import pytest

from blackjack_rl.agents.baselines import BaselineAgent, strategy_action
from blackjack_rl.env import BlackjackSessionEnv
from blackjack_rl.evaluation.runner import evaluate
from blackjack_rl.evaluation.statistics import proportion_interval
from blackjack_rl.reporting.report import build_report
from blackjack_rl.telemetry.writer import TableWriter, create_run, read_table
from blackjack_rl.training.runner import train


def test_baseline_reference_cells(config):
    env = BlackjackSessionEnv(config)
    actions = env.actions
    for total, soft, dealer, double, expected in [
        (11, False, "10", True, actions.double),
        (11, False, "A", True, actions.hit),
        (16, False, "6", True, actions.stand),
        (16, False, "10", True, actions.hit),
        (18, True, "6", True, actions.double),
        (18, True, "6", False, actions.stand),
        (18, True, "9", False, actions.hit),
    ]:
        public = {
            "total": total,
            "soft": soft,
            "dealer_rank": config.shoe.ranks.index(dealer),
            "initial": True,
            "can_double": double,
        }
        assert strategy_action(public, actions, config) == expected


def test_evaluation_and_report(config, tmp_path):
    config.plots.static_formats = []
    run = create_run(config, "basic", tmp_path)
    summary = evaluate(
        config, BaselineAgent("basic", config), run / "test", sessions=5, progress=False
    )
    assert summary["sessions"] == 5
    assert summary["ruin_ci"][1] > 0
    rows = read_table(run / "test", "episodes")
    assert len(rows) == 5
    decisions = read_table(run / "test", "decisions")
    for row in decisions.itertuples():
        assert row.legal_mask[row.action_id]
        assert sum(row.probabilities) == pytest.approx(1)
    report = build_report(run, config)
    assert report.exists()
    catalog = json.loads((report.parent / "catalog.json").read_text())
    assert len(catalog["charts"]) > 30


def test_chunked_tables(config, tmp_path):
    with TableWriter(tmp_path, flush_rows=2) as writer:
        for i in range(5):
            writer.add("test", {"i": i, "late": None if i < 2 else i})
    assert sorted(read_table(tmp_path, "test").i) == list(range(5))


def test_wilson_zero_events():
    low, high = proportion_interval(0, 100, 0.95)
    assert low == 0 and high > 0


@pytest.mark.parametrize("vectorization", ["dummy", "subproc"])
def test_training_save_load(config, tmp_path, vectorization):
    config.training.run_root = str(tmp_path)
    config.training.total_timesteps = config.training.n_steps * config.training.n_envs
    config.training.vectorization = vectorization
    config.training.progress_bar = False
    config.training.final_test = False
    config.evaluation.validation_sessions = 2
    config.evaluation.validation_trace_sessions = 1
    run = train(config, config.training.seeds[0])
    assert (run / "latest.zip").exists() and (run / "best.zip").exists()
    assert json.loads((run / "status.json").read_text())["status"] == "completed"
    from sb3_contrib import MaskablePPO

    model = MaskablePPO.load(run / "best.zip")
    env = BlackjackSessionEnv(config)
    obs, _ = env.reset(seed=7)
    action, _ = model.predict(obs, action_masks=env.action_masks(), deterministic=True)
    assert env.action_masks()[int(action)]
    episodes = read_table(run / "training", "episodes")
    assert not episodes.empty


def test_declared_worker_seed_is_preserved(config):
    from blackjack_rl.training.runner import make_env

    wrapped = make_env(config, 123)
    expected = BlackjackSessionEnv(config)
    reference, _ = expected.reset(seed=123)
    actual, _ = wrapped.reset(seed=999)
    assert all((reference[k] == actual[k]).all() for k in reference)
    assert wrapped.unwrapped.engine.shoe.cards == expected.engine.shoe.cards


def test_evaluation_refuses_population_mix(config, tmp_path):
    evaluate(config, BaselineAgent("basic", config), tmp_path, sessions=1, progress=False)
    with pytest.raises(FileExistsError):
        evaluate(config, BaselineAgent("basic", config), tmp_path, sessions=1, progress=False)


def test_filtered_report_omits_whole_population_summary(config, tmp_path):
    config.plots.static_formats = []
    run = create_run(config, "basic", tmp_path)
    evaluate(config, BaselineAgent("basic", config), run / "test", sessions=1, progress=False)
    report = build_report(run, config, filters=["soft=true"])
    assert "Whole-session metrics omitted" in report.read_text()
