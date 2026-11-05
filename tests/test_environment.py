import numpy as np
import pytest
from hypothesis import given, settings, strategies as st

from blackjack_rl.config import load_config
from blackjack_rl.env import BlackjackSessionEnv
from blackjack_rl.env.rewards import terminal_penalties
from blackjack_rl.game.cards import Card
from conftest import rig


def test_phase_masks_and_dedup(config):
    env = BlackjackSessionEnv(config)
    env.reset(seed=1)
    amounts = env.actions.amounts(env.engine.settled)
    legal = np.flatnonzero(env.action_masks())
    assert len({amounts[i] for i in legal}) == len(legal)
    assert not env.action_masks()[env.actions.hit]
    with pytest.raises(ValueError):
        env.step(env.actions.hit)
    rig(env.engine, ["5", "10", "6", "8", "K"])
    obs, reward, done, _, _ = env.step(0)
    assert reward == 0 and not done
    assert env.action_masks()[env.actions.double]
    assert not env.action_masks()[: env.actions.bet_count].any()
    assert env.observation_space.contains(obs)


@pytest.mark.parametrize(
    "bankroll,total", [(15000, 0.5), (10000, 0), (8000, -0.6), (500, -12.85), (0, -13)]
)
def test_reward_examples(config, bankroll, total):
    wealth = bankroll / config.money.initial_bankroll_minor - 1
    assert wealth + sum(terminal_penalties(bankroll, config).values()) == pytest.approx(total)


@given(seed=st.integers(min_value=0, max_value=100000))
@settings(max_examples=30, deadline=None)
def test_random_legal_trajectory_invariants(seed):
    config = load_config("configs/experiments/smoke.yaml")
    env = BlackjackSessionEnv(config)
    obs, _ = env.reset(seed=seed)
    rng = np.random.default_rng(seed)
    reward_sum = 0
    settled = 0
    while True:
        mask = env.action_masks()
        assert mask.any()
        action = int(rng.choice(np.flatnonzero(mask)))
        obs, reward, done, truncated, info = env.step(action)
        assert env.engine.bank.cash >= 0
        assert env.observation_space.contains(obs)
        assert all(np.isfinite(value).all() for value in obs.values())
        reward_sum += reward
        settled += "round" in info
        if done:
            assert not truncated
            expected = env.engine.settled / config.money.initial_bankroll_minor - 1
            expected += sum(terminal_penalties(env.engine.settled, config).values())
            assert reward_sum == pytest.approx(expected)
            assert settled == env.engine.completed
            with pytest.raises(ValueError):
                env.step(0)
            break


def test_observation_hidden_independence(config):
    env = BlackjackSessionEnv(config)
    env.reset(seed=7)
    rig(env.engine, ["5", "10", "6", "8", "K", "4"])
    env.step(0)
    before, mask = env.observation(), env.action_masks()
    hole = env.engine.dealer[-1]
    env.engine.dealer[-1] = Card(hole.uid, config.shoe.ranks.index("9"))
    env.engine.shoe.cards.reverse()
    after = env.observation()
    assert all(np.array_equal(before[k], after[k]) for k in before)
    assert np.array_equal(mask, env.action_masks())


def test_no_count_history_leak(config):
    config.features.counting = "none"
    env = BlackjackSessionEnv(config)
    env.reset(seed=7)
    before = env.observation()
    env.engine.ledger.counts[0] += 1
    env.engine.shoe.drawn += 1
    after = env.observation()
    assert all(np.array_equal(before[k], after[k]) for k in before)


def test_seed_replay(config):
    def run():
        env = BlackjackSessionEnv(config)
        env.reset(seed=71)
        rewards = []
        while True:
            _, reward, done, _, info = env.step(int(np.flatnonzero(env.action_masks())[0]))
            rewards.append(reward)
            if done:
                return rewards, info["session"]

    assert run() == run()


def test_scale_equivalence(config):
    def run(cfg):
        env = BlackjackSessionEnv(cfg)
        env.reset(seed=23)
        trace = []
        while True:
            _, reward, done, _, _ = env.step(int(np.flatnonzero(env.action_masks())[0]))
            trace.append(reward)
            if done:
                return trace

    original = run(config)
    config.money.initial_bankroll_minor *= 10
    config.money.minimum_bet_minor *= 10
    config.money.bet_increment_minor *= 10
    assert run(config) == pytest.approx(original)


def test_checker(config):
    from stable_baselines3.common.env_checker import check_env
    from blackjack_rl.env.checking import LegalCheckAdapter

    check_env(LegalCheckAdapter(BlackjackSessionEnv(config)))


def test_minimum_residual_and_last_round_ruin(config):
    config.session.rounds = 1
    config.money.initial_bankroll_minor = 1500
    env = BlackjackSessionEnv(config)
    env.reset(seed=1)
    rig(env.engine, ["10", "10", "7", "8"])
    env.step(0)
    _, reward, done, _, info = env.step(env.actions.stand)
    assert done and info["session"]["ruin"]
    assert env.engine.settled == 500
    assert info["ruin_penalty"] == -config.reward.ruin_weight


def test_control_modes(config):
    for mode in ("play", "betting"):
        config.features.control = mode
        env = BlackjackSessionEnv(config)
        env.reset(seed=1)
        rig(env.engine, ["5", "10", "6", "8", "K"])
        if mode == "play":
            assert env.action_masks().sum() == 1
        env.step(0)
        if mode == "betting":
            assert env.action_masks().sum() == 1
            assert env.action_masks()[env.actions.double]
