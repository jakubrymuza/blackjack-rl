import pytest
from pydantic import ValidationError

from blackjack_rl.config import load_config


@pytest.mark.parametrize(
    "override",
    [
        "training.batch_size=13",
        "rules.allow_split=true",
        "money.initial_bankroll_minor=1",
        "shoe.penetration_fraction=1",
        "training.seeeds=[1]",
        "shoe.hilo_weights=[1]",
        "money.bet_increment_minor=3",
        "features.counting=oracle",
        "reward.ruin_weight=-1",
    ],
)
def test_invalid(override):
    with pytest.raises(ValidationError):
        load_config("configs/experiments/smoke.yaml", [override])


def test_cycle(tmp_path):
    path = tmp_path / "cycle.yaml"
    path.write_text("include: [cycle.yaml]")
    with pytest.raises(ValueError, match="cycle"):
        load_config(path)


def test_resolved_roundtrip(config, tmp_path):
    path = tmp_path / "config.json"
    path.write_text(config.json())
    assert load_config(path).config_hash == config.config_hash
