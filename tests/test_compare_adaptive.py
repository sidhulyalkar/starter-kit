"""Fail-closed tests for comparative performance reports."""
import json

import pytest

from scripts.compare_adaptive import compare


def _write(root, building, weather, policy, reward, steps=8640):
    d = root / building / weather / policy
    d.mkdir(parents=True, exist_ok=True)
    (d / "summary.json").write_text(json.dumps({
        "total_reward": reward, "total_steps": steps,
    }))


def test_paired_loss_and_sign(tmp_path):
    for building in ("building_1", "building_2"):
        _write(tmp_path, building, "phoenix_1", "pid_water", -864.0)
        _write(tmp_path, building, "phoenix_1", "adaptive", -648.0)
    report = compare(tmp_path, expected=2)
    assert report["scenarios"] == 2
    assert report["reference_mean_loss"] == pytest.approx(0.1)
    assert report["candidate_mean_loss"] == pytest.approx(0.075)
    assert report["relative_improvement_pct"] == pytest.approx(25.0)
    assert report["wins"] == 2


def test_missing_scenario_is_rejected(tmp_path):
    _write(tmp_path, "building_1", "denver_1", "adaptive", -80)
    _write(tmp_path, "building_2", "denver_2", "pid_water", -100)
    with pytest.raises(ValueError, match="mismatched"):
        compare(tmp_path, expected=1)


@pytest.mark.parametrize("steps", [1, 288, 8641])
def test_episode_lengths_not_comparable(tmp_path, steps):
    _write(tmp_path, "building_1", "atlanta_1", "adaptive", -10, steps)
    _write(tmp_path, "building_1", "atlanta_1", "pid_water", -20)
    with pytest.raises(ValueError, match="exactly 8640"):
        compare(tmp_path, expected=1)


@pytest.mark.parametrize("reward", [float("nan"), float("inf"), 50.0, True])
def test_bad_rewards_are_rejected(tmp_path, reward):
    _write(tmp_path, "building_1", "atlanta_1", "adaptive", reward)
    _write(tmp_path, "building_1", "atlanta_1", "pid_water", -20)
    with pytest.raises(ValueError, match="(invalid|nonnumeric) reward"):
        compare(tmp_path, expected=1)
