"""Unit tests: strictly aligned, leak-resistant thermal transition archives."""
import gzip
import json

import numpy as np
import pytest

from smart_buildings.experiments.thermal_records import (
    TransitionWriter, validate_transition, verify_archive,
)


@pytest.fixture
def metadata():
    return {
        "observation_field_names": ["zone_temp", "outside_temp"],
        "n_continuous_action": 2,
        "n_discrete_action": 1,
        "observation_normalization_metadata": {
            "zone_air_temperature_sensor": {
                "sample_mean": np.float64(293.15),
                "sample_std": np.float64(10.0),
                "sample_variance": np.float64(100.0),
            }
        },
        "action_normalization_metadata": {},
        "temp_indices": [0],
        "pressure_indices": [1],
        "recirculation_indices": [],
        "water_indices": [],
        "zone_action_indices": {},
        "ahu_zone_assignments": {"ahu_1": ["room_1"]},
        "floor_plan": np.zeros((2, 2), dtype=np.int32),
        "zone_map": np.ones((2, 2), dtype=np.int32),
    }


def action():
    return {
        "continuous_action": np.array([-0.25, 0.75], dtype=np.float32),
        "discrete_action": np.array([1], dtype=np.int32),
    }


def test_roundtrip_manifest_and_labels(tmp_path, metadata):
    file = tmp_path / "rollout.jsonl.gz"
    with TransitionWriter(file, metadata,
                          {"scenario": "building_train_1/atlanta_2",
                           "policy": "pid_water"}) as writer:
        writer.write(0, [0.0, 1.0], action(), [0.01, 1.01], -0.05)
        writer.write(1, [0.01, 1.01], action(), [0.02, 1.02], -0.08)
    manifest = verify_archive(file)
    assert manifest["status"] == "complete"
    assert manifest["record_count"] == 2
    assert manifest["floor_plan_digest"]["shape"] == [2, 2]
    assert manifest["observation_normalization_metadata"][
        "zone_air_temperature_sensor"]["sample_mean"] == 293.15
    with gzip.open(file, "rt") as f:
        row = json.loads(next(f))
    assert row["observation_t"] == [0.0, 1.0]
    assert row["observation_tp1"] == [0.01, 1.01]
    assert "scenario" not in row
    assert "weather" not in row
    assert row["reward_t"] == -0.05


@pytest.mark.parametrize("bad_action", [
    {"continuous_action": [1.2, 0.0], "discrete_action": [1]},
    {"continuous_action": [0.0, 0.0], "discrete_action": [2]},
    {"continuous_action": [float("nan"), 0.0], "discrete_action": [1]},
])
def test_invalid_action_rejected(bad_action):
    with pytest.raises(ValueError):
        validate_transition(0, [1.0, 2.0], bad_action, [2.0, 3.0],
                            -0.1, 2, 2, 1)


def test_invalid_next_observation_rejected():
    with pytest.raises(ValueError, match="obs_tp1"):
        validate_transition(0, [1.0, 2.0], action(), [float("inf"), 3.0],
                            -0.1, 2, 2, 1)


def test_noncontiguous_step_and_aborted_manifest(tmp_path, metadata):
    file = tmp_path / "aborted.jsonl.gz"
    with pytest.raises(ValueError, match="Noncontiguous"):
        with TransitionWriter(file, metadata, {"policy": "pid_water"}) as writer:
            writer.write(0, [0.0, 0.0], action(), [0.0, 0.0], -0.1)
            writer.write(2, [0.0, 0.0], action(), [0.0, 0.0], -0.1)
    assert json.loads(file.with_suffix(".manifest.json").read_text())["status"] == "aborted"
    with pytest.raises(ValueError, match="not completed"):
        verify_archive(file)


def test_corrupt_archive_rejected(tmp_path, metadata):
    file = tmp_path / "corrupt.jsonl.gz"
    with TransitionWriter(file, metadata, {}) as writer:
        writer.write(0, [0.0, 0.0], action(), [0.0, 0.0], -0.1)
    file.write_bytes(file.read_bytes() + b"corruption")
    with pytest.raises(ValueError, match="checksum"):
        verify_archive(file)


def test_duplicate_field_names_rejected(tmp_path, metadata):
    metadata["observation_field_names"] = ["duplicate", "duplicate"]
    with pytest.raises(ValueError, match="not unique"):
        TransitionWriter(tmp_path / "bad.jsonl.gz", metadata, {})


def test_invalid_reward_rejected():
    with pytest.raises(ValueError, match="Reward"):
        validate_transition(0, [1.0, 2.0], action(), [2.0, 3.0],
                            True, 2, 2, 1)
