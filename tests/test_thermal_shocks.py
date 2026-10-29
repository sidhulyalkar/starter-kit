"""Locate a synthetic seven-Kelvin transition and its exact action context."""
from pathlib import Path

import numpy as np
import pytest

from smart_buildings.experiments.thermal_records import TransitionWriter
from smart_buildings.experiments.thermal_shocks import inspect_shocks


def metadata():
    return {
        "observation_field_names": [
            "VAV_1_zone_air_temperature_sensor",
            "ahs_ahu_1_outside_air_temperature_sensor",
        ],
        "n_continuous_action": 4, "n_discrete_action": 2,
        "observation_normalization_metadata": {
            "zone_air_temperature_sensor": {
                "sample_mean": 290.0, "sample_std": 10.0,
                "sample_variance": 100.0
            }
        },
        "action_normalization_metadata": {},
        "temp_indices": [0], "pressure_indices": [1],
        "recirculation_indices": [], "water_indices": [],
        "zone_action_indices": {
            "VAV_1": {
                "supply_air_damper_percentage_command": 2,
                "reheat_valve_percentage_command": 3
            }
        },
        "ahu_zone_assignments": {"ahu_1": ["room_1"]},
        "floor_plan": None, "zone_map": None,
    }


def make_archive(path: Path):
    with TransitionWriter(path, metadata(), {
        "scenario": "building_0410/denver_8", "policy": "pid_water"
    }) as writer:
        for step, (now, following) in enumerate([(0.0, 0.01), (0.01, 0.71), (0.71, 0.72)]):
            writer.write(
                step, [now, 280.0], {
                    "continuous_action": np.asarray([0.0, 0.5, -0.2, 1.0]),
                    "discrete_action": np.asarray([1, 1])
                }, [following, 280.0], -0.1
            )


def test_detect_exact_large_transition(tmp_path):
    p = tmp_path / "x.jsonl.gz"
    make_archive(p)
    d = inspect_shocks(p, topn=2)
    assert d["total_zone_transitions"] == 3
    assert d["threshold_counts"]["ge_5_kelvin"] == 1
    assert d["threshold_counts"]["ge_2_kelvin"] == 1
    first = d["top_observed_transitions"][0]
    assert first["step"] == 1
    assert first["minute"] == 5
    assert first["zone"] == "VAV_1"
    assert first["ahu"] == "ahu_1"
    assert first["delta_kelvin"] == pytest.approx(7.0)
    assert first["ahu_run"] == 1
    assert first["zone_reheat_command"] == 1.0
    assert first["outside_kelvin"] == 280.0


def test_threshold_validation(tmp_path):
    p = tmp_path / "x.jsonl.gz"
    make_archive(p)
    with pytest.raises(ValueError):
        inspect_shocks(p, topn=0)
    with pytest.raises(ValueError):
        inspect_shocks(p, thresholds_k=(-1.0,))
