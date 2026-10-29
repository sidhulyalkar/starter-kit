"""Analytic persistence errors for a controlled linear thermal trajectory."""
import numpy as np
import pytest

from scripts.evaluate_persistence import score_persistence_archive
from smart_buildings.experiments.thermal_records import TransitionWriter


def metadata(zone_name="VAV_1_zone_air_temperature_sensor"):
    return {
        "observation_field_names": [zone_name, "outside_air_temp"],
        "n_continuous_action": 1,
        "n_discrete_action": 1,
        "observation_normalization_metadata": {
            "zone_air_temperature_sensor": {
                "sample_mean": 293.15,
                "sample_std": 10.0,
                "sample_variance": 100.0
            }
        },
        "action_normalization_metadata": {},
        "temp_indices": [0],
        "pressure_indices": [],
        "recirculation_indices": [],
        "water_indices": [],
        "zone_action_indices": {},
        "ahu_zone_assignments": {"ahu_1": ["room_1"]},
        "floor_plan": None, "zone_map": None,
    }


def record(path, meta):
    with TransitionWriter(path, meta, {"scenario": "building_0250/atlanta_1",
                                       "policy": "pid_water"}) as writer:
        for t in range(15):
            obs = [t * 0.01, 285.0]
            nxt = [(t + 1) * 0.01, 285.0]
            action = {"continuous_action": np.zeros(1, dtype=np.float32),
                      "discrete_action": np.ones(1, dtype=np.int32)}
            writer.write(t, obs, action, nxt, -0.01)


def test_known_kelvin_change_at_multiple_horizons(tmp_path):
    path = tmp_path / "x.jsonl.gz"
    record(path, metadata())
    scores = score_persistence_archive(path)
    values = scores["horizons_5_min_steps"]
    assert values["1"]["rmse_kelvin"] == pytest.approx(0.1)
    assert values["3"]["rmse_kelvin"] == pytest.approx(0.3)
    assert values["12"]["rmse_kelvin"] == pytest.approx(1.2)
    assert values["12"]["n_zone_forecasts"] == 4
    assert scores["zone_count"] == 1


def test_missing_zone_rejected(tmp_path):
    path = tmp_path / "x.jsonl.gz"
    record(path, metadata("not_a_zone"))
    with pytest.raises(ValueError, match="No zone-temperature"):
        score_persistence_archive(path)


def test_insufficient_horizon_marked_unavailable(tmp_path):
    path = tmp_path / "x.jsonl.gz"
    record(path, metadata())
    values = score_persistence_archive(path, (16,))["horizons_5_min_steps"]["16"]
    assert values["mae_kelvin"] is None
    assert values["n_zone_forecasts"] == 0
