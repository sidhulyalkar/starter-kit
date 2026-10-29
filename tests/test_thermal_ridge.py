"""Test a shared thermal model, unit conversion and leakage barriers."""
import numpy as np
import pytest

from smart_buildings.experiments.thermal_records import TransitionWriter
from smart_buildings.experiments.thermal_ridge import (
    FEATURE_NAMES, ThermalRidge, ZoneFeatureMap, cross_building_evaluate,
    load_examples, metrics,
)


def metadata():
    names = [
        "VAV_1_zone_air_temperature_sensor",
        "ahs_ahu_1_outside_air_temperature_sensor",
        "hod_sin_000", "hod_cos_000", "dow_sin_000", "dow_cos_000",
    ]
    return {
        "observation_field_names": names,
        "n_continuous_action": 4, "n_discrete_action": 2,
        "observation_normalization_metadata": {
            "zone_air_temperature_sensor": {
                "sample_mean": 293.15, "sample_std": 10.0,
                "sample_variance": 100.0}},
        "action_normalization_metadata": {},
        "temp_indices": [0], "pressure_indices": [1],
        "recirculation_indices": [], "water_indices": [],
        "zone_action_indices": {
            "VAV_1": {
                "supply_air_damper_percentage_command": 2,
                "reheat_valve_percentage_command": 3}},
        "ahu_zone_assignments": {"ahu_1": ["room_1"]},
        "floor_plan": None, "zone_map": None,
    }


def make_archive(path, building, outside=285.0, n=100,
                 break_continuity=False):
    meta = metadata()
    normalized_t = 0.0
    with TransitionWriter(path, meta, {
        "scenario": f"{building}/atlanta_2", "policy": "pid_water"
    }) as writer:
        for t in range(n):
            reheat = 1.0 if (t // 4) % 2 else -1.0
            obs = [normalized_t, outside, 0.0, 1.0, 0.0, 1.0]
            if break_continuity and t == 4:
                obs[0] += 0.2
            continuous = [-0.5, 0.1, 0.8, reheat]
            # Synthetic linear zone dynamics, no hidden future weather.
            temp_k = 293.15 + normalized_t * 10.0
            next_k = temp_k + 0.005 * (outside - temp_k) + 0.02 * reheat
            next_normalized = (next_k - 293.15) / 10.0
            next_obs = [next_normalized, outside, 0.0, 1.0, 0.0, 1.0]
            writer.write(t, obs, {
                "continuous_action": continuous, "discrete_action": [1, 1]
            }, next_obs, -0.1)
            normalized_t = next_normalized


def test_map_converts_zone_temperature_to_kelvin(tmp_path):
    p = tmp_path / "a.jsonl.gz"
    make_archive(p, "building_0250", n=3)
    manifest, x, current, target = load_examples(p)
    assert current[0] == pytest.approx(293.15)
    assert x.shape == (3, len(FEATURE_NAMES))
    assert x[0, 0] == pytest.approx(285.0 - 293.15)
    assert target[0] == pytest.approx(293.15 + 0.005 * (285.0 - 293.15) - 0.02)


def test_ridge_fits_a_linear_physics_response():
    rng = np.random.default_rng(22)
    x = rng.normal(size=(2000, len(FEATURE_NAMES)))
    baseline = np.full(len(x), 295.0)
    target = baseline + 0.02 * x[:, 0] - 0.007 * x[:, 2] + 0.03 * x[:, 8]
    ridge = ThermalRidge.fit(x, baseline, target, alpha=0.01)
    report = metrics(target, ridge.predict(x, baseline))
    assert report["mae_kelvin"] < 0.00001


def test_cross_building_pilot_on_synthetic_physics(tmp_path):
    train = tmp_path / "train.jsonl.gz"
    evaluation = tmp_path / "eval.jsonl.gz"
    make_archive(train, "building_0250", outside=285, n=100)
    make_archive(evaluation, "building_0410", outside=290, n=100)
    report = cross_building_evaluate([train], [evaluation], alpha=0.01)
    assert report["train_buildings"] == ["building_0250"]
    assert report["eval_buildings"] == ["building_0410"]
    assert report["eval_zone_observations"] == 100
    assert np.isfinite(report["ridge"]["mae_kelvin"])
    assert report["persistence"]["mae_kelvin"] > 0


def test_train_eval_building_overlap_rejected(tmp_path):
    a = tmp_path / "a.jsonl.gz"
    b = tmp_path / "b.jsonl.gz"
    make_archive(a, "building_0250", n=10)
    make_archive(b, "building_0250", outside=290, n=10)
    with pytest.raises(ValueError, match="Building-level leakage"):
        cross_building_evaluate([a], [b])


def test_noncontinuous_recording_rejected(tmp_path):
    a = tmp_path / "a.jsonl.gz"
    make_archive(a, "building_0250", n=10, break_continuity=True)
    with pytest.raises(ValueError, match="Discontinuous sequence"):
        load_examples(a)


def test_nonphysical_outside_sensor_rejected(tmp_path):
    p = tmp_path / "a.jsonl.gz"
    make_archive(p, "building_0250", outside=400, n=2)
    with pytest.raises(ValueError, match="plausible Kelvin"):
        load_examples(p)


def test_bad_dimensions_rejected():
    with pytest.raises(ValueError):
        ThermalRidge.fit(np.ones((6, 2)), np.ones(6), np.ones(6))
    with pytest.raises(ValueError):
        metrics(np.array([1]), np.array([np.inf]))
