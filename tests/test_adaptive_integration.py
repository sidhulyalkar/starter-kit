"""Real simulator contract tests. These are NOT evidence of score improvement."""
from pathlib import Path

import numpy as np
import pytest

from smart_buildings.evaluate import (
    create_building_meta_data, load_hybrid_environment, validate_action,
)
from smart_buildings.experiments.adaptive_policy import (
    AdaptivePIDPolicy, ForecastOnlyPolicy, OutlierOnlyPolicy, SensorOnlyPolicy,
)
from smart_buildings.smart_control.utils.scenario_generator import SimulationTracker


@pytest.fixture(scope="module")
def environment_metadata():
    directory = Path(__file__).resolve().parents[1] / "smart_buildings" / "training_scenarios"
    scenarios = sorted(directory.glob("*/*/config.gin"))
    assert scenarios, "No official generated scenario configurations present"
    path = scenarios[0]
    env = load_hybrid_environment(path)
    meta = create_building_meta_data(env, path.parent)
    return env, meta


@pytest.mark.parametrize("cls", [
    AdaptivePIDPolicy, SensorOnlyPolicy, OutlierOnlyPolicy, ForecastOnlyPolicy,
])
def test_actual_simulator_action_contract(cls, environment_metadata):
    env, metadata = environment_metadata
    tracker = SimulationTracker(env, vmin=280, vmax=310, capture_snapshots=False)
    tracker.reset()
    policy = cls(metadata, env.action_spec())
    obs = tracker.last_obs

    for _ in range(3):
        action = policy.get_action(obs)
        validated = validate_action(action, env.action_spec())
        continuous = validated["continuous_action"]
        discrete = validated["discrete_action"]
        assert continuous.dtype == np.float32
        assert discrete.dtype == np.int32
        assert np.all(np.isfinite(continuous))
        assert np.all(continuous >= -1) and np.all(continuous <= 1)
        assert np.all((discrete == 0) | (discrete == 1))
        obs, _, terminated, truncated, _ = tracker.step(validated)
        if terminated or truncated:
            break

    assert policy._last_zone_k
    policy.reset()
    assert not policy._last_zone_k


def test_bad_observation_rejected(environment_metadata):
    env, metadata = environment_metadata
    tracker = SimulationTracker(env, vmin=280, vmax=310, capture_snapshots=False)
    tracker.reset()
    policy = AdaptivePIDPolicy(metadata, env.action_spec())
    observation = np.asarray(tracker.last_obs, dtype=np.float64)

    with pytest.raises(ValueError, match="shape"):
        policy.get_action(observation[:-1])
    bad = observation.copy()
    bad[0] = np.nan
    with pytest.raises(ValueError, match="Nonfinite"):
        policy.get_action(bad)
    assert not policy._last_zone_k
