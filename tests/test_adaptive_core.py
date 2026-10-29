"""Fast deterministic tests of the controller's proposed changes."""
from dataclasses import replace

import pytest

from smart_buildings.experiments.adaptive_core import (
    AdaptiveConfig, aggregate_demand, deadband_error, projected_temperature,
    zone_commands, zone_request,
)


def test_deadband_sign():
    assert deadband_error(289.0, 293.0, 298.0) == 4.0
    assert deadband_error(301.0, 293.0, 298.0) == -3.0
    assert deadband_error(295.0, 293.0, 298.0) == 0.0
    assert deadband_error(293.0, 293.0, 298.0) == 0.0


@pytest.mark.parametrize("lo,hi", [(300, 290), (294, 294), (float("nan"), 300)])
def test_invalid_band(lo, hi):
    with pytest.raises(ValueError):
        deadband_error(295.0, lo, hi)


@pytest.mark.parametrize("change", [-2.0, 0.0, 2.0])
def test_projection_is_bounded(change):
    cfg = AdaptiveConfig()
    result = projected_temperature(296.0, 296.0 - change, cfg)
    assert abs(result - 296.0) <= cfg.max_forecast_delta_k + 1e-12


def test_no_history_has_zero_forecast_shift():
    cfg = AdaptiveConfig()
    assert projected_temperature(294.0, None, cfg) == 294.0
    assert zone_request(292.0, None, 293.0, 298.0, cfg) == 1.0


def test_anticipation_preserves_sign():
    cfg = AdaptiveConfig()
    warming = zone_request(292.0, 291.95, 293.0, 298.0, cfg)
    cooling = zone_request(292.0, 292.05, 293.0, 298.0, cfg)
    assert warming < cooling
    assert 0 < warming < cooling


def test_outlier_ablation():
    cfg = AdaptiveConfig()
    vals = [0.0, 0.0, 3.0]
    assert aggregate_demand(vals, cfg) > 1.0
    off = replace(cfg, max_outlier_weight=0.0)
    assert aggregate_demand(vals, off) == 1.0


def test_opposing_zones_are_not_hidden_by_mean():
    cfg = AdaptiveConfig()
    assert aggregate_demand([5.0, -1.0, -1.0, -1.0, -1.0], cfg) > 0


def test_empty_demand_is_zero():
    assert aggregate_demand([], AdaptiveConfig()) == 0.0


@pytest.mark.parametrize("demand", [-100.0, -2.0, 0.0, 2.0, 100.0])
def test_normalized_zone_actuators(demand):
    damper, reheat = zone_commands(demand, AdaptiveConfig())
    assert -1.0 <= damper <= 1.0
    assert -1.0 <= reheat <= 1.0
    if demand <= 0:
        assert reheat == -1.0


@pytest.mark.parametrize("config", [
    dict(horizon_steps=-1),
    dict(max_observed_delta_k=0.0),
    dict(forecast_weight=1.1),
    dict(max_outlier_weight=-0.01),
])
def test_invalid_config(config):
    with pytest.raises(ValueError):
        AdaptiveConfig(**config)


def test_nonfinite_requests_are_rejected():
    with pytest.raises(ValueError):
        aggregate_demand([float("nan")], AdaptiveConfig())
    with pytest.raises(ValueError):
        zone_commands(float("inf"), AdaptiveConfig())
