"""Verify observable-only selective cap, reset and cold-state overrides."""
from dataclasses import replace

import numpy as np
import pytest

from smart_buildings.baselines.pid_plus import PIDWaterResetPolicy
from smart_buildings.experiments.selective_reheat import (
    SelectiveReheatConfig, SelectiveReheatPolicy,
    SelectiveReheat75Policy, decide_reheat_cap,
)


def cfg():
    return SelectiveReheatConfig()


@pytest.mark.parametrize("current,low,previous,was,expected", [
    (292.5, 294.0, None, False, False),  # no preceding observation
    (291.5, 294.0, 290.0, False, False),  # >2K cold: full authority
    (292.5, 294.0, 292.6, False, False),  # temperature falling
    (292.5, 294.0, 292.1, False, True),   # fast rise moderately cold
    (292.5, 294.0, 292.42, True, True),  # hysteresis: hold with slow rise
    (292.5, 294.0, 292.42, False, False),# same rise, no engagement
    (292.5, 294.0, 292.49, True, False),# below exit threshold
    (293.2, 294.0, 292.5, True, False), # near target: no cap
])
def test_gate_conditions(current,low,previous,was,expected):
    actual, reason = decide_reheat_cap(current,low,previous,was,cfg())
    assert actual is expected
    assert isinstance(reason,str) and reason


def test_fraction_and_threshold_invariants():
    assert SelectiveReheat75Policy.CONFIG.cap_native_fraction==0.75
    for params in (
        {"cap_native_fraction":-1},
        {"cap_native_fraction":1.2},
        {"max_shortfall_k":0.5},
        {"enter_rise_k_per_step":0.04},
        {"exit_rise_k_per_step":0.4},
    ):
        with pytest.raises(ValueError):
            SelectiveReheatConfig(**params)
    with pytest.raises(ValueError):
        SelectiveReheatConfig(cap_native_fraction=float("nan"))


def fake_policy():
    # Construct only the action guard; a separate CI smoke test constructs
    # the complete policy with official simulator metadata and AHU logic.
    policy = object.__new__(SelectiveReheatPolicy)
    policy.field_names = ["VAV_1_zone_air_temperature_sensor",
                          "VAV_1_zone_air_heating_temperature_setpoint_sensor"]
    policy.zone_air_temperature_sensor_mean = 293.15
    policy.zone_air_temperature_sensor_std = 10.0
    policy._temp_idx = 0
    policy._heating_setpoint_idx = 1
    policy._reheat_action_idx = 2
    policy._cfg = cfg()
    policy._cap_cmd = np.float32(0.0)
    policy._last_zone_temp_k = None
    policy._gate_active = False
    policy.intervention_modified_steps = 0
    policy.gate_active_steps = 0
    policy.gate_reason_counts = {}
    return policy


def test_only_target_reheat_index_changes(monkeypatch):
    def base_action(_self, obs):
        return {"continuous_action":
                np.asarray([-1.,0.3,1.,0.7,-0.6], dtype=np.float32),
                "discrete_action":np.asarray([1,0], dtype=np.int32)}
    monkeypatch.setattr(PIDWaterResetPolicy, "get_action", base_action)
    p = fake_policy()
    warm_up = np.asarray([(292.0-293.15)/10,294.0])
    rising = np.asarray([(292.5-293.15)/10,294.0])
    first = p.get_action(warm_up)
    second = p.get_action(rising)
    assert first["continuous_action"][2] == 1.0
    assert second["continuous_action"][2] == 0.0
    np.testing.assert_array_equal(
        second["continuous_action"][[0,1,3,4]],
        np.asarray([-1.,0.3,0.7,-0.6],dtype=np.float32),
    )
    np.testing.assert_array_equal(second["discrete_action"],[1,0])
    assert p.intervention_modified_steps == 1
    assert p.gate_active_steps == 1
    assert p.gate_reason_counts["warming_moderately_cold_zone"] == 1


def test_deep_cold_restores_original_heating(monkeypatch):
    monkeypatch.setattr(PIDWaterResetPolicy,"get_action",
                        lambda _self,obs: {
                            "continuous_action":np.asarray([0.,1.,1.],dtype=np.float32),
                            "discrete_action":np.asarray([1],dtype=np.int32)})
    p = fake_policy()
    p._last_zone_temp_k = 290.0
    obs=np.asarray([(291.0-293.15)/10,294.0])
    action=p.get_action(obs)
    assert action["continuous_action"][2] == 1.0
    assert p.intervention_modified_steps == 0
    assert p.gate_reason_counts["deep_cold_preserve_authority"] == 1


def test_nonfinite_observation_fails_closed(monkeypatch):
    monkeypatch.setattr(PIDWaterResetPolicy,"get_action",lambda self,obs:None)
    p=fake_policy()
    with pytest.raises(ValueError,match="Nonfinite"):
        p.get_action(np.asarray([float("nan"),294.0]))
    with pytest.raises(ValueError,match="shape"):
        p.get_action(np.asarray([293.0]))
