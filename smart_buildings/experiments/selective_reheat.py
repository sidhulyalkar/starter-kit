"""Experimental observation-only selective reheat limiter.

Scientific motivation: a fixed 50% reheat cap suppressed severe thermal
jumps but worsened full-month comfort, energy/carbon and overall reward.
This controller keeps unmodified PID-water heating whenever a target zone
is cooling, deeply below its heating setpoint, or has no valid momentum
estimate. It ONLY caps a moderately underheated zone already warming
rapidly, where continued large reheat is a candidate overshoot hazard.

No simulator thermal source, heat capacity, occupancy future, rewards,
privileged plant dynamics, or other inaccessible field is read. All control
input features are observation and metadata that the policy already receives.

This is a hypothesis, NOT a safety guarantee or competition-selected policy.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import numpy as np

from smart_buildings.baselines.pid_plus import PIDWaterResetPolicy


@dataclass(frozen=True)
class SelectiveReheatConfig:
    vav_id: str = "VAV_1"
    cap_native_fraction: float = 0.50
    min_shortfall_k: float = 1.00
    max_shortfall_k: float = 2.00
    enter_rise_k_per_step: float = 0.25
    exit_rise_k_per_step: float = 0.05

    def __post_init__(self) -> None:
        if not self.vav_id.startswith("VAV_"):
            raise ValueError("Target must be a VAV device")
        for key in ("cap_native_fraction", "min_shortfall_k",
                    "max_shortfall_k", "enter_rise_k_per_step",
                    "exit_rise_k_per_step"):
            value = getattr(self, key)
            if not math.isfinite(value):
                raise ValueError(f"{key} nonfinite")
        if not 0.0 <= self.cap_native_fraction <= 1.0:
            raise ValueError("Native cap fraction outside [0,1]")
        if not 0 <= self.min_shortfall_k < self.max_shortfall_k:
            raise ValueError("Invalid cold-shortfall window")
        if not 0 <= self.exit_rise_k_per_step < self.enter_rise_k_per_step:
            raise ValueError("Exit threshold must be below enter threshold")


def decide_reheat_cap(
    temperature_k: float,
    heating_setpoint_k: float,
    previous_temperature_k: float | None,
    was_gated: bool,
    config: SelectiveReheatConfig,
) -> tuple[bool, str]:
    """Pure gate used both by the policy and its independent tests."""
    if not all(math.isfinite(float(v)) for v in
               (temperature_k, heating_setpoint_k)):
        return False, "invalid_current_sensor"
    if previous_temperature_k is None or not math.isfinite(
        float(previous_temperature_k)
    ):
        return False, "no_valid_momentum"
    shortfall = heating_setpoint_k - temperature_k
    if shortfall < config.min_shortfall_k:
        return False, "reheat_near_target_or_unneeded"
    if shortfall > config.max_shortfall_k:
        return False, "deep_cold_preserve_authority"
    observed_rise = temperature_k - previous_temperature_k
    if observed_rise <= 0:
        return False, "not_warming_preserve_authority"
    min_rise = (config.exit_rise_k_per_step if was_gated
                else config.enter_rise_k_per_step)
    if observed_rise < min_rise:
        return False, "warming_too_slow"
    return True, "warming_moderately_cold_zone"


class SelectiveReheatPolicy(PIDWaterResetPolicy):
    """PID-water with one optional observation-only VAV reheat cap."""

    CONFIG = SelectiveReheatConfig()
    POLICY_NAME = "vav1_momentum50"

    @property
    def name(self) -> str:
        return self.POLICY_NAME

    def __init__(self, building_meta_data: dict[str, Any], action_spec):
        super().__init__(building_meta_data, action_spec)
        cfg = self.CONFIG
        self._cfg = cfg
        names = list(self.field_names)
        temp_field = f"{cfg.vav_id}_zone_air_temperature_sensor"
        heat_field = (
            f"{cfg.vav_id}_zone_air_heating_temperature_setpoint_sensor"
        )
        if temp_field not in names or heat_field not in names:
            raise ValueError("Required controller-visible zone sensors missing")
        index_map = building_meta_data["zone_action_indices"]
        if cfg.vav_id not in index_map:
            raise ValueError("Requested VAV has no action metadata")
        self._temp_idx = names.index(temp_field)
        self._heating_setpoint_idx = names.index(heat_field)
        if "reheat_valve_percentage_command" not in index_map[cfg.vav_id]:
            raise ValueError("Requested VAV is missing reheat action")
        self._reheat_action_idx = index_map[cfg.vav_id][
            "reheat_valve_percentage_command"]
        if not 0 <= self._reheat_action_idx < self.n_continuous:
            raise ValueError("Reheat index is outside continuous action array")
        self._cap_cmd = np.float32(2.0 * cfg.cap_native_fraction - 1.0)
        self._last_zone_temp_k: float | None = None
        self._gate_active = False
        self.intervention_modified_steps = 0
        self.gate_active_steps = 0
        self.gate_reason_counts: dict[str, int] = {}

    def reset(self) -> None:
        super().reset()
        if hasattr(self, "_water_integral"):
            self._water_integral = 0.0
        self._last_zone_temp_k = None
        self._gate_active = False
        self.intervention_modified_steps = 0
        self.gate_active_steps = 0
        self.gate_reason_counts = {}

    def get_action(self, obs: np.ndarray) -> dict:
        # Preserve the original sensor array dtype and values when calling
        # the baseline. Casting float32 observations to float64 changed a
        # no-op controller's reward by a few billionths in a public pilot.
        x = np.asarray(obs)
        if x.ndim != 1 or len(x) != len(self.field_names):
            raise ValueError("Observation shape mismatch")
        if not np.isfinite(x).all():
            raise ValueError("Nonfinite observation; cannot safely control")
        # Exactly one call to the underlying stateful controller per step.
        action = super().get_action(x)
        continuous = np.asarray(action["continuous_action"], dtype=np.float32).copy()
        discrete = np.asarray(action["discrete_action"], dtype=np.int32).copy()
        if not np.isfinite(continuous).all():
            raise ValueError("Baseline emitted nonfinite command")
        current_k = float(self._denormalize_zone_temp(x[self._temp_idx]))
        heating_setpoint_k = float(x[self._heating_setpoint_idx])
        should_gate, reason = decide_reheat_cap(
            current_k, heating_setpoint_k, self._last_zone_temp_k,
            self._gate_active, self._cfg
        )
        self.gate_reason_counts[reason] = self.gate_reason_counts.get(reason, 0) + 1
        if should_gate:
            self.gate_active_steps += 1
            original = continuous[self._reheat_action_idx]
            continuous[self._reheat_action_idx] = min(original, self._cap_cmd)
            if continuous[self._reheat_action_idx] < original:
                self.intervention_modified_steps += 1
        self._gate_active = should_gate
        self._last_zone_temp_k = current_k
        return {"continuous_action": continuous, "discrete_action": discrete}


class SelectiveReheat75Policy(SelectiveReheatPolicy):
    CONFIG = SelectiveReheatConfig(cap_native_fraction=0.75)
    POLICY_NAME = "vav1_momentum75"
