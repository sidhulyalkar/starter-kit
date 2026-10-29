"""Experimental PID-water extensions; NOT the competition-selected policy.

No future observations or hidden simulator attributes are used. The initial
controller retains the official economizer, water reset, and action indexing,
while varying zone demand aggregation and bounded temperature anticipation.
"""
from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np

from smart_buildings.baselines.pid_plus import PIDWaterResetPolicy
from smart_buildings.experiments.adaptive_core import (
    AdaptiveConfig, aggregate_demand, zone_commands, zone_request,
)


class AdaptivePIDPolicy(PIDWaterResetPolicy):
    """An experimental, topology-independent comparison arm."""

    CONFIG = AdaptiveConfig()
    USE_SENSOR_BAND = True
    USE_WORST_ZONE = False

    @property
    def name(self) -> str:
        return "adaptive"

    def __init__(self, building_meta_data: dict[str, Any], action_spec: Any):
        super().__init__(building_meta_data, action_spec)
        self.config = self.CONFIG
        self._expected_obs = len(self.field_names)
        self._last_zone_k: dict[str, float] = {}
        if self.n_discrete != len(self.ahu_controllers) + 1:
            raise ValueError("Unexpected AHU/pump discrete-action layout")

    def reset(self) -> None:
        """Clear persistent state on every new episode."""
        super().reset()
        self._last_zone_k = {}
        if hasattr(self, "_water_integral"):
            self._water_integral = 0.0

    def _measure(self, obs: np.ndarray, vav: str, idx: int,
                 low: float, high: float) -> tuple[float, float, float]:
        temp = float(self._denormalize_zone_temp(float(obs[idx])))
        lower, upper = self._zone_band(obs, vav, low, high)
        if not np.isfinite(lower) or not np.isfinite(upper) or lower >= upper:
            lower, upper = low, high
        return temp, float(lower), float(upper)

    def _demand(self, obs: np.ndarray, vav: str, idx: int,
                low: float, high: float) -> float:
        temp, lower, upper = self._measure(obs, vav, idx, low, high)
        return zone_request(temp, self._last_zone_k.get(vav),
                            lower, upper, self.config)

    def _ahu_error(self, obs: np.ndarray, ctrl: dict,
                   low: float, high: float) -> float:
        requests = [
            self._demand(obs, vav, idx, low, high)
            for vav, idx, _ in ctrl["zones"]
        ]
        return float(aggregate_demand(requests, self.config))

    def _command_zones(self, continuous: np.ndarray, zones: list,
                       obs: np.ndarray, low: float, high: float) -> None:
        for vav, idx, actions in zones:
            demand = self._demand(obs, vav, idx, low, high)
            damper, reheat = zone_commands(demand, self.config)
            damper_idx = actions.get("supply_air_damper_percentage_command")
            reheat_idx = actions.get("reheat_valve_percentage_command")
            if damper_idx is not None:
                continuous[damper_idx] = np.float32(damper)
            if reheat_idx is not None:
                continuous[reheat_idx] = np.float32(reheat)

    def get_action(self, obs: np.ndarray) -> dict[str, np.ndarray]:
        obs = np.asarray(obs, dtype=np.float64)
        if obs.ndim != 1 or obs.size != self._expected_obs:
            raise ValueError(f"Observation shape mismatch: {obs.shape}")
        if not np.all(np.isfinite(obs)):
            raise ValueError("Nonfinite observation; cannot produce valid command")

        action = super().get_action(obs)
        continuous = np.asarray(action["continuous_action"], dtype=np.float32)
        discrete = np.asarray(action["discrete_action"], dtype=np.int32)
        if continuous.shape != (self.n_continuous,) or discrete.shape != (self.n_discrete,):
            raise ValueError("Action shape mismatch")
        if not np.all(np.isfinite(continuous)):
            raise ValueError("Nonfinite continuous action")
        if np.any(continuous < -1.000001) or np.any(continuous > 1.000001):
            raise ValueError("Continuous action outside [-1, 1]")
        if np.any((discrete != 0) & (discrete != 1)):
            raise ValueError("Invalid discrete run command")

        # Store only the current observed state after choosing the action.
        updated = {}
        for ctrl in self.ahu_controllers:
            for vav, idx, _ in ctrl["zones"]:
                updated[vav] = float(self._denormalize_zone_temp(float(obs[idx])))
        self._last_zone_k = updated
        return {
            "continuous_action": np.clip(continuous, -1, 1).astype(np.float32),
            "discrete_action": discrete,
        }


class SensorOnlyPolicy(AdaptivePIDPolicy):
    CONFIG = replace(AdaptivePIDPolicy.CONFIG, forecast_weight=0.0,
                     max_outlier_weight=0.0)

    @property
    def name(self) -> str:
        return "adaptive_sensor"


class OutlierOnlyPolicy(AdaptivePIDPolicy):
    CONFIG = replace(AdaptivePIDPolicy.CONFIG, forecast_weight=0.0)

    @property
    def name(self) -> str:
        return "adaptive_outlier"


class ForecastOnlyPolicy(AdaptivePIDPolicy):
    CONFIG = replace(AdaptivePIDPolicy.CONFIG, max_outlier_weight=0.0)

    @property
    def name(self) -> str:
        return "adaptive_forecast"
