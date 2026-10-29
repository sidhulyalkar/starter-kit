"""Controlled, simulator-only local VAV actuator interventions.

The intervention modifies only named VAV action indices *after* the official
PID-water policy has computed its commands. All other controller logic is
unchanged, and each arm receives an independently reset simulator. These are
research treatments, not proposed safe competition policies.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import numpy as np

from smart_buildings.baselines.pid_plus import PIDWaterResetPolicy


@dataclass(frozen=True)
class VAVCaps:
    vav_id: str = "VAV_1"
    damper_max_fraction: float = 1.0
    reheat_max_fraction: float = 1.0

    def __post_init__(self) -> None:
        if not self.vav_id or not self.vav_id.startswith("VAV_"):
            raise ValueError("Must specify a real VAV device ID")
        for label, value in (
            ("damper_max_fraction", self.damper_max_fraction),
            ("reheat_max_fraction", self.reheat_max_fraction),
        ):
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{label} must be finite and within [0,1]")

    @staticmethod
    def normalize_fraction(fraction: float) -> np.float32:
        if not math.isfinite(fraction) or not 0 <= fraction <= 1:
            raise ValueError("Expected native 0-1 actuator fraction")
        return np.float32(2.0 * fraction - 1.0)


def cap_vav_command(
    action: dict[str, np.ndarray],
    metadata: dict[str, Any],
    limits: VAVCaps,
) -> dict[str, np.ndarray]:
    """Return an independent command with *only* target VAV changes.

    The official output is normalized [-1,1], while the physical VAV
    damper/reheat positions are [0,1]. A 0.75 native fraction maps to +0.5
    normalized, not +0.75 normalized.
    """
    zones = metadata["zone_action_indices"]
    if limits.vav_id not in zones:
        raise ValueError(f"Target VAV absent in this building: {limits.vav_id}")
    indices = zones[limits.vav_id]
    damper_index = indices.get("supply_air_damper_percentage_command")
    reheat_index = indices.get("reheat_valve_percentage_command")
    if damper_index is None or reheat_index is None:
        raise ValueError(f"Missing target actuator indices: {limits.vav_id}")
    c = np.asarray(action["continuous_action"], dtype=np.float32).copy()
    d = np.asarray(action["discrete_action"], dtype=np.int32).copy()
    if c.ndim != 1 or d.ndim != 1 or not np.isfinite(c).all():
        raise ValueError("Unexpected action shape/finite values")
    if not (0 <= damper_index < len(c) and 0 <= reheat_index < len(c)):
        raise ValueError("Target action index out of range")
    if damper_index == reheat_index:
        raise ValueError("Damper and reheat indices must not alias")
    if np.any(np.abs(c) > 1.000001) or np.any((d!=0)&(d!=1)):
        raise ValueError("Invalid baseline action")
    c[damper_index] = min(
        c[damper_index], VAVCaps.normalize_fraction(limits.damper_max_fraction)
    )
    c[reheat_index] = min(
        c[reheat_index], VAVCaps.normalize_fraction(limits.reheat_max_fraction)
    )
    return {"continuous_action": c, "discrete_action": d}


def intervention_policy_class(
    name: str,
    limits: VAVCaps,
) -> type[PIDWaterResetPolicy]:
    """Build an experiment arm; validates the zone before running a rollout."""
    if not name or name == "pid_water":
        raise ValueError("Use a unique experimental policy name")

    class LocalVAVTreatment(PIDWaterResetPolicy):
        @property
        def name(self) -> str:
            return name

        def __init__(self, building_meta_data, action_spec):
            super().__init__(building_meta_data, action_spec)
            self._limits = limits
            target = building_meta_data["zone_action_indices"].get(limits.vav_id)
            if not target or not {"supply_air_damper_percentage_command",
                                  "reheat_valve_percentage_command"}.issubset(target):
                raise ValueError("Missing intervention VAV device/actions")

        def get_action(self, obs):
            result = super().get_action(obs)
            return cap_vav_command(result, self.metadata, self._limits)

    LocalVAVTreatment.__name__ = "LocalVAVTreatment_" + name
    return LocalVAVTreatment


EXPERIMENT_ARMS = {
    "pid_water": None,
    "vav1_sham": VAVCaps(),
    "vav1_reheat75": VAVCaps(reheat_max_fraction=0.75),
    "vav1_reheat50": VAVCaps(reheat_max_fraction=0.50),
    "vav1_damper75": VAVCaps(damper_max_fraction=0.75),
    "vav1_both75": VAVCaps(damper_max_fraction=0.75,
                           reheat_max_fraction=0.75),
}
