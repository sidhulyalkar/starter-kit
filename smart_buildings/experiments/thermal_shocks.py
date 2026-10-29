"""Locate large *observed* 5-minute thermal transitions in a verified archive.

This is a diagnostic, not a forecast model or causal explanation. It exposes
the exact time step, zone, measured temperature change and contemporaneous
commands for investigation of errors that dominate RMSE.
"""
from __future__ import annotations

import gzip
import heapq
import json
import math
from pathlib import Path

import numpy as np

from smart_buildings.experiments.thermal_records import verify_archive
from smart_buildings.experiments.thermal_ridge import ZoneFeatureMap, _sorted_ahu_ids


def inspect_shocks(path: Path, topn: int = 20,
                   thresholds_k: tuple[float, ...] = (1.0, 2.0, 5.0)) -> dict:
    if topn < 1 or not thresholds_k or any(
        not math.isfinite(v) or v <= 0 for v in thresholds_k
    ):
        raise ValueError("Invalid shock inspection thresholds/limits")
    manifest = verify_archive(path)
    mapping = ZoneFeatureMap.from_manifest(manifest)
    ahu_names = _sorted_ahu_ids(manifest["ahu_zone_assignments"])
    heap: list[tuple[float, int, dict]] = []
    counts = {f"ge_{v:g}_kelvin": 0 for v in thresholds_k}
    per_step: dict[int, int] = {}
    next_tie = 0
    total = 0
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for record in f:
            row = json.loads(record)
            step = int(row["step"])
            obs = np.asarray(row["observation_t"], dtype=float)
            future = np.asarray(row["observation_tp1"], dtype=float)
            action = np.asarray(row["continuous_action_t"], dtype=float)
            discrete = np.asarray(row["discrete_action_t"], dtype=int)
            for zone_i, zone_item in enumerate(mapping.zone_items):
                (sensor_idx, ahu_no, damper_idx, reheat_idx, supply_idx,
                 pressure_idx, recirc_idx, outside_idx) = zone_item
                current_k = float(obs[sensor_idx] * mapping.std_kelvin +
                                  mapping.mean_kelvin)
                next_k = float(future[sensor_idx] * mapping.std_kelvin +
                               mapping.mean_kelvin)
                delta = next_k - current_k
                magnitude = abs(delta)
                total += 1
                for threshold in thresholds_k:
                    if magnitude >= threshold:
                        counts[f"ge_{threshold:g}_kelvin"] += 1
                if magnitude >= 2:
                    per_step[step] = per_step.get(step, 0) + 1
                row_info = {
                    "step": step, "minute": step * 5,
                    "zone": mapping.zone_names[zone_i],
                    "ahu": ahu_names[ahu_no],
                    "current_kelvin": current_k,
                    "next_kelvin": next_k,
                    "delta_kelvin": delta,
                    "outside_kelvin": (
                        float(obs[outside_idx]) if outside_idx is not None else None
                    ),
                    "ahu_run": int(discrete[ahu_no]),
                    "ahu_supply_command": float(action[supply_idx]),
                    "ahu_pressure_command": (
                        float(action[pressure_idx])
                        if pressure_idx is not None else None
                    ),
                    "zone_damper_command": (
                        float(action[damper_idx])
                        if damper_idx is not None else None
                    ),
                    "zone_reheat_command": (
                        float(action[reheat_idx])
                        if reheat_idx is not None else None
                    ),
                }
                if len(heap) < topn:
                    heapq.heappush(heap, (magnitude, next_tie, row_info))
                elif magnitude > heap[0][0]:
                    heapq.heapreplace(heap, (magnitude, next_tie, row_info))
                next_tie += 1
    if not total:
        raise ValueError("Empty transition archive")
    return {
        "source_archive": str(path),
        "scenario": manifest["provenance"]["scenario"],
        "steps": manifest["record_count"],
        "zone_count": len(mapping.zone_items),
        "total_zone_transitions": total,
        "threshold_counts": counts,
        "threshold_fraction": {
            key: value / total for key, value in counts.items()
        },
        "top_observed_transitions": [
            x[2] for x in sorted(heap, reverse=True)
        ],
        "steps_with_multiple_above_2k": sorted(
            [{"step": step, "zone_count": count}
             for step, count in per_step.items() if count >= 2],
            key=lambda row: (-row["zone_count"], row["step"]))[:20],
        "notes": (
            "Observed jumps, not model residuals. Causes unknown: inspect "
            "simulation schedule, sensor normalization, occupancy, HVAC state "
            "and initial condition before diagnosing physics."),
    }
