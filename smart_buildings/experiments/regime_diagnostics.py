"""Time-stratified, no-retraining attribution of a failed full-month cap.

Inputs are matched saved OFFLINE simulator physics traces, not controller
features. For every day compare temperature, reheat saturation, source input,
and shock distribution on the same scenario and identical 5-minute indexes.

This deliberately cannot infer occupancy-weighted reward by day from these
archives: reward components were summarized over the full episode. Do not
invent day-level reward, carbon or comfort numbers.
"""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np

STEPS_PER_DAY = 288


def _finite(raw: Any, field: str) -> float:
    if isinstance(raw, bool):
        raise ValueError(f"Boolean {field}")
    v = float(raw)
    if not math.isfinite(v):
        raise ValueError(f"Nonfinite {field}")
    return v


def _trace(trace: list[dict], steps: int) -> dict[int, dict]:
    if not isinstance(trace, list) or len(trace) != steps:
        raise ValueError(f"Expected {steps} complete trace rows")
    indexed: dict[int, dict] = {}
    for row in trace:
        idx = row["step"]
        if not isinstance(idx, int) or idx in indexed or not 0 <= idx < steps:
            raise ValueError(f"Invalid/duplicate step: {idx!r}")
        if int(row["minute"]) != idx * 5:
            raise ValueError("Misaligned five-minute timestamp")
        for k in ("temp_t_kelvin", "temp_tp1_kelvin",
                  "delta_kelvin", "realized_reheat_fraction",
                  "realized_damper_fraction", "realized_zone_air_flow_m3_s",
                  "source_power_w_reported_by_simulator"):
            _finite(row[k], k)
        if abs((row["temp_tp1_kelvin"] - row["temp_t_kelvin"]) -
               row["delta_kelvin"]) > 1e-4:
            raise ValueError("Temperature delta inconsistent with trace")
        if not (0 <= row["realized_reheat_fraction"] <= 1.00001 and
                0 <= row["realized_damper_fraction"] <= 1.00001):
            raise ValueError("Actuator fraction outside native 0-1 range")
        indexed[idx] = row
    if set(indexed) != set(range(steps)):
        raise ValueError("Incomplete trace step set")
    return indexed


def _sample_summary(rows: list[dict]) -> dict:
    t = np.asarray([_finite(r["temp_tp1_kelvin"], "end temp") for r in rows])
    dt = np.asarray([_finite(r["delta_kelvin"], "delta") for r in rows])
    heat = np.asarray([_finite(r["realized_reheat_fraction"], "reheat") for r in rows])
    damper = np.asarray([_finite(r["realized_damper_fraction"], "damper") for r in rows])
    watts = np.asarray([_finite(r["source_power_w_reported_by_simulator"], "source")
                        for r in rows])
    return {
        "mean_end_temperature_kelvin": float(t.mean()),
        "min_end_temperature_kelvin": float(t.min()),
        "max_end_temperature_kelvin": float(t.max()),
        "mean_reheat_native_fraction": float(heat.mean()),
        "reheat_at_95pct_fraction": float(np.mean(heat >= .95)),
        "mean_damper_native_fraction": float(damper.mean()),
        "mean_signed_source_watts": float(watts.mean()),
        "abs_temperature_change_ge_2k": int(np.count_nonzero(np.abs(dt) >= 2)),
        "abs_temperature_change_ge_5k": int(np.count_nonzero(np.abs(dt) >= 5)),
        "p95_abs_step_temperature_change_kelvin": float(np.quantile(np.abs(dt), .95)),
        "max_abs_step_temperature_change_kelvin": float(np.max(np.abs(dt))),
        "steps": len(rows),
    }


def _longest_run(mask: np.ndarray) -> int:
    longest = current = 0
    for value in mask:
        current = current + 1 if bool(value) else 0
        longest = max(longest, current)
    return longest


def analyze_pair(reference: Mapping[str, Any],
                 treatment: Mapping[str, Any],
                 *, steps: int = 8640,
                 temperature_gap_k: float = 1.0) -> dict:
    """Compare two saved traces, preserving time order and source metadata."""
    if steps < 1 or steps % STEPS_PER_DAY != 0:
        raise ValueError("Require complete 288-step days")
    if temperature_gap_k <= 0 or not math.isfinite(temperature_gap_k):
        raise ValueError("Invalid cold-offset threshold")
    for key in ("scenario", "config_sha256", "zone", "room_floor_area_m2",
                "room_heat_capacity_j_per_kelvin"):
        if reference[key] != treatment[key]:
            raise ValueError(f"Mismatched {key} between paired episodes")
    if reference["policy"] != "pid_water":
        raise ValueError("Reference must be original pid_water")
    if treatment["policy"] == "pid_water":
        raise ValueError("Treatment is not a distinct controller")
    a = _trace(reference["trace"], steps)
    b = _trace(treatment["trace"], steps)
    aligned_ref = [a[i] for i in range(steps)]
    aligned_treatment = [b[i] for i in range(steps)]
    temp_diff = np.asarray([
        b[i]["temp_tp1_kelvin"] - a[i]["temp_tp1_kelvin"]
        for i in range(steps)
    ], dtype=float)
    ref_deltas = np.asarray([a[i]["delta_kelvin"] for i in range(steps)])
    treatment_deltas = np.asarray([b[i]["delta_kelvin"] for i in range(steps)])
    day_rows = []
    for start in range(0, steps, STEPS_PER_DAY):
        end = start + STEPS_PER_DAY
        x = _sample_summary(aligned_ref[start:end])
        y = _sample_summary(aligned_treatment[start:end])
        d = temp_diff[start:end]
        day_rows.append({
            "day_0_based": start // STEPS_PER_DAY,
            "step_start": start, "step_end_exclusive": end,
            "reference": x, "treatment": y,
            "mean_treatment_minus_reference_temp_kelvin": float(d.mean()),
            "minimum_treatment_minus_reference_temp_kelvin": float(d.min()),
            "steps_treatment_at_least_1k_colder": int(np.count_nonzero(
                d <= -temperature_gap_k)),
            "longest_treatment_colder_run_steps": _longest_run(
                d <= -temperature_gap_k),
            "change_in_abs_2k_shocks": (
                y["abs_temperature_change_ge_2k"] -
                x["abs_temperature_change_ge_2k"]
            ),
            "change_in_abs_5k_shocks": (
                y["abs_temperature_change_ge_5k"] -
                x["abs_temperature_change_ge_5k"]
            ),
            "change_in_mean_reheat_fraction": (
                y["mean_reheat_native_fraction"] -
                x["mean_reheat_native_fraction"]
            ),
        })
    first_day = day_rows[0]
    following = day_rows[1:]
    result = {
        "scope": "offline_time_stratified_single_zone_physics",
        "scenario": reference["scenario"],
        "target_zone": reference["zone"],
        "reference_policy": reference["policy"],
        "treatment_policy": treatment["policy"],
        "steps": steps, "days": len(day_rows),
        "threshold_temperature_gap_k": temperature_gap_k,
        "all_steps": {
            "reference": _sample_summary(aligned_ref),
            "treatment": _sample_summary(aligned_treatment),
            "mean_treatment_minus_reference_temp_kelvin": float(temp_diff.mean()),
            "time_steps_treatment_at_least_1k_colder": int(np.count_nonzero(
                temp_diff <= -temperature_gap_k)),
            "longest_colder_run_steps": _longest_run(
                temp_diff <= -temperature_gap_k),
            "reference_2k_target_zone_shocks": int(np.count_nonzero(
                np.abs(ref_deltas) >= 2)),
            "treatment_2k_target_zone_shocks": int(np.count_nonzero(
                np.abs(treatment_deltas) >= 2)),
        },
        "first_day": first_day,
        "later_days": {
            "days": len(following),
            "mean_temperature_gap_kelvin": (
                float(np.mean([x["mean_treatment_minus_reference_temp_kelvin"]
                               for x in following])) if following else None),
            "days_with_negative_mean_temperature_gap": sum(
                x["mean_treatment_minus_reference_temp_kelvin"] < 0
                for x in following),
            "days_with_fewer_2k_shocks": sum(
                x["change_in_abs_2k_shocks"] < 0 for x in following
            ),
        },
        "per_day": day_rows,
        "unavailable_from_saved_trace": (
            "Per-day occupancy-weighted reward components and all-zone "
            "severe comfort cannot be inferred from a single-zone physics "
            "trace. Use episode-level paired report only. A future recorder "
            "must explicitly capture them for time-resolved attribution."),
        "interpretation": (
            "Within-simulator paired comparison of VAV_1 temperature and "
            "actuator dynamics, not a causal decomposition of the official "
            "reward. Episodes differ through closed-loop feedback and shared "
            "hot-water allocation. Do not tune and re-test on the same "
            "scenario as if it were held out."),
    }
    return result


def load_and_analyze(root: Path, *, steps: int = 8640) -> dict:
    paths = sorted(Path(root).rglob("*_physics.json"))
    reports = [json.loads(p.read_text()) for p in paths]
    reference = [r for r in reports if r.get("policy") == "pid_water"]
    treatment = [r for r in reports if r.get("policy") == "vav1_reheat50"]
    if len(reference) != 1 or len(treatment) != 1:
        raise ValueError("Exactly one reference and one fixed-cap trace required")
    return analyze_pair(reference[0], treatment[0], steps=steps)


def write_reports(report: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    csv_path = path.with_suffix(".csv")
    fields = ("day_0_based", "mean_treatment_minus_reference_temp_kelvin",
              "minimum_treatment_minus_reference_temp_kelvin",
              "steps_treatment_at_least_1k_colder",
              "longest_treatment_colder_run_steps",
              "change_in_abs_2k_shocks",
              "change_in_abs_5k_shocks",
              "change_in_mean_reheat_fraction")
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for day in report["per_day"]:
            writer.writerow({k: day[k] for k in fields})
