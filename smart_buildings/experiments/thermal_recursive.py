"""Conservative recursive thermal forecast under FROZEN initial inputs.

A deliberately restricted multi-step diagnostic: only the anchor observation
and anchor action are visible at forecast time. Non-zone observations, clock
and commands remain fixed throughout the forecast. This is NOT a forecast
under the actual later control sequence, and it is NOT evidence of MPC
readiness. Crucially it cannot leak future zone observations.

The aim is to reveal recursion instability on previously unseen buildings.
"""
from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import Any

import numpy as np

from smart_buildings.experiments.thermal_records import verify_archive
from smart_buildings.experiments.thermal_ridge import (
    ThermalRidge, ZoneFeatureMap, load_examples, metrics
)


def frozen_context_rollout(
    model: ThermalRidge,
    mapping: ZoneFeatureMap,
    initial_observation: list[float],
    continuous_action: list[float],
    discrete_action: list[int],
    horizon: int,
) -> np.ndarray:
    """Predict one zone vector at t+h, never accessing t+1... observations."""
    if not isinstance(horizon, int) or horizon < 1:
        raise ValueError("Horizon must be a positive integer")
    state = np.asarray(initial_observation, dtype=np.float64).copy()
    idx = np.asarray([row[0] for row in mapping.zone_items], dtype=int)
    for _ in range(horizon):
        x, current = mapping.featurize(state, continuous_action, discrete_action)
        future = model.predict(x, current)
        if not np.isfinite(future).all():
            raise ValueError("Nonfinite recursive model forecast")
        # Other sensor fields are deliberately not replaced with future data.
        state[idx] = (future - mapping.mean_kelvin) / mapping.std_kelvin
    return future


def evaluate_recursive(
    train_archives: list[Path],
    eval_archives: list[Path],
    alpha: float = 100.0,
    horizons: tuple[int, ...] = (1, 3, 12),
    anchor_stride: int = 12,
) -> dict[str, Any]:
    if not train_archives or not eval_archives or anchor_stride < 1:
        raise ValueError("Nonempty archive sets and positive stride required")
    if not horizons or any(not isinstance(h, int) or h < 1 for h in horizons):
        raise ValueError("Invalid evaluation horizon")
    train = [load_examples(path) for path in train_archives]
    training_ids = {r[0]["provenance"]["scenario"].split("/")[0] for r in train}
    evaluation_ids = {
        verify_archive(path)["provenance"]["scenario"].split("/")[0]
        for path in eval_archives
    }
    if training_ids & evaluation_ids:
        raise ValueError("Building-level leakage between train and evaluation")
    x = np.concatenate([r[1] for r in train])
    base = np.concatenate([r[2] for r in train])
    target = np.concatenate([r[3] for r in train])
    model = ThermalRidge.fit(x, base, target, alpha=alpha)

    reports = []
    for path in eval_archives:
        manifest = verify_archive(path)
        mapping = ZoneFeatureMap.from_manifest(manifest)
        with gzip.open(path, "rt", encoding="utf-8") as f:
            rows = [json.loads(line) for line in f]
        idx = [row[0] for row in mapping.zone_items]
        per_horizon = {}
        for horizon in sorted(set(horizons)):
            true, ridge, persistence = [], [], []
            # Anchors are separated to limit strongly overlapping examples.
            # This is still descriptive, not an independent-sample CI.
            for t in range(0, len(rows)-horizon+1, anchor_stride):
                initial = rows[t]
                observed_t = np.asarray(initial["observation_t"], dtype=np.float64)
                initial_kelvin = observed_t[idx]*mapping.std_kelvin+mapping.mean_kelvin
                forecast = frozen_context_rollout(
                    model, mapping, initial["observation_t"],
                    initial["continuous_action_t"], initial["discrete_action_t"],
                    horizon
                )
                future_obs = np.asarray(
                    rows[t+horizon-1]["observation_tp1"], dtype=np.float64)
                truth = future_obs[idx]*mapping.std_kelvin+mapping.mean_kelvin
                true.append(truth)
                persistence.append(initial_kelvin)
                ridge.append(forecast)
            if not true:
                per_horizon[str(horizon)] = {"anchors": 0, "available": False}
                continue
            truth_flat = np.concatenate(true)
            pred_flat = np.concatenate(ridge)
            persist_flat = np.concatenate(persistence)
            per_horizon[str(horizon)] = {
                "anchors": len(true),
                "available": True,
                "persistence": metrics(truth_flat, persist_flat),
                "frozen_action_ridge": metrics(truth_flat, pred_flat),
                "mae_improvement_pct": (
                    100 * (np.mean(np.abs(truth_flat-persist_flat))
                          - np.mean(np.abs(truth_flat-pred_flat))) /
                    np.mean(np.abs(truth_flat-persist_flat))
                    if np.mean(np.abs(truth_flat-persist_flat)) > 0 else None
                )
            }
        reports.append({
            "scenario": manifest["provenance"]["scenario"],
            "zones": len(mapping.zone_items),
            "steps": len(rows),
            "by_horizon": per_horizon
        })
    return {
        "experiment": "frozen_context_recursive_thermal_ridge",
        "train_buildings": sorted(training_ids),
        "eval_buildings": sorted(evaluation_ids),
        "alpha_fixed": alpha,
        "anchor_stride_steps": anchor_stride,
        "horizon_steps": sorted(set(horizons)),
        "per_archive": reports,
        "interpretation": (
            "At each anchor, only t observation and t action are inputs; "
            "future zone temperatures are scoring targets ONLY. All nonzone "
            "sensors, time features, weather and commands are frozen through "
            "recursion; later real HVAC actions may differ. This tests "
            "stability under frozen-context forecasting, NOT counterfactual "
            "control or true forward simulator accuracy under changed actions."),
    }
