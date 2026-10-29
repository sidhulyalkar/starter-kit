"""Cross-building, action-conditioned ridge baseline for one-step zone temperature.

Scope: a transparent *research baseline*, not a graph network or a control
policy. Training requires verified transition archives from whole buildings;
evaluation refuses any overlap in building identity. Features at time t use
only observation_t, action_t, and published static metadata; observation_tp1
is used strictly as a supervised target.
"""
from __future__ import annotations

from dataclasses import dataclass
import gzip
import json
import math
import re
from pathlib import Path
from typing import Mapping

import numpy as np

from smart_buildings.experiments.thermal_records import verify_archive

FEATURE_NAMES = (
    "outside_minus_zone_k", "ahu_mean_minus_zone_k",
    "zone_minus_295k", "ahu_supply_cmd", "ahu_pressure_cmd",
    "ahu_recirculation_cmd", "ahu_run", "damper_cmd", "reheat_cmd",
    "water_temp_cmd", "water_pump_run", "supply_times_damper",
    "reheat_times_pump", "hod_sin", "hod_cos", "dow_sin", "dow_cos",
    "outside_available",
)
VAV_SUFFIX = "_zone_air_temperature_sensor"


def _sorted_ahu_ids(mapping: Mapping[str, object]) -> list[str]:
    def number(name: str):
        match = re.search(r"(\d+)", name)
        return int(match.group(1)) if match else 0
    return sorted(mapping, key=number)


def _finite_float(x: float, label: str) -> float:
    v = float(x)
    if not math.isfinite(v):
        raise ValueError(f"Nonfinite {label}")
    return v


@dataclass(frozen=True)
class ZoneFeatureMap:
    """Compiled metadata indices, reused across all rows of one archive."""

    observation_dim: int
    action_dim: int
    discrete_dim: int
    zone_items: tuple[tuple[int, int, int | None, int | None, int, int | None, int | None, int | None], ...]
    # zone obs index, AHU number, damper index, reheat index,
    # AHU temp cmd index, pressure index, recirculation index, outside obs index
    ahu_observation_groups: tuple[tuple[int, ...], ...]
    water_temp_idx: int | None
    outside_name: str
    mean_kelvin: float
    std_kelvin: float
    clock_indices: tuple[int | None, ...]
    zone_names: tuple[str, ...]

    @classmethod
    def from_manifest(cls, manifest: Mapping) -> "ZoneFeatureMap":
        names = list(manifest["observation_field_names"])
        lookup = {name: idx for idx, name in enumerate(names)}
        if len(lookup) != len(names):
            raise ValueError("Duplicate observation field names")
        dims = manifest["dimensions"]
        means = manifest["observation_normalization_metadata"][
            "zone_air_temperature_sensor"]
        mean = _finite_float(means["sample_mean"], "normalization mean")
        std = _finite_float(means["sample_std"], "normalization std")
        if std <= 0:
            raise ValueError("Zone normalization std must be positive")
        indices = manifest["action_indices"]
        zones = indices["zone_action_indices"]
        ahus = manifest["ahu_zone_assignments"]
        ahu_ids = _sorted_ahu_ids(ahus)
        temp_idxs = indices["temp_indices"]
        pressure_idxs = indices["pressure_indices"]
        recirc_idxs = indices["recirculation_indices"]
        water_idxs = indices["water_indices"]

        zone_items = []
        groups = []
        zone_names = []
        for ahu_no, ahu_id in enumerate(ahu_ids):
            if ahu_no >= len(temp_idxs):
                raise ValueError("Missing AHU supply temperature action")
            group = []
            for room in ahus[ahu_id]:
                vav = room.replace("room_", "VAV_")
                field = vav + VAV_SUFFIX
                if field not in lookup:
                    continue
                index = lookup[field]
                group.append(index)
                zone_names.append(vav)
                action_indices = zones.get(vav, {})
                outside_name = f"ahs_{ahu_id}_outside_air_temperature_sensor"
                zone_items.append((
                    index, ahu_no,
                    action_indices.get("supply_air_damper_percentage_command"),
                    action_indices.get("reheat_valve_percentage_command"),
                    temp_idxs[ahu_no],
                    pressure_idxs[ahu_no] if ahu_no < len(pressure_idxs) else None,
                    recirc_idxs[ahu_no] if ahu_no < len(recirc_idxs) else None,
                    lookup.get(outside_name),
                ))
            groups.append(tuple(group))
        if not zone_items:
            raise ValueError("No AHU-assigned zone sensors discovered")
        return cls(
            observation_dim=int(dims["observation"]),
            action_dim=int(dims["continuous_action"]),
            discrete_dim=int(dims["discrete_action"]),
            zone_items=tuple(zone_items),
            ahu_observation_groups=tuple(groups),
            water_temp_idx=water_idxs[0] if water_idxs else None,
            outside_name="ahu_outside_air_temperature",
            mean_kelvin=mean, std_kelvin=std,
            clock_indices=tuple(lookup.get(n) for n in
                                ("hod_sin_000", "hod_cos_000",
                                 "dow_sin_000", "dow_cos_000")),
            zone_names=tuple(zone_names),
        )

    def featurize(self, observation: list[float], continuous: list[float],
                  discrete: list[int]) -> tuple[np.ndarray, np.ndarray]:
        obs = np.asarray(observation, dtype=float)
        cont = np.asarray(continuous, dtype=float)
        disc = np.asarray(discrete, dtype=float)
        if obs.shape != (self.observation_dim,) or \
           cont.shape != (self.action_dim,) or disc.shape != (self.discrete_dim,):
            raise ValueError("Mismatch between archive record and metadata")
        if not np.isfinite(obs).all() or not np.isfinite(cont).all() or \
           not np.isfinite(disc).all():
            raise ValueError("Nonfinite model input")
        temps = obs * self.std_kelvin + self.mean_kelvin
        means = [
            float(np.mean(temps[list(group)])) if group else self.mean_kelvin
            for group in self.ahu_observation_groups
        ]
        water = float(cont[self.water_temp_idx]) if self.water_temp_idx is not None else 0.0
        pump = float(disc[-1]) if len(disc) else 0.0
        clock = [float(obs[idx]) if idx is not None else 0.0
                 for idx in self.clock_indices]
        x, t = [], []
        for obs_idx, ahu_no, damper_idx, reheat_idx, supply_idx, pressure_idx, recirc_idx, oat_idx in self.zone_items:
            zone = float(temps[obs_idx])
            outside = float(obs[oat_idx]) if oat_idx is not None else zone
            if outside < 160 or outside > 350:
                # Outside air sensor in the starter kit is expressed in K.
                # If an unexpected normalization/range appears, fail closed
                # rather than silently fitting a nonsense thermal coefficient.
                raise ValueError("Outdoor temperature not in plausible Kelvin range")
            run = float(disc[ahu_no])
            damper = float(cont[damper_idx]) if damper_idx is not None else -1.0
            reheat = float(cont[reheat_idx]) if reheat_idx is not None else -1.0
            supply = float(cont[supply_idx])
            x.append([
                outside - zone,
                means[ahu_no] - zone,
                zone - 295.0,
                supply, float(cont[pressure_idx]) if pressure_idx is not None else -1.0,
                float(cont[recirc_idx]) if recirc_idx is not None else 1.0,
                run, damper, reheat, water, pump,
                run * supply * ((damper + 1) / 2),
                pump * ((reheat + 1) / 2),
                *clock, float(oat_idx is not None),
            ])
            t.append(zone)
        return np.asarray(x, dtype=float), np.asarray(t, dtype=float)


def load_examples(path: Path) -> tuple[dict, np.ndarray, np.ndarray, np.ndarray]:
    manifest = verify_archive(path)
    mapping = ZoneFeatureMap.from_manifest(manifest)
    features, baselines, targets = [], [], []
    with gzip.open(path, "rt", encoding="utf-8") as f:
        previous_next = None
        for line in f:
            row = json.loads(line)
            if previous_next is not None and not np.allclose(
                row["observation_t"], previous_next, rtol=0, atol=1e-6
            ):
                raise ValueError("Discontinuous sequence: observation_tp1 != next observation_t")
            x, t = mapping.featurize(
                row["observation_t"], row["continuous_action_t"],
                row["discrete_action_t"])
            next_obs = np.asarray(row["observation_tp1"], dtype=float)
            target = next_obs[
                [entry[0] for entry in mapping.zone_items]
            ] * mapping.std_kelvin + mapping.mean_kelvin
            features.append(x)
            baselines.append(t)
            targets.append(target)
            previous_next = row["observation_tp1"]
    return (manifest,
            np.concatenate(features, axis=0),
            np.concatenate(baselines, axis=0),
            np.concatenate(targets, axis=0))


@dataclass(frozen=True)
class ThermalRidge:
    mean: np.ndarray
    scale: np.ndarray
    coefficients: np.ndarray
    alpha: float

    @classmethod
    def fit(cls, x: np.ndarray, baseline_t: np.ndarray, target_t: np.ndarray,
            alpha: float = 100.0) -> "ThermalRidge":
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(target_t, dtype=np.float64) - np.asarray(
            baseline_t, dtype=np.float64)
        if x.ndim != 2 or x.shape[1] != len(FEATURE_NAMES) or len(y) != len(x):
            raise ValueError("Invalid regression feature/target dimensions")
        if not np.isfinite(x).all() or not np.isfinite(y).all() or \
           not math.isfinite(alpha) or alpha <= 0:
            raise ValueError("Nonfinite data or invalid regularization")
        center = x.mean(axis=0)
        scale = x.std(axis=0)
        scale = np.where(scale < 1e-8, 1.0, scale)
        normalized = (x - center) / scale
        design = np.column_stack((np.ones(len(x)), normalized))
        gram = design.T @ design
        reg = np.eye(gram.shape[0]) * alpha
        reg[0, 0] = 0
        coefficient = np.linalg.solve(gram + reg, design.T @ y)
        return cls(center, scale, coefficient, alpha)

    def predict(self, x: np.ndarray, baseline_t: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        baseline = np.asarray(baseline_t, dtype=np.float64)
        if x.ndim != 2 or x.shape[1] != len(FEATURE_NAMES) or \
           baseline.shape != (len(x),):
            raise ValueError("Invalid prediction dimensions")
        normalized = (x - self.mean) / self.scale
        delta = self.coefficients[0] + normalized @ self.coefficients[1:]
        return baseline + delta


def metrics(truth: np.ndarray, predicted: np.ndarray) -> dict:
    err = np.asarray(truth, dtype=float) - np.asarray(predicted, dtype=float)
    if err.size == 0 or not np.isfinite(err).all():
        raise ValueError("Invalid forecast residuals")
    return {
        "zone_observations": int(len(err)),
        "mae_kelvin": float(np.mean(np.abs(err))),
        "rmse_kelvin": float(np.sqrt(np.mean(err ** 2))),
        "bias_kelvin": float(np.mean(err)),
        "p95_abs_kelvin": float(np.quantile(np.abs(err), 0.95)),
    }


def cross_building_evaluate(train_archives: list[Path],
                            evaluation_archives: list[Path],
                            alpha: float = 100.0) -> dict:
    if not train_archives or not evaluation_archives:
        raise ValueError("Must provide nonempty train and evaluation archives")
    train_loaded = [load_examples(Path(p)) for p in train_archives]
    evaluation_loaded = [load_examples(Path(p)) for p in evaluation_archives]

    def building(result):
        provenance = result[0].get("provenance", {})
        scenario = provenance.get("scenario", "")
        if scenario.count("/") != 1:
            raise ValueError("Archive lacks usable building provenance")
        return scenario.split("/")[0]

    trained = {building(result) for result in train_loaded}
    evaluated = {building(result) for result in evaluation_loaded}
    if trained & evaluated:
        raise ValueError("Building-level leakage: train/eval share building IDs")
    xtrain = np.concatenate([result[1] for result in train_loaded])
    ybase = np.concatenate([result[2] for result in train_loaded])
    ytrain = np.concatenate([result[3] for result in train_loaded])
    model = ThermalRidge.fit(xtrain, ybase, ytrain, alpha=alpha)
    details = []
    for manifest, x, baseline, target in evaluation_loaded:
        forecast = model.predict(x, baseline)
        p0 = metrics(target, baseline)
        p1 = metrics(target, forecast)
        details.append({
            "scenario": manifest["provenance"]["scenario"],
            "reference_persistence": p0,
            "ridge": p1,
        })
    forecast_all = np.concatenate([
        model.predict(x, baseline) for _, x, baseline, target in evaluation_loaded
    ])
    true_all = np.concatenate([target for _, x, baseline, target in evaluation_loaded])
    base_all = np.concatenate([baseline for _, x, baseline, target in evaluation_loaded])
    p0 = metrics(true_all, base_all)
    p1 = metrics(true_all, forecast_all)
    return {
        "experiment": "action_conditioned_pooled_one_step_thermal_ridge",
        "model": "shared linear ridge predicting change in zone temperature (Kelvin)",
        "alpha_fixed_without_eval_tuning": alpha,
        "features": list(FEATURE_NAMES),
        "train_buildings": sorted(trained),
        "eval_buildings": sorted(evaluated),
        "train_zone_observations": int(len(xtrain)),
        "eval_zone_observations": int(len(true_all)),
        "persistence": p0,
        "ridge": p1,
        "mae_reduction_pct": 100 * (
            p0["mae_kelvin"] - p1["mae_kelvin"]
        ) / p0["mae_kelvin"] if p0["mae_kelvin"] else None,
        "per_archive": details,
        "evaluation_scope": (
            "Cross-building. If source buildings are public, this is a "
            "public pilot, NOT procedurally generated unseen-building transfer."),
        "limitations": (
            "1-step regression from observations/actions at t. No multi-step "
            "closed-loop rollouts, calibrated uncertainty or causal impact "
            "on control proven."),
    }
