"""Physical-Kelvin persistence forecast baseline from verified transition records.

No learned model. Future observations are used ONLY as scoring targets.
An unchanged-temperature predictor is a necessary and surprisingly strong
reference for slowly evolving HVAC systems.
"""
from __future__ import annotations

import argparse
import gzip
import json
import math
from pathlib import Path

import numpy as np

from smart_buildings.experiments.thermal_records import verify_archive


def score_persistence_archive(path: Path,
                              horizons: tuple[int, ...] = (1, 3, 12)) -> dict:
    manifest = verify_archive(path)
    names = manifest["observation_field_names"]
    zone_indices = [i for i, name in enumerate(names)
                    if name.endswith("_zone_air_temperature_sensor")]
    if not zone_indices:
        raise ValueError("No zone-temperature observations in this archive")
    norm = manifest["observation_normalization_metadata"][
        "zone_air_temperature_sensor"]
    scale = float(norm["sample_std"])
    shift = float(norm["sample_mean"])
    if not math.isfinite(scale) or scale <= 0 or not math.isfinite(shift):
        raise ValueError("Missing or invalid Kelvin temperature normalization")
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream]
    current = np.asarray([r["observation_t"] for r in rows], dtype=float)
    next_obs = np.asarray([r["observation_tp1"] for r in rows], dtype=float)
    t0 = current[:, zone_indices] * scale + shift
    t1 = next_obs[:, zone_indices] * scale + shift

    results = {}
    for horizon in horizons:
        if not isinstance(horizon, int) or horizon <= 0:
            raise ValueError("Forecast horizon must be a positive integer")
        if horizon > len(rows):
            results[str(horizon)] = {"n_zone_forecasts": 0, "mae_kelvin": None,
                                     "rmse_kelvin": None, "note": "not enough transitions"}
            continue
        # At timestamp t the persistence model predicts T(t+h) = T(t).
        # Target at t+h comes from row (t+h-1)'s post-action observation.
        difference = t1[horizon - 1:] - t0[:len(rows) - horizon + 1]
        errors = difference.reshape(-1)
        results[str(horizon)] = {
            "n_zone_forecasts": int(errors.size),
            "mae_kelvin": float(np.mean(np.abs(errors))),
            "rmse_kelvin": float(np.sqrt(np.mean(np.square(errors)))),
            "max_abs_kelvin": float(np.max(np.abs(errors))),
        }
    return {
        "predictor": "temperature_persistence",
        "source_archive": str(path),
        "zone_count": len(zone_indices),
        "transition_steps": len(rows),
        "horizons_5_min_steps": results,
        "note": "Diagnostic on this trajectory; not a held-out generalization result",
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("archive", type=Path)
    ap.add_argument("--horizons", type=int, nargs="+", default=[1, 3, 12])
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    scores = score_persistence_archive(args.archive, tuple(args.horizons))
    body = json.dumps(scores, indent=2) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(body)
    print(body)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
