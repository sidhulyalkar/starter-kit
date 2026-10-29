"""Auditable streaming transition records for thermal system identification.

Rows are ONLY (observation_t, action_t, observation_t+1, reward_t, step).
Scenario name, generator seed, weather ID and environment details are stored in
a separate manifest, never appended as model input features.

A row's next observation is a training *target*, not a policy input.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np

SCHEMA_VERSION = 1


def _python(value: Any) -> Any:
    """Convert simple nested numpy metadata to strict JSON values."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(k): _python(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_python(v) for v in value]
    return value


def _as_vector(value: Any, name: str, width: int) -> np.ndarray:
    v = np.asarray(value)
    if v.ndim != 1 or len(v) != width:
        raise ValueError(f"{name}: expected flat width {width}, got {v.shape}")
    if not np.issubdtype(v.dtype, np.number):
        raise ValueError(f"{name}: expected numeric values")
    if not np.all(np.isfinite(v)):
        raise ValueError(f"{name}: nonfinite values")
    return v


def validate_transition(step: int, obs_t: Any, action: Mapping[str, Any],
                        obs_tp1: Any, reward_t: Any,
                        obs_width: int, continuous_width: int,
                        discrete_width: int) -> dict[str, Any]:
    if not isinstance(step, int) or step < 0:
        raise ValueError("Step index must be nonnegative integer")
    x = _as_vector(obs_t, "obs_t", obs_width)
    y = _as_vector(obs_tp1, "obs_tp1", obs_width)
    a = _as_vector(action["continuous_action"], "continuous_action",
                   continuous_width)
    d = _as_vector(action["discrete_action"], "discrete_action",
                   discrete_width)
    if np.any(a < -1.000001) or np.any(a > 1.000001):
        raise ValueError("Continuous action outside [-1, 1]")
    if np.any((d != 0) & (d != 1)):
        raise ValueError("Discrete action must be binary")
    if isinstance(reward_t, (bool, np.bool_)) or not isinstance(
        reward_t, (int, float, np.number)
    ) or not math.isfinite(float(reward_t)):
        raise ValueError("Reward must be a finite number")

    return {
        "step": step,
        "observation_t": x.astype(float).tolist(),
        "continuous_action_t": a.astype(float).tolist(),
        "discrete_action_t": d.astype(int).tolist(),
        "observation_tp1": y.astype(float).tolist(),
        "reward_t": float(reward_t),
    }


def _array_digest(value: Any) -> dict[str, Any] | None:
    if value is None:
        return None
    array = np.asarray(value)
    return {"shape": list(array.shape),
            "dtype": str(array.dtype),
            "sha256": hashlib.sha256(array.tobytes()).hexdigest()}


class TransitionWriter:
    """Write compressed records with separately verifiable schema/provenance.

    A manifest is marked complete only when the context exits successfully.
    Missing/incomplete manifests MUST NOT be used for model training.
    """

    def __init__(self, output_path: Path, metadata: Mapping[str, Any],
                 provenance: Mapping[str, Any]):
        self.output_path = Path(output_path)
        self.manifest_path = self.output_path.with_suffix(".manifest.json")
        self.metadata = metadata
        self.provenance = dict(provenance)
        self.obs_width = len(metadata["observation_field_names"])
        self.cont_width = int(metadata["n_continuous_action"])
        self.disc_width = int(metadata["n_discrete_action"])
        self.count = 0
        self._handle = None
        fields = list(metadata["observation_field_names"])
        if len(set(fields)) != len(fields):
            raise ValueError("Observation field names are not unique")

    def __enter__(self) -> "TransitionWriter":
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = gzip.open(self.output_path, "wt", encoding="utf-8",
                                 compresslevel=3)
        return self

    def write(self, step: int, obs_t: Any, action: Mapping[str, Any],
              obs_tp1: Any, reward_t: Any) -> None:
        if self._handle is None:
            raise RuntimeError("Writer must be used in a context manager")
        if step != self.count:
            raise ValueError(f"Noncontiguous step index: {step} != {self.count}")
        record = validate_transition(
            step, obs_t, action, obs_tp1, reward_t,
            self.obs_width, self.cont_width, self.disc_width
        )
        self._handle.write(json.dumps(record, allow_nan=False,
                                      separators=(",", ":")) + "\n")
        self.count += 1

    def __exit__(self, exc_type, exc, traceback) -> None:
        if self._handle is not None:
            self._handle.close()
            self._handle = None
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "status": "complete" if exc_type is None else "aborted",
            "record_count": self.count,
            "archive_name": self.output_path.name,
            "archive_sha256": hashlib.sha256(
                self.output_path.read_bytes()).hexdigest(),
            "provenance": _python(self.provenance),
            "observation_field_names": list(self.metadata["observation_field_names"]),
            "observation_normalization_metadata": _python(
                self.metadata["observation_normalization_metadata"]),
            "action_normalization_metadata": _python(
                self.metadata["action_normalization_metadata"]),
            "action_indices": _python({
                name: self.metadata[name] for name in (
                    "temp_indices", "pressure_indices", "recirculation_indices",
                    "water_indices", "zone_action_indices"
                )
            }),
            "ahu_zone_assignments": _python(
                self.metadata["ahu_zone_assignments"]),
            "floor_plan_digest": _array_digest(self.metadata.get("floor_plan")),
            "zone_map_digest": _array_digest(self.metadata.get("zone_map")),
            "dimensions": {
                "observation": self.obs_width,
                "continuous_action": self.cont_width,
                "discrete_action": self.disc_width,
            },
            "notes": ("Only observation_t, action_t, and explicitly allowed "
                      "building metadata may be features. observation_tp1 is "
                      "a supervised target; reward_t is a training diagnostic. "
                      "Never inject provenance IDs as policy/model input."),
        }
        self.manifest_path.write_text(
            json.dumps(manifest, indent=2, allow_nan=False) + "\n"
        )


def verify_archive(path: Path) -> dict:
    """Fail closed before any training job ingests an archive."""
    path = Path(path)
    manifest_path = path.with_suffix(".manifest.json")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Unknown transition schema")
    if manifest.get("status") != "complete":
        raise ValueError("Transition archive was not completed")
    if hashlib.sha256(path.read_bytes()).hexdigest() != manifest.get("archive_sha256"):
        raise ValueError("Transition archive checksum mismatch")
    expected = manifest["record_count"]
    if expected < 1:
        raise ValueError("Empty archive")
    dims = manifest["dimensions"]
    count = 0
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            if set(obj) != {"step", "observation_t", "continuous_action_t",
                            "discrete_action_t", "observation_tp1", "reward_t"}:
                raise ValueError("Unrecognized transition fields")
            validate_transition(
                count,
                obj["observation_t"],
                {"continuous_action": obj["continuous_action_t"],
                 "discrete_action": obj["discrete_action_t"]},
                obj["observation_tp1"], obj["reward_t"],
                dims["observation"], dims["continuous_action"],
                dims["discrete_action"])
            if obj["step"] != count:
                raise ValueError("Archive has noncontiguous steps")
            count += 1
    if count != expected:
        raise ValueError("Archive record count mismatch")
    return manifest
