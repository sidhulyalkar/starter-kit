"""Frozen, building-level split checks for thermal dynamics research.

Splits are based on the GENERATOR SEED, not individual transitions. Reusing
weather strips between seed splits only tests structural transfer.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Mapping, Any

SEED_RANGES = {
    "train": (200, 399),
    "validation": (400, 449),
    "locked_test": (500, 549),
}
_GENERATED_PATTERN = re.compile(r"^building_(\d{4})$")


def validate_seed_ranges(ranges: Mapping[str, tuple[int, int]]) -> None:
    if set(ranges) != {"train", "validation", "locked_test"}:
        raise ValueError("Expected train, validation, locked_test splits")
    occupied = set()
    for label, (lo, hi) in ranges.items():
        if not isinstance(lo, int) or not isinstance(hi, int) or lo < 0 or hi < lo:
            raise ValueError(f"Invalid seed range for {label}")
        seeds = set(range(lo, hi + 1))
        if occupied & seeds:
            raise ValueError("Building seed ranges overlap")
        occupied.update(seeds)


def classify_building(building: str,
                      ranges: Mapping[str, tuple[int, int]] = SEED_RANGES) -> str:
    validate_seed_ranges(ranges)
    match = _GENERATED_PATTERN.fullmatch(building)
    if not match:
        raise ValueError(
            f"Not a generated-building seed ID: {building!r}; official "
            "training buildings must not be represented as an unseen-building holdout")
    seed = int(match.group(1))
    labels = [label for label, (lo, hi) in ranges.items() if lo <= seed <= hi]
    if len(labels) != 1:
        raise ValueError(f"Building seed {seed} not assigned to a frozen split")
    return labels[0]


def partition_manifests(manifests: list[Mapping[str, Any]],
                        ranges: Mapping[str, tuple[int, int]] = SEED_RANGES
                        ) -> dict[str, list[Mapping[str, Any]]]:
    """Reject overlapping or unassigned buildings before fitting a model.

    Scenario identity comes from manifest PROVENANCE for validation only. It
    must never be fed as a machine-learning feature.
    """
    validate_seed_ranges(ranges)
    result: dict[str, list[Mapping[str, Any]]] = {
        "train": [], "validation": [], "locked_test": [],
    }
    seen_archive = set()
    for item in manifests:
        if item.get("status") != "complete":
            raise ValueError("Incomplete archive in model dataset")
        provenance = item.get("provenance") or {}
        scenario = provenance.get("scenario")
        if not isinstance(scenario, str) or scenario.count("/") != 1:
            raise ValueError("Missing/invalid scenario provenance")
        building, _weather = scenario.split("/")
        split = classify_building(building, ranges)
        identifier = (scenario, provenance.get("policy"),
                      item.get("archive_sha256"))
        if identifier in seen_archive:
            raise ValueError("Duplicate dataset archive")
        seen_archive.add(identifier)
        result[split].append(item)

    # A single generated building maps deterministically to one split.
    across: dict[str, str] = {}
    for split, records in result.items():
        for item in records:
            building = item["provenance"]["scenario"].split("/")[0]
            if building in across and across[building] != split:
                raise ValueError("Building appears in two different splits")
            across[building] = split
    return result


def manifest_from_archive(path: Path) -> dict:
    """Reuse the strict checksum and sequence validation at ingestion time."""
    from smart_buildings.experiments.thermal_records import verify_archive
    return verify_archive(path)
