"""Disjoint structural-transfer split tests."""
import pytest

from smart_buildings.experiments.thermal_splits import (
    SEED_RANGES, classify_building, partition_manifests, validate_seed_ranges,
)


def manifest(building, weather="atlanta_1", policy="pid_water", digest="hash"):
    return {"status": "complete",
            "provenance": {"scenario": f"{building}/{weather}", "policy": policy},
            "archive_sha256": digest}


def test_seed_partition():
    assert classify_building("building_0200") == "train"
    assert classify_building("building_0399") == "train"
    assert classify_building("building_0400") == "validation"
    assert classify_building("building_0449") == "validation"
    assert classify_building("building_0500") == "locked_test"
    assert classify_building("building_0549") == "locked_test"


@pytest.mark.parametrize("building", [
    "building_train_1", "building_0450", "building_0550",
    "building_050", "building_0001", "../building_0400"
])
def test_unassigned_or_official_building_rejected(building):
    with pytest.raises(ValueError):
        classify_building(building)


def test_overlapping_ranges_are_rejected():
    bad = dict(SEED_RANGES)
    bad["validation"] = (399, 449)
    with pytest.raises(ValueError, match="overlap"):
        validate_seed_ranges(bad)


def test_manifest_partition_by_building_not_weather():
    partitions = partition_manifests([
        manifest("building_0250", "atlanta_1", "pid_water", "first"),
        manifest("building_0250", "phoenix_1", "adaptive", "second"),
        manifest("building_0410", "atlanta_1", "pid_water", "third"),
        manifest("building_0510", "atlanta_1", "adaptive", "fourth"),
    ])
    assert {k: len(v) for k, v in partitions.items()} == {
        "train": 2, "validation": 1, "locked_test": 1,
    }


def test_duplicate_archive_rejected():
    with pytest.raises(ValueError, match="Duplicate"):
        partition_manifests([
            manifest("building_0250"),
            manifest("building_0250"),
        ])


def test_incomplete_archive_rejected():
    m = manifest("building_0410")
    m["status"] = "aborted"
    with pytest.raises(ValueError, match="Incomplete"):
        partition_manifests([m])
