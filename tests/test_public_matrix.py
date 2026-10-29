"""Fail-closed public scenario inventory, independent of the simulation run."""
import json
from pathlib import Path

import pytest

from scripts.run_public_matrix import published_scenarios


REFERENCE = Path(__file__).resolve().parents[1] / "baseline_returns.json"


def test_inventory_is_thirty_pairs():
    scenarios = published_scenarios(REFERENCE)
    assert len(scenarios) == 30
    assert len(set(scenarios)) == 30
    counts = {building: sum(b == building for b, _ in scenarios)
              for building, _ in scenarios}
    assert len(counts) == 6
    assert set(counts.values()) == {5}


def test_missing_scenario_rejected(tmp_path):
    d = json.loads(REFERENCE.read_text())
    d.pop(next(iter(d)))
    p = tmp_path / "missing.json"
    p.write_text(json.dumps(d))
    with pytest.raises(ValueError, match="30"):
        published_scenarios(p)


def test_duplicate_weather_station_rejected(tmp_path):
    d = json.loads(REFERENCE.read_text())
    key = next(k for k in d if k.startswith("building_train_1/atlanta"))
    d["building_train_1/atlanta_999"] = d.pop(key)
    key2 = next(k for k in d if k.startswith("building_train_1/denver"))
    d["building_train_1/atlanta_888"] = d.pop(key2)
    p = tmp_path / "duplicate.json"
    p.write_text(json.dumps(d))
    with pytest.raises(ValueError, match="Repeated station"):
        published_scenarios(p)
