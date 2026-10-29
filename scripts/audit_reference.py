"""Reproduce the official published reference scores without importing the simulator.

This audits the *shipped baseline_returns.json*, not a fresh evaluation. It
rejects malformed/missing scenarios and computes the competition's loss metric:
mean across scenarios of abs(total_reward / 8640), lower is better.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import mean

BASELINES = ("bang_bang", "fixed_setpoint", "pid", "pid_water")
STEPS = 8640
STATIONS = {"atlanta", "denver", "fairbanks", "phoenix", "singapore"}


def audit_reference(data: dict, expected_scenarios: int = 30) -> dict:
    if not isinstance(data, dict) or len(data) != expected_scenarios:
        raise ValueError(f"Expected {expected_scenarios} scenarios; received "
                         f"{len(data) if isinstance(data, dict) else 'non-object'}")
    values: dict[str, list[float]] = {name: [] for name in BASELINES}
    building_stations: dict[str, set[str]] = {}
    for scenario, returns in sorted(data.items()):
        if not isinstance(scenario, str) or scenario.count("/") != 1:
            raise ValueError(f"Invalid scenario key: {scenario!r}")
        building, weather = scenario.split("/")
        if not building.startswith("building_") or not weather:
            raise ValueError(f"Invalid scenario key: {scenario!r}")
        if not isinstance(returns, dict) or set(returns) != set(BASELINES):
            raise ValueError(f"{scenario}: expected four reference policies")
        station = weather.rsplit("_", 1)[0]
        if station not in STATIONS:
            raise ValueError(f"{scenario}: unknown weather station")
        if station in building_stations.setdefault(building, set()):
            raise ValueError(f"{scenario}: repeated weather station on one building")
        building_stations[building].add(station)
        for name in BASELINES:
            reward = returns[name]
            if isinstance(reward, bool) or not isinstance(reward, (float, int)):
                raise ValueError(f"{scenario}/{name}: nonnumeric reward")
            if not math.isfinite(reward) or reward > 0:
                raise ValueError(f"{scenario}/{name}: expected finite nonpositive reward")
            values[name].append(abs(float(reward)) / STEPS)
    if expected_scenarios == 30:
        if len(building_stations) != 6 or any(stations != STATIONS for stations in building_stations.values()):
            raise ValueError("Published training set requires six buildings and five stations each")
    return {
        "scenarios": len(data),
        "steps_per_scenario": STEPS,
        "reference_scores": {name: mean(series) for name, series in values.items()},
        "best_reference": min(BASELINES, key=lambda name: mean(values[name])),
        "source": "published baseline_returns.json; no simulator run",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--path", type=Path,
                        default=Path(__file__).resolve().parents[1] / "baseline_returns.json")
    parser.add_argument("--expect", type=int, default=30)
    args = parser.parse_args()
    try:
        if args.expect <= 0:
            raise ValueError("--expect must be positive")
        result = audit_reference(json.loads(args.path.read_text()), args.expect)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        parser.exit(1, f"Reference audit failed: {exc}\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
