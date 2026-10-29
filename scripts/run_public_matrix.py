"""Run exactly the shipped 30 scenario pairs, never a guessed cross product.

Supports a six-building CI matrix and a dry-run manifest. Full official
comparison requires 8640 steps and 30 paired results. This script profiles
without plotting to reduce cost; evaluate.py remains the official reference.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

from scripts.profile_control import profile_one


def published_scenarios(reference_file: Path) -> list[tuple[str, str]]:
    raw = json.loads(reference_file.read_text())
    if not isinstance(raw, dict) or len(raw) != 30:
        raise ValueError("Expected exactly 30 published scenario pairs")
    rows: list[tuple[str, str]] = []
    per_building: dict[str, set[str]] = {}
    stations = {"fairbanks", "denver", "atlanta", "phoenix", "singapore"}
    for pair in sorted(raw):
        if not isinstance(pair, str) or pair.count("/") != 1:
            raise ValueError(f"Bad scenario name: {pair!r}")
        building, weather = pair.split("/")
        station = weather.rsplit("_", 1)[0]
        if station not in stations or not building.startswith("building_train_"):
            raise ValueError(f"Unexpected station or building: {pair}")
        if station in per_building.setdefault(building, set()):
            raise ValueError(f"Repeated station on building: {pair}")
        per_building[building].add(station)
        rows.append((building, weather))
    if len(per_building) != 6 or any(s != stations for s in per_building.values()):
        raise ValueError("Published pairs are not six buildings x five stations")
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--reference-file", type=Path, default=Path("baseline_returns.json"))
    ap.add_argument("--building", help="Optional single building for CI sharding")
    ap.add_argument("--policies", nargs="+", default=["pid_water", "adaptive"])
    ap.add_argument("--steps", type=int, default=8640)
    ap.add_argument("--results-dir", type=Path,
                    default=Path("smart_buildings/results/g1_public_full"))
    ap.add_argument("--dry-run", action="store_true",
                    help="Validate scenario inventory without running simulation")
    args = ap.parse_args()
    if args.steps <= 0:
        ap.error("--steps must be positive")

    all_rows = published_scenarios(args.reference_file)
    valid_buildings = sorted({row[0] for row in all_rows})
    if args.building and args.building not in valid_buildings:
        ap.error(f"--building must be one of {valid_buildings}")
    rows = [r for r in all_rows if not args.building or r[0] == args.building]
    if len(rows) != (5 if args.building else 30):
        raise ValueError("Incomplete public scenario matrix")

    manifest = {
        "scope": "single-building" if args.building else "30-scenario-public",
        "scenario_pairs": ["/".join(r) for r in rows],
        "policies": args.policies,
        "steps_requested_per_episode": args.steps,
        "published_reference_sha256": hashlib.sha256(
            args.reference_file.read_bytes()).hexdigest(),
        "commit": os.environ.get("GITHUB_SHA", "not-reported"),
        "provenance": "public training scenario list; no hidden evaluation access",
        "status": "planned" if args.dry_run else "running",
    }
    args.results_dir.mkdir(parents=True, exist_ok=True)
    output = args.results_dir / f"manifest_{args.building or 'all'}.json"
    output.write_text(json.dumps(manifest, indent=2) + "\n")
    if args.dry_run:
        print(json.dumps(manifest, indent=2))
        return 0

    for building, weather in rows:
        config_dir = Path("smart_buildings/training_scenarios") / building / weather
        for policy in args.policies:
            result = profile_one(config_dir, policy, args.steps, args.results_dir)
            print(f"COMPLETED {building}/{weather}/{policy}: "
                  f"loss={result['loss_per_step']:.7f}; "
                  f"wall={result['wall_seconds']:.1f}s", flush=True)

    manifest["status"] = "complete"
    output.write_text(json.dumps(manifest, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
