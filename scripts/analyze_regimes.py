"""Analyze *saved* G4 month-long VAV physics traces without rerunning the plant.

Example:
  python -m scripts.analyze_regimes \
    --artifact-dir smart_buildings/results/g5_source \
    --steps 8640 --out smart_buildings/results/g5_regimes/regimes.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from smart_buildings.experiments.regime_diagnostics import (
    load_and_analyze, write_reports,
)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--artifact-dir", required=True, type=Path)
    p.add_argument("--steps", required=True, type=int)
    p.add_argument("--out", required=True, type=Path)
    args = p.parse_args()
    result = load_and_analyze(args.artifact_dir, steps=args.steps)
    write_reports(result, args.out)
    summary = {
        "scenario": result["scenario"],
        "steps": result["steps"],
        "days": result["days"],
        "all_steps": result["all_steps"],
        "first_day": {
            k: v for k, v in result["first_day"].items()
            if k not in ("reference", "treatment")
        },
        "later_days": result["later_days"],
        "unavailable_from_saved_trace": result["unavailable_from_saved_trace"],
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
