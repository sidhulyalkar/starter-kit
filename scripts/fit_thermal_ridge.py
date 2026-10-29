"""Evaluate a shared, action-conditioned thermal ridge model on other buildings.

Example:
  python -m scripts.fit_thermal_ridge \
    --train smart_buildings/results/ridge_pilot/building_train_1/atlanta_2/pid_water/transitions.jsonl.gz \
    --eval smart_buildings/results/ridge_pilot/building_train_2/denver_8/pid_water/transitions.jsonl.gz \
    --output smart_buildings/results/ridge_pilot/ridge_report.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from smart_buildings.experiments.thermal_ridge import cross_building_evaluate


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--train", nargs="+", type=Path, required=True)
    ap.add_argument("--eval", nargs="+", type=Path, required=True)
    ap.add_argument("--alpha", type=float, default=100.0,
                    help="Fixed ridge regularization; never fit using --eval")
    ap.add_argument("--output", type=Path)
    args = ap.parse_args()
    try:
        report = cross_building_evaluate(args.train, args.eval, alpha=args.alpha)
    except (OSError, KeyError, ValueError, ArithmeticError) as exc:
        ap.exit(1, f"Thermal ridge evaluation refused: {exc}\n")
    text = json.dumps(report, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
