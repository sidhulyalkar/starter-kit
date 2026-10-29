"""Report observed extreme five-minute zone-temperature changes.

Example:
  python -m scripts.inspect_thermal_shocks path/to/transitions.jsonl.gz \
    --top 20 --output path/to/shock_diagnostics.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from smart_buildings.experiments.thermal_shocks import inspect_shocks


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("archive", type=Path)
    ap.add_argument("--top", type=int, default=20)
    ap.add_argument("--output", type=Path)
    args = ap.parse_args()
    try:
        result = inspect_shocks(args.archive, topn=args.top)
    except (ValueError, OSError, KeyError) as exc:
        ap.exit(1, f"Shock inspection refused: {exc}\n")
    data = json.dumps(result, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(data)
    print(data)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
