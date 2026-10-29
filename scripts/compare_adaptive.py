"""Paired full-episode comparison. Lower loss is better.

Unlike the official score tool, this requires results for BOTH policies on
identical scenario keys. It never compares a short rollout to a 30-day return.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import mean


def results(root: Path, policy: str) -> dict[str, dict]:
    found: dict[str, dict] = {}
    for path in root.glob(f"building_*/*/{policy}/summary.json"):
        scenario = f"{path.parts[-4]}/{path.parts[-3]}"
        if scenario in found:
            raise ValueError(f"Duplicate scenario {scenario}")
        found[scenario] = json.loads(path.read_text())
    return found


def compare(root: Path, candidate: str = "adaptive",
            reference: str = "pid_water", expected: int = 30,
            steps: int = 8640) -> dict:
    if expected <= 0 or steps <= 0 or candidate == reference:
        raise ValueError("Invalid comparison arguments")
    cand, base = results(root, candidate), results(root, reference)
    if len(cand) != expected or len(base) != expected or set(cand) != set(base):
        raise ValueError(
            f"Incomplete/mismatched scenario coverage: {candidate}={len(cand)}, "
            f"{reference}={len(base)}, intersection={len(set(cand) & set(base))}, "
            f"expected each={expected}")
    rows = []
    for key in sorted(cand):
        pair = []
        for policy, data in ((candidate, cand[key]), (reference, base[key])):
            if int(data.get("total_steps", -1)) != steps:
                raise ValueError(f"{key}/{policy}: expected exactly {steps} steps")
            total = data.get("total_reward")
            if isinstance(total, bool) or not isinstance(total, (int, float)):
                raise ValueError(f"{key}/{policy}: nonnumeric total_reward")
            if not math.isfinite(total) or total > 0:
                raise ValueError(f"{key}/{policy}: invalid reward")
            pair.append(abs(total) / steps)
        closs, rloss = pair
        rows.append({"scenario": key, "candidate_loss": closs,
                     "reference_loss": rloss, "improvement": rloss - closs})
    deltas = [r["improvement"] for r in rows]
    cmean = mean(r["candidate_loss"] for r in rows)
    bmean = mean(r["reference_loss"] for r in rows)
    return {
        "candidate": candidate, "reference": reference,
        "scenarios": expected, "steps": steps,
        "candidate_mean_loss": cmean, "reference_mean_loss": bmean,
        "relative_improvement_pct": 100 * (bmean - cmean) / bmean if bmean else None,
        "mean_improvement": mean(deltas),
        "wins": sum(v > 0 for v in deltas),
        "losses": sum(v < 0 for v in deltas),
        "ties": sum(v == 0 for v in deltas),
        "worst_regression": min(deltas),
        "rows": rows,
        "note": "Observed comparison; neither a safety guarantee nor hidden leaderboard score",
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("results_dir", type=Path)
    ap.add_argument("--candidate", default="adaptive")
    ap.add_argument("--reference", default="pid_water")
    ap.add_argument("--expect", type=int, default=30)
    ap.add_argument("--steps", type=int, default=8640)
    ap.add_argument("--csv", type=Path)
    args = ap.parse_args()
    try:
        summary = compare(args.results_dir, args.candidate, args.reference,
                          args.expect, args.steps)
    except (OSError, ValueError) as e:
        ap.exit(1, f"Comparison refused: {e}\n")
    if args.csv:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with args.csv.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(summary["rows"][0]))
            writer.writeheader()
            writer.writerows(summary["rows"])
    print(json.dumps({k: v for k, v in summary.items() if k != "rows"},
                     indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
