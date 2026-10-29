"""Measure repeatability of identical controller/config full simulator rollouts.

This tests whether comparisons are confounded by stochastic simulator
conditions across policy runs. The published baseline comparison must be
handled separately: old reference returns may have been generated using a
different simulator version.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts.profile_control import profile_one


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--building", default="building_train_1")
    p.add_argument("--weather", default="atlanta_2")
    p.add_argument("--policy", default="pid_water")
    p.add_argument("--steps", type=int, default=288)
    p.add_argument("--results-dir", type=Path,
                   default=Path("smart_buildings/results/replay_determinism"))
    p.add_argument("--atol", type=float, default=1e-8)
    args = p.parse_args()
    config = Path("smart_buildings/training_scenarios") / args.building / args.weather
    reports = [
        profile_one(config, args.policy, args.steps,
                    args.results_dir / f"independent_run_{i}")
        for i in (1, 2)
    ]
    difference = abs(reports[0]["total_reward"] - reports[1]["total_reward"])
    passed = difference <= args.atol
    result = {
        "policy": args.policy,
        "scenario": f"{args.building}/{args.weather}",
        "steps": args.steps,
        "run_1_reward": reports[0]["total_reward"],
        "run_2_reward": reports[1]["total_reward"],
        "absolute_reward_difference": difference,
        "tolerance_absolute_reward": args.atol,
        "repeatability_pass": passed,
        "scope": "Two independent simulator instances; one short or full trajectory",
        "note": "This does not compare against prior published baseline versions",
    }
    args.results_dir.mkdir(parents=True, exist_ok=True)
    (args.results_dir / "replay_determinism.json").write_text(
        json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    if not passed:
        raise SystemExit("Identical-policy replay was not deterministic; avoid "
                         "unqualified single-run causal controller claims.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
