"""Check that fast experimental profiler matches the *official* evaluator.

A same-commit, same-scenario, same-policy total-reward comparison determines
whether the fast runner is score-compatible. It does NOT independently
explain why historical baseline_returns.json may differ on this commit.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from scripts.profile_control import profile_one
from scripts.run_adaptive_experiments import POLICY_MAP
from smart_buildings.evaluate import evaluate_policy_for_config


def verify_parity(profile: dict, official: dict, tolerance: float=1e-7) -> dict:
    if not math.isfinite(tolerance) or tolerance < 0:
        raise ValueError("Invalid numerical tolerance")
    expected=int(profile["total_steps"])
    actual=int(official["total_steps"])
    if expected != actual or expected <= 0:
        raise ValueError(f"Rollout horizon mismatch: {expected} vs {actual}")
    fast=float(profile["total_reward"])
    ref=float(official["total_reward"])
    if not math.isfinite(fast) or not math.isfinite(ref):
        raise ValueError("Nonfinite evaluator reward")
    delta=fast-ref
    if abs(delta)>tolerance:
        raise ValueError(
            f"Official/profiler reward mismatch: "
            f"profile={fast}, official={ref}, delta={delta}"
        )
    return {
        "status":"matched",
        "policy":profile["policy"],
        "scenario":profile["scenario"],
        "steps":expected,
        "fast_profiler_reward":fast,
        "official_evaluator_reward":ref,
        "absolute_reward_difference":abs(delta),
        "absolute_tolerance":tolerance,
        "interpretation": (
            "Numerical reward agreement for these conditions. Historical "
            "published baselines may use a different code/config version, "
            "which is not checked by this parity test."
        )
    }


def main() -> int:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--building",default="building_train_1")
    p.add_argument("--weather",default="atlanta_2")
    p.add_argument("--policy",default="pid_water",choices=sorted(POLICY_MAP))
    p.add_argument("--steps",type=int,default=288)
    p.add_argument("--atol",type=float,default=1e-7)
    p.add_argument("--results-dir",type=Path,
                   default=Path("smart_buildings/results/g1_runner_parity"))
    args=p.parse_args()
    if args.steps < 1:
        p.error("--steps must be positive")
    config=Path("smart_buildings/training_scenarios")/args.building/args.weather
    profile=profile_one(config,args.policy,args.steps,args.results_dir/"fast")
    official=evaluate_policy_for_config(
        scenario_results_dir=args.results_dir/"official"/args.building/args.weather,
        input_dir=config,policy_key=args.policy,
        policy_class=POLICY_MAP[args.policy],num_steps=args.steps,
        save_video=False
    )
    result=verify_parity(profile,official,args.atol)
    args.results_dir.mkdir(parents=True,exist_ok=True)
    (args.results_dir/"parity_report.json").write_text(
        json.dumps(result,indent=2)+"\n"
    )
    print(json.dumps(result,indent=2))
    return 0


if __name__=="__main__":
    raise SystemExit(main())
