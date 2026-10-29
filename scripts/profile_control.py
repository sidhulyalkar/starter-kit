"""Profile real HVAC rollouts without expensive charts or snapshot buffers.

This is an experimental instrumentation harness, NOT a substitute for the
official Kaggle evaluator. Its result summaries use the same total_reward and
total_steps fields so the strict paired comparator can validate their coverage.

Example:
  python -m scripts.profile_control --building building_train_1 \
    --weather atlanta_2 --policies pid_water adaptive --steps 288 \
    --results-dir smart_buildings/results/g1_profile
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import resource
import sys
import time
from pathlib import Path

import numpy as np

# Register adaptive variants at import time, while keeping official
# evaluate.POLICY_MAP unchanged on disk.
from scripts.run_adaptive_experiments import POLICY_MAP
from smart_buildings.evaluate import (
    create_building_meta_data,
    get_policy_action,
    initialize_policy,
    load_hybrid_environment,
    validate_action,
)
from smart_buildings.smart_control.utils.scenario_generator import SimulationTracker


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _quantile(data: list[float], q: float) -> float:
    return float(np.quantile(np.asarray(data, dtype=np.float64), q))


def _peak_rss_mb() -> float:
    """Maximum process RSS, diagnostic only; includes imported libraries."""
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    divisor = 1024 * 1024 if sys.platform == "darwin" else 1024
    return float(value / divisor)


def profile_one(config_dir: Path, policy_key: str, steps: int,
                output_root: Path) -> dict:
    if steps <= 0:
        raise ValueError("steps must be positive")
    if policy_key not in POLICY_MAP:
        raise ValueError(f"Unknown policy: {policy_key}")
    config_path = config_dir / "config.gin"
    if not config_path.is_file():
        raise FileNotFoundError(config_path)

    # Create a fresh environment per policy, matching evaluate.py.
    init_start = time.perf_counter()
    env = load_hybrid_environment(config_path)
    tracker = SimulationTracker(env, vmin=280, vmax=310,
                                capture_snapshots=False)
    env.metrics_path = None
    env._occupancy_normalization_constant = 125.0
    metadata = create_building_meta_data(env, config_dir)
    policy = initialize_policy(POLICY_MAP[policy_key], metadata, env.action_spec())
    tracker.reset()
    obs = tracker.last_obs
    init_seconds = time.perf_counter() - init_start
    policy_times: list[float] = []
    sim_times: list[float] = []

    start = time.perf_counter()
    for _ in range(steps):
        tick = time.perf_counter()
        action = validate_action(get_policy_action(policy, obs),
                                 env.action_spec())
        if not np.all(np.isfinite(action["continuous_action"])):
            raise ValueError("Nonfinite continuous action")
        if np.any(np.abs(action["continuous_action"]) > 1.000001):
            raise ValueError("Continuous action outside normalized limits")
        if np.any((action["discrete_action"] != 0) &
                  (action["discrete_action"] != 1)):
            raise ValueError("Discrete actions are not binary")
        action_done = time.perf_counter()
        obs, _, terminated, truncated, _ = tracker.step(action)
        done = time.perf_counter()
        policy_times.append(action_done - tick)
        sim_times.append(done - action_done)
        if terminated or truncated:
            break
    wall_seconds = time.perf_counter() - start

    if len(tracker.rewards) != steps:
        raise RuntimeError(f"Expected {steps} steps, got {len(tracker.rewards)}")
    total_reward = float(np.sum(tracker.rewards))
    if not np.isfinite(total_reward) or total_reward > 1e-8:
        raise ValueError("Invalid total episode reward")

    report = {
        "policy": policy_key,
        "scenario": f"{config_dir.parent.name}/{config_dir.name}",
        "total_steps": steps,
        "total_reward": total_reward,
        "avg_reward": total_reward / steps,
        "loss_per_step": abs(total_reward / steps),
        "initialization_seconds": init_seconds,
        "wall_seconds": wall_seconds,
        "policy_seconds_total": sum(policy_times),
        "simulator_seconds_total": sum(sim_times),
        "wall_seconds_per_step": wall_seconds / steps,
        "policy_seconds_p50": _quantile(policy_times, 0.50),
        "policy_seconds_p95": _quantile(policy_times, 0.95),
        "simulator_seconds_p95": _quantile(sim_times, 0.95),
        "max_process_rss_mb": _peak_rss_mb(),
        "observation_dim": len(metadata["observation_field_names"]),
        "continuous_action_dim": metadata["n_continuous_action"],
        "discrete_action_dim": metadata["n_discrete_action"],
        "scenario_config_sha256": _sha256(config_path),
        "python": platform.python_version(),
        "note": ("Smoke/profile result, NOT comparable to the full competition "
                 "score" if steps != 8640 else
                 "Full-horizon single-scenario result, NOT a hidden leaderboard"),
    }
    outdir = output_root / config_dir.parent.name / config_dir.name / policy_key
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--building", required=True, help="Official building stem")
    parser.add_argument("--weather", required=True, help="Official weather strip stem")
    parser.add_argument("--policies", nargs="+", required=True,
                        choices=sorted(POLICY_MAP))
    parser.add_argument("--steps", type=int, default=288)
    parser.add_argument("--scenarios-dir", type=Path,
                        default=Path("smart_buildings/training_scenarios"))
    parser.add_argument("--results-dir", type=Path,
                        default=Path("smart_buildings/results/g1_profile"))
    args = parser.parse_args()
    config = args.scenarios_dir / args.building / args.weather
    results = []
    for policy_key in args.policies:
        result = profile_one(config, policy_key, args.steps, args.results_dir)
        results.append(result)
        print(json.dumps({
            key: result[key]
            for key in ("scenario", "policy", "total_steps", "loss_per_step",
                        "initialization_seconds", "wall_seconds",
                        "policy_seconds_p95", "max_process_rss_mb")
        }, indent=2), flush=True)
    index = args.results_dir / "profile_index.json"
    index.parent.mkdir(parents=True, exist_ok=True)
    index.write_text(json.dumps(results, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
