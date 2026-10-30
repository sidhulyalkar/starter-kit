"""Fail-closed paired VAV policy evaluation; never promote on one scenario.

Use only complete, equal-horizon, identity-sham-validated simulator runs.
Separate evidentiary validity from whether an experimental cap performed well.
Even a clean public 30-scenario result cannot demonstrate generated-building
transfer without a separately frozen validation set.

Example:
  python -m scripts.assess_vav_qualification results/paired_interventions.json \
    --expect-scenarios 1 --steps 8640 --target-arm vav1_reheat50 \
    --output results/qualification.json
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

REQUIRED_COMPONENTS = (
    "normalized_productivity_regret",
    "normalized_energy_cost",
    "normalized_carbon_emission",
)


def number(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} is boolean, not a score")
    v = float(value)
    if not math.isfinite(v):
        raise ValueError(f"{label} is nonfinite")
    return v


def assess_reports(reports: list[dict], *, expected_scenarios: int,
                   steps: int, target_arm: str = "vav1_reheat50",
                   allow_inactive: bool = False,
                   expected_inventory: set[str] | None = None) -> dict:
    """Return quality report. Regressions are results, NOT CI failures."""
    if expected_scenarios < 1 or steps < 1:
        raise ValueError("Positive expected scenario count and horizon required")
    if len(reports) != expected_scenarios:
        raise ValueError(
            f"Incomplete paired scenario coverage: expected {expected_scenarios}, "
            f"got {len(reports)}")
    seen: set[str] = set()
    comparisons = []
    for report in reports:
        scenario = report["scenario"]
        if not isinstance(scenario, str) or scenario.count("/") != 1 or scenario in seen:
            raise ValueError(f"Invalid or duplicate scenario: {scenario!r}")
        seen.add(scenario)
        if report.get("sham_reproduces_reference") is not True:
            raise ValueError(f"Sham not validated: {scenario}")
        arms = report["arms"]
        arm_map = {a["arm"]: a for a in arms}
        if len(arm_map) != len(arms) or not {"pid_water", "vav1_sham", target_arm}.issubset(arm_map):
            raise ValueError(f"Duplicate/missing reference, sham, target: {scenario}")
        base = arm_map["pid_water"]
        sham = arm_map["vav1_sham"]
        treatment = arm_map[target_arm]
        for arm in (base, sham, treatment):
            if int(arm["total_steps"]) != steps:
                raise ValueError(f"Truncated or mismatched horizon: {scenario}")
            if number(arm["loss_per_step"], "loss") < 0:
                raise ValueError("Negative reported episode loss")
            if number(arm["total_reward"], "reward") > 0:
                raise ValueError("Positive simulator reward unsupported")
            if abs(number(arm["total_reward"], "reward") +
                   steps * number(arm["loss_per_step"], "loss")) > 1e-6:
                raise ValueError("Total reward inconsistent with loss-per-step")
            for key in ("target_zone_shocks_ge_2_kelvin",
                        "all_zone_shocks_ge_2_kelvin",
                        "all_zone_shocks_ge_5_kelvin",
                        "all_zone_comfort_excursion_rate",
                        "all_zone_comfort_excursion_difference",
                        "modified_action_steps"):
                number(arm[key], key)
            if not 0 <= arm["all_zone_comfort_excursion_rate"] <= 1:
                raise ValueError("Comfort rate outside [0,1]")
            if any(number(arm["reward_component_totals"][k], k) !=
                   arm["reward_component_totals"][k] for k in REQUIRED_COMPONENTS):
                raise ValueError("Invalid component total")
        if int(base["modified_action_steps"]) != 0 or int(sham["modified_action_steps"]) != 0:
            raise ValueError("Reference or sham unexpectedly modified actions")
        if int(treatment["modified_action_steps"]) <= 0:
            if not allow_inactive:
                raise ValueError(f"Intervention inactive: {scenario}")
            if (abs(number(treatment["total_reward"], "reward") -
                    number(base["total_reward"], "reward")) > 1e-8 or
                treatment["all_zone_shocks_ge_5_kelvin"] != base["all_zone_shocks_ge_5_kelvin"] or
                treatment["all_zone_shocks_ge_2_kelvin"] != base["all_zone_shocks_ge_2_kelvin"] or
                treatment["all_zone_comfort_excursion_rate"] != base["all_zone_comfort_excursion_rate"]):
                raise ValueError(f"Unexercised cap inexplicably changed outcome: {scenario}")
        if number(sham["total_reward"], "sham reward") != number(base["total_reward"], "reference reward"):
            raise ValueError(f"Sham changed reward: {scenario}")
        if (number(sham["all_zone_comfort_excursion_rate"], "sham comfort") !=
                number(base["all_zone_comfort_excursion_rate"], "reference comfort")):
            raise ValueError(f"Sham changed measured comfort: {scenario}")
        for key in REQUIRED_COMPONENTS:
            if abs(number(base["reward_component_totals"][key], key) -
                   number(sham["reward_component_totals"][key], key)) > 1e-8:
                raise ValueError(f"Sham changed reward component {key}")
            if abs(number(treatment["reward_component_differences"][key], key) -
                   (number(treatment["reward_component_totals"][key], key) -
                    number(base["reward_component_totals"][key], key))) > 1e-8:
                raise ValueError(f"Incorrect reward component difference {key}")
        reference_loss = number(base["loss_per_step"], "reference loss")
        intervention_loss = number(treatment["loss_per_step"], "intervention loss")
        delta = intervention_loss - reference_loss
        ref_comfort = number(base["all_zone_comfort_excursion_rate"], "reference comfort")
        treatment_comfort = number(treatment["all_zone_comfort_excursion_rate"], "intervention comfort")
        comps = {
            "scenario": scenario,
            "steps": steps,
            "modified_action_steps": int(treatment["modified_action_steps"]),
            "treatment_exercised": bool(int(treatment["modified_action_steps"]) > 0),
            "loss_baseline": reference_loss,
            "loss_treatment": intervention_loss,
            "loss_delta": delta,
            "loss_reduction_pct": (100 * (reference_loss - intervention_loss) /
                                   reference_loss if reference_loss else None),
            "all_zone_5k_shocks_baseline": int(base["all_zone_shocks_ge_5_kelvin"]),
            "all_zone_5k_shocks_treatment": int(treatment["all_zone_shocks_ge_5_kelvin"]),
            "all_zone_2k_shocks_baseline": int(base["all_zone_shocks_ge_2_kelvin"]),
            "all_zone_2k_shocks_treatment": int(treatment["all_zone_shocks_ge_2_kelvin"]),
            "comfort_proxy_baseline": ref_comfort,
            "comfort_proxy_treatment": treatment_comfort,
            "comfort_proxy_delta": treatment_comfort - ref_comfort,
            "reward_component_delta": {
                key: number(treatment["reward_component_differences"][key], key)
                for key in REQUIRED_COMPONENTS
            },
            "experimental_checks": {
                "loss_nonworsening": delta <= 0,
                "comfort_proxy_nonworsening": treatment_comfort <= ref_comfort,
                "extreme_5k_shocks_nonincreasing": (
                    treatment["all_zone_shocks_ge_5_kelvin"] <=
                    base["all_zone_shocks_ge_5_kelvin"]
                ),
            },
        }
        comparisons.append(comps)
    if expected_inventory is not None and seen != expected_inventory:
        missing = sorted(expected_inventory - seen)
        extra = sorted(seen - expected_inventory)
        raise ValueError(f"Published scenario coverage mismatch; missing={missing}, unexpected={extra}")
    comparisons.sort(key=lambda c: c["scenario"])
    baseline_mean = sum(c["loss_baseline"] for c in comparisons) / len(comparisons)
    candidate_mean = sum(c["loss_treatment"] for c in comparisons) / len(comparisons)
    worst_loss = max(c["loss_delta"] for c in comparisons)
    worst_comfort = max(c["comfort_proxy_delta"] for c in comparisons)
    all_check = all(all(c["experimental_checks"].values()) for c in comparisons)
    return {
        "status": "valid_paired_evidence_not_controller_qualification",
        "target_arm": target_arm,
        "scenario_count": len(comparisons),
        "steps_per_scenario": steps,
        "mean_reference_loss_per_step": baseline_mean,
        "mean_treatment_loss_per_step": candidate_mean,
        "mean_loss_reduction_pct": (
            100 * (baseline_mean - candidate_mean)/baseline_mean
            if baseline_mean else None
        ),
        "worst_scenario_loss_delta": worst_loss,
        "worst_scenario_comfort_proxy_delta": worst_comfort,
        "scenario_check_results_all_pass": all_check,
        "decision": (
            "experiment_concerning_regression_or_tradeoff" if not all_check
            else "preliminary_constraints_passed_not_qualified"
        ),
        "qualification": (
            "Not qualified: needs independent held-out generated buildings, "
            "multi-climate 8640-step coverage, and official occupancy-aware "
            "comfort/reward diagnostics. Short public results cannot be "
            "declared generalization."
        ),
        "per_scenario": comparisons,
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("reports", type=Path, nargs="+")
    p.add_argument("--expect-scenarios", type=int, required=True)
    p.add_argument("--steps", type=int, default=8640)
    p.add_argument("--target-arm", default="vav1_reheat50")
    p.add_argument("--output", type=Path)
    p.add_argument("--allow-inactive", action="store_true",
                   help="Count an unexercised cap as explicit no-effect evidence")
    p.add_argument("--verify-public-inventory", type=Path,
                   help="Require exact six-building / five-weather public scenario list")
    args = p.parse_args()
    inventory = None
    if args.verify_public_inventory is not None:
        from scripts.run_public_matrix import published_scenarios
        inventory = {"/".join(pair) for pair in
                     published_scenarios(args.verify_public_inventory)}
    result = assess_reports(
        [json.loads(path.read_text()) for path in args.reports],
        expected_scenarios=args.expect_scenarios, steps=args.steps,
        target_arm=args.target_arm, allow_inactive=args.allow_inactive,
        expected_inventory=inventory,
    )
    body = json.dumps(result, indent=2, allow_nan=False) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(body)
    print(body)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
