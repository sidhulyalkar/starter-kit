"""Counterfactual *simulator intervention* study of local Denver VAV shocks.

Fresh independent simulator per arm, identical scenario and starting policy
state. The only deliberate intervention is a native [0,1] cap on VAV_1's
reheat and/or damper output, applied after official PID-water chooses actions.
Subsequent states and all other PID decisions may change due to feedback.
No intervention is installed in the competition submission policy.

A sham cap of 100% must match the reference exactly, or the experiment fails.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from scripts.diagnose_zone_physics import diagnose
from scripts.run_adaptive_experiments import POLICY_MAP
from smart_buildings.experiments.vav_interventions import (
    EXPERIMENT_ARMS, intervention_policy_class,
)


def paired_intervention_report(reports: dict[str, dict], *,
                               allow_inactive_treatment: bool = False) -> dict:
    if "pid_water" not in reports or "vav1_sham" not in reports:
        raise ValueError("Missing reference or sham controller")
    ref = reports["pid_water"]
    sham = reports["vav1_sham"]
    base = ref["summary"]
    a, b = ref["trace"], sham["trace"]
    if not a or len(a) != len(b):
        raise ValueError("Reference and sham have incompatible traces")
    if not all(a[i]["step"] == b[i]["step"]
               and abs(a[i]["temp_tp1_kelvin"]-b[i]["temp_tp1_kelvin"])<=1e-6
               and abs(a[i]["temp_t_kelvin"]-b[i]["temp_t_kelvin"])<=1e-6
               for i in range(len(a))):
        raise ValueError("Sham diverged: invalid counterfactual pairing")
    if abs(base["total_reward"]-sham["summary"]["total_reward"])>1e-8:
        raise ValueError("Sham reward differs from reference")
    keys = ("scenario", "config_sha256", "zone")
    if any(ref[k]!=sham[k] for k in keys):
        raise ValueError("Sham used a different scenario")
    result = []
    for name, report in sorted(reports.items()):
        if any(report[k]!=ref[k] for k in keys):
            raise ValueError(f"Arm {name} has different configuration/zone")
        s = report["summary"]
        if s["steps"]!=base["steps"]:
            raise ValueError(f"Arm {name} does not have the same horizon")
        if name not in ("pid_water", "vav1_sham") and \
           s["intervention_modified_steps"] < 1:
            if not allow_inactive_treatment:
                raise ValueError(f"No actuator intervention actually executed: {name}")
            # No modified action is valid evidence for warm climates, but it
            # cannot be credited with improving anything.
            if (abs(s["total_reward"]-base["total_reward"]) > 1e-8 or
                s["all_zone_transitions_ge_2_kelvin"] != base["all_zone_transitions_ge_2_kelvin"] or
                s["all_zone_transitions_ge_5_kelvin"] != base["all_zone_transitions_ge_5_kelvin"] or
                s["all_zone_comfort_excursions_ge_2_kelvin"] != base["all_zone_comfort_excursions_ge_2_kelvin"]):
                raise ValueError(f"Inactive treatment unexpectedly changed outcome: {name}")
        row = {
            "arm": name,
            "modified_action_steps": s["intervention_modified_steps"],
            "treatment_exercised": bool(s["intervention_modified_steps"] > 0)
                                  if name not in ("pid_water","vav1_sham") else None,
            "total_steps": s["steps"],
            "total_reward": s["total_reward"],
            "loss_per_step": s["loss_per_step"],
            "relative_loss_change_pct": (
                100*(s["loss_per_step"]-base["loss_per_step"])/
                base["loss_per_step"] if base["loss_per_step"] else None
            ),
            "target_zone_shocks_ge_2_kelvin": s["shock_count"],
            "target_zone_shock_difference": s["shock_count"]-base["shock_count"],
            "all_zone_shocks_ge_2_kelvin": s["all_zone_transitions_ge_2_kelvin"],
            "all_zone_shocks_ge_5_kelvin": s["all_zone_transitions_ge_5_kelvin"],
            "all_zone_shocks_ge_2_difference": (
                s["all_zone_transitions_ge_2_kelvin"] -
                base["all_zone_transitions_ge_2_kelvin"]
            ),
            "all_zone_comfort_excursion_rate": s["all_zone_comfort_excursion_rate"],
            "all_zone_comfort_excursion_difference": (
                s["all_zone_comfort_excursions_ge_2_kelvin"] -
                base["all_zone_comfort_excursions_ge_2_kelvin"]
            ),
            "max_target_zone_absolute_change_kelvin": s["max_absolute_delta_kelvin"],
            "reward_component_totals": dict(s["reward_component_sums"]),
            "reward_component_differences": {
                key: s["reward_component_sums"][key] -
                     base["reward_component_sums"][key]
                for key in base["reward_component_sums"]
            },
        }
        if any(isinstance(x,float) and not math.isfinite(x) for x in row.values()):
            raise ValueError(f"Nonfinite outcome: {name}")
        result.append(row)
    return {
        "scenario": ref["scenario"], "target_zone": ref["zone"],
        "status": "paired_simulator_intervention_diagnostic",
        "sham_reproduces_reference": True,
        "arms": result,
        "interpretation": (
            "Same initial simulator scenario with distinct intervention policies. "
            "Other commands may change through closed-loop state feedback, "
            "so only the controlled cap assignment differs initially. "
            "One public scenario cannot establish transfer, safe control or "
            "real-building causality. Comfort proxy is unweighted sensor "
            "band excursions, NOT official occupancy-weighted comfort reward."
        )
    }


def main() -> int:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--building",default="building_train_2")
    p.add_argument("--weather",default="denver_8")
    p.add_argument("--zone",default="VAV_1")
    p.add_argument("--steps",type=int,default=288)
    p.add_argument("--threshold-k",type=float,default=2.0)
    p.add_argument("--allow-inactive-treatment",action="store_true",
                   help="Valid no-effect evidence when warm climates never invoke reheat")
    p.add_argument("--arms",nargs="+",default=list(EXPERIMENT_ARMS),
                   choices=list(EXPERIMENT_ARMS))
    p.add_argument("--results-dir",type=Path,
                   default=Path("smart_buildings/results/g3_vav_counterfactual"))
    args=p.parse_args()
    if "pid_water" not in args.arms or "vav1_sham" not in args.arms:
        p.error("Always include pid_water and vav1_sham for matched controls")
    if args.zone!="VAV_1":
        p.error("This fixed-arm causal experiment currently targets VAV_1 only")
    for name,limits in EXPERIMENT_ARMS.items():
        if limits is not None:
            POLICY_MAP[name]=intervention_policy_class(name,limits)
    reports={}
    for name in args.arms:
        reports[name]=diagnose(
            args.building,args.weather,args.zone,name,args.steps,
            args.threshold_k,args.results_dir
        )
    summary=paired_intervention_report(
        reports,allow_inactive_treatment=args.allow_inactive_treatment
    )
    args.results_dir.mkdir(parents=True,exist_ok=True)
    (args.results_dir/"paired_interventions.json").write_text(
        json.dumps(summary,indent=2,allow_nan=False)+"\n"
    )
    print("PAIRED_INTERVENTION_RESULTS")
    print(json.dumps(summary,indent=2,allow_nan=False))
    return 0


if __name__=="__main__":
    raise SystemExit(main())
