"""Predeclared selective-reheat test arms using real public building physics.

Do not modify official TransferPolicy or evaluate.py. All interventions
wrap the unchanged original PIDWaterResetPolicy. Store *negative* outcomes.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts.diagnose_zone_physics import diagnose
from scripts.run_adaptive_experiments import POLICY_MAP
from scripts.run_vav_interventions import paired_intervention_report
from smart_buildings.experiments.selective_reheat import (
    SelectiveReheatPolicy, SelectiveReheat75Policy,
)
from smart_buildings.experiments.vav_interventions import (
    VAVCaps, intervention_policy_class,
)

POLICY_MAP.update({
    "vav1_sham": intervention_policy_class("vav1_sham", VAVCaps()),
    "vav1_reheat50": intervention_policy_class(
        "vav1_reheat50", VAVCaps(reheat_max_fraction=.5)),
    "vav1_momentum50": SelectiveReheatPolicy,
    "vav1_momentum75": SelectiveReheat75Policy,
})
PILOT_ARMS = (
    "pid_water", "vav1_sham", "vav1_reheat50",
    "vav1_momentum50", "vav1_momentum75",
)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--building",default="building_train_2")
    ap.add_argument("--weather",default="denver_8")
    ap.add_argument("--steps",type=int,default=288)
    ap.add_argument("--arms",nargs="+",choices=PILOT_ARMS,
                    default=list(PILOT_ARMS))
    ap.add_argument("--results-dir",type=Path,
                    default=Path("smart_buildings/results/g6_pilot"))
    args=ap.parse_args()
    if not {"pid_water","vav1_sham"}.issubset(args.arms):
        ap.error("Must include original PID-water and identity sham")
    reports={}
    for arm in args.arms:
        reports[arm] = diagnose(
            args.building,args.weather,"VAV_1",arm,args.steps,
            threshold_k=2.0,output_root=args.results_dir
        )
    paired=paired_intervention_report(
        reports,allow_inactive_treatment=True
    )
    args.results_dir.mkdir(parents=True,exist_ok=True)
    (args.results_dir/"paired_selective_pilot.json").write_text(
        json.dumps(paired,indent=2,allow_nan=False)+"\n")
    print("SELECTIVE_REHEAT_PAIRED_REPORT")
    print(json.dumps(paired,indent=2,allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
