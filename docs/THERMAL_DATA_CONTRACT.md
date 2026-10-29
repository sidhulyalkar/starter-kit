# G2: Thermal-transition data qualification

Status: first implementation. No model accuracy claims, no real dataset beyond the CI smoke sample yet.

This branch builds on the adaptive profiling branch and uses a stacked PR. We separate collecting valid trajectories from training a graph model.

## Transition schema

Each compressed JSON Lines record includes exactly:

- step: zero-based, contiguous transition index
- observation_t: flat observation supplied to the controller at t
- continuous_action_t and discrete_action_t: validated commands sent
- observation_tp1: next observation from the simulator, a **training target only**
- reward_t: training diagnostic, NOT a physics-model input

The separate manifest contains observation field names, action indices, normalization constants, AHU-zone topology, scenario config SHA256, floor-plan/zone-map checksums, dimensions and run provenance. It is **not** used as a dynamic controller feature. Static building metadata genuinely given to the policy (like topology) may be used. Scenario/weather IDs, generator seeds and config paths are never model features.

Every training archive must pass checksum, status, dimension, numeric-value, field-name, and contiguous-step checks before ingestion.

## Example collection

~~~bash
python -m pip install -e . pytest
python -m pytest -q tests/test_thermal_records.py
python -m scripts.profile_control \
  --building building_train_1 --weather atlanta_2 \
  --policies pid_water adaptive --steps 288 \
  --record-transitions \
  --results-dir smart_buildings/results/g2_smoke
python - <<'PY'
from pathlib import Path
from smart_buildings.experiments.thermal_records import verify_archive
for path in Path("smart_buildings/results/g2_smoke").rglob("transitions.jsonl.gz"):
    print(path, verify_archive(path)["record_count"])
PY
~~~

Use 8640 steps for the full horizon. Short rollouts test collection integrity, **not** prediction, policy performance, or transfer.

Do not commit generated archives. Store hashed and versioned trajectory artifacts externally and reference them from experiment manifests.

## Proposed frozen building splits

Audit collisions with existing generated building configs and freeze these ranges *before* collecting data:

- Train building seeds 200–399
- Validation building seeds 400–449
- Locked structural holdout building seeds 500–549

Reusing the same recorded weather strips across these building splits tests **structural** transfer, not unseen weather. An explicitly disjoint weather-strip holdout is required for a weather-transfer claim.

## First thermal model experiment

1. Temperature persistence: next temperature equals current.
2. Ridge/gray-box temperature-change model with available measurements and control commands.
3. AHU-pooled model with shared parameters.
4. Nonlinear residual dynamics only if simpler models plateau.
5. Typed graph dynamics only if they improve held-out multi-step validity.

Evaluate on whole held-out buildings. Report Kelvin MAE/RMSE at 5, 15 and 60 minutes using *recursive*, not teacher-forced, predictions. Report calibrated uncertainty and physical plausibility. Later, verify whether better predictions yield better closed-loop control.

No reinforcement learning is justified by a dataset collection smoke test.
