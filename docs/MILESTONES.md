# Research milestone board

GitHub Issues is currently disabled on this fork (API returned 410).
This document serves as the initial issue backlog until the repository owner
enables Issues in repository Settings → General → Features. Each milestone
should become a separate issue/PR once issue tracking is enabled.

| Gate | Status | Proposed target | Depends on |
|---|---|---|---|
| G0 baseline qualification | IN PROGRESS | 2026-11-03 | published scorer |
| G1 adaptive PID ablation | NOT STARTED in fork | 2026-11-13 | G0 |
| G2 rollout corpus | NOT STARTED | 2026-11-25 | G0 |
| G3 gray-box dynamics | NOT STARTED | 2026-12-01 | G2 |
| G4 graph & MPC | NOT STARTED | 2026-12-20 | G3 |
| G5 residual RL (conditional) | NOT STARTED | 2027-01-05 | G4 qualified |
| G6 Kaggle notebook and paper | NOT STARTED | 2027-01-15 | qualified controller |

## G0: Trustworthy evaluation

- [ ] Verify dependency versions, upstream commit and environment hash.
- [ ] Audit all 30 published reference returns (script in this PR).
- [ ] Run PID-water baseline on the real simulator over full 8,640-step episodes.
- [ ] Store per-building/weather loss and reward components; compare reference values.
- [ ] Test the actual hybrid action interface at 1 step / 288 steps / full horizon.
- [ ] Quantify wall time and RAM use to set a compute budget.

Exit: reproducible full-length baseline; no missing scenarios or invalid actions.

## G1: Adaptive PID candidate

- [ ] Integrate existing experimental candidate in its own branch, not directly on main.
- [ ] Keep PID-water unmodified for paired controls.
- [ ] Evaluate sensor-only, outlier-only, forecast-only and combined candidates.
- [ ] Validate on newly generated buildings not used for tuning.
- [ ] Report all scenarios; assess worst-zone comfort tail and worst regressions.

Exit: proposed promotion gate in PROJECT_STRATEGY.md or a clear negative result.

## G2: Experimental dataset

- [ ] Freeze train/validation/test building-seed manifests.
- [ ] Log transitions with exact observation/action timing and physical units.
- [ ] Collect trajectories from multiple policies under bounded simulated perturbations.
- [ ] Check train/inference feature boundary and weather-strip overlap.

Exit: reproducible transitions and provenance manifest; no future information in inputs.

## G3: Thermal dynamics

- [ ] Persistence, linear RC/autoregressive, and nonlinear residual comparisons.
- [ ] Held-out 5-minute / 15-minute / 1-hour forecast error and calibration.
- [ ] Recursive rollout tests for error accumulation, not teacher-forced forecasts only.
- [ ] Compare per-zone and AHU-pooled representations before any graph model.

Exit: dependable multi-step forecast that improves on simple baselines.

## G4: Graph MPC

- [ ] Represent HWS, AHUs and zones with typed topology edges.
- [ ] Compare graph versus pooled models in identical training settings.
- [ ] Add constrained short-horizon optimization with deterministic PID-water fallback.
- [ ] Measure closed-loop improvement, comfort tail and inference latency.

Exit: proven control improvement, not just prediction improvement.

## G5: Residual RL (only if justified)

- [ ] Decide using G4 results whether RL has a realistic opportunity.
- [ ] Use bounded residual actions around a qualified reference controller.
- [ ] Test policy collapse, tail-risk regressions and generalization.

Exit: beat strongest non-RL controller or discard RL for final submission.

## G6: Submission

- [ ] Package notebook, weights and required Kaggle Policy inference interface.
- [ ] Prove clean offline execution, variable-dimensional actions, deterministic init.
- [ ] Meet competition CPU/GPU runtime <=9 hours and confirm current competition rules.
- [ ] Freeze reproducible experiment report and final artifact before January 15, 2027.
