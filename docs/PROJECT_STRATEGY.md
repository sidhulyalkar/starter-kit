# Smart Buildings: transferable control research plan

Status: **research plan, not performance evidence**. Base: official NeurIPS 2026 starter kit. Target: January 15, 2027 final submission; November 13, 2026 paper round is optional.

## North-star hypothesis

Given only the allowed building metadata, the current observation and its own past observations/actions, a topology-conditioned controller with online thermal-state estimation can reduce held-out control loss relative to a strong reference controller without building-specific manual tuning or unsafe comfort regressions.

We are testing this hypothesis in steps. Better leaderboard numbers on the six provided buildings alone are **not** evidence of transfer.

## Decision: retain this fork

- Retain the upstream simulator, official baselines, evaluator, package layout, Apache 2.0 attribution and license.
- Keep \`main\` stable. Branch by research milestone and submit narrowly scoped PRs.
- Add an \`experiments/\` namespace for candidate policies and model training; expose the final selected controller through the official \`TransferPolicy\` interface only after qualification.
- Sync official upstream changes deliberately, with regression tests.
- Do **not** create a separate general-purpose graph-control library until evidence shows that multiple tasks benefit from the abstraction.

## Objective and controls

A five-minute decision selects AHU/pump on-off states and normalized setpoints, with variable-dimensional actions. The only permitted runtime context is the submitted policy's metadata, incoming observations and state maintained across its own calls. No scenario IDs, hidden simulator state, future weather, or future occupancy at inference.

Official local loss is mean over scenarios of absolute mean per-step reward, with full 8,640-step trajectories. Lower is better. The provided baseline_returns.json is a fixed 30-scenario, published reference, **not** a hidden evaluation or observed result of our code.

Track secondary metrics separately: comfort-regret component, worst-zone/long-tail comfort violation, energy cost, carbon, runtime, memory, failed actions, and per-scenario score. Evaluate with the same building, weather, horizon, seed and program version when comparing two policies.

## Promotion gates

### G0 — Reproducibility and compatibility

1. Pin a known upstream SHA and Python/dependency environment.
2. Verify published baseline_returns.json: 30 scenarios, all four named baselines, finite nonpositive reward, computed scores.
3. Run the real policy metadata/action interface in 1-, 288- and 8,640-step scenarios; validate dtypes, shapes, finite action range, legal discrete values and state reset.
4. Run PID-water on all 30 public scenarios and investigate any differences from published reference values. Record software SHA, Python, dependencies and simulator configuration.
5. Store artifacts under ignored results directories; publish small manifests and reports, not enormous rollouts or generated datasets.

**Gate:** a complete and reproducible baseline audit. Short rollouts are smoke tests only, never leaderboard comparisons.

### G1 — Prove or reject adaptive PID

Control arms: (A0) official PID-water; (A1) measured per-zone comfort band; (A2) outlier-aware AHU demand; (A3) bounded temperature-trend anticipation; (A4) combined candidate. Run all arms on the same scenarios and inspect **per-scenario paired** loss changes, reward components and worst regressions.

Initial research promotion thresholds (design choices, not competition rules): >=5% improvement in validation mean loss over PID-water, no >10% increase in severe comfort violations, no invalid actions, and no gross worst-case regression. If an ablation fails, remove it rather than just increasing model complexity. Repeated candidate selection on the same scenarios invalidates their use as a clean test.

**Gate:** measured improvement on generated validation buildings, not only public training pairs.

### G2 — Build a leakage-resistant rollout corpus

Generate building configurations under recorded seed ranges, split **by building seed**, and record building metadata, topology, envelope attributes, weather-strip identifier, policy/seed, observations, actions, rewards and next observations. Weather strips are historical recorded windows; do not pretend reused strips are novel weather. Keep a separate, explicitly labeled weather-strip holdout where the available input data permits.

Proposed split (freeze before experiment): training building seeds 200–399; validation 400–449; final locked structural test 500–549. Check that these seeds don't collide with official generator assets or prior generated experiments; revise once before collection if they do. Stratify by AHU count, zones, floor count and envelope properties. Do not train or tune on locked test results.

Collect from multiple control behaviors: official PID-water, schedule controls and bounded randomized perturbations *in simulation only*, subject to actuator constraints. The model needs action coverage to learn response, not just the observational distribution of one controller.

**Gate:** provenance manifest, deterministic train/validation/test split, schema checks, coverage diagnostics and zero future-data leakage in training features.

### G3 — Thermal system identification (before GNN)

Start with a per-zone gray-box thermal model: heat balance with outside conduction, neighboring-zone coupling, equipment heating/cooling and occupancy proxy. Fit a simple linear/regularized autoregressive model first. Then compare a compact nonlinear residual model.

Prediction target: physical zone-temperature change (Kelvin) at +1 step (5 min), +3 steps (15 min), +12 steps (1 h), and longer closed-loop rollouts. Features: observed zone temperature, outdoor temperature, actual AHU state, relevant AHU/VAV/HWS actions and measured flows, permitted schedules and metadata; history only. Distinguish **commanded** actuator values from **measured** values, and only use values known at the prediction time.

Tests: held-out building MAE/RMSE in Kelvin, calibration and coverage of predictive intervals, rollouts with recursive model predictions, performance by climate, and clear improvement over persistence and simple linear RC models. One-step MAE alone is not sufficient for predictive control.

**Gate:** improved multi-step forecast calibration and accuracy on held-out buildings at a feasible inference cost.

### G4 — Graph dynamics and predictive control

First graph: global HWS node, one node per AHU, one node per zone; typed HWS→AHU, AHU→zone edges. Add physical adjacency edges only when grounded in floor-plan geometry. The goal is generalization across variable counts and assignments, not merely visually attractive attention maps.

Compare (a) independent-zone dynamics, (b) AHU-pooled model, (c) typed graph dynamics. This ablation is critical: accept a graph only if it improves held-out dynamics or control.

Controller: short-horizon optimization around the PID-water reference, with forecast uncertainty, actuator bounds, limited action changes, and a deterministic reference fallback when an optimizer fails. Incorporate occupancy scheduling using current observed flags, not privileged future knowledge. Optimize expected regret while monitoring severe comfort violations separately.

**Gate:** paired improvements on unseen buildings without unacceptable comfort tail risk or excess inference latency.

### G5 — Residual reinforcement learning (conditional)

Only attempt policy-gradient/offline RL after G4. Initialize with the reference controller, limit action residuals, train across randomized building/weather scenarios and preserve a fallback. Evaluate policy collapse, safety constraint violations, intervention count and out-of-distribution building behavior.

**Gate:** RL must beat the strongest G4 non-RL policy on locked validation and cost less to run than the submission budget. Otherwise submit G4.

### G6 — Kaggle and reproducibility

Prepare a Kaggle notebook that implements the required inference-server policy interface. The code competition requires notebooks, internet disabled and CPU/GPU run-time <=9 hours; confirm the live competition rules again before submission. Test clean kernel imports, packaged model weights, variable shapes, deterministic initialization, runtime and memory.

Document the exact selected checkpoint, dependencies, validation protocol, model card, negative experiments, and final submission artifact. Keep public validation scores distinct from private leaderboard results.

## Execution cadence

- **Oct 28–Nov 3:** G0 baseline qualification, adapter tests, profiling.
- **Nov 4–Nov 13:** G1 adaptive PID ablations. Optional November paper only if results are reproducible.
- **Nov 14–Dec 1:** G2 data and G3 linear/gray-box dynamics.
- **Dec 2–Dec 20:** G3 residual models, G4 graph ablations.
- **Dec 21–Jan 5:** MPC/control qualification; optional G5 if justified.
- **Jan 6–Jan 15:** frozen controller, clean notebook packaging, final submission and paper.

Dates are planning targets, not promises. Prefer fewer credible comparisons to a large, unvalidated method catalog.

## Decisions to prevent misleading results

- Keep a *frozen* test set separate from all tuning and use it sparingly.
- Report **every scenario**, not just the mean or best climate.
- Use matched pairs for controller comparisons. When bootstrapping, account for clustering by building/weather where possible; naive independent-scenario bootstrap intervals are descriptive only.
- For safety-related claims, report the actual severe-discomfort metric and its definition, not just leaderboard loss.
- Do not equate good single-step prediction with stable closed-loop control.
- Avoid oversized RL training before measuring simulation throughput and policy inference latency.
- Archive failed hypotheses, including cases where the reference PID is better.
