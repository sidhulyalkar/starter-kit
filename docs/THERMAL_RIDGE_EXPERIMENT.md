# G3: First action-conditioned thermal dynamics baseline

Status: experimental one-step predictor on **public** building trajectories. Not a qualified GNN, MPC controller, or Kaggle policy.

## Scientific question

Can a global, shared, regularized linear dynamics model beat "the temperature will stay the same" on a *different* building, given only:

- current zone/outdoor temperature observations;
- commands issued at the current five-minute timestep;
- static AHU-to-zone topology and registered action indices;
- current time-of-day and day-of-week cycle from observation inputs?

No hidden future weather, next observations, simulation state internals, scenario IDs or reward components become model inputs. Physical zone temperatures are reconstructed in Kelvin from the shipped observation normalizer.

## Model

For zone z and action a at time t:

    T[z,t+1] = T[z,t] + b + beta.T @ standardized_features(zone,z,t,a)

Features cover outdoor-to-zone temperature gradient, mean served-zone temperature gradient, current AHU supply/pressure/recirculation command, AHU run state, VAV damper/reheat command, water setpoint/pump, current time cycles, and modest physically motivated interactions. A **single shared coefficient vector** fits all zones; it does not encode a building-specific ID or learned zone embedding.

Ridge regularization alpha=100 is fixed before evaluating on target building. Training/evaluation refuse any overlap in building IDs, even if the weather strip is different. A prior baseline reports persistence error at t+1 in Kelvin. Both models are evaluated on exactly the same target values.

## Running the research pilot

The first pilot is automated in GitHub Actions:

- Training: building_train_1/atlanta_2, PID-water, first 288 transitions
- Evaluation: building_train_2/denver_8, PID-water, first 288 transitions

This is intentionally a **combined building + weather distribution shift** on two *public* buildings, not hidden test/generalization evidence. More trajectories, paired weather regimes, and procedurally generated held-outs are required for a transferable-control claim.

Command:

~~~bash
python -m scripts.fit_thermal_ridge \
  --train smart_buildings/results/g3_pilot/building_train_1/atlanta_2/pid_water/transitions.jsonl.gz \
  --eval smart_buildings/results/g3_pilot/building_train_2/denver_8/pid_water/transitions.jsonl.gz \
  --alpha 100 \
  --output smart_buildings/results/g3_pilot/cross_building_ridge.json
~~~

## Guardrails and limits

- **No test-set tuning:** alpha and feature list are selected before reading evaluation performance. Do not select a new alpha using the target building and report the same target as a clean test.
- **No action counterfactual claim:** the regression identifies correlations under the data-collection policy; it does not establish causal HVAC effectiveness without sufficient action diversity.
- **No multi-step validity yet:** one-step errors may be extremely small because the thermal system moves slowly. We must verify open-loop recursive forecasts and calibration before any MPC.
- **No structural OOD claim:** new procedural building seeds must be reserved for final validation.
- **No performance promotion:** a predictor can beat persistence yet worsen closed-loop rewards.

## Next gates

1. Expand training to multiple buildings, weathers, and bounded safe simulation excitation, keeping frozen building-seed splits.
2. Compare against persistence, linear RC/ridge, AHU-pooled and nonlinear residual models under identical held-outs.
3. Check multi-step recursive Kelvin RMSE/MAE at 5, 15 and 60 minutes and confidence interval coverage.
4. Only then train a typed graph-conditioned model or allow the model into a constrained MPC.

## First observed public cross-building result (2026-10-29)

[Successful GitHub Actions pilot](https://github.com/sidhulyalkar/starter-kit/actions/runs/40835021698), on one source/target pair:

- Training: building_train_1 / atlanta_2, 288 steps, 39 zone sensors = 11,232 zone observations.
- Evaluation: building_train_2 / denver_8, 288 steps, 85 zone sensors = 24,480 zone observations.
- Persistence MAE: 0.08144898 K; shared ridge MAE: 0.07722960 K (5.18% lower).
- Persistence RMSE: 0.36475677 K; shared ridge RMSE: 0.36520366 K (slightly **worse**).
- Persistence 95th-percentile absolute error: 0.32944 K; ridge: 0.31713 K.

**Decision:** not qualified for model-based control. One-step MAE improved but
RMSE did not; the model has no recursive 15/60-minute validation, uncertainty
calibration, or data diversity across policies. Also, building and climate
changed together in this first comparison. A follow-up evaluates the second
public building under Atlanta as well, and reports per-zone tail failures.
This is a public pilot only, not procedural holdout or private competition score.

An apparently promising average cannot mask larger squared errors. Do not
reuse these public results for unconstrained hyperparameter selection and
then describe them as an untouched held-out test.
