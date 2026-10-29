# G3 physical jump audit and recursive forecast qualification

## Scope

This is diagnostic research, not a qualified competition policy. Our earlier
fixed ridge model improved one-step MAE on public building 2, but its squared
error regressed on the Denver weather strip. The recorded Denver sequence
contained 21 zone-temperature changes above 5 K in one 5-minute interval.
Some of the largest repeatedly affected VAV_1.

## A: Physical energy-response diagnosis

Use scripts/diagnose_zone_physics.py to run the official PID-water controller
and inspect the simulator **offline**.

The report includes observed current/next zone temperatures, normalized
commands, realized VAV damper/reheat settings, airflow, AHU and hot-water
setpoints, room floor area, internal heat capacity and reported heat input.

Research-only simulator state is NOT available to the competition policy and
must not be used as model input at inference. The source-power-equivalent
temperature change is only an approximation: thermal diffusion, wall losses,
implicit air-stream coupling and convection are additional terms. It is
not an exact energy-conservation residual.

~~~bash
python -m scripts.diagnose_zone_physics \
  --building building_train_2 --weather denver_8 \
  --policy pid_water --zone VAV_1 --steps 288 \
  --results-dir smart_buildings/results/g3_physics
~~~

## B: Frozen-context recursive forecasting

Use scripts/evaluate_recursive_ridge.py to predict temperatures at 5, 15
and 60 minutes. Only the **initial** observed state and action can enter the
rollout. Zone temperatures are replaced with the model's own successive
predictions. Other sensors, climate, time and control actions remain frozen
at the anchor state. Future measurements appear only as scoring targets.

This tests prediction stability and leaks no future observations. It
does **not** predict the real future policy's changing actions/weather, so
long-horizon model errors also include this frozen-context mismatch.
It does not validate counterfactual action effects or MPC readiness.

~~~bash
python -m scripts.evaluate_recursive_ridge \
  --train smart_buildings/results/g3_recursive/building_train_1/atlanta_2/pid_water/transitions.jsonl.gz \
  --eval smart_buildings/results/g3_recursive/building_train_2/denver_8/pid_water/transitions.jsonl.gz \
  --horizons 1 3 12 --stride 12 --alpha 100 \
  --out smart_buildings/results/g3_recursive/recursive_report.json
~~~

## Model promotion gates

1. Multi-policy training on many building seeds with diverse weather regimes.
2. Disjoint held-out **procedural buildings** and separate weather-strip checks.
3. Recursive multi-step Kelvin error better than persistence and gray-box models,
   without worse tail behavior and with uncertainty calibration.
4. Plausible response to new actions, tested with simulator intervention.
5. Closed-loop control benefit and deterministic fallback when optimizer fails.

A GNN or MPC should be attempted only after these gates, with negative
evidence preserved and clearly reported.
