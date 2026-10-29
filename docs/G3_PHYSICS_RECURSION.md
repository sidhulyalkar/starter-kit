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

## Verified run 2026-10-29: measured results (public pilot)

[Successful Actions run](https://github.com/sidhulyalkar/starter-kit/actions/runs/40956495749)
includes unit tests, the actual 288-step physics trace, independent public
training/evaluation trajectories, and a frozen-context recursive report. See
the Actions artifact g3-physics-and-recursive-evidence.

### Denver VAV_1 physical response (official PID-water, 288 steps)

- Room floor area **61.36 m²**; simulator room heat capacity **222,000.48 J/K**.
- **44 of 288** time steps have >2 K *absolute* temperature change.
- Step 198: **285.6333 → 292.7598 K**, observed **+7.1265 K**.
- During this heating response, both realized VAV damper and reheat were
  **100%**, with realized airflow **0.63246 m³/s**. AHU supply setpoint was
  **295.27 K**, hot-water supply sensor **316.03 K**.
- Simulator-reported thermal source for this room was **10,120.7 W**. A
  *source-only* equivalent temperature increment is **13.68 K**, whereas the
  observed change was **7.13 K**. This demonstrates why source power divided
  by room heat capacity is NOT an exact timestep prediction: the solver
  includes air coupling, diffusion, convection and losses. It does show
  the event coincided with strong heating authority and a relatively small
  effective room heat capacity.
- Repeated swings at steps 210 and 221 also show max damper/reheat with
  ~10 kW source input. **No definitive causal mechanism identified.**

Only the offline audit can access source power and physical heat capacity.
Do not use either privileged quantity as a model feature at inference.

### Frozen-input recursive forecast (one public target building)

Training: public building_train_1/atlanta_2, 288 steps, fixed ridge alpha=100.
Evaluation: public building_train_2/denver_8, 288 steps and **24 anchor
times** spaced 12 steps apart, 85 zones = **2,040 zone forecasts per horizon**.

| Horizon | Persistence MAE K | Ridge MAE K | Persistence RMSE K | Ridge RMSE K |
| --- | ---: | ---: | ---: | ---: |
| 5 min | 0.067695 | **0.063567** | **0.345966** | 0.346966 |
| 15 min | 0.127975 | **0.116970** | **0.482688** | 0.486264 |
| 60 min | 0.236041 | **0.194646** | 0.424950 | **0.412049** |

MAE improves **6.10% / 8.60% / 17.54%**, respectively. RMSE worsens
at 5 and 15 minutes but improves at 60 minutes. Individual 15-minute
errors exceed **8.86 K**; 60-minute max errors are about **5.00 K**,
compared with 4.62 K for persistence. The tails remain an issue.

**Interpretation:** We have observed some recursive signal beyond one step
but have NOT demonstrated safe counterfactual model prediction. The
forecast explicitly holds all exogenous state and actions constant while
the recorded real controller may change them. Anchors and zones within
one building are highly correlated, so do not treat 2,040 zone predictions
as independent samples or compute naïve significance from them.

**Decision:** keep the ridge model outside the controller. G3 is not
qualified for MPC. Next prioritize data diversity, regime-specific
failures, and a policy-feasible control rollout with uncertainty bounds.
