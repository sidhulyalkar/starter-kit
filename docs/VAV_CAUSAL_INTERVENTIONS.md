# G3-C: Controlled intervention study for recurrent Denver VAV_1 shocks

Status: experimental simulation research, not a competition-controller change.

## Motivation

Public building_train_2 / denver_8 repeatedly shows five-minute temperature
jumps over 5 K in VAV_1. The preceding research audit found a roughly
61 m² room, 222 kJ/K simulated effective heat capacity, and 10 kW reported
source power with both reheat and damper saturated. Correlation with heater
saturation is not yet causal attribution.

## Predeclared intervention arms

Every arm creates an independent simulator, starts from the same scenario
config, and runs the unmodified PIDWaterResetPolicy everywhere **except**
the indicated VAV_1 action cap. Percentages below are **physical native
[0,1] fractions**, not normalized competition command values:

| Arm | Maximum VAV_1 reheat | Maximum VAV_1 damper |
| --- | ---: | ---: |
| pid_water (reference) | 100% | 100% |
| vav1_sham (identity cap) | 100% | 100% |
| vav1_reheat75 | 75% | 100% |
| vav1_reheat50 | 50% | 100% |
| vav1_damper75 | 100% | 75% |
| vav1_both75 | 75% | 75% |

The 100% sham MUST have exactly identical observed temperature traces and
total reward to the reference. Every nontrivial arm must actually modify
at least one controller action, or the comparison is refused. All continuous
action indices other than VAV_1 damper/reheat, and all discrete actions, are
unchanged **for the same policy input/state**.

Subsequent trajectory-dependent feedback in all controller outputs is
allowed to differ. This identifies effects of the *closed-loop policy
intervention*, not a fixed-action local impulse response.

## Measurements

Predeclared target:
1. Count of VAV_1 five-minute absolute temperature changes >=2 K and its
   maximum jump; note recurring >5 K events.
2. Total reference loss per step (lower is better).
3. Count of >=2 K five-minute shocks across **all** VAVs (to catch shifting
   the problem to another room).
4. Count and fraction of end-step zone temperatures >=2 K **outside each
   sensor-reported comfort band**, unweighted. This is a *comfort proxy*,
   NOT the official occupancy-weighted productivity/reward.
5. Number of time steps at which the treatment truly modified a command.

Stop/prevent promotion when the sham fails, an arm is inactive or a required
actuator field is missing. Do not equate the reduction of one-zone shocks
with improved comfort or overall reward.

## Execute

~~~bash
python -m pytest -q tests/test_vav_interventions.py
python -m scripts.run_vav_interventions \
  --building building_train_2 --weather denver_8 \
  --zone VAV_1 --steps 288 --threshold-k 2 \
  --results-dir smart_buildings/results/g3_vav_counterfactual
~~~

Each arm writes a full internal-physics trace, and the experiment produces
paired_interventions.json. The CI job archives all evidence.

## Interpretation / next tests

A single seeded simulator scenario can support a **within-simulator controlled
counterfactual** if the reference and sham are identical, but is not evidence
that the intervention is universally beneficial, safe, or transferable to real
HVAC plants. Capping reheat may reduce large positive jumps yet increase
persistent undershoot or degrade energy/comfort objectives.

Keep this diagnostic outside Kaggle submission. Repeat on other cold-weather
strips and on independently generated buildings before considering a
design change. If reheat caps help, study proportional/deadband/hysteresis
variants and AHU interactions; if not, investigate thermal source gradients,
convection/solver timing and zone-level feedback oscillations. Never tune
to the exact first 288-step Denver trace then report it as a held-out test.
