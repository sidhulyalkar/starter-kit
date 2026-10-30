# G4: Full-month VAV reheat-cap qualification

Status: first 8640-step simulation is an experiment, not controller promotion.

## Scientific question and frozen treatment

Does the preselected 50% native VAV_1 reheat cap reduce official loss over a
complete 30-day episode, **without** worsening occupancy-weighted
productivity, extreme five-minute thermal shocks, or severe room-level
comfort excursions?

The trial runs three independently constructed simulator environments on
public building_train_2/denver_8 with identical code/config/horizons:

1. Original official PIDWaterResetPolicy.
2. Sham cap at 100% native VAV_1 reheat, which must reproduce PID-water
   exactly in trace and reward or the experimental comparison is refused.
3. Frozen 50% native VAV_1 reheat cap, leaving all other policy command
   indices untouched at the same input state. Subsequent feedback may
   cause different downstream AHU and water-loop decisions.

The intervention is not part of TransferPolicy and must never use
privileged simulator state. Recorded internal physics is offline only.

## Preserve independent outcome dimensions

Official reward (lower loss is better) is recorded alongside the
original signed normalized productivity-regret, energy-cost and carbon
components. In the official source file
smart_buildings/smart_control/reward/setpoint_energy_carbon_regret.py,
normalized productivity regret equals
(actual productivity - minimum)/(maximum - minimum) - 1 when occupied,
so values closer to zero are preferable. This is not identical to the
unweighted severe-comfort proxy below.

Separately count every end-step zone temperature >=2 K outside its
individual sensor-reported setpoint band; track global five-minute
zone-temperature changes >=2 K and >=5 K. The comfort proxy is not the
official occupancy-weighted productivity component.

## Evidence validity versus controller quality

scripts/assess_vav_qualification.py checks exact scenario coverage,
unique scenarios, an active intervention, identical declared 8640-step
horizons, nonpositive and finite reward, reward/loss consistency, sham
identical official reward component totals, and exact reconciliation of
signed component differences. It fails closed on missing or malformed data.

A negative result is NOT a broken CI run: the evaluator reports it as
a scientifically meaningful adverse outcome and retains artifacts.

Predeclared per-scenario diagnostics are nonincreasing loss,
nonincreasing unweighted severe comfort rate, and nonincreasing >=5 K
whole-building shock count. Even when all checks pass on one public
building, controller status remains NOT QUALIFIED for generalization.

## Reproduce

~~~bash
python -m pytest -q tests/test_vav_qualification.py tests/test_vav_interventions.py
python -m scripts.run_vav_interventions \
  --building building_train_2 --weather denver_8 \
  --zone VAV_1 --steps 8640 --threshold-k 2 \
  --arms pid_water vav1_sham vav1_reheat50 \
  --results-dir smart_buildings/results/g4_full_month
python -m scripts.assess_vav_qualification \
  smart_buildings/results/g4_full_month/paired_interventions.json \
  --expect-scenarios 1 --steps 8640 --target-arm vav1_reheat50 \
  --output smart_buildings/results/g4_full_month/qualification.json
~~~

## Next gates

1. Full 30 official public building/weather pairs at 8640 steps with
   fail-closed per-scenario coverage and failure-preserving reports.
2. Locked generated-building seeds and independent weather-window
   structural and meteorological transfer studies.
3. Occupancy-weighted official productivity versus severe per-zone
   discomfort and energy/carbon tradeoffs.
4. If justified, train a new robust/hysteretic reheat strategy on
   separate development buildings and evaluate on untouched holdouts.
5. Only then consider TransferPolicy integration, with known-good
   fallback, action constraints and explicit comfort safety budget.

The 50% cap was selected after inspecting the initial 288-step public
Denver result, so further public comparisons are development
replications, not an independent hidden test.
