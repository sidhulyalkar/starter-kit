# G6: Observation-only selective reheat prototype

**Status:** unqualified experimental policy. No edits to main, official
TransferPolicy or the simulator.

## Motivation

The fixed 50% VAV_1 reheat cap looked beneficial on three separate
*public one-day pilot* trajectories but failed a complete 30-day test:
loss was 0.223% worse; occupancy-weighted signed productivity regret,
energy and carbon were all worse; severe comfort outliers increased,
although >=5 K thermal jumps decreased.

A no-rerun day-by-day physics analysis confirmed the fixed cap leaves
VAV_1 ~0.59 K colder on average across every day of that month.
Fewer rapid thermal swings do not imply a better control policy.

## Explicit test hypothesis, not a guarantee

A *conditional* cap should preserve the heating authority that the
fixed cap suppressed while still limiting potential overshoots.

The new prototype inherits exactly one unchanged PIDWaterResetPolicy
call at each step and only modifies the VAV_1 reheat command. It uses
three controller-observable inputs:

- current normalized VAV_1 zone-air temperature converted to Kelvin,
- current VAV_1 heating setpoint sensor, already given to the policy in K,
- previous observed VAV_1 temperature stored as local policy state.

Predeclared intervention conditions:

1. The target zone is **1.0 to 2.0 K below its current heating setpoint**.
2. The zone has risen by at least **0.25 K in one five-minute step**.
3. Once the cap engages, it may remain engaged while the zone continues
   rising by at least **0.05 K per five-minute step**, within the same
   shortfall window.
4. Temperature falling/stagnant, no prior observation, missing/invalid
   data, and shortfall beyond 2.0 K all **preserve original PID-water
   authority**. Input shape/nonfinite values fail closed.
5. Two candidates cap the actuator's native reheat fraction at **50%**
   or **75%** respectively. Fractions are normalized to official
   [-1,1] actions; no damper, AHU, water or other controller action
   index is deliberately altered.

The 75% variant probes a weaker version of the same rule, rather than
tuning after observing the first 288-step result.

## Pilot controls and endpoints

The first pilot evaluates two *public development* scenarios:

- building_train_2/denver_8 (previous cap-discovery scenario);
- building_train_4/denver_2 (earlier one-day replication).

Five arms per scenario, each with a fresh simulator:

- original PID-water;
- identity 100% sham, required to match original trace and reward;
- rejected fixed 50% cap negative control;
- selective 50% cap, only under rising moderate-shortfall conditions;
- selective 75% cap, same conditions.

Compare per-step official loss; signed productivity-regret, normalized
energy and carbon; VAV_1 shocks; all-zone shocks; and the strict
unweighted severe comfort proxy. Record how frequently a candidate
**actually changes** the VAV command. If an intervention never
activates, record *no effect*, never an invented benefit.

A positive development result is NOT an untouched holdout result and
cannot justify TransferPolicy changes.

## Scientific limitations

- A temperature trend is not a causal thermal-response model.
- The policy uses no future observations but can still destabilize
  a shared hydronic heating system by changing closed-loop feedback.
- A local VAV_1 rule may not transfer to other zones, AHUs or buildings.
- Dynamic hysteresis thresholds and 50%/75% are fixed in advance.
  Do not tune them on these public test episodes, then call those same
  episodes independently validated.
- Never assert real-world HVAC safety from simulation-only results.

## Promotion gate

First compare the two public 288-step episodes and preserve *all*
adverse comfort/reward outcomes. Reject candidate immediately if
it cannot meaningfully change the command, violates shape/bounds,
or cannot reproduce the identity sham.

Only if the candidate is not dominated should it receive a new
full 8640-step *public* trial and independent generated-building
validation. Even then report confidence intervals at the building
level, not from correlated zone-step samples.

## Reproduce

~~~bash
python -m pytest -q tests/test_selective_reheat.py
python -m scripts.run_selective_reheat \
  --building building_train_2 --weather denver_8 \
  --steps 288 --results-dir smart_buildings/results/g6_pilot
~~~
