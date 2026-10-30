# G5: Explain why 288-step fixed reheat cap gains reversed over 8640 steps

## Established evidence

The original 50% VAV_1 cap looked better on the first 288 steps of the
public building_train_2/denver_8 scenario. Two independent 8640-step
replications then showed **0.223% worse loss**, a worse unweighted severe
comfort proxy and worse official normalized productivity, cost and carbon
components, even as >=5 K all-zone temperature jumps dropped by 33.3%.

Source runs:
- https://github.com/sidhulyalkar/starter-kit/actions/runs/40992831555
- https://github.com/sidhulyalkar/starter-kit/actions/runs/40993164516

The cap is rejected for the competition submission and should not be
retuned against the first 288 steps as though that were unseen data.

## Question for the next experiment

Is the fixed cap mainly valuable during short-lived rapid-heating episodes
and harmful during sustained cold deficits? Can we identify the latter
using *controller-visible* temperature, heating setpoint and recent
observed temperature change rather than privileged simulator heat input?

First test this mechanistic hypothesis using **previously saved traces**.
This is a descriptive offline attribution, not a randomized causal
decomposition or a new policy.

## Saved evidence and limitations

G4 retained per-step **VAV_1** temperature, reheat fraction, damper
fraction, airflow and reported simulator source power. It stored
episode-wide official reward components and all-zone comfort/shock counts,
but **not time-resolved** energy/carbon/productivity or occupancy-weighted
comfort. Never reconstruct those day-specific quantities from a single-zone
trace.

The analysis is purely retrospective and must never introduce internal
simulator thermal source power into the online policy.

## G5 offline analysis

The workflow downloads the exact completed G4 artifact through GitHub
Actions read-only permissions and produces time-stratified JSON and CSV
with:

- day-by-day mean, minimum and maximum target-zone temperatures;
- daily >=2 K and >=5 K target-zone temperature jump counts;
- average realized VAV reheat and its near-saturation fraction;
- fraction and longest uninterrupted duration that treatment VAV_1 is
  at least 1 K colder than the reference;
- average target-zone source power as a diagnostic only;
- first-day versus remaining-day comparisons;
- alignment, provenance and finite-value validation.

The original G4 reports remain authoritative for official full-episode
rewards, comfort and global shocks.

## Decision after inspection

Only after attributing the temperature regimes do we implement a
**candidate** observation-only conditional reheat cap: retain baseline
heating authority under sustained or fast-worsening cold, and limit heating
only during a short-lived rising-temperature overshoot-risk state.

That candidate must be predeclared and compared against *both* PID-water
and the rejected fixed cap on multiple public development scenarios.
A positive public result remains a development result, not a qualified
submission. Preserve comfort-tail and worse-building cases.

## Commands

~~~bash
python -m pytest -q tests/test_regime_diagnostics.py
python -m scripts.analyze_regimes \
  --artifact-dir smart_buildings/results/g5_source \
  --steps 8640 \
  --out smart_buildings/results/g5_regimes/daily_thermal_regimes.json
~~~
