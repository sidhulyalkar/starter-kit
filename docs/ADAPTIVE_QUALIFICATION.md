# G0/G1 Experiment Runbook

Status: **experimental controller with interface and smoke-test checks**.
Not a competition submission, no claimed improvement until full benchmarks run.
The official PID-water baseline is unchanged; the starter kit's TransferPolicy
remains unchanged. Research design: docs/PROJECT_STRATEGY.md in PR #1.

## Installation

From your fork's working tree, with Python 3.12:

```bash
python -m pip install -e . pytest
python -m pytest -q tests/test_adaptive_core.py tests/test_compare_adaptive.py
python -m pytest -q tests/test_adaptive_integration.py
```

The integration tests load the **real** official simulator. No mock environment
is substituted. A successfully completed test establishes action validity, not
long-term comfort or energy performance.

## G0: Reproduce the published baseline first

```bash
python -m smart_buildings.evaluate --policies pid_water \
  --results-dir smart_buildings/results/g0_reference --steps 8640
python score_leaderboard.py smart_buildings/results/g0_reference \
  --policy pid_water --expect 30
```

The published `pid_water` score is about **0.0541413**. That number comes
from the shipped `baseline_returns.json`, not from a run of this branch.
Inspect per-scenario returns and reward component columns if a replay differs.

## G1: Fast closed-loop smoke tests

```bash
python -m scripts.run_adaptive_experiments \
  --policies pid_water adaptive_sensor adaptive_outlier adaptive_forecast adaptive \
  --buildings building_train_1 --weathers atlanta_2 \
  --steps 288 --results-dir smart_buildings/results/g1_smoke
python -m scripts.compare_adaptive smart_buildings/results/g1_smoke \
  --candidate adaptive --reference pid_water --expect 1 --steps 288
```

**Never** compare these 288-step episode scores against 8,640-step
published reference returns. A short episode misses many occupied and
eco-setback cycles, so results are solely diagnostic.

## G1: Paired full-horizon trials

Start with paired **reference and candidate** on the official set, before
running every ablation to manage CPU cost:

```bash
python -m scripts.run_adaptive_experiments \
  --policies pid_water adaptive --steps 8640 \
  --results-dir smart_buildings/results/g1_public
python -m scripts.compare_adaptive smart_buildings/results/g1_public \
  --candidate adaptive --reference pid_water --expect 30 --steps 8640 \
  --csv smart_buildings/results/g1_public/paired.csv
python score_leaderboard.py smart_buildings/results/g1_public \
  --policy adaptive --expect 30
```

The comparator refuses absent policies, incomplete paired coverage, unequal
episode horizons, nonnumeric or nonfinite reward, and positive rewards.

After first-pass qualification, run all four ablations to isolate gains and
regressions. Keep per-scenario files, physical-unit comfort metrics and the
reward component breakdown; the mean alone can hide severe failures.

## Structural transfer (do not tune on locked test seeds)

The competition provides a procedural building generator. Freeze your generator
version and split manifest BEFORE model fitting. Proposed building seed
partitions, pending collision audit:

- Train: 200–399
- Validation: 400–449
- Locked structural test: 500–549

These are **design proposals**, not already generated or executed datasets.
Using the same publicly shipped weather strips on new buildings is structural
transfer, not truly unseen-weather transfer. Reserve separate weather strips
where available and label the distinction transparently.

Example generation commands (exercise on a small validation sample first):

```bash
python -m smart_buildings.run_floor_generator --start-seed 400 --count 2
python -m smart_buildings.run_scenario_generator \
  --buildings building_0400 building_0401 \
  --weathers fairbanks_1 phoenix_6
python -m scripts.run_adaptive_experiments \
  --policies pid_water adaptive --steps 8640 \
  --buildings building_0400 building_0401 \
  --weathers fairbanks_1 phoenix_6 \
  --results-dir smart_buildings/results/g1_validation
python -m scripts.compare_adaptive smart_buildings/results/g1_validation \
  --candidate adaptive --reference pid_water --expect 4 --steps 8640
```

Check generator semantics before running large cross-products: a bigger building
with more AHUs increases simulator cost.

## Required promotion report

Record upstream commit, fork commit, Python and package lock, seed manifest,
weather-strip IDs, actual run horizon, wall-clock time, policy variants and
per-scenario summary JSON. Include:

1. Official loss mean and paired difference, with all scenario rows
2. Comfort-regret distribution and severe comfort excursion frequency
3. Energy and carbon contributions separately
4. Worst positive and negative deviations versus PID-water
5. Latency and memory for controller execution
6. Negative findings and failures

Tentative **research** gate: at least 5% lower mean loss on independently
generated validation buildings without >10% increase in severe comfort
violations. This is *not* an official competition criterion, and a single
validation improvement is not proof of safety or global superiority.

## Next G2: Data contract before fitting dynamics

Collect aligned `(observation_t, action_t, observation_t+1)` transitions
from multiple controllers. Label normalized vs native sensor values. Do not
include next-observation values, future occupancy, generated scenario IDs,
or reward internals in runtime policy features. Before any graph model, compare
persistence and simple linear/RC models on disjoint building-seed splits.
