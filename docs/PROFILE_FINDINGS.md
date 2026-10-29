# G1 measured profiling findings

First nontrivial real-simulator profile:
[GitHub Actions run](https://github.com/sidhulyalkar/starter-kit/actions/runs/40824523084).

**Public scenario:** building_train_1/atlanta_2  
**Horizon:** 288 five-minute steps / 24 hours  
**Important:** a single 288-step rollout is not comparable to 8,640-step official
full-horizon returns and is not evidence of transfer.

| Measured item | PID-water | Combined adaptive |
| --- | ---: | ---: |
| 288-step mean absolute reward loss | 0.0000281706009 | 0.0000295063881 |
| Rollout wall seconds | 10.7766 | 10.6673 |
| Initialization seconds | 3.6238 | 3.2648 |
| Policy p95 call seconds | 0.0002630 | 0.0006735 |
| Max process RSS MB | 724.24 | 744.02 |

On this specific smoke scenario the combined adaptive was **4.74% worse**
than the PID-water reference. That must be kept in the experimental record.
We cannot tell which individual ablation causes it until comparable paired
trials across climates and buildings are available.

The dominant cost here is the simulator, not policy inference. A naive linear
extrapolation suggests about 5.3 wall-hours for 30 × 2 × 8,640 steps *on this
particular configuration*, before varying building sizes, initializations,
rendering, concurrency and CI overhead. This is only a rough capacity planning
calculation, not an observed 30-scenario benchmark.

### Immediate actions

- [x] 288-step paired real-simulator profiling.
- [x] Strict paired 288-step comparator and saved Actions artifact.
- [ ] One complete 8,640-step paired public scenario, to validate throughput
  extrapolation and reveal longer-horizon behavior.
- [ ] Full 30-scenario public paired ablation.
- [ ] Disjoint generated-building validation with weather-strip controls.
- [ ] Reward-component and severe-discomfort diagnostics before promotion.

Do not declare a winning controller or tune gains on a single short trajectory.

## Full 8,640-step public scenario result

[Successful 30-day full-horizon workflow](https://github.com/sidhulyalkar/starter-kit/actions/runs/40824822807):

- building_train_1/atlanta_2, PID-water mean loss: 0.02490677.
- Same scenario, combined adaptive mean loss: 0.01605735.
- Paired 8,640-step improvement: **35.53% lower loss**.
- Recorded policy p95 latency: 0.576 ms reference, 1.332 ms adaptive.
- Do NOT claim transfer on one known public scenario.

**Reproducibility caveat:** the committed published baseline_returns.json
lists PID-water reward -208.20547 (loss 0.02409786) on this exact scenario,
whereas our profile measured -215.19453 (loss 0.02490677), **3.36% higher**.
Investigate version, configuration and simulator conditions before claiming
exact published-reference reproduction. Both members of our paired local test
were run using the same current code revision and scenario.

## Same-policy replay check

[Two independent PID-water 288-step runs](https://github.com/sidhulyalkar/starter-kit/actions/runs/40835233468)
produced exactly the same total reward: -0.008113133068036404 in each run,
absolute difference 0. This supports short-horizon determinism on this
configuration but does not prove full-month reproducibility across hosts,
versions, or seeds.

Next: full 30-scenario matched paired evaluation and failure-preserving tail
metrics. Do not tune the controller to this one successful public scenario.
