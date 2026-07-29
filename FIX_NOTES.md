# SGB 0.3.0 Methodological and Runtime Fixes

## Status

- Full source syntax check: passed.
- Automated tests: **181 passed**.
- P9 calibration-only smoke run: passed.
- Targeted Experiment 2 run: nontrivial immediate loss and multi-step recovery.
- Targeted Experiment 3 run: nonzero interior thresholds selected by balanced accuracy.
- Targeted Experiment 4 run: ordered low-to-high maturity response retained.

The sandbox did not contain the exact Mesa installation from the project virtual environment, so the test suite was executed with a minimal API-compatible Mesa test stub. Run the same suite once in the project `.venv` before the final research pipeline.

## Corrections

1. **Removed long-run dimension collapse**
   - Failure magnitude reduced from `0.012` to `0.003`.
   - Recovery increased to `gap_fraction_per_step: 0.05` and `maximum_adjustment_per_step: 0.005`.
   - Recovery now targets each organization's sampled reference profile instead of one common population mean.

2. **Removed activation-order bias**
   - All exchange outcomes are applied first.
   - Every organization then receives exactly one recovery update per simulation step.

3. **Aligned exchange and threshold logic**
   - Pair success uses the less-ready endpoint as the bottleneck.
   - Experiment 3 uses the same minimum-endpoint rule.

4. **Fixed Experiment 3 temporal leakage**
   - Exchange records now store sender and receiver D1-D5/SVC values before the outcome.
   - Threshold evaluation uses those event-time dimensions instead of one terminal score map for the entire observation window.
   - Default optimization changed from F1 to balanced accuracy.
   - Tie-breaking now prefers the higher threshold.

5. **Made Experiment 2 meaningful**
   - Recovery is measured over shock-affected organizations rather than diluted over the full ecosystem.
   - Recovery requires restoration of 90% of shock loss, subject to a 0.001 absolute floor.
   - One-hop propagation avoids automatically affecting every node in the N=50 Barabási-Albert graph.

6. **Corrected sensitivity design**
   - P2-P7 dimension-weight sensitivity rescales shared baseline snapshots post hoc, avoiding stochastic path-divergence contamination.
   - Threshold sensitivity uses event-time endpoint scores and balanced accuracy.
   - P9 is now a public-source calibration item, not a simulation scenario. Organization type is currently descriptive and does not impose operational behavior, so rerunning six P9 mixtures was computationally expensive but scientifically inert.
   - Added live sensitivity progress bars.

7. **Aligned final seed counts**
   - Experiments 1-4, time convergence, population convergence, and full sensitivity now use 50 seeds, matching the previous seed-convergence recommendation.

8. **Corrected reporting**
   - Final report no longer states that P9 is pending when P9 metadata reports completion.

## Previous runtime

The previous experiments were not skipped. Their row counts matched the configured design:

- Experiment 1: 30 seeds × 3 frameworks = 90 rows.
- Experiment 2: 30 seeds × 3 framework recovery views = 90 rows.
- Experiment 3: 30 seeds with full coarse/refined threshold scans.
- Experiment 4: 30 seeds × 5 levels × 3 frameworks = 450 rows.

The shorter experiment runtime compared with older code is therefore acceptable. The long stage was full sensitivity because it reran many scenarios and seeds sequentially. Removing simulation-inert P9 scenarios reduces that cost, but full sensitivity remains the longest stage.

## Required rerun

The dynamics and experiment definitions changed, so old convergence and experiment outputs must not be mixed with version 0.3.0 outputs.

```bash
cd /home/darien/Uni/SGB_Project
source .venv/bin/activate

python -m compileall -q sgb tests
python -m pytest -q
```

Archive old results:

```bash
timestamp="$(date +%Y%m%d_%H%M%S)"
mv outputs "outputs_v0.2.0_${timestamp}"
mkdir -p outputs
```

Run the full pipeline:

```bash
PYTHONUNBUFFERED=1 \
python -m sgb.pipeline \
2>&1 | tee outputs/full_pipeline_console.log
```

A scientifically rejected sensitivity result is not a software crash. The pipeline intentionally allows final reporting to record both accepted and rejected robustness decisions.
