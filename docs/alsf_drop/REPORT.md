# ALSF-Drop: implementation and measured comparison

**Genuine evaluation savings are achieved, but target-HV efficiency does not
improve over baseline in these three seed pairs.** At 200 generations Drop saves
1,002-1,043 real model calls per run, while its final HV/IGD+ are slightly worse
than baseline and better than Resample. All three Drop runs eventually attain
the predefined target, but require more evaluations than their paired baselines.

## Implementation and verification

`run_di_barba_drop.py` uses `src/offspring_drop.py`. A normal duplicate-eliminated
mating batch is generated exactly once per generation, then forbidden candidates
are removed before objective evaluation. No Drop replacements are generated.
Existing parents merge with accepted offspring for ordinary NSGA-II survival.
Zero accepted offspring preserve the parents and still advance the callback,
termination and generation counter. Initialization is unfiltered. Defaults,
physical tolerances, thresholds, floors, operators and seed semantics are preserved.
The first classification is after generation 50; its policy first affects batch 51.

**49 automated tests pass** on the final integrated branch, including the existing suite
and the preceding convex-validation tests (42 passed before that base update). An instrumented real
pymoo objective records every evaluated coordinate; synthetic forbidden proposals
never appear in its calls. Partial-drop and all-drop cases verify exactly one
mating batch per generation, fewer/zero new model calls, valid parent size and
completion without premature termination. Other tests cover archive updates,
seed determinism, empty-policy equality with ordinary NSGA-II, no overwrite,
unchanged evidence when sparse-cell reporting is disabled, independent exact
sensitivity and the complete comparison plot workflow.

The benchmark and convex reference calculation were not modified. Baseline and
Resample executables, `OffspringRepair`, local variation and exclusion-policy
implementations are unchanged. The existing diagnostic opt-out for sparse cell reporting is reused without
changing the shared classifier on the updated base branch. The old baseline
and Resample results and reference were copied unchanged from validated source
commit `1e456cbf6d523bef4a29578d56bf0b39826f7622`, with hashes in
[protocol.json](protocol.json). Checks confirm identical histories through the
first 50 generations for all methods/seeds. No sensitivity parameters were tuned.

## Requested smoke test

60 generations, population/offspring 40, flag start 10, seed 42:

- Proposed: **2,400**; dropped without replacement: **22**.
- Accepted, actual model calls and evaluator count: **2,378** each.
- Genuine saved evaluations: **22 (0.917%)**.
- Parent population stays 40; final nondominated set has 40 points.
- Cache hits, archive repeat evaluations and proposal shortfalls: zero.
- Normal duplicate elimination removed 80 raw candidates; these are **not**
  added to ALSF savings. Runtime about 3.25 seconds.

The smoke run was completed and verified before full Drop experiments began.
See [smoke summary](smoke/summary.json) and [process log](smoke/process.csv).

## A. Fixed generation budget: 200 generations

Each mode uses population/offspring 100, the same operators, seed and common
reference. Baseline and Resample execute 20,000 model calls per run. Drop returns
20,000 actual proposals including initialization, then executes only accepted
ones. No caches or normal mating shortfalls contribute to reported savings.

| Seed | Drop model calls | True saved calls | Baseline HV | Resample HV | Drop HV |
|---|---:|---:|---:|---:|---:|
| 13 | 18998 | 1002 (5.010%) | 1.127048 | 1.126237 | 1.126904 |
| 42 | 18957 | 1043 (5.215%) | 1.127097 | 1.126245 | 1.126607 |
| 97 | 18978 | 1022 (5.110%) | 1.127440 | 1.126145 | 1.126613 |

Across Drop runs, **56,933 actual calls versus 60,000 proposed**, saving **3,067
calls (5.112%)**. Every final parent population and final Pareto set has 100
members. Drop histories contain 200 generation records and exactly 199 mating
batches, without refilling rejected candidates.

Mean final HV: baseline 1.127195, Resample
1.126209, Drop 1.126708.
Mean IGD+: baseline 0.002495, Resample
0.002944, Drop 0.002777.
Drop's mean HV is -0.043%
relative to baseline, and mean IGD+ is
11.3% higher.
Thus its HV remains close, but the cost reduction comes with measurable changes
in front quality/distribution. No equivalence margin or statistical significance
threshold was predefined; “approximately comparable” is not asserted as a formal
success criterion. Relative to Resample, Drop achieves better final HV and IGD+
in every measured seed with fewer calls.

## B. Objective-evaluation efficiency

The primary target was fixed before inspecting Drop results at **99.5% of the
common reference HV**, namely **1.1265765048106742**. First attainment uses actual
recorded model calls, not generations or theoretical population budgets:

| Seed | Baseline evaluations to target | Resample | Drop evaluations to target |
|---|---:|---|---:|
| 13 | 11600 | not reached by 20,000 | 16185 |
| 42 | 15800 | not reached by 20,000 | 18139 |
| 97 | 11100 | not reached by 20,000 | 14048 |

The target requires more Drop evaluations than baseline for every paired seed.
Resample observations are right-censored at 20,000 calls, not assigned a fictitious
successful attainment. First crossings and final fronts are different measures;
finite-population HV can fluctuate after a crossing.

Convergence curves share actual-evaluation axes. Common allowed budgets of
5,000, 10,000, 15,000 and the shared completed horizon (18,957) are also compared.
At the 15,000-call allowance, mean observed HV is
1.126901 baseline,
1.125442 Resample and
1.126435 Drop; corresponding IGD+ values are
0.002619,
0.003273 and
0.002856.

**Budget granularity limitation:** these checkpoints use the last completed
NSGA-II generation at or below each allowed budget. Drop has variable batch
sizes, so its actual checkpoint can undershoot the allowance; the largest
undershoot here is **91 calls**. Actual checkpoint counts and unused budget are
exported. No curve is fabricated by interpolation or by using a state beyond
the cap. This is an allowed-budget comparison, not a separately rerun experiment
with exactly synchronized intra-generation stopping. Tiny differences within
this resolution should not support superiority claims. The target-attainment
differences are substantially larger than one batch.

## Actual sensitivity and runtime

The same independent exact convex-box construction audits up to 25 objective-
ranked final designs per run: bounded least-squares minima and 1024-vertex f1
maxima, separable f2 extrema, the same symmetric denominator and 0.001 floors.
All **75 selected Drop final designs are sensitive** at the 10% threshold, as
are the corresponding 75 baseline and 75 Resample audit designs. There is no
observed binary robustness improvement in these selected subsets. Continuous
R1/R2 values are exported and show tradeoffs, not a consistent joint improvement.
No exact diagnostic evaluations are inserted into the archive or counted as
optimizer model calls. Unsampled designs are not certified by these audits.

Mean optimization runtimes (including archive processing and indicators):
baseline 1.83 s,
Resample 185.42 s,
Drop 180.12 s. Drop archive processing averages
177.76 s. On this cheap analytical
model, the overhead outweighs saved objective time. Timing is descriptive on a
shared cloud machine, not a controlled hardware comparison. Expensive objective
functions could change the cost tradeoff, but that is not established here.

## Interpretation and reproducibility

Drop implements the intended mechanism correctly and saves about five percent
of evaluations at fixed generations. That is a real saving, independently
separated from duplicate elimination, operator surplus, caches and shortfalls.
It does not establish faster attainment of desired Pareto quality: baseline
reaches the fixed target with fewer calls in all three seeds. Drop nevertheless
performs better than the previous Resample variant under the measured fixed-
generation protocol. These three seeds support an exploratory result for this
benchmark, not universal claims. Neither nominal convexity nor archive-stable
labels guarantee actual box robustness.

- [Method and runnable commands](../../README_ALSF_DROP.md).
- [Verification and accounting checks](verification.json).
- [Fixed-generation metrics](plots/fixed_generations.csv),
  [actual-evaluation checkpoints](plots/fixed_evaluation_checkpoints.csv),
  [target-HV attainment](plots/target_hv_attainment.csv).
- [Unchanged reference CSV](reference/reference.csv).
- [Complete Drop result CSVs/settings/summaries (ZIP)](full_drop_results.zip),
  including both required archive files for every full run.
- `full/<method>_seed<seed>/` contains measured histories, settings, process logs,
  final fronts, summaries and exact-sensitivity audits. Historical baseline and
  Resample artifacts retain their original bytes and are identified by source
  hashes. Their original output directories were never touched.
- [Quality versus actual evaluations PNG](plots/01_quality_vs_actual_evaluations.png) /
  [PDF](plots/01_quality_vs_actual_evaluations.pdf).
- [Final fronts PNG](plots/02_final_fronts_fixed_generations.png) /
  [PDF](plots/02_final_fronts_fixed_generations.pdf).
- [Evaluations to target PNG](plots/03_evaluations_to_target_hv.png) /
  [PDF](plots/03_evaluations_to_target_hv.pdf).
- [True savings PNG](plots/04_true_drop_savings.png) /
  [PDF](plots/04_true_drop_savings.pdf).
- [Final quality and calls PNG](plots/05_quality_and_evaluations_fixed_generations.png) /
  [PDF](plots/05_quality_and_evaluations_fixed_generations.pdf).
