# Twenty paired seeds on both benchmarks: 120 measured runs

**ALSF-Drop's outcome depends on the problem.** On the two-dimensional Gaussian
benchmark it saves about **25.3%** of model evaluations at 200 generations, with
no statistically clear final HV/IGD+ difference from baseline in this sample.
This does **not** prove quality equivalence. On the ten-dimensional convex
benchmark it saves about **5.7%**, but has lower HV and higher IGD+ than baseline,
with paired confidence intervals excluding zero and adjusted tests below 0.05.
Drop versus Resample shows no clear final-quality difference on either problem
in the 20-seed sample; the apparent three-seed advantage was not stable.

## Design and completeness

Both problems use baseline NSGA-II, ALSF-Resample and ALSF-Drop. Each method uses
20 paired seeds: **1-17, 42, 71, 97**. There are 60 runs per problem, **120 runs
total**, all newly executed for this protocol. All complete 200 generations with
population/offspring proposals 100. Physical tolerances, operators, thresholds,
floors and flag start 50 are unchanged. The objective functions are unchanged.
All modes use identical rounding/clipping, no objective cache, and consistent
actual model-call accounting. Unique archive coordinates are counted separately.

Total measured optimizer model calls: **2,276,245** versus a nominal 2,400,000.
Drop accounts for **123,755** genuinely unevaluated proposals; no duplicate
elimination, cache hits or diagnostic evaluations are attributed as savings.
These totals span different problems and are not a pooled scientific effect.
Every run has 200 process records and a valid 100-member parent population.
Missing runs are rejected by analysis; no seeds were excluded based on outcome.

The paired design uses the same initial seed for all methods. Their histories
are identical through generation 50, before the first exclusion affects batch 51.
Verification covers every paired seed and the actual evaluation counters.
**53 tests pass**, including the previous suites, independent neighborhood-count
oracles, the analytical 2D reference, matched smoke trajectories and complete
analysis/plotting checks.

## Computational optimization, not methodological retuning

A redundant all-archive neighbor-count query has been replaced with exact
incremental counts: every newly observed unordered neighbor pair increments both
endpoints, while self-counts start at one. Existing cache-reset rules are
preserved. Full-tree pair oracles test reordered/truncated data and duplicates.
The full 200-generation trajectories for solenoid seed 42 reproduce the older
baseline, Resample and Drop histories **bit for bit** (evaluations, HV and IGD+).
No extra objective evaluations or changed sensitivity/exclusion rules are used.

Pointwise evidence updates every active generation. DBSCAN is diagnostic and is
computed only at completion; sparse cell metadata is disabled. Removing repeated
diagnostic clustering does not change selection. Under this implementation the
old multi-minute filtered solenoid run takes about 6-7 seconds in the pilot.
This timing improvement is a software/cache optimization, not evidence of an
ALSF search advantage. The measured experiment uses two concurrent workers on a
two-core CPU quota and one BLAS thread per worker. Wall times are descriptive;
compare evaluation counts and quality as the primary scientific outcomes.

## Two-dimensional Gaussian benchmark

Final quality cells show **median [25th, 75th percentile]**, not best runs.
Higher HV and lower IGD+ are better; all normalizations come from the same
independent reference for this problem.

| Method | Mean actual model calls | Final HV | Final IGD+ |
|---|---:|---|---|
| NSGA-II | 20000.00 | 1.037822 [1.037684, 1.037894] | 0.002545 [0.002502, 0.002606] |
| ALSF-Resample | 20000.00 | 1.037904 [1.037807, 1.038019] | 0.002511 [0.002451, 0.002543] |
| ALSF-Drop | 14946.80 | 1.037977 [1.037692, 1.038041] | 0.002463 [0.002445, 0.002605] |

Drop saves an average **5053.20 calls (25.266%)**
per run, ranging from 4844 to 5313. Baseline and
Resample each execute 20,000 model calls. Calls are counted even for previously
visited designs; unique-coordinate archives do not inflate local evidence.

Paired Drop-minus-baseline effects (same seed):

| Outcome | Mean paired difference | Pointwise 95% paired-bootstrap CI | Holm-adjusted p |
|---|---:|---|---:|
| hypervolume | 9.90181e-05 | [-2.5799931e-05, 0.00022077708] | 1 |
| igd_plus | -4.3968486e-05 | [-0.00010014655, 1.3697483e-05] | 1 |
| actual_model_calls | -5053.2 | [-5105.0513, -5003.8] | 3.43323e-05 |

Target attainment, without imputing censored successes:

| Locked fraction of reference HV | NSGA-II | Resample | Drop |
|---|---:|---:|---:|
| 99.0% | 20/20 | 20/20 | 20/20 |
| 99.5% | 20/20 | 20/20 | 20/20 |
| 99.9% | 0/20 | 0/20 | 0/20 |

[Convergence PNG](analysis/test1_01_convergence_median_IQR.png) /
[PDF](analysis/test1_01_convergence_median_IQR.pdf).
[Final distributions PNG](analysis/test1_02_final_distributions.png) /
[PDF](analysis/test1_02_final_distributions.pdf).
[Target attainment PNG](analysis/test1_03_target_attainment.png) /
[PDF](analysis/test1_03_target_attainment.pdf).
[Evaluation-saving/quality tradeoff PNG](analysis/test1_04_savings_quality_tradeoff.png) /
[PDF](analysis/test1_04_savings_quality_tradeoff.pdf).
## Ten-dimensional convex solenoid

Final quality cells show **median [25th, 75th percentile]**, not best runs.
Higher HV and lower IGD+ are better; all normalizations come from the same
independent reference for this problem.

| Method | Mean actual model calls | Final HV | Final IGD+ |
|---|---:|---|---|
| NSGA-II | 20000.00 | 1.127371 [1.127085, 1.127453] | 0.002449 [0.002390, 0.002533] |
| ALSF-Resample | 20000.00 | 1.126490 [1.126243, 1.126738] | 0.002850 [0.002775, 0.002956] |
| ALSF-Drop | 18865.45 | 1.126617 [1.126216, 1.126856] | 0.002799 [0.002646, 0.002991] |

Drop saves an average **1134.55 calls (5.673%)**
per run, ranging from 937 to 1526. Baseline and
Resample each execute 20,000 model calls. Calls are counted even for previously
visited designs; unique-coordinate archives do not inflate local evidence.

Paired Drop-minus-baseline effects (same seed):

| Outcome | Mean paired difference | Pointwise 95% paired-bootstrap CI | Holm-adjusted p |
|---|---:|---|---:|
| hypervolume | -0.00075787033 | [-0.0009595306, -0.00055756012] | 4.95911e-05 |
| igd_plus | 0.00036302375 | [0.00027589738, 0.00045055809] | 3.43323e-05 |
| actual_model_calls | -1134.55 | [-1202.3, -1075.05] | 0.00106139 |

Target attainment, without imputing censored successes:

| Locked fraction of reference HV | NSGA-II | Resample | Drop |
|---|---:|---:|---:|
| 99.0% | 20/20 | 20/20 | 20/20 |
| 99.5% | 20/20 | 10/20 | 13/20 |
| 99.9% | 0/20 | 0/20 | 0/20 |

[Convergence PNG](analysis/solenoid_01_convergence_median_IQR.png) /
[PDF](analysis/solenoid_01_convergence_median_IQR.pdf).
[Final distributions PNG](analysis/solenoid_02_final_distributions.png) /
[PDF](analysis/solenoid_02_final_distributions.pdf).
[Target attainment PNG](analysis/solenoid_03_target_attainment.png) /
[PDF](analysis/solenoid_03_target_attainment.pdf).
[Evaluation-saving/quality tradeoff PNG](analysis/solenoid_04_savings_quality_tradeoff.png) /
[PDF](analysis/solenoid_04_savings_quality_tradeoff.pdf).
## Interpretation of the target-HV results

The primary target was locked at **99.5%** of the corresponding reference HV,
with **99% and 99.9%** predefined secondary targets. The thresholds were not
retuned after observing method outcomes.

On Test 1, every primary-target first crossing occurs between **1,700 and 4,600
model calls**, before filtering starts after 5,000 calls. Therefore the three
methods have identical first-attainment counts seed by seed. This target is
uninformative about filtering on Test 1. The fixed-generation 25% saving is real,
but is not proof of faster achievement of that target. A formally justified
quality-equivalence margin or a different prospectively chosen stopping target
would need a separate follow-up; nonsignificance is not equivalence.

On the solenoid, baseline reaches the primary target in **20/20**, Drop in
**13/20**, and Resample in **10/20** runs. Conditional mean/median attainment
counts among successful runs are exported, but must not be treated as unbiased
method averages when other runs are censored. All three methods reach 99%; none
reaches 99.9% in the tested configuration. Non-attainment is never assigned an
invented successful evaluation count. The results do not demonstrate a Drop
advantage in primary target-HV efficiency on the solenoid.

## Statistical and evaluation-budget details

Every comparison is paired by the same seed. Descriptive data include means,
sample standard deviations, medians and quartiles. Reported CIs use 10,000
paired bootstrap resamples of the **mean difference**, with fixed resampling seed.
They are pointwise 95% intervals, not simultaneous adjusted confidence regions.
Two-sided SciPy Wilcoxon tests use automatic tie/zero handling; Holm correction
covers all **18** final-HV, IGD+ and model-call comparisons across the two problems
and three method pairs. All-zero differences get p=1. These tests do not establish
universality beyond the two problems, parameters and seed sample.

Convergence curves show median and IQR for the **same 20-run cohort** at every
checkpoint. They use the most recent completed generation at or below a common
allowed evaluation budget, not interpolation of unobserved performance. The
balanced horizon is **14,687 calls for Test 1** and **18,474 for the solenoid**,
so no curve silently drops shorter completed runs. Common checkpoint tables
retain all 20 seeds per method; a nominal 15k checkpoint is omitted on Test 1
because some Drop runs finish earlier. The maximum unused allowance is **91
calls**. These are allowed-budget observations at generation boundaries, not
separately run synchronized intra-generation stopping experiments. Tiny quality
differences at that granularity require caution. Final-generation plots separately
show every method's actual terminal budget and all 20 individual outcomes.

## References and independent sensitivity

The solenoid uses the unchanged validated convex reference. For Test 1 an
independent analytical objective-space Pareto envelope is derived by fixing
q=x1²+x2² and minimizing the convex expression in s=x1+x2. Its 2,001 actual
feasible representative designs are checked against independent scalar
minimization. One symmetric decision-space branch suffices for the objective
front. A doubled-resolution check gives maximum normalized polyline discrepancy
about 1.05e-4 and changes final IGD+ by at most 1.82e-6. Paired HV uses the same
reference-only endpoint normalization, so method comparisons do not depend on
choosing per-method scales. The experiment's targets remain locked to the
original reference; the finer diagnostic does not rewrite them.

Each final front receives an independent audit of up to 25 objective-ranked
members. In Test 1 this is a **21x21 finite perturbation grid**: violations found
are genuine counterexamples, while their absence is not a robustness guarantee.
The average violating audit fraction is 82.6% for baseline, 79.6% for Resample,
and 79.0% for Drop. These are descriptive selected-subset grid results, not
exact robustness rates over whole populations. The solenoid uses exact convex
minima/exhaustive vertices and separable extrema with the same 0.001 floors;
**all 1,500 selected solenoid designs across its 60 runs are sensitive** at the
10% criterion. No binary robustness improvement is demonstrated on that subset.
Diagnostic model evaluations are separate from optimization budgets and never
change archive flags or exclusion decisions.

## Artifacts and reproduction

- [Protocol and fixed seed list](protocol.json), [environment](environment.json),
  [verification checks](verification.json), [reference resolution](reference_resolution.json).
- [All 120 run summaries](analysis/all_runs.csv), [descriptive statistics](analysis/descriptive.csv),
  [paired CIs/tests](analysis/paired_effects.csv), [all target outcomes](analysis/target_hv_attainment.csv),
  [conditional target differences](analysis/conditional_target_differences.csv),
  [balanced evaluation checkpoints](analysis/evaluation_checkpoints.csv).
- [Test 1 reference](reference/test1.csv), [solenoid reference](reference/solenoid.csv).
- `runs/<benchmark>/<method>/seed<seed>/` retains each settings file, summary,
  generation history, process/evidence log, final Pareto set and sensitivity audit.
- Full unique-coordinate archives are retained locally in
  `work/twenty_seed_study/runs/`; they are omitted from Git to avoid hundreds of
  megabytes of duplicate experiment data. All plotted results and statistical
  tables can be reproduced from the committed files without these raw archives.
- [Workflow, mathematical reference derivation and runnable commands](../../README_REPEATED_STUDY.md).

No previous experiment outputs were overwritten. The 120-run root is new, and
provisional analyses were preserved separately. The optimization objectives,
physical assumptions, thresholds and operators were not tuned for favorable
results.
