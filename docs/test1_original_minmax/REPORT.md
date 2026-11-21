# Restored Test 1: minimize f1, maximize f2

## Scientific result

ALSF-Drop saved **7.31%** of actual model calls on average in this new, corrected 60-run experiment. Quality changes and uncertainty are reported below; savings alone do not establish efficiency or equivalence. These results replace no historical experiment. The earlier 25.3% saving belongs exclusively to the both-minimization study.

| mode | actual_model_calls | hypervolume | igd_plus | rejected_fraction | sensitive_audit_fraction |
| --- | --- | --- | --- | --- | --- |
| baseline | 20000 | 0.813886 | 0.00362022 | 0 | 0.248 |
| resample | 20000 | 0.813733 | 0.0037044 | 0.0794375 | 0.254 |
| drop | 18538.5 | 0.813744 | 0.00369548 | 0.073075 | 0.256 |

The 99% HV target was reached by every method after 900–1,900 evaluations (mean 1,290), before filtering starts; paired target costs are identical. Neither higher target was reached by any method. Therefore these targets provide **no evidence of faster attainment at comparable quality**. Final paired HV/IGD+ confidence intervals contain zero, which does not establish equivalence.

Across all 60 final fronts, **none contains a point within radius 0.18 of the Gaussian center**. Drop rejected **zero** candidates there; Resample rejected **three screening checks**. All 29,230 Drop discards therefore occurred outside the predefined peak neighborhood. Thus the observed savings cannot be attributed to successful filtering of the sensitive peak. The peak was sparsely explored, and most archived nearby designs remained NEI. This central scientific hypothesis is not demonstrated by this experiment.

The Gaussian has a genuine but extremely narrow efficient segment. Its center is **dominated**, not Pareto optimal. Global HV/IGD+ can obscure omission of this segment. Peak-specific evidence below is essential to interpret the headline savings.

## Exact problem and provenance

Domain [-5,10]^2. Minimize f1=x1²+x2²; maximize f2=(x1−5)²+(x2−5)²+100 exp(−((x1−2.5)²+(x2−2.5)²)/0.01). At (2.5,2.5), (f1,f2)=(12.5,112.5). The physical model and Gaussian are unchanged. The explicit adapter `benchmark_test1_minmax.py` supplies g=(f1,−f2) to pymoo. Archives, fronts and physical plots retain positive f2. Dominance, HV and IGD+ use g. Sensitivity is calculated from raw physical values; tests verify sign invariance of relative differences, cached evidence and flags.

Source: [JCAM manuscript](https://github.com/tamasorosz/Archive-Based-Local-Sensitivity-Filtering-for-Robust-Design-Optimisation-of-Electric-Machines/blob/09d3c15ffeaa36acd4696b4d4f747f2d6d39e04e/elsarticle-template-num.tex), Results and Discussion → Analytical bi-objective optimization problem: “The first objective is subject to minimization, whereas the second is subject to maximization.” No manuscript file was modified. The historical `benchmark_test1.py` documentation and min-min helpers are retained for old experiment reproducibility; the new entry point is explicit.

## Independent reference and accuracy

Put q=f1 and s=x1+x2. At fixed q, f2=q−10s+50+100 exp(−(q−5s+12.5)/0.01), which is convex in s. For 0≤q≤50, both endpoints s=±√(2q) are feasible, attained at the two diagonals. Thus their maximum is the full feasible upper envelope, not an assumption about NSGA-II output. For q>50, some coordinate must exceed 5; the quadratic background is ≤125 and the Gaussian ≤100 exp(−1250). These points are dominated by (−5,−5), with (50,200). Mixed-direction nondominance filtering is applied after constructing the envelope, including its locally decreasing portion.

The efficient positive-diagonal segment starts at q=12.479999998932 and ends at q=12.497499843740, where f2=112.501250132828. The negative branch catches this maximum at q=12.500416713258. The endpoint slightly preceding the Gaussian center dominates that center.

The reference uses 10,001 global q values plus 4,001 adaptively located peak values and exact transition points; 13437 nondominated rows remain. Doubling both resolutions gives 26870 rows and changes reference HV by 2.51e-05 (absolute). There are 532 independent SLSQP starts over 37 q levels, including peak refinement, and 65,537 polar angles per level. Maximum feasible SLSQP envelope excess is 3.25e-10; maximum gap of the best feasible multistart value is 2.84e-14; maximum polar gap is 2.84e-14. Two optimizer starts report failure; they remain in `reference/multistart.csv` with residuals and feasibility, rather than being silently discarded. The convex-envelope proof, not multistart success, establishes global coverage. These numerical checks validate the implementation separately.

Normalization is fixed from the corrected reference: ideal g=(0,−200), range=(50,150), HV reference=(1.1,1.1). Global IGD+ uses 501 uniformly normalized-arclength reference points, avoiding accidental weighting by peak refinement density. Linear interpolation bridges the negligible nondominated gap; reference resolution error is documented above. `protocol.json` locks HV targets at 99%, 99.5%, 99.9% of independent reference HV **before** these 60 optimization runs. No old absolute targets or old normalization are reused.

## Protocol and counting

20 paired seeds: 1–17,42,71,97. All three methods use population=100, offspring=100, 200 generations (including initialization), SBX probability .9/eta 5, PM probability 1/eta 5, tolerances (.1,.1), sensitivity thresholds (.1,.1), floors (.001,.001), scaled Euclidean neighborhood and exclusion radius 1, minimum support 10, rounding 8 decimals, no objective cache. Evidence is updated after generation 50; filtering affects the next batch. Resample retains max 1,000 replacement attempts. Drop calls mating once then discards forbidden proposals without replacement; the existing policy and algorithm files are unchanged. Two workers with single-threaded BLAS were used. Wall times include observation and indicators and are descriptive, not expensive-solver timings.

Each actual objective request counts, including repeated designs; the archive stores unique rounded designs. Each Drop run satisfies accepted=evaluator calls=actual calls and dropped+actual calls=20,000. Resample and baseline each evaluate 20,000 designs. Resample screening counts include replacement trials and mating duplicate handling; they are not model calls or unique rejected offspring. `all_runs.csv`, `runs/*/seed*/summary.json`, `process.csv` and `history.csv` contain every run, final labels, rejection counts/fractions and per-generation evidence. For Resample, `rejected_fraction` is rejected initial repair inputs divided by all initial repair inputs (including normal mating duplicate retries); total rejected replacement checks and their denominator are exported separately. These counts are not directly interchangeable with Drop's single post-mating rejection fraction. Drop's rejection fraction includes the 100 unfiltered initialization points in its denominator. `resample_forbidden_checks` counts rejected original repair inputs; replacement trial totals are separate in `process.csv`.

## Paired statistics and comparable quality

`descriptive.csv` provides mean, median, sample standard deviation, quartiles and IQR. `paired_effects.csv` provides B−A effects and 95% paired percentile bootstrap mean CIs (10,000 resamples), plus two-sided Wilcoxon tests, Holm-corrected across all nine primary comparisons (3 metrics × 3 pairs). A nonsignificant difference does not establish equivalence; no equivalence margin was specified.

| metric | method_a | method_b | mean_difference | ci95_low | ci95_high | p_holm |
| --- | --- | --- | --- | --- | --- | --- |
| hypervolume | baseline | resample | -0.000152453 | -0.00035812 | 5.62277e-05 | 1 |
| hypervolume | baseline | drop | -0.00014148 | -0.000319817 | 3.77822e-05 | 1 |
| hypervolume | resample | drop | 1.09731e-05 | -0.000196645 | 0.000215138 | 1 |
| igd_plus | baseline | resample | 8.41798e-05 | -2.15004e-05 | 0.000189249 | 1 |
| igd_plus | baseline | drop | 7.52572e-05 | -1.48048e-05 | 0.000164812 | 1 |
| igd_plus | resample | drop | -8.92262e-06 | -0.000115703 | 0.000100391 | 1 |
| actual_model_calls | baseline | resample | 0 | 0 | 0 | 1 |
| actual_model_calls | baseline | drop | -1461.5 | -1486.65 | -1435.45 | 1.71661e-05 |
| actual_model_calls | resample | drop | -1461.5 | -1486.65 | -1435.45 | 1.71661e-05 |

Target attainment counts (out of 20):

| mode | 0.99 | 0.995 | 0.999 |
| --- | --- | --- | --- |
| baseline | 20 | 0 | 0 |
| resample | 20 | 0 | 0 |
| drop | 20 | 0 | 0 |

At the 99% target, paired evaluation differences B−A (conditional only when both attain; confidence intervals descriptive):

| target_fraction | method_a | method_b | n_both | n_total | mean_difference | ci95_low | ci95_high |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0.99 | baseline | resample | 20 | 20 | 0 | 0 | 0 |
| 0.99 | baseline | drop | 20 | 20 | 0 | 0 | 0 |
| 0.99 | resample | drop | 20 | 20 | 0 | 0 | 0 |

Unreached targets remain censored at the actual final evaluation budget. No failure is assigned a fictitious attainment time. Conditional paired target differences are descriptive only; restricting to joint successes can bias comparison. `evaluation_checkpoints.csv` and the convergence figure compare all 20 seeds at identical allowed evaluation budgets up to the minimum final budget across every run. Values are held at the last completed generation; unused budget is recorded. This supports assessment at comparable quality without changing the cohort or extrapolating beyond available data.

## Peak sensitivity, evidence and actual rejection

Near-peak means Euclidean distance≤0.18 from (2.5,2.5), fixed before the experiment. The independent 21×21 perturbation grid over the prescribed ±0.1 box finds sensitive witnesses for **100%** of sampled reference peak-efficient points (`reference/peak_independent_audit.csv`). This is a finite witness test; passing it would not certify stability. All actually archived near-peak points are independently audited in `peak_archive_independent_audit.csv` with their final observed archive flag. Nominal efficiency, archive evidence, independent sensitivity and actual rejection are separate quantities.

Aggregated screening evidence across 20 seeds (checks can repeat candidates):

| mode | checks | rejected | allowed_sparse | evaluated |
| --- | --- | --- | --- | --- |
| baseline | 13 | 0 | 13 | 13 |
| resample | 26 | 3 | 23 | 23 |
| drop | 13 | 0 | 13 | 14 |

For Drop, this table counts only `drop_proposal` events, after mating. Its empty rounding repair is excluded. For Resample it counts `repair_check` events, including replacement attempts. Baseline checks use an empty policy and are unfiltered, so its sparse-support counts have no scientific classification meaning. `allowed_sparse` means fewer than10 existing archive neighbors and no exclusion evidence, **not** an assigned flag on an unevaluated candidate. A single verified violation can exclude a point even with sparse support. Allowed candidates are never called robust solely because they were allowed. Actual archived NEI, stable-observed and verified-violation counts are separate in `peak_archive_labels.csv`.

The near-peak independent audit contains 50 unique-within-run archived designs; 50 have independent sensitive witnesses. Every audited near-peak design has a sensitive witness, including those whose archive flag remains NEI; no nearby archived design is labeled locally stable. These are posthoc audits, not evidence supplied to the filtering policy. Empty/no-rejection observations do not establish successful peak exclusion. Archive flags use observed pairs within scaled Euclidean tolerance balls; the independent audit uses a grid spanning a coordinate box, a different and larger perturbation domain.

Peak-restricted IGD+ (same global normalization, unweighted dense efficient segment) and mean number of final points within radius.18:

| mode | peak_igd_plus | final_points_near_peak |
| --- | --- | --- |
| baseline | 0.00371657 | 0 |
| resample | 0.00344524 | 0 |
| drop | 0.00365214 | 0 |

A point within radius.18 need not approximate the extremely thin nominally efficient segment. Peak IGD+ is an objective-space metric and cannot certify capture of the positive decision branch when the negative branch has nearly identical objectives. The supplied fronts and peak IGD+ are required to assess its approximation; global HV alone cannot justify preservation of the narrow feature. No global metric result establishes robustness or expensive-model speedup.

## Figures

All figures have PDF and PNG versions. Physical f2 is always maximized.

1. [Objective surfaces](01_objective_surfaces.png), including a peak zoom.
2. [Independent reference](02_independent_reference.png), including the efficient segment and dominated center.
3. [All20 final fronts per method](03_all_final_fronts.png).
4. [Decision archives and final flags, seed42](04_archive_labels_seed42.png).
5. [Peak zoom and actual rejected points, seed42](05_peak_zoom_seed42.png). A prespecified illustration, not a best run.
6. [Savings versus final quality](test1_04_savings_quality_tradeoff.png).
7. [Per-seed HV/IGD+/calls](test1_02_final_distributions.png).
8. [Accepted and dropped offspring](08_offspring_evolution.png), median/IQR across 20 seeds. Resample rejects then replaces; its accepted count remains100.

Also: [3D objective surfaces](01b_objective_surfaces_3d.png), [quality vs actual evaluation budget](test1_01_convergence_median_IQR.png), [censored target attainment](test1_03_target_attainment.png).

## Limitations and interpretation

These are 20 finite stochastic runs on one analytical problem. The Gaussian-efficient segment is exceptionally small, potentially missed by all methods. Observed stability is empirical and does not certify a continuous tolerance box. Global IGD+ and HV have limited resolution for this feature. Bootstrap intervals are finite-sample uncertainty, not an equivalence test. Evaluation savings trade off against any front degradation; assess paired quality, common-budget curves, target attainment and local peak evidence jointly. This experiment does not establish transfer to expensive electromagnetic models. The solenoid was not rerun and its code/results and all previous studies remain unchanged. Existing files are verified byte-for-byte against the parent commit; all restoration changes are new files.

## Reproduce

From the repository root, using the existing Python requirements/environment:

```bash
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
python prepare_test1_minmax.py --root work/test1_original_minmax
python run_test1_original_minmax.py --root work/test1_original_minmax --workers 2
python analyze_test1_original_minmax.py --root work/test1_original_minmax --outdir docs/test1_original_minmax
python -m pytest -q
python verify_test1_original_minmax.py
```

Use a new root to lock a fresh protocol and an absent output directory for analysis. Completed optimization runs resume only when settings match. Raw archives are gzip-compressed under `runs/`; all other per-run CSV/JSON files are retained. The new unit and pymoo integration tests cover physical exports, mixed dominance, sign invariance, full-domain envelope, center dominance, call accounting and observational instrumentation preserving decisions and random draws. No JCAM manuscript edits are included. See `suggested_subsection.tex` for a separate insertion proposal.
