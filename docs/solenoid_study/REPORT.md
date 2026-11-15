# Solenoid validation: measured results

**On this defined smooth convex benchmark, the measured three-seed comparison
shows no ALSF advantage in nominal convergence or binary 10% box robustness.**
ALSF is active and changes the search, but its final HV is lower and IGD+ higher
for every paired seed. Continuous sensitivity levels show mixed changes, and
archive processing is expensive relative to this inexpensive analytical model.
The first two-variable benchmark was not modified.

## Completed phases

1. Validated the existing convex bounded least-squares approach and adaptively
   refined its reference front: 389 coarse and 776 fine nondominated points.
2. Validated exact relative box sensitivity using extrema, independent grids,
   independent vertex enumeration and a separate minimum solver. Used a small
   constructed archive to verify classification semantics.
3. Completed a matched 192-evaluation baseline/ALSF smoke pair. The filter was
   inactive and the evaluated trajectories were identical.
4. Passed numerical gates, matched-budget checks and implementation tests.
5. Completed baseline and ALSF for seeds **13, 42, 97**, each with **20,000 actual
   optimizer model evaluations**: 120,000 full-run evaluations in total. No full
   experiment began before the earlier phases passed. Old experiment outputs
   were preserved; all study directories were new.

## Reference accuracy and numerical checks

All weighted solves converged without bound violations. Maximum projected KKT
residual: 7.09e-13. Maximum analytical-gradient finite-difference discrepancy:
2.01e-10. Hessian minimum eigenvalues are about -1.90e-17 (numerical roundoff)
and 0.2. The field matrix condition number is about 2.9e13; very small residuals
are interpreted only to solver precision.

The maximum fine-to-coarse normalized polyline distance is 4.27e-5. Coarse and
fine discrete reference HV values are 1.131721 and 1.132238. Recomputing all final
run indicators with the coarse reference changes their HV by zero and IGD+ by at
most 2.75e-6, much smaller than the paired method differences observed here.
Reference normalization comes only from the independent front. The exact
zero-current minimum-loss endpoint is retained. Every solver diagnostic is saved.

**40 tests pass**, including all previous tests and the complete Figures A-E
workflow. See [numerical checks](numerical_checks.json), [environment](environment.json),
and [method and reproduction commands](../SOLENOID_VALIDATION.md).

## Nominal performance

Higher HV is better; lower IGD+ and distance are better. Distances and indicators
use fixed reference-based objective normalization. Runtime includes the optimizer,
archive processing and indicators, with separate timing columns in each summary.

| Seed | Method | HV | IGD+ | Mean distance to reference | Run seconds | Regenerated |
|---|---|---:|---:|---:|---:|---:|
| 13 | baseline | 1.127048 | 0.002521 | 0.002345 | 1.79 | 0 |
| 13 | alsf | 1.126237 | 0.002870 | 0.002701 | 194.32 | 1276 |
| 42 | baseline | 1.127097 | 0.002566 | 0.002640 | 1.81 | 0 |
| 42 | alsf | 1.126245 | 0.002958 | 0.003062 | 178.72 | 1433 |
| 97 | baseline | 1.127440 | 0.002398 | 0.002312 | 1.90 | 0 |
| 97 | alsf | 1.126145 | 0.003003 | 0.003265 | 183.23 | 1126 |

Across these three seeds, mean HV is 1.127195 for baseline and
1.126209 for ALSF (-0.087% change).
Mean IGD+ is 0.002495 and 0.002944, respectively
(18.0% higher for ALSF). Mean distance to the
reference is 0.002432 and 0.003009.
The paired seed sample is small; these are descriptive results for this protocol,
not a general statistical proof about every problem or parameter choice.

Both modes use identical population/offspring counts, SBX/PM operators, bounds,
thresholds, floors and seeds. Their indicator histories match exactly through
generation 50 before the first active repair. All optimizer requests execute the
same analytical model once. Baseline archives contain 20,000 unique coordinates;
ALSF archives contain 19,900 each. The unchanged seeded uniform repair stream
revisits 100 previously evaluated coordinates during early resampling. These
repeat evaluations count toward both evaluator/model budgets but do not inflate
archive neighbor counts. This is a limitation of the existing seeded repair
implementation and is reported rather than silently changing its random stream
or granting ALSF an extra budget.

## Actual robustness and empirical-label accuracy

The 61 sampled convex-reference designs are all sensitive under the defined
10% relative criterion and physical +/-0.1 MA/m² box. The independent reference
curve is a nominal optimum, not a robustness optimum.

Across all six final fronts, 25 objective-ranked Pareto designs per run were
independently audited: **all 150 selected designs are actually sensitive**.
Thus no binary 10% robustness improvement was demonstrated in the selected
subsets. This is not an exhaustive statement about every feasible design or
every unsampled front member. Median continuous R1/R2 values vary by seed and
method; changes do not consistently improve both objectives. Mean per-run
median R1 is 0.4553 baseline versus 0.4575 ALSF;
R2 is 0.3937 versus 0.3878.

For archive-label audits, select up to 25 objective-ranked members per empirical
class per run. **All 55 empirically stable points in the three ALSF archives are
included and all 55 fail the exact box criterion**: false-stable rate 100% for
these actual stable-labeled archives. Baseline has 87 stable-labeled members;
72 were selected and all 72 are actually sensitive. The sampled baseline rate
is 100%, without a claim about its remaining 15 stable-labeled members.
All 75 selected verified-sensitive points per method are confirmed sensitive
(precision 100% on that subset). NEI is an abstention, not a stable prediction;
all 75 selected NEI points per method are also sensitive. Because the audit
contains no actually stable points, specificity cannot be estimated. Recall
and coverage values in the JSON apply to the selected audit and must not be
extrapolated as population rates from stratified sampling.

The constructed 66-point validation archive separately gives 12 false-stable,
14 verified-sensitive and 40 NEI points, all actually sensitive. It is a semantic
counterexample/control dataset and is not pooled with optimization performance.

Exact diagnostics never add points to the optimization archive, change flags,
or alter exclusion. There are no Monte Carlo approximations in the reported
box extrema; least-squares minima remain subject to numerical tolerance.

## Ten-dimensional coverage and cost

The final ALSF archives aggregate 56,220 NEI,
3,425 verified-sensitive and 55
empirically stable observations: NEI fraction
94.17%. Baseline posthoc classification has
52,003 NEI, 7,910
verified-sensitive and 87 stable observations:
NEI fraction 86.67%.

Local median observation count is one, including self, in all runs. The final
95th percentile is three for ALSF and 9-10 for baseline. ALSF DBSCAN finds only
4-10 density-connected components with nearly all samples marked noise; baseline
has 15-16 components. These are connectivity statistics, not uniform stability
regions. No high-dimensional grid cells are created or interpreted.

ALSF regenerates 1,126-1,433 candidates per run, about 5.49-6.99% of candidates
checked. It therefore is active, despite high NEI coverage. Mean run time is
185.42 seconds versus 1.83 seconds for baseline.
ALSF archive processing alone averages 183.12 seconds.
This overhead dominates a cheap analytical model; it should not be directly
transferred to expensive FEM objectives. Timing is descriptive on a shared
cloud CPU, with single-threaded BLAS, rather than controlled hardware benchmarking.
Posthoc exact audits and baseline posthoc classification are timed separately.

## Scientific implications and limits

ALSF's verified pair detection and exclusion operate correctly, but activity is
not evidence of optimization improvement. On this protocol, nominal convergence
is slightly worse, the audited binary robustness is unimproved, and computation
cost is much higher. The benign local archive samples do not adequately cover
independent perturbation directions, even when a count threshold is met. Smooth
convexity does not make archive-supported stability a worst-case certificate.
Relative sensitivity near small objective values, fixed denominator floors and
the physical tolerance are consequential parts of this conclusion; none were
retuned to favor ALSF. More seeds, other tolerances or expensive physical models
would be separate experiments. No surrogate was introduced.

## Artifacts

- [Fine reference CSV](reference/reference.csv), [coarse reference](reference/reference_coarse.csv),
  [fine solver diagnostics](reference/solver_fine.csv), [reference checks](reference/validation.json).
- [Exact reference sensitivity](sensitivity/reference_sensitivity.csv),
  [constructed archive audit](sensitivity/empirical_vs_exact.csv),
  [per-run stratified audits](archive_audit/summary.json).
- [All run summaries](plots/run_summary.csv), [paired differences](plots/paired_differences.csv),
  [reference-resolution indicator checks](plots/indicator_reference_resolution.csv).
- Individual settings, histories, evidence/repair logs, final fronts, exact Pareto
  audits and neighbor distributions are under `full/<method>_seed<seed>/`.
- Figure A: [PNG](plots/A_reference_and_final_fronts.png) / [PDF](plots/A_reference_and_final_fronts.pdf).
- Figure B: [PNG](plots/B_convergence_vs_evaluations.png) / [PDF](plots/B_convergence_vs_evaluations.pdf).
- Figure C: [PNG](plots/C_exact_reference_sensitivity.png) / [PDF](plots/C_exact_reference_sensitivity.pdf).
- Figure D: [PNG](plots/D_empirical_vs_exact.png) / [PDF](plots/D_empirical_vs_exact.pdf).
- Figure E: [PNG](plots/E_repair_and_evidence.png) / [PDF](plots/E_repair_and_evidence.pdf).

Complete unique-coordinate optimization archives are retained locally under
`work/solenoid_study/full/`; they are omitted from the committed report to avoid
large duplicate datasets. All measured plots can be regenerated from committed
artifacts with `python plot_solenoid_validation.py --study docs/solenoid_study
--outdir figures/solenoid_replot` (one command line). To regenerate full archives
or choose another exact-audit subset, rerun the documented seeded experiments
into new output directories.
