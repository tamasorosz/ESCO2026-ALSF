# Independent convex-solenoid validation

This study concerns only the defined ten-variable, Di Barba-inspired analytical
model. It is not an exact reproduction of the published solenoid problem. The
first two-variable Gaussian benchmark and its plotting utilities are unchanged.

## Reproducible phases and preserved methodology

Use a fresh study root; every phase/run refuses an existing output directory.
Install `requirements.txt` and `pytest` in an activated virtual environment.

```
python -m pytest -q
python solenoid_validation.py reference --outdir results/solenoid_study/reference
python solenoid_validation.py sensitivity --reference results/solenoid_study/reference/reference.csv --outdir results/solenoid_study/sensitivity
```

Inspect both `validation.json` files: `validated` must be true before optimization.
Then run a matched smoke pair:

```
python solenoid_validation.py run --mode baseline --seed 42 --generations 8 --pop-size 24 --offsprings 24 --flag-start 3 --reference results/solenoid_study/reference/reference.csv --outdir results/solenoid_study/smoke/baseline
python solenoid_validation.py run --mode alsf --seed 42 --generations 8 --pop-size 24 --offsprings 24 --flag-start 3 --reference results/solenoid_study/reference/reference.csv --outdir results/solenoid_study/smoke/alsf
```

After checking budgets, counters, sensitivity and runtime, run each full pair
with the same seed, defaults (100 population, 100 offspring, 200 generations,
flag start 50), and separate output directories. Example:

```
python solenoid_validation.py run --mode baseline --seed 42 --reference results/solenoid_study/reference/reference.csv --outdir results/solenoid_study/full/baseline_seed42
python solenoid_validation.py run --mode alsf --seed 42 --reference results/solenoid_study/reference/reference.csv --outdir results/solenoid_study/full/alsf_seed42
python solenoid_validation.py audit --study results/solenoid_study --outdir results/solenoid_study/archive_audit
python plot_solenoid_validation.py --study results/solenoid_study --outdir figures/solenoid_study
```

Repeat matched full pairs for independent seeds. Never pool unpaired seeds. The
plotter uses measured results and validates paired seeds and evaluation axes.
The old fixed-path runners remain available for compatibility; this new harness
is the supported route for phased multi-seed validation and non-overwriting output.

SBX probability 0.9, eta 5; PM probability 1, eta 5; identical random sampling,
rounding/clipping, duplicate elimination and seeds in both modes. Both use
`copy_algorithm=False`. The baseline repair policy is permanently empty; ALSF
uses the existing `OffspringRepair`, `LocalVariationState`, and pointwise
classifier. The tolerance is 0.1 in each normalized current coordinate
(0.1 MA/m² physically), local and exclusion radii are 1, minimum observations
10 including self, objective thresholds 0.1, and denominator floors 0.001.
DBSCAN radius/density remain separate and fixed at 1/10.

Optimization evaluations execute the analytical model once per pymoo evaluator
request, with a checked budget `pop_size+(generations-1)*offsprings`. The archive
stores unique rounded coordinates, preserving original ALSF evidence semantics;
repeated optimizer evaluations cannot inflate local observation counts. Model
calls, evaluator requests, and unique coordinates are reported separately. Model
calls used in exact posthoc diagnostics are separate from the optimization budget.
The optimizer's objective cache is not mixed with pymoo's internal method cache.
Objective data are buffered in memory and written at completion, removing per-row
CSV overhead from the comparison. Completed run artifacts include settings,
full labeled archive, final front, generation indicators, evidence/repair counters,
neighbor distributions and selected exact sensitivity audits.

## Convex reference

The existing bounded least-squares scalarization is retained:
`||Kx-1||²/m + lambda ||x||²/10`, with bounds [-2,2]. Start with lambda=0,
241 log-spaced weights from 1e-8 through 1e7, and the exact zero-current
minimum-loss endpoint. Refine scalar weights geometrically where adjacent
solutions have large normalized objective-space gaps. This prioritizes coverage
of the curve over uniform weight increments. A second construction starts with
481 weights and a smaller gap target. Diagnostics retain **every solve**, even
when its point is removed from the plotted front as numerically duplicated or
dominated. No surrogate is used.

Analytical Hessians are `2 K.T K / m` and `2 I / 10`. Small negative computed
eigenvalues at machine precision are roundoff, not physical nonconvexity. The
field matrix is severely ill-conditioned (about 2.9e13); tiny field-error
differences near the minimum are interpreted only at solver precision, never
as evidence of a scientifically distinct optimum.
Projected-gradient KKT residuals, bound violations, solver status, analytical
gradient finite differences, nondominance and cross-resolution polyline distances
are checked. Successful scalarized solves provide global solutions to numerical
solver tolerance; finite front discretization remains an approximation of the
continuous curve. Every plotted solved point is an actual model design.

## Exact independent sensitivity

The existing `exact_box_sensitivity` name still returns **absolute** deviations
for compatibility. `box_objective_extrema` now exposes the extrema;
`exact_relative_box_sensitivity` uses precisely ALSF's symmetric denominator:

`r_j = |f_j(u)-f_j(x)| / max(|f_j(u)|,|f_j(x)|,0.001)`.

For nonnegative objective values and fixed center value, the relative difference
is monotone away from the center on either side. Its maximum therefore occurs
at the minimum or maximum achievable objective value. For f1 the convex maximum
is attained at one of the 1024 box vertices, and the minimum is a bounded least-
squares problem. For f2 the closest/farthest absolute coordinate values give the
extrema analytically. The perturbation box is intersected with the original
bounds. An infeasible center is rejected; zero tolerance gives exactly zero
variation. No artificial padding outside the box is allowed. “Exact” refers to
this exhaustive/convex construction, with numerical least-squares tolerance.

Tests independently compare extrema with exhaustive low-dimensional grids,
independently enumerated ten-dimensional vertices, a separate L-BFGS-B minimum,
and random feasible perturbations. Posthoc exact calculations never enter the
ALSF archive or exclusion policy. Each final front is sorted by f1/f2, then at
most 25 evenly spaced row ranks including endpoints are audited; the same rule
applies to both modes. The reference sensitivity plot uses 61 deterministic
reference ranks. A separate stratified final-archive audit selects up to 25
objective-ranked observations per empirical class and run; its rates are
class-conditional descriptions of that subset, not population recall estimates. These are descriptive subsets, not exhaustive population claims.

A separate constructed 66-point archive tests the interpretation of stable,
verified sensitive and NEI labels. It deliberately includes tight benign local
samples and observed violations, and is clearly separated from optimization
results. Its class proportions must not be presented as population estimates.
False-stable rate is sensitive ground-truth points divided by empirically stable
points in the audit. Precision applies to predicted-sensitive points; recall
conditional on classified points excludes NEI abstentions. Detection over all
actually sensitive audited points and classification coverage are also reported.
Undefined ratios are JSON null, never fabricated zero accuracy.

## Indicators, timing and dimensionality

All runs share an ideal point, objective ranges and HV reference point determined
only from the independent fine reference. Normalized HV uses the fixed point
(1.1,1.1); points outside it do not contribute. IGD+ uses 501 points uniformly
interpolated in normalized arclength along that same fine polyline, avoiding
unequal log-weight density. These interpolated indicator points approximate the
continuous objective curve; they are not claimed to be newly solved designs.
Final distance is Euclidean distance to the normalized reference polyline.
Reference resolutions and their HV discrepancy are reported for context.

Convergence plots use actual objective evaluations. Curves show per-generation
median and interquartile range across independent seeds, without invented warmup
classifications. Runtime includes optimizer, archive processing and indicators;
objective, archive, indicator, posthoc-label and exact-audit times are also logged
separately. Resampling fractions divide regenerated candidates by candidates
actually checked, which includes initialization and duplicate-elimination retries;
it can differ from a fraction of the objective-evaluation budget.

The high-dimensional classifier disables **only** spatial-cell reporting via
`partition_width=None`. It creates no sparse cell identities or representative
objects. This opt-out is tested to leave pointwise flags, local counts, observed
variations and DBSCAN labels unchanged; two-dimensional defaults are preserved.
DBSCAN components, local-count distributions, NEI fractions, verified endpoints
and repair activity are recorded without targeting a cluster count.

Figures A-E export PNG and PDF. Full trade-off panels retain endpoints; the
explicit positive-error logarithmic detail omits zero and restricts its x-window.
Results should distinguish nominal convergence, continuous sensitivity levels,
binary 10% robustness, and total computation cost. A small seed sample supports
an exploratory paired comparison, not a general statistical superiority claim.
