# Tolerance-aware Box-IGD: existing 20-seed solenoid results

This is a supplementary **post-processing diagnostic**, not IGD+ and not a new
optimization experiment. The existing 776 reference designs
and 60 final fronts (20 paired seeds per method) were used without modification.
Every reference row receives equal weight. Existing HV, IGD+ and actual model-call
values are retained and independently checked against the stored common reference.

## Definitions and units

For each reference design r, the decision-space half-width is **h_i=0.1** in all
ten coordinates x_i=J_i/(1 MA/m²). This corresponds to independent **±0.1 MA/m²**
physical current-density tolerance; the full box side is 0.2, not 0.1. The distance
is `max_i max(0,abs(p_i-r_i)-h_i)`, minimized over obtained designs and averaged
over reference rows. Coverage counts exactly zero distances, including boundaries.
Values approach symmetric decision-space L-infinity IGD at zero half-width, not
Euclidean IGD+. The diagnostic is not automatically Pareto-compliant.

Objective-space rectangles use the existing `box_objective_extrema` routine on
[r-0.1,r+0.1] intersected with [-2,2]^10. f1 minima are bounded least squares,
maxima exhaust 1,024 vertices; f2 extrema are separable. If the existing TRF
minimum solver does not converge, the analysis uses bounded BVLS with a checked
projected-gradient KKT residual, on the identical least-squares problem. This
fallback is logged per reference design. The nominal feasible design is included
as a minimum/maximum candidate, with no tolerance padding or hidden coverage epsilon. The existing routines
were previously validated and are regression-tested here. The lower/upper bounds
are normalized by **the reference objective ranges only**, identically for every
method: scales=[0.9999999999999695, 1.1549624507083016], ideal=[3.05322158243951e-14, 0.0].
Distance is `max_j max(lower_j-p_j,0,p_j-upper_j)` in these normalized units.
The rectangle is a **coordinatewise bounding approximation** to the joint
attainable objectives. Different coordinates' extrema may require different
perturbations; an interior vector need not be jointly attainable.

Across all reference points, BVLS fallbacks: 2; nominal feasible-candidate inclusions: 1. All fallback KKT checks passed. See per-reference solver diagnostics.

## Results

Each cell is **mean / median across 20 seeds**. Decision distances are numerically
in normalized-current units (MA/m²); objective distances are reference-normalized.

| Method | Decision Box-IGD | Decision coverage | Objective Box-IGD | Objective coverage |
|---|---:|---:|---:|---:|
| NSGA-II | 0.173567 / 0.178179 | 40.79% / 35.57% | 0.002193 / 0.001890 | 96.77% / 96.26% |
| ALSF-Resample | 0.179932 / 0.184499 | 34.26% / 29.77% | 0.001419 / 0.000476 | 97.67% / 98.13% |
| ALSF-Drop | 0.179837 / 0.180093 | 37.35% / 32.60% | 0.001971 / 0.001193 | 96.93% / 96.97% |

Within-run 95th-percentile and maximum reference distances are in the per-seed
CSV and both distribution figures, alongside all 20 individual outcomes. These
are tail distances **within one reference set**, not confidence intervals over
seeds. HV, IGD+ and actual evaluation counts are also in that CSV.

Paired Drop-minus-baseline differences, with 20,000-resample paired-bootstrap
95% mean-difference intervals (fixed seed 20261116):

| Metric | Mean difference | Pointwise 95% CI |
|---|---:|---:|
| decision_box_igd | 0.006270068 | [-0.001622883, 0.01407641] |
| objective_box_igd | -0.0002222356 | [-0.0009303586, 0.0004367885] |
| decision_tolerance_coverage | -0.03447165 | [-0.07545264, 0.01121134] |
| objective_tolerance_coverage | 0.001610825 | [-0.003801546, 0.007152062] |

All three method pairs and all requested per-run metrics are exported. Bootstrap
CIs are pointwise, not simultaneous multiplicity-adjusted intervals. A CI
containing zero is not proof of practical equivalence. Reference rows are not
bootstrap replicates: the independent paired unit is the optimizer seed.

## What the tolerance-aware results show

In decision space, baseline covers **40.79%**
of reference boxes and Drop **37.35%**.
Mean excess distances are **0.17357 versus
0.17984 MA/m²**. These are excesses beyond the allowed
box, not distances from its center. The paired Drop-minus-baseline difference is
0.006270, CI [-0.001623, 0.014076]. The
ill-conditioned field map makes design matching and performance matching
fundamentally different questions.

Objective-rectangle coverage averages **96.77%
baseline, 97.67% Resample and
96.93% Drop**. Mean objective Box-IGD
is **0.002193 baseline versus
0.001971 Drop**; the paired difference is
-0.000222, CI [-0.000930, 0.000437]. Interpret
this interval on its own scale; neither matching nor a nonsignificant difference
establishes guaranteed robustness or general practical equivalence.

The tails still matter: the mean within-run maximum objective excess is
**0.0853 baseline and
0.0831 Drop**. Drop's worst run has a
maximum of **0.1694** reference-normalized
units. Small means can hide missed reference-endpoint regions. All reference
distances are retained; no endpoint was trimmed for a favorable score.

The following Drop-versus-baseline outcomes pass the illustrative, Holm-adjusted
equivalence margins: **objective_tolerance_coverage**. Outcomes not listed do not pass the
adjusted family test (or are not estimable). This is not an overall engineering-
equivalence certificate: the margins are proposals, and the objective rectangles
bound rather than exactly characterize the jointly attainable uncertainty set.
The measured evaluation saving is **5.67%**; its engineering
acceptability must consider all metrics, tails and approved acceptance margins.

## Explicit, illustrative equivalence analysis

Engineering acceptance margins were not supplied. The following are **proposals
fixed in the analysis code before computing new Box metrics**, not validated
engineering specifications: mean decision Box-IGD difference ±0.01 MA/m² (10% of
one physical half-width); mean objective Box-IGD difference ±0.001 normalized
units (0.1% of one reference range); coverage difference ±0.05 (five percentage
points). The same definitions apply to every pair. They require engineering
approval and cannot establish physical equivalence by themselves.

An exploratory paired t-TOST at alpha=0.05 evaluates the paired population mean
against both sides of each margin, assuming a t approximation for the paired
mean. Its matching 90% t-interval and p-value are saved. Holm correction covers
all estimable proposed-margin tests across both spaces and three method pairs.
Zero paired variance is marked not estimable, rather than inventing population
certainty from constant observations. This is a separate equivalence question,
not a reinterpretation of nonsignificance in a difference test.

| Drop versus baseline outcome | Proposed symmetric margin | Holm-adjusted TOST result |
|---|---:|---|
| decision_box_igd | ±0.01 | does not pass |
| objective_box_igd | ±0.001 | does not pass |
| decision_tolerance_coverage | ±0.05 | does not pass |
| objective_tolerance_coverage | ±0.05 | passes |

## Interpretation constraints

Decision matching within manufacturing tolerance does not guarantee identical
objective performance or robustness. Conversely, the ill-conditioned solenoid
field matrix permits different current distributions with similar objectives,
so poor decision-box matching need not imply poor objective performance.
Objective-rectangle matching is more permissive and can hide differences through
its joint-set bounding approximation. Neither diagnostic overrides the previous
exact-sensitivity result that sampled final designs are sensitive at the 10%
criterion. These metrics supplement, rather than replace, conventional HV/IGD+.
The mean optimization call saving of Drop is 5.67%.

The stored reference is adaptively sampled in weight/objective space, not uniformly
in decision coordinates or arclength. The requested uniform stored-row average
therefore reflects that sampling density, including its low-error endpoint.
`reference_distribution.csv`, per-reference distances and rectangle bounds allow
audit of this effect. No method-specific normalization or favorable relabeling
was performed. Differences and tail distances must be considered alongside
coverage; Box-IGD alone is not an engineering acceptance decision.

## Reproduce and audit

```
python analyze_box_igd.py --study docs/twenty_seed_study --outdir docs/twenty_seed_study/box_igd
```

The output directory must not already exist. Use a new `--outdir` for repeat
analysis. No NSGA-II optimization or new Pareto-front construction is launched. The only new model work
is independent post-processing of reference uncertainty extrema and consistency
checks of stored objective values; these calls are not optimization evaluations.
All pre-existing study files are SHA-256 checked before/after analysis.

- [Per-seed results](box_igd_per_seed.csv), [summaries](box_igd_summary.csv),
  [paired effects/equivalence checks](box_igd_paired_effects.csv).
- [Decision-space PNG](box_igd_decision_space.png) / [PDF](box_igd_decision_space.pdf).
- [Objective-space PNG](box_igd_objective_space.png) / [PDF](box_igd_objective_space.pdf).
- [Box-IGD versus model calls PNG](box_igd_vs_evaluations.png) / [PDF](box_igd_vs_evaluations.pdf).
- [Exact rectangle bounds](objective_tolerance_rectangles.csv),
  [per-reference solver diagnostics](objective_extrema_solver_diagnostics.csv),
  [settings and proposed margins](settings.json), [source integrity](verification.json).
- `per_reference_distances.npz`: ordered `run_keys` and decision/objective arrays,
  each of shape (60, number_of_reference_rows).
