# ALSF: independent evidence, connectivity, and bounded reporting regions

ALSF reports **empirical archive evidence**, not guaranteed worst-case robustness.
The classifier uses no additional objective evaluations. Both benchmark objectives
are minimized; their equations and bounds remain in `benchmark_test1.py`.

## Independent controls

| Meaning | Python classifier | Test 1 CLI | Default |
|---|---|---|---|
| Physical coordinate tolerances | `tolerances` | `--tol` | each coordinate 0.1 |
| DBSCAN radius | `dbscan_eps` | `--dbscan-eps` | 1.0 |
| DBSCAN density | `dbscan_min_samples` | `--dbscan-min-samples` | 10 |
| Local evidence radius | `radius` | `--local-radius` | 1.0 |
| Minimum local observations, including self | `min_samples` | `--min-samples` | 10 |
| Offspring buffer radius | `VerifiedViolationFilter.radius` | `--exclusion-radius` | 1.0 |
| Reporting cell width | `partition_width` | `--partition-width` | 1.0 |

Every geometric radius is in tolerance-normalized Chebyshev distance:
`d(x,y) = max_i |x_i-y_i|/Delta_i`. Changing DBSCAN does not change Delta,
local evidence, or the exclusion rule. In the compatibility file API and
`FlagCallback`, `d_cluster` and `min_samples` control DBSCAN;
`local_radius` and `local_min_samples` control evidence. Old callers that used
nondefault density to change evidence must explicitly pass `local_min_samples`.
A reused incremental state with mismatched evidence parameters raises an error.

For every archived pair with `d(x,y) <= local_radius`, compute

`r_j = |f_j(x)-f_j(y)| / max(|f_j(x)|, |f_j(y)|, 0.001)`.

An endpoint is `non-robust` if any directly observed pair exceeds 0.10 in
any objective. Otherwise it is `robust` (empirically locally stable) when its
local observation count reaches the minimum, or `NEI` otherwise. Verified
endpoints persist in an append-only archive; adding observations updates both
old and new endpoints. Sparse or directionally biased observations can miss
sensitivity even when the count reaches ten.

## DBSCAN summaries and a bounded alternative

DBSCAN remains a description of density connectivity. Its `cluster_status` is
`robust`, `non-robust`, or `NEI` only when every member has that pointwise label;
otherwise it is `mixed`. Noise has status `noise`, not a shared component.
`cluster_fraction_robust`, `cluster_fraction_non_robust`, and
`cluster_fraction_nei` report proportions within each non-noise component;
these fractions are undefined (NaN) for noise. No label propagates along chains.

For interpretable local reporting, independently partition normalized coordinates
`z_i=x_i/Delta_i` into origin-anchored half-open cells:

`k_i = floor(z_i / w)`, with `w=partition_width`.

A cell has Chebyshev diameter at most `w`, unlike a DBSCAN component whose
diameter can grow without bound along a chain. Cells include DBSCAN noise and
have `region_id` and `region_status`. They never merge observations arbitrarily
or optimize a target count. The default width corresponds to the local radius,
but it does not replace centered sensitivity neighborhoods: neighboring points
can fall on opposite cell boundaries, and all such pairs are still checked.
This method is deliberately a separate spatial partition, not a claim to recover
physical basins or connected robust subsets. IDs are local to each classification,
not persistent region identities over generations.

Within each occupied cell containing stable observations, select the actual
stable member nearest the mean of those stable members in normalized L-infinity
distance (first archive row breaks ties). `is_stable_representative` records this
plotting choice. A representative does not certify the whole cell. Legacy
`is_medoid` is retained only for wholly stable DBSCAN components and denotes an
approximate geometric representative, not an exact pairwise medoid.

Tradeoffs: fixed cells impose an origin and orientation, may split nearby points,
and can proliferate in high dimensions. Varying cell width/origin should be part
of subsequent large-archive analysis. DBSCAN remains useful for connectivity;
the cells are useful for bounded spatial resolution. Neither defines sensitivity.

## Offspring repair and logging

Only points within `exclusion_radius` of a **verified violating endpoint** are
forbidden. Neither cluster membership, region membership, nor NEI status causes
exclusion. Uniform resampling, the fixed seeded generator, failure on exhaustion,
relative variation, thresholds, floors, and previous counters are preserved.
`pop_diagnostics()` additionally exposes repair calls, offspring seen, forbidden
candidates, resampling attempts, regenerations, and current exclusion centers.
The callback writes these with separate pointwise, DBSCAN, and region counts.
Both ALSF pymoo runners use `copy_algorithm=False` to keep callback and repair
attached to the same algorithm instance. The Test 1 evaluation cache is separate
from pymoo's own method cache, so cache metadata cannot inflate evaluation counts.

## Reproducing figures and auditing existing data

The three figures show the same measured archive and representative indices in
decision and objective space. A/B annotations are illustrative reference regions,
not learned clusters; objective-space highlights map actual archive rows. Full
archive axes are the default. Evolution has separate component/region, pointwise,
and offspring-repair panels. Missing warmup rows are not filled with invented
zeros. PNG and PDF are produced; vector text/axes accompany rasterized dense
scatter layers. No official journal-format compliance is claimed.

```
python run_test1.py --mode alsf --generations 8 --pop-size 24 --flag-start 3 --tag new_smoke
python plot_three_figures.py --archive results/test1_alsf_seed42_new_smoke/archive.csv --process results/test1_alsf_seed42_new_smoke/process.csv --outdir figures/new_smoke --annotate-ab
python validate_alsf.py --archive results/test1_alsf_seed42_new_smoke/archive.csv --outdir work/new_validation
```

The validation command refuses to reuse an output directory and never modifies
its input. It compares DBSCAN eps 1, 1.25, 1.5, 1.75, 2 while asserting unchanged
pointwise results, saves a classified copy, and audits **all** empirically stable
A/B archive samples against a separate 21-by-21 perturbation grid over their
physical tolerance boxes. Additional objective evaluations occur only in this
explicit diagnostic command, never in ALSF classification. Finite perturbations
can establish counterexamples; absence of a grid violation is not a proof.
The original-style `plot_original_three_ALSF.py`, `plot_original_3.py` and
`plot_alsf_elsevier.py` also accept revised run directories and share the revised
plot implementation (Elsevier retains its legacy log-only mode).
