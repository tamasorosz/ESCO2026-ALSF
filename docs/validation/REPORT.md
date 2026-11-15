# Initial smoke validation report

The full archive was subsequently supplied by the user. See the
[full archive analysis](full_archive/REPORT.md) for measured 20,000-point
results, revised figures, and the independent B-region counterexamples.

Validation used Python 3.12, pymoo 0.6.2, seed 42. The original base checkout
had 18 passing tests and two failures: mixed-component medoid handling and the
missing `pop_diagnostics` implementation. The revised suite has **30 passing
tests**, including actual NSGA-II repair and four plotting command entrypoints.

## Independent perturbations

A separate 21-by-21 tensor grid spans each center's physical +/-0.1 tolerance
box. The symmetric relative formula and fixed 0.001 floor are unchanged.

| Region center | Maximum sampled r1 | Maximum sampled r2 | Threshold violation |
|---|---:|---:|---|
| A: (2.5, 2.5) | 0.078400 | 0.777302 | yes, f2 |
| B: (1.8, 1.87) | 0.105983 | 0.062185 | yes, f1 |

The B test constructs ten nearly coincident evaluated samples, all classified
empirically stable, then finds independent violating perturbations for all ten.
This is a reproducible counterexample to interpreting sufficient archive count
and no observed violation as guaranteed robustness. It is not a claim about
which points in the unavailable full archive were mislabeled.

## Measured short run

```
python run_test1.py --mode alsf --generations 8 --pop-size 24 --flag-start 3 --tag evidence_validated
```

The run produced 192 unique archived evaluations, a Pareto set of 24, and six
classification records (generations 3 through 8). All 192 observations are NEI;
there are no verified violation endpoints or DBSCAN components at the default
density. There are 183 occupied bounded cells. Across the recorded intervals,
repair was called 14 times and saw 206 candidates (including initialization and
duplicate elimination); none were forbidden and none were regenerated.
Counters before the first classification are accumulated in its first row.
This short run verifies integration and schema, not optimization performance.

A separate integration test seeds four forbidden initial candidates into real
NSGA-II, verifies they are regenerated before evaluation, checks every evaluated
candidate against the policy, and repeats the run with the same seed to verify
identical evaluated coordinates. This supplies the rejection evidence that the
sparse smoke run cannot supply.

Changing DBSCAN eps from 1 to 2 on the short archive leaves every pointwise flag,
local count, and maximum observed variation identical. All eps values yield zero
components here, so the comparison is uninformative about large-archive chaining.
A separate chained-data regression changes connectivity while asserting identical
pointwise evidence and exclusion behavior; mixed fractions are checked explicitly.
For all bounded regions, the maximum normalized coordinate span cannot exceed
the chosen cell width by construction, regardless of DBSCAN chaining.

## Reproducible artifacts

- [Measured archive](smoke/archive.csv), [process](smoke/process.csv),
  [settings](smoke/settings.json), [summary](smoke/summary.json).
- [Independent perturbation results](validation.json).
- [Short-archive partition comparison](partition_comparison.csv).
- Decision space: [PNG](smoke_plots/design_space_clusters.png),
  [PDF](smoke_plots/design_space_clusters.pdf).
- Evolution: [PNG](smoke_plots/test_evolution.png),
  [PDF](smoke_plots/test_evolution.pdf).
- Objective space: [PNG](smoke_plots/pareto_front.png),
  [PDF](smoke_plots/pareto_front.pdf).

The figures use measured smoke data and visibly state that they contain only
192 evaluations. Consequently they contain no sensitive or stable representative
markers; the plotting code supports these categories and is tested with a
mixed archive. They are not replacements for the full experiment figures.

## Remaining limitations

The 20,000-point two-variable archive and its generation history were absent
from the supplied repository. Its reported 6,795 stable / 1,564 sensitive /
11,641 NEI observations could not be reclassified or plotted. No expensive full
optimization was launched and no previous results were overwritten. Once the
full archive is available, run `validate_alsf.py` on it and regenerate the figures
from the corresponding measured process history. Old component histories that
lack mixed counts cannot be reconstructed from only a final archive.
Finite perturbation grids can miss extrema; robust labels remain empirical.
Fixed cells are reporting regions with an arbitrary origin and orientation,
not physical basins, certificates, or exclusion zones. High-dimensional cell
counts and full-run runtime/quality still require separate evaluation.
