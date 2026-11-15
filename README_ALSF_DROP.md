# ALSF-Drop: rejection before evaluation, without replacement

This experimental strategy tests evaluation savings on the defined ten-current
Di Barba-inspired convex solenoid model. It does not claim exact reproduction
of the published physical problem. The analytical objectives, bounds, convex
reference calculation, baseline and ALSF-Resample executables are unchanged.

## Run

Activate your Python environment and install the project requirements and pytest.
Use Python 3.11+; the implementation was inspected and tested on pymoo 0.6.2.

```
python -m pytest -q
python run_di_barba_drop.py --generations 60 --pop-size 40 --offsprings 40 --flag-start 10 --seed 42 --tag smoke1
```

Every invocation requires a fresh output directory. Results are written to
`results/di_barba_drop_seed42_smoke1/`; use another tag for another run.
The runner creates `archive.csv`, `archive_classified.csv`, `pareto.csv`,
`process.csv`, `settings.json` and `summary.json`. Optional `--outdir` chooses
another fresh directory. Existing directories are refused before any writes.

For convergence indicators, supply the same reference used in the comparisons:

```
python run_di_barba_drop.py --generations 200 --pop-size 100 --offsprings 100 --flag-start 50 --seed 42 --tag full1 --reference docs/alsf_drop/reference/reference.csv
```

Repeat with seeds 13 and 97 and unique tags. The optional reference affects only
indicators; it is never consulted by the filter. No objective calls are made by
sensitivity classification. Preserve previous output files instead of rerunning
into their directories. Plot the committed measured comparison without rerunning
optimization:

```
python plot_di_barba_drop_comparison.py --study docs/alsf_drop --outdir figures/drop_comparison1
```

Both PNG and PDF are generated. The comparison requires baseline, resample and
drop histories/fronts for the same seeds and the common unchanged reference;
`protocol.json` records the target HV, checkpoints and historical provenance.

## Lifecycle and scientific parameters

`OffspringDropFilter` in `src/offspring_drop.py` wraps the existing
`VerifiedViolationFilter`. `DropNSGA2._infill()` calls the ordinary pymoo mating
operator **once**, then filters its completed proposal batch. Rejected members
are removed before `Evaluator.eval`. There is no second mating call to replace
ALSF rejections, and no uniform resampling in this component.

Normal mating can internally retry duplicates to construct its usual batch;
these retries occur before ALSF filtering and are counted separately. Genetic
operators can also overproduce/truncate candidates. Neither duplicate removal
nor surplus truncation is claimed as ALSF saving. `n_proposed` counts the actual
batch returned by normal duplicate-eliminated mating; initialization counts its
initial candidates separately in the first row. Standard initialization remains
unfiltered so there is a parent population and initial archive evidence.

An accepted batch can have any size from zero through the configured offspring
count. For a nonempty batch the inherited NSGA-II survival selects from parents
plus accepted offspring. With an empty batch, `_advance` preserves the existing
parents and lets pymoo perform its ordinary termination/callback/generation
update. It never sets `force_termination`, evaluates forbidden solutions,
refills a batch or loops indefinitely. A normal empty mating batch is also
handled without early termination; its proposal shortfall is not ALSF saving.
The parent population size is checked every generation. Generation 1 is
initialization, so 200 generations contain 199 mating batches.

The default tolerance remains +/-0.1 MA/m² in each of ten normalized current
coordinates. Local and exclusion radii are 1 in normalized Chebyshev distance.
An endpoint is verified sensitive when an observed neighbor differs by more
than 10% in either objective using
`|fa-fb| / max(|fa|, |fb|, 0.001)`. The minimum local observation count is ten,
including self. DBSCAN remains diagnostic (radius 1, min samples 10) and never
determines rejection. `partition_width=None` disables only sparse cell-reporting
objects in 10D; its default behavior is unchanged and a regression verifies
identical evidence and DBSCAN labels with reporting enabled/disabled.

SBX probability 0.9, eta 5; polynomial mutation probability 1.0, eta 5;
identical rounding/clipping to eight decimals and standard duplicate elimination.
The initial reference seed is 42. The exact same seed reproduces a run.
`copy_algorithm=False` keeps algorithm, filter and callback attached to the same
objects. An empty filter reproduces ordinary NSGA-II trajectories in tests.

As in the existing ALSF callback, classification occurs **after** generation
`flag_start` finishes. At the default 50, the first potentially filtered offspring
batch is generation 51. `n_exclusion_centers_used` records the policy that actually
filtered the batch; `n_exclusion_centers` records the updated policy for the next
batch. Before classification, empirical statistics are blank, never invented
zeros. Previously observed violations persist as the archive grows.

## Accounting

The runner instruments actual analytical model calls. Every accepted optimizer
request executes the model once; there is no objective cache. The archive retains
unique coordinates, and repeated evaluated coordinates are counted explicitly
instead of inflating neighborhood evidence. Objective evaluations used in exact
posthoc audits belong to a separate diagnostic budget.

The principal identities checked at completion are:

```
total_proposed = total_dropped + total_accepted
actual_model_calls = total_accepted = evaluator_evaluations
saved_evaluations = total_dropped
baseline_expected_evaluations = pop_size + (generations-1)*offsprings
total_missing_evaluations = saved_evaluations + non_alsf_shortfall
```

`non_alsf_shortfall` includes an ordinary mating batch failing to return all
configured proposals; it is explicitly excluded from saved evaluations.
`cache_hits` is zero by design. Duplicate-elimination counts and raw genetic
candidate counts are separate from the proposal/evaluation budget.

Per-generation logs include proposed, dropped, accepted, evaluated, cumulative
proposed/dropped/accepted/evaluated, verified endpoints, stable/NEI counts,
exclusion-center counts, population size, duplicate removals and archive time.
The final summary includes all required savings counters, actual calls, expected
baseline evaluations, drop fraction and Pareto size. Timings separate objective,
archive processing and indicators.

## Experimental comparison and limits

The measured study first completes the requested smoke test, checks counters,
then runs Drop for seeds 13, 42 and 97 at 200 generations. Existing baseline and
Resample results from the validated solenoid study are copied byte-for-byte and
identified by source commit and SHA-256 hashes; the originals are not changed.
The common reference CSV and its solver diagnostics are reused unchanged.
All physical/algorithm parameters and the actual-call accounting match. The
previous baseline and Resample runs use the same in-memory scientific harness;
Resample uses the unchanged `OffspringRepair` implementation.

Protocol A compares final quality and actual model calls at 200 generations.
Protocol B uses actual-evaluation convergence axes, first observed attainment of
a predefined target HV (99.5% of the common reference HV), and common allowed
budgets at 5k/10k/15k calls plus the shared completed-run horizon. Each checkpoint
uses the last **completed generation** at or below the allowed budget. Since
batches have variable sizes, exact snapshot call counts can differ by less than
one batch. The CSV exports both actual checkpoint calls and unused allowance;
there is no fictitious interpolation or use of a state after the budget.
These are conservative allowed-budget comparisons, not newly run optimizers
truncated at an exactly synchronized intra-generation call count. Small quality
differences comparable with this granularity need caution.

Target HV was fixed before looking at Drop results. Non-attainment is right-
censored at the actual completed model-call count; it is never assigned a
fabricated successful evaluation count. First attainment can precede a later
quality decline, so final quality is reported separately. Three seeds support
an exploratory comparison, not general statistical superiority. A nominal convex
reference is not a robust optimum. Up to 25 objective-ranked final Pareto designs
per run receive an independent exact relative-box sensitivity audit; subset
results are not claims about every feasible design.

On this inexpensive analytical benchmark, repeated archive processing can cost
far more than the saved model calls. Real evaluation savings therefore do not
by themselves establish a wall-time benefit or faster attainment of a desired
front quality. The hypothesis is tested without retuning thresholds, tolerance,
operators, the reference or empirical flags to favor Drop.
