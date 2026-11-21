# Paired twenty-seed study on both benchmarks

The measured results are in [docs/twenty_seed_study/REPORT.md](docs/twenty_seed_study/REPORT.md).
There are 20 independent seeds per benchmark and method: **120 complete runs**.
All three methods use the same seed list: 1-17, 42, 71, 97. Genetic operators,
physical tolerances, objective thresholds/floors and 200-generation / 100-member
settings are fixed across methods. No benchmark equation was changed.

## Reproduce

Use an activated environment with the project requirements and pytest installed.
The recorded experiment uses pymoo 0.6.2 and Python 3.12. On Linux/macOS:

```
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python run_repeated_study.py --root results/twenty_seed_study1 --workers 2
python analyze_repeated_study.py --root results/twenty_seed_study1 --outdir figures/twenty_seed_study1
```

On Windows/PowerShell set the environment variables before invoking Python:

```
$env:OPENBLAS_NUM_THREADS="1"
$env:OMP_NUM_THREADS="1"
python run_repeated_study.py --root results/twenty_seed_study1 --workers 2
python analyze_repeated_study.py --root results/twenty_seed_study1 --outdir figures/twenty_seed_study1
```

Two workers match the measured machine's CPU quota; use one worker to avoid
concurrent-run contention when wall-clock timing is the primary outcome. Each
worker uses one BLAS thread. Outputs are isolated by problem, method and seed.
Completed cases can be resumed with exactly the same experiment settings; a
mismatched protocol is rejected. An incomplete existing case directory is never
overwritten or silently counted as a success. Analysis refuses missing cases.
All previous experiment directories are preserved.

A short verification run uses a separate root:

```
python run_repeated_study.py --root results/repeated_smoke1 --seeds 1 2 --generations 8 --pop-size 24 --flag-start 3
```

To regenerate the committed result plots, without rerunning optimization:

```
python analyze_repeated_study.py --root docs/twenty_seed_study --outdir figures/twenty_seed_replot
```

## Identical search and sensitivity rules

Both problems use baseline NSGA-II, ALSF-Resample and ALSF-Drop with paired seeds.
Population and offspring proposals are 100, generations 200, flag start 50.
SBX probability/eta: 0.9/5; PM probability/eta: 1.0/5. All modes use the same
rounding/clipping to eight decimals and duplicate elimination. The objective
cache is disabled consistently: each accepted evaluator request executes the
model once. Unique archive coordinates and actual model calls remain separate.
Resample retains its seeded uniform replacement policy; Drop rejects without
replacement. Initialization is unfiltered; the first policy update follows
generation 50 and first affects batch 51.

Local/exclusion radii are 1 in tolerance-normalized Chebyshev distance. Every
coordinate tolerance is 0.1; the solenoid physical tolerance is 0.1 MA/m².
Pointwise observation minimum is ten including self, thresholds are 10%, and
objective denominator floors are 0.001. The maximum replacement attempts remain
1,000 for Test 1 and 100 for the solenoid, matching their previous configurations.

The shared incremental state now maintains neighbor counts using already checked
unordered pairs: every new pair increments both endpoint counts once, and each
point initially counts itself once. This eliminates a redundant all-archive
neighbor query on every generation. Reordering, changed data and truncation reset
all caches as before. Tests compare counts, violations and maximum variations
against an independent full-tree/pair oracle, including duplicate points.
Full 200-generation baseline/Resample/Drop trajectories for solenoid seed 42 are
bit-for-bit identical to the earlier validated histories. This is an equivalent
computational optimization, not a sensitivity or exclusion-method change.

DBSCAN and sparse cells never determine rejection. Pointwise evidence updates
every active generation. This study computes DBSCAN diagnostics once at the end
and omits high-dimensional cell metadata. Consequently runtime numbers should
not be compared directly with the old diagnostic-heavy runners. All modes in
this study share the same computational implementation. Per-generation evidence,
repair/drop counters, final labels, final fronts and independent audits are saved.

## Independent references and audits

The solenoid reuses the previously verified convex reference unchanged. Test 1
uses an analytical objective-space Pareto envelope. Write q=x1²+x2², s=x1+x2:

`f2 = q - 10s + 50 + 100 exp(-(q - 5s + 12.5)/0.01)`.

Any q>50 is dominated by (5,5), which has objectives (50,0). For q in [0,50],
minimizing the convex function of s over s<=sqrt(2q) gives
`s=min(sqrt(2q),(q+12.5-0.01*log(5000))/5)`. The reconstructed design
`((s+sqrt(2q-s²))/2,(s-sqrt(2q-s²))/2)` is feasible and achieves the global
minimum f2 at that q. The envelope decreases monotonically. The symmetric swapped
design has identical objectives; only one branch is needed for objective-space
indicators. The implementation is checked against independent scalar minimization
and a doubled reference resolution. Neither original objective function changes.

Each run uses reference-only normalization, fixed HV reference (1.1,1.1), and
501 arclength-spaced IGD+ reference points. All methods use the same references.
Independent final sensitivity audits select up to 25 objective-ranked designs
per run. The 10D benchmark uses exhaustive vertices / convex minima, giving exact
box sensitivity up to solver tolerance. Test 1 uses a separate 21x21 perturbation
grid: finding a violation is a valid counterexample, but not finding one does not
certify robustness. These two audit types are explicitly distinguished.

## Statistical analysis

- Summaries include means, sample standard deviations, medians and quartiles.
- Paired comparisons use the same seed. CIs are 10,000-resample, deterministic
  paired-bootstrap **mean-difference** intervals. These 95% intervals are
  pointwise, not simultaneous multiplicity-adjusted intervals.
- Two-sided SciPy Wilcoxon tests use its automatic tie/zero handling. Holm
  adjustment covers all 18 final-HV / IGD+ / actual-call comparisons across
  both problems and the three method pairs. All-zero differences yield p=1.
- The primary target is 99.5% of the locked reference HV; 99% and 99.9% are
  predefined secondary targets. First attainment is recorded at an observed
  completed generation. Non-attainment remains censored, never assigned an
  invented successful time. Conditional differences among successful pairs are
  labeled selection-biased when other pairs are censored.
- Convergence summaries use the latest complete generation at or below a common
  allowed evaluation budget. A balanced common horizon retains all 20 seeds per
  method; no curve silently drops unfinished cases. Actual checkpoints and unused
  budget are exported. These are not synchronized intra-generation stopping runs.
- Failure to reject a quality difference is not proof of equivalence. No formal
  noninferiority/equivalence margin was predefined, and no threshold was retuned
  after seeing outcomes. Conclusions apply to these problems and settings.
