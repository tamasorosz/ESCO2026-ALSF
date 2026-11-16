# Modified Gaussian benchmark

This branch adds a separate min-f1/max-f2 Gaussian experiment on top of
`feature/test1-original-minmax`. It preserves the original benchmarks, algorithms,
previous experimental results and JCAM manuscript.

The completed primary study contains **60 runs: 20 paired seeds for each of
NSGA-II, ALSF-Resample and ALSF-Drop**, all completing 200 generations.

See [the full report](docs/test1_modified_gaussian/REPORT.md),
[the locked protocol](docs/test1_modified_gaussian/protocol.json),
[all per-run results](docs/test1_modified_gaussian/analysis/all_runs.csv),
[figures in PNG and PDF](docs/test1_modified_gaussian/figures/), and the separate
[suggested manuscript subsection](docs/test1_modified_gaussian/suggested_subsection.tex).

ALSF-Drop saved 23.78% of optimization evaluations on average. Actual rejections
near the Gaussian peak were demonstrated, but nominal hypervolume and narrow
Pareto-segment coverage decreased. The report separates discovery, empirical
archive evidence, independently assessed sensitivity and actual rejection.

## Reproduce in a new directory

```sh
python -m pip install -r requirements.txt pytest
python prepare_modified_gaussian.py --root results/test1_modified_gaussian_reproduction
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python run_test1_modified_gaussian.py --root results/test1_modified_gaussian_reproduction --workers 2
python analyze_modified_gaussian.py --root results/test1_modified_gaussian_reproduction
python -m pytest -q
```

Exact tested dependency versions and source hashes are recorded in the study's
`environment.json`. Preparation refuses to overwrite a locked protocol. The
runner resumes completed runs with matching settings and refuses to overwrite
partially written run directories. The report documents how to recover an
incomplete run without excluding unsuccessful outcomes.

The optional wider-Gaussian study has not been run; the primary width remains
exactly 0.01.
