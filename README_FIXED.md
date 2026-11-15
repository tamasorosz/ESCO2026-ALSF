# ESCO2026 Test 1: diagnostic fixes, 2026-11-14

This is the full, **source-only** 2-D Test 1 project with fixes to the figures and offspring repair diagnostics.

Changes:

1. A `robust medoid` is plotted only if its **entire DBSCAN component** is robust. Mixed and non-robust clusters have no robust medoid. `cluster_status` is now explicit in `archive.csv`.
2. A missing `n_regenerated` or `n_attempts` column is a **hard error**, not silently interpreted as zero. Implausible / negative counters are rejected.
3. `src/preprocessing.py` counts repair invocations, number of offspring checked, rejected/forbidden candidates, successful regenerations, attempts, and active exclusion centers. `src/postprocessing.py` logs them per generation in `process.csv`.
4. Per-generation counts reset throughout warmup. If optimization is active and repair has not seen any offspring, a RuntimeError prevents a misleading run report.
5. The plots explain explicitly when zero regenerations are *reported*. The completed run's `summary.json` contains aggregate repair diagnostics.

## Setup (Ubuntu / Python 3.10+ recommended)

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pytest -q
```

## Short verified run

```bash
python run_test1.py --mode alsf --generations 12 --pop-size 24 --flag-start 3 --seed 42 --tag fixed_smoke
python plot_three_figures.py \
 --archive results/test1_alsf_seed42_fixed_smoke/archive.csv \
 --process results/test1_alsf_seed42_fixed_smoke/process.csv \
 --outdir figures/fixed_smoke --annotate-ab
```

## Full run

```bash
python run_test1.py --mode alsf --generations 200 --pop-size 100 --flag-start 50 --seed 42 --tag fixed_full
python plot_three_figures.py \
 --archive results/test1_alsf_seed42_fixed_full/archive.csv \
 --process results/test1_alsf_seed42_fixed_full/process.csv \
 --outdir figures/fixed_full --annotate-ab
```

Use `--reference-axes` ONLY when reproducing screenshot axis limits; it will warn about clipped archive points.

## Diagnostics

After running the full script:

```bash
python - <<'PY'
import json
p='results/test1_alsf_seed42_fixed_full/summary.json'
a=json.load(open(p))
for name,value in a.items():
    if name.startswith('total_') or name.endswith('_centers'):
        print(name, value)
PY
```

`total_n_offspring_seen > 0` confirms `OffspringRepair._do()` ran. `total_n_forbidden_candidates > 0` confirms it actually rejected candidates. `total_n_regenerated > 0` confirms successful regeneration. If the latter two are zero despite many exclusion centers, it is now a **measurable method behavior** rather than a missing-data plotting artifact.

## Methodological limit

The local relative sensitivity expression and the 10% thresholds are unchanged. Therefore the method can still label many near-zero-objective designs as non-robust; this is a **scientific definition to validate separately**, not a plot-level coding bug. `robust` means stable within observed archive pairs only, NOT a worst-case mathematical guarantee.

### Testing status

The pure Python unit tests and plotting-only smoke tests run without pymoo. The pymoo-specific test is skipped in environments without pymoo; run it locally with pymoo installed. The full 200-generation optimization is **not** pre-validated by this package.
