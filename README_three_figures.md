# ESCO2026 Test 1 — Three figure reproduction

This plotting-only script reproduces the **layout and plot types** of:

- `design_space_clusters.png` — decision-space archive with non-robust points and robust cluster medoids;
- `test_evolution.png` — cluster status over generations, offspring regeneration/resampling;
- `pareto_front.png` — **objective-space archived points** (not the final nondominated-only Pareto set).

## Required real input files

For the **original two-dimensional Test 1**, supply the **ALSF archive CSV** (`X1,X2,Obj1,Obj2,flag,cluster_id,is_medoid`) and the associated **process CSV** (`generation,n_robust_clusters,n_non_robust_clusters,n_regenerated,n_attempts`).

CSV files with historical column aliases are accepted; `is_medoid` and `cluster_id` are optional but necessary for colored medoid markers. The program rejects 10D archives rather than silently plotting an irrelevant 2D projection.

## Run

```bash
python -m pip install numpy pandas matplotlib scipy
python plot_three_figures.py \
    --archive results/test1_alsf_seed42_full/archive.csv \
    --process results/test1_alsf_seed42_full/process.csv \
    --outdir figures/test1_reproduced \
    --reference-axes --annotate-ab
```

Outputs: 3 PNG + 3 PDF plots. If the screenshot axis ranges would hide actual points, omit `--reference-axes` to autoscale without clipping.

If you don't yet have a full archive, run your `run_test1.py --mode alsf` first. For a 200-generation, 100-population run:

```bash
python run_test1.py --mode alsf --generations 200 --pop-size 100 --seed 42 --tag full
```

**Important research caveats**:
- These visuals are *recreations from the supplied run CSVs*, not numeric digitizations of screenshots. For an exact recreation of older figures you need the original older CSV files; re-running changed clustering/repair policies produces different data.
- The plotting medoids are representatives of `robust` archive rows in each DBSCAN group. They are not the basis of local stability in the Pavel variant.
- Panel 3 contains all archived objective vectors, *not just nondominated vectors*. Avoid captioning it "the Pareto front" without qualification.
- The `n_attempts` series counts total resampling attempts; the second-panel red cap shows the **additional** attempts beyond the successful regenerated offspring, avoiding double-counting. The displayed stack reaches total attempts.
