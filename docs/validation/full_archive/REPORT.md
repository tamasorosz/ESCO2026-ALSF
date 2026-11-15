# Uploaded full archive: validation and revised figures

The user supplied the final 20,000-row archive and 100-row Pareto CSV after the
initial implementation. No process history was supplied. These figures use
those measured points; no optimization was run and no pointwise label was changed.
The raw uploaded files are not added to Git.

## Data checks

- All 100 supplied Pareto coordinates and objective pairs match archived observations.
- Archive labels: 6,795 empirically stable, 1,564 verified sensitive, 11,641 NEI.
- Benchmark objective values agree with the analytical equations within 5.7e-14.
- Independent reclassification exactly reproduces every flag, verified endpoint,
  and local observation count in the uploaded archive.
- The five DBSCAN radius comparisons exactly reproduce the previously supplied
  component counts, noise counts, and largest-component sizes. Changing radius
  leaves local classifications, counts, and observed relative variations unchanged.
- At eps=1: 72 components, of which 67 mixed and 5 entirely sensitive; no wholly
  stable components. There are 10,740 bounded cells, of which 683 mixed. Connectivity
  and bounded partition counts measure different objects and now use separate panels.

## Independent A/B perturbations

The predefined A circle contains eight archive samples, all verified sensitive.
The B ellipse contains 169 samples: 160 empirically stable, five verified
sensitive, four NEI. A separate 21-by-21 tensor grid over each stable point's
+/-0.1 decision box finds violating center-to-perturbation pairs for **130 of
B's 160 stable archive points (81.25%)**. See `perturbation_audit.csv` for row-level
results. Diagnostic evaluations were not inserted into the archive or used to
relabel it. Absence of a sampled violation at the remaining 30 is not a proof.
This is a measured limitation of archive coverage, not merely an aesthetic issue.

## Verification

All 33 unit/integration/plotting tests pass. Added checks cover deterministic
representative display thinning without data mutation, separate evolution scales,
measured pointwise percentages summing to 100, and scatter-only CLI use with a
Pareto overlay and no fabricated process history.

## Visual changes

- Small gray/green/orange points encode NEI/stable/sensitive observations directly.
- The 1,828 stored stable representatives are retained in data. Only 32 deterministic
  representatives are marked, with small outlined diamonds. The identical row subset
  appears in both spaces and the legend explicitly calls it a display subset.
- The optional final Pareto set is shown as blue hollow circles. This is the final
  NSGA-II nondominated set, not a guaranteed robust front.
- Both figures have a full-archive overview and separate A/B details. Objective-space
  detail outlines refer to the actual archive members of the indicated decision region,
  not guessed ellipses in objective space. The other points in those windows remain
  visible as context. Detail-window clipping never affects the full overview.
- Legends are outside the data, colors are consistent, and PDF text/axes remain vector.
- Evolution plotting now separates DBSCAN components and bounded cells into different
  panels and shows pointwise proportions normalized by the measured archive size.
  Regeneration/extra-attempt counts remain measured, unnormalized counts.

[Decision-space PNG](../full_archive_plots/design_space_clusters.png) /
[PDF](../full_archive_plots/design_space_clusters.pdf).
[Objective-space PNG](../full_archive_plots/pareto_front.png) /
[PDF](../full_archive_plots/pareto_front.pdf).

## Reproduce

```
python plot_three_figures.py --archive archive.csv --pareto pareto.csv --outdir figures/full_archive --annotate-ab
```

Add `--process process.csv` to also produce the revised evolution figure. Without
that file, no generation history is inferred from row ordering or screenshot pixels.
The initial archive remains unchanged and all original representative indices remain
available for analysis. Subsequent large-run sensitivity research should address
neighborhood coverage and independent validation, rather than interpreting a count
threshold as a worst-case certificate.
