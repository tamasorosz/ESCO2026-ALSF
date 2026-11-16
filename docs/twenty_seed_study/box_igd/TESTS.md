# Validation

The complete repository suite passed: **74 tests**. The new Box-IGD suite
contains **21 analytical/unit/integration cases** and also passed after the
figure-generation refactor.

```
python -m pytest -q
python -m pytest -q tests/test_box_igd.py
```

Checks include zero inside/on box boundaries, exact excess, scalar and anisotropic
half-widths, full-width conversion, zero-width Chebyshev limit, symmetric penalty
for dominating solutions, arbitrary dimension and chunking against an independent
scalar oracle, empty solutions, invalid inputs, asymmetric rectangles, fixed-scale
normalization, the BVLS fallback against a dense independent grid, inclusion of a
known feasible nominal candidate, and explicit equivalence tests separate from
nonsignificance.

The end-to-end test constructs a separate temporary study, disables NSGA-II
optimization and Pareto-reference construction with failing mocks, generates all
outputs, verifies exact zero for matching inputs, checks source SHA-256 integrity,
and verifies refusal to overwrite an existing output directory. No original
experiment is rerun by any new test or by the Box-IGD analysis.
