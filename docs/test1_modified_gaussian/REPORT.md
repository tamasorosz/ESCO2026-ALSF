# Modified Gaussian benchmark: independent reference and paired 60-run study

## Problem and scope

Physical objectives: min f1=x1²+x2²; max f2=−[(x1−5)²+(x2−5)²]+100 exp(−[(x1−2.5)²+(x2−2.5)²]/0.01), with x∈[−5,10]².
All optimizer dominance, HV, IGD+ and distance calculations use g=(f1,−f2). Physical objectives are exported as Obj1,Obj2. Sensitivity uses physical values and is invariant under the sign transform. This is a separate experiment; previous implementations, results and the manuscript are untouched.

## Independent Pareto reference and complete domain

At fixed q=x1²+x2², put s=x1+x2. Then f2=−q+10s−50+100 exp(−[q−5s+12.5]/0.01), whose derivative with respect to s is strictly positive. For 0≤q≤50, s≤√(2q), attained at x1=x2=√(q/2)∈[0,5], inside the feasible bounds. Thus every off-diagonal point in this range is dominated at the same f1 by its diagonal counterpart.
For q>50, distance to the Gaussian center is at least √q−√12.5. The negative quadratic is nonpositive, so f2≤100 exp(−[√q−√12.5]²/0.01)<10^−540. Such points have f1>50 and f2 below the peak endpoint and are dominated. This covers the entire square, including negative coordinates and q up to 200.
On the diagonal, f1=2t² and f2=−2(t−5)²+100 exp(−2(t−2.5)²/0.01). Its derivative is positive on [0,2.5]. Complete stationary-root enumeration finds t=[2.5002499781263476, 2.6826131853037105, 5.0]; checking these and both endpoints establishes the global maximum at **t*=2.50024997812635**, f1=12.502499906242, f2=87.501249882822. The derivative stays positive up to t*, and all later diagonal points are dominated by t*. The true Pareto set is exactly (t,t), 0≤t≤t*.

The independent adaptive reference has 4001 rows and its refinement has 8001. Reference HV changes by 5.7395376e-05; final-run HV changes by at most 1.110223e-16, IGD+ by at most 1.2966982e-08. Stationary residual is 5.52e-13. An independent one-dimensional quadrature of the continuous-front dominated area gives limiting HV 0.442999203638 (quadrature error estimate 8.29e-13), a gap of 0.00011477518 above the locked discrete reference HV. The original target values remain fixed; they are fractions of that discrete reference, not of an exact continuous HV. Numerical cross-checks use 250,000 uniform full-domain points and five global differential-evolution searches explicitly initialized at the peak. These supplement the proof; stochastic searches alone do not certify the maximum. Root solves use float64, not interval arithmetic. Reference coordinates, objective values, all stationary roots and validation are in `reference/`.

## Locked protocol and methods

Seeds: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 42, 71, 97]. Exactly 20 paired seeds per algorithm, 60 runs, 200 generations each, population=offspring count=100. Filtering evidence is first updated at generation 50 and affects generation 51 onward, matching the inherited harness. SBX probability .9, eta 5; PM probability 1, eta 5. Decision tolerance (.1,.1), relative thresholds (.1,.1), fixed objective floors (.001,.001), at least 10 local observations including self for the archive stable label. Eight-decimal coordinate rounding and duplicate elimination match the earlier experiment. No objective caching.
The existing LocalVariationState, OffspringRepair, OffspringDropFilter and DropNSGA2 are reused without modification. Neighborhood and exclusion distances use tolerance-normalized **L∞**, as implemented on the source branch. One observed violating pair establishes empirical sensitivity of both endpoints. Resample retains the inherited uniform replacement procedure with up to 1000 attempts; Drop discards the actual post-mating proposals before evaluation, without replacement. Initial population is unfiltered. An empty accepted batch still advances the generation. The benchmark parameters and methods were not tuned after outcomes.

Normalization: ideal=[0.0, -87.50124988282204], span=[12.502499906241603, 137.50124988282204], HV point=(1.1,1.1). IGD+ uses 501 equally spaced points along normalized objective arclength of the independent reference, avoiding a Gaussian-knot-density weighting. Nominal front error is mean shortest normalized Euclidean distance from final front to the reference polyline. The narrow segment is fixed as diagonal t∈[2.4,t*]; coverage is the proportion of 501 arc-uniform reference points within normalized Euclidean distance .01 of a final solution.

Every objective call, including duplicates, is stored in evaluated.csv.gz; archive.csv contains unique rounded designs and empirical classifications. Every screening check is stored in screening.csv.gz, including outside the peak; stage distinguishes original inputs from repeated replacement attempts. Resample repair checks may include proposals later removed as duplicates; they must not be interpreted as scheduled objective calls saved. Drop events correspond to the actual duplicate-eliminated proposal batch. Actual savings are 20000 minus actual optimization evaluations, verified against dropped batches. Independent posthoc sensitivity calls are outside the optimizer budget.

## All 60 results

| mode | seed | actual_model_calls | hypervolume | igd_plus | saved_percent | narrow_coverage | sensitive_audit_fraction | closest_evaluated_endpoint | closest_final_endpoint | run_seconds |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | 1 | 20000 | 0.43863 | 0.00211823 | 0 | 0.866267 | 0.86 | 0.00114089 | 0.00114089 | 5.43997 |
| drop | 1 | 15204 | 0.435397 | 0.00342077 | 23.98 | 0.431138 | 0.83 | 0.0116843 | 0.0116843 | 5.12554 |
| resample | 1 | 20000 | 0.435377 | 0.00342258 | 0 | 0.431138 | 0.83 | 0.0116843 | 0.0116843 | 5.52801 |
| baseline | 2 | 20000 | 0.438711 | 0.00206595 | 0 | 0.952096 | 0.87 | 0.00121893 | 0.00121893 | 5.98799 |
| drop | 2 | 15296 | 0.430874 | 0.00392429 | 23.52 | 0.552894 | 0.85 | 0.031194 | 0.031194 | 5.12124 |
| resample | 2 | 20000 | 0.430936 | 0.00389071 | 0 | 0.552894 | 0.84 | 0.031194 | 0.031194 | 6.25063 |
| baseline | 3 | 20000 | 0.438657 | 0.00212583 | 0 | 0.840319 | 0.89 | 0.00156297 | 0.00156297 | 6.16545 |
| drop | 3 | 15156 | 0.435776 | 0.00373584 | 24.22 | 0.449102 | 0.82 | 0.00163717 | 0.00163717 | 5.13733 |
| resample | 3 | 20000 | 0.435794 | 0.0037252 | 0 | 0.449102 | 0.84 | 0.00163717 | 0.00163717 | 5.61326 |
| baseline | 4 | 20000 | 0.438636 | 0.00209832 | 0 | 0.860279 | 0.88 | 0.00305921 | 0.00305921 | 5.96734 |
| drop | 4 | 15226 | 0.430205 | 0.00399215 | 23.87 | 0.630739 | 0.83 | 0.0338289 | 0.0338289 | 5.97383 |
| resample | 4 | 20000 | 0.430153 | 0.00401599 | 0 | 0.630739 | 0.81 | 0.0338289 | 0.0338289 | 7.17894 |
| baseline | 5 | 20000 | 0.43854 | 0.00214446 | 0 | 0.878244 | 0.87 | 0.00294681 | 0.00294681 | 6.7675 |
| drop | 5 | 15448 | 0.437817 | 0.00228156 | 22.76 | 0.698603 | 0.82 | 0.0090236 | 0.0090236 | 5.26122 |
| resample | 5 | 20000 | 0.437957 | 0.00221236 | 0 | 0.698603 | 0.84 | 0.0090236 | 0.0090236 | 6.74805 |
| baseline | 6 | 20000 | 0.438644 | 0.00212813 | 0 | 0.882236 | 0.88 | 0.00134636 | 0.00134636 | 4.88885 |
| drop | 6 | 15276 | 0.435707 | 0.00342215 | 23.62 | 0.592814 | 0.84 | 0.00932751 | 0.00932751 | 5.70019 |
| resample | 6 | 20000 | 0.435795 | 0.00337087 | 0 | 0.592814 | 0.83 | 0.00932751 | 0.00932751 | 6.81371 |
| baseline | 7 | 20000 | 0.438847 | 0.00202732 | 0 | 0.914172 | 0.86 | 1.36398e-05 | 1.36398e-05 | 6.3111 |
| drop | 7 | 15375 | 0.437598 | 0.00236675 | 23.125 | 0.828343 | 0.86 | 0.00943283 | 0.00943283 | 4.96921 |
| resample | 7 | 20000 | 0.437733 | 0.00230212 | 0 | 0.828343 | 0.85 | 0.00943283 | 0.00943283 | 6.84349 |
| baseline | 8 | 20000 | 0.438462 | 0.00221211 | 0 | 0.862275 | 0.91 | 0.0007854 | 0.0007854 | 5.96815 |
| drop | 8 | 15384 | 0.437 | 0.00237799 | 23.08 | 0.772455 | 0.85 | 0.0135492 | 0.0135492 | 6.13973 |
| resample | 8 | 20000 | 0.437016 | 0.0023674 | 0 | 0.772455 | 0.83 | 0.0135492 | 0.0135492 | 7.49913 |
| baseline | 9 | 20000 | 0.438903 | 0.0019953 | 0 | 0.874251 | 0.89 | 0.00132677 | 0.00132677 | 5.44668 |
| drop | 9 | 15299 | 0.433815 | 0.00329214 | 23.505 | 0.453094 | 0.81 | 0.0214532 | 0.0214532 | 4.29572 |
| resample | 9 | 20000 | 0.433792 | 0.00330692 | 0 | 0.453094 | 0.79 | 0.0214532 | 0.0214532 | 6.15 |
| baseline | 10 | 20000 | 0.438281 | 0.00230567 | 0 | 0.878244 | 0.91 | 0.000531497 | 0.000531497 | 5.21145 |
| drop | 10 | 15387 | 0.438348 | 0.00228869 | 23.065 | 0.688623 | 0.82 | 0.00127253 | 0.00127253 | 4.59841 |
| resample | 10 | 20000 | 0.438344 | 0.00228975 | 0 | 0.688623 | 0.82 | 0.00127253 | 0.00127253 | 5.76721 |
| baseline | 11 | 20000 | 0.43849 | 0.00210127 | 0 | 0.894212 | 0.86 | 0.00494934 | 0.00494934 | 5.63463 |
| drop | 11 | 15090 | 0.435469 | 0.00271553 | 24.55 | 0.744511 | 0.83 | 0.0181482 | 0.0181482 | 4.76788 |
| resample | 11 | 20000 | 0.435435 | 0.00273543 | 0 | 0.746507 | 0.82 | 0.0181482 | 0.0181482 | 5.65027 |
| baseline | 12 | 20000 | 0.438664 | 0.00209273 | 0 | 0.908184 | 0.88 | 0.00284931 | 0.00284931 | 5.18594 |
| drop | 12 | 15320 | 0.436615 | 0.00241448 | 23.4 | 0.732535 | 0.83 | 0.0150563 | 0.0150563 | 4.69128 |
| resample | 12 | 20000 | 0.436649 | 0.00240088 | 0 | 0.732535 | 0.84 | 0.0150563 | 0.0150563 | 5.68261 |
| baseline | 13 | 20000 | 0.43872 | 0.0020797 | 0 | 0.868263 | 0.88 | 0.00113916 | 0.00113916 | 5.68392 |
| drop | 13 | 15131 | 0.436213 | 0.00286679 | 24.345 | 0.467066 | 0.82 | 0.0131314 | 0.0131314 | 4.21519 |
| resample | 13 | 20000 | 0.436207 | 0.00287071 | 0 | 0.467066 | 0.81 | 0.0131314 | 0.0131314 | 5.6801 |
| baseline | 14 | 20000 | 0.438758 | 0.00206412 | 0 | 0.818363 | 0.89 | 0.00155086 | 0.00155086 | 4.96271 |
| drop | 14 | 15280 | 0.437402 | 0.0026967 | 23.6 | 0.698603 | 0.83 | 0.00578358 | 0.00578358 | 4.35852 |
| resample | 14 | 20000 | 0.437347 | 0.00271613 | 0 | 0.698603 | 0.84 | 0.00578358 | 0.00578358 | 5.80038 |
| baseline | 15 | 20000 | 0.438647 | 0.00211628 | 0 | 0.870259 | 0.89 | 0.000628882 | 0.000628882 | 5.2786 |
| drop | 15 | 15318 | 0.437731 | 0.00246882 | 23.41 | 0.720559 | 0.85 | 0.00616053 | 0.00616053 | 4.29388 |
| resample | 15 | 20000 | 0.43771 | 0.00250109 | 0 | 0.720559 | 0.86 | 0.00616053 | 0.00616053 | 5.84736 |
| baseline | 16 | 20000 | 0.438476 | 0.0021848 | 0 | 0.872255 | 0.89 | 0.00299642 | 0.00299642 | 5.24965 |
| drop | 16 | 14899 | 0.434302 | 0.00369477 | 25.505 | 0.413174 | 0.82 | 0.0151029 | 0.0151029 | 4.52809 |
| resample | 16 | 20000 | 0.434248 | 0.00372174 | 0 | 0.413174 | 0.83 | 0.0151029 | 0.0151029 | 5.66681 |
| baseline | 17 | 20000 | 0.438291 | 0.00228442 | 0 | 0.928144 | 0.88 | 0.00286984 | 0.00286984 | 5.20333 |
| drop | 17 | 15275 | 0.435913 | 0.0028094 | 23.625 | 0.532934 | 0.82 | 0.0152238 | 0.0152238 | 4.28493 |
| resample | 17 | 20000 | 0.435938 | 0.00278905 | 0 | 0.532934 | 0.79 | 0.0152238 | 0.0152238 | 5.88923 |
| baseline | 42 | 20000 | 0.438671 | 0.00208696 | 0 | 0.9002 | 0.87 | 0.00292527 | 0.00292527 | 5.03146 |
| drop | 42 | 15227 | 0.438356 | 0.00226678 | 23.865 | 0.706587 | 0.83 | 0.00292527 | 0.00292527 | 4.4081 |
| resample | 42 | 20000 | 0.438231 | 0.00233075 | 0 | 0.706587 | 0.85 | 0.00292527 | 0.00292527 | 5.54548 |
| baseline | 71 | 20000 | 0.438685 | 0.00210342 | 0 | 0.914172 | 0.87 | 0.000578203 | 0.000578203 | 5.68173 |
| drop | 71 | 15039 | 0.424879 | 0.00677532 | 24.805 | 0.383234 | 0.82 | 0.0409851 | 0.0409851 | 3.96679 |
| resample | 71 | 20000 | 0.424874 | 0.00677167 | 0 | 0.383234 | 0.82 | 0.0409851 | 0.0409851 | 5.26724 |
| baseline | 97 | 20000 | 0.438684 | 0.00210684 | 0 | 0.898204 | 0.88 | 0.00088085 | 0.00088085 | 4.96105 |
| drop | 97 | 15235 | 0.436855 | 0.00281453 | 23.825 | 0.546906 | 0.81 | 0.00892307 | 0.00892307 | 4.06082 |
| resample | 97 | 20000 | 0.436842 | 0.00282672 | 0 | 0.546906 | 0.83 | 0.00892307 | 0.00892307 | 5.00913 |

Algorithm aggregates (equal weight per seed):

| mode | n | mean_HV | mean_IGD_plus | mean_evaluations | mean_saved_percent | mean_seconds | mean_sensitive_fraction | mean_narrow_coverage |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | 20 | 0.43862 | 0.00212209 | 20000 | 0 | 5.55137 | 0.8805 | 0.884032 |
| resample | 20 | 0.435319 | 0.0031284 | 20000 | 0 | 6.02155 | 0.8285 | 0.602295 |
| drop | 20 | 0.435314 | 0.00313127 | 15243.2 | 23.7838 | 4.79489 | 0.8295 | 0.602196 |

## Peak discovery, retained solutions and actual rejection

Euclidean discovery radii are fixed at .03, .10 and .18 around (2.5,2.5). General-neighborhood discovery is not accurate approximation: consult closest evaluated/final endpoint distance and narrow-segment coverage per run above. Final Pareto counts are optimizer nondominated approximations, not a claim that off-diagonal designs are mathematically Pareto-optimal.

| mode | radius | runs_evaluated | evaluated | runs_retained | final_pareto | checked | rejected | accepted | original_rejected | replacement_rejected | rejected_outside |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | 0.03 | 20 | 1468 | 20 | 86 | 0 | 0 | 0 | 0 | 0 | 0 |
| baseline | 0.1 | 20 | 12931 | 20 | 517 | 0 | 0 | 0 | 0 | 0 | 0 |
| baseline | 0.18 | 20 | 23874 | 20 | 802 | 0 | 0 | 0 | 0 | 0 | 0 |
| resample | 0.03 | 17 | 151 | 17 | 52 | 2313 | 2012 | 301 | 2011 | 1 | 116556 |
| resample | 0.1 | 20 | 1315 | 20 | 343 | 17211 | 14968 | 2243 | 14950 | 18 | 103600 |
| resample | 0.18 | 20 | 2961 | 20 | 616 | 32405 | 27626 | 4779 | 27581 | 45 | 90942 |
| drop | 0.03 | 17 | 151 | 17 | 53 | 1109 | 958 | 151 | 958 | 0 | 94177 |
| drop | 0.1 | 20 | 1315 | 20 | 344 | 10804 | 9489 | 1315 | 9489 | 0 | 85646 |
| drop | 0.18 | 20 | 2961 | 20 | 615 | 21919 | 18959 | 2960 | 18959 | 0 | 76176 |

Rejected proposals outside each radius are also shown; these radii overlap and their totals must not be added. Peak rejection events, original inputs and replacement checks remain separate. Evidence for peak filtering is limited to the actual rejection counts, and any reduction in narrow coverage represents a loss of nominally efficient peak-region solutions. See `analysis/peak_classification_history.csv` for robust/stable, sensitive and NEI counts at each radius and generation from activation onward. Observed stability means sufficient sampled neighbors without a violation; NEI means insufficient evidence. Neither establishes worst-case robustness.

## Independently assessed sensitivity

For every final nondominated solution, intersect [x−.1,x+.1] with [−5,10]² and compute both objective minima/maxima. f1 is separable convex: its minimum is at the coordinatewise closest-to-zero point and maximum at a corner. For f2 enumerate all corners, edge stationary points and interior stationary points. Interior stationarity implies x1=x2=2.5+z, 0<z<2.5. On each edge the free-coordinate derivative has the form h(z)=5−2z−200/0.01 z exp(−[kz²+d²]/0.01), with k=1 for edges and k=2 on the diagonal. Its derivative's root equation is log(2u−1)−u=log(0.01/100)+d²/0.01, u=kz²/0.01. This function increases up to u=1.5 then decreases, so all critical points can be bracketed. Splitting h at them enumerates all roots; negative z and z>2.5 have no roots. The finite list is mathematically complete; numeric roots use double precision, without interval-certified rounding bounds.
Absolute S_j=max(|fmin−f(x)|,|fmax−f(x)|) is saved alongside the inherited symmetric relative deviation |a−b|/max(|a|,|b|,.001). Relative maxima additionally consider b=−a and floor breakpoints inside the attained interval; checking just the absolute extrema is insufficient when f2 changes sign. Sensitive means either maximal relative deviation exceeds .1. All final fronts are assessed, rather than a favorable subsample.
For archive/proposal comparison, systematically select at most 31 near-peak evaluated archive points and 31 near-peak rejected events per run by observed order, without selecting by sensitivity. These are diagnostic samples and cannot represent all rejected/evaluated designs. Rejected coordinates are assessed only posthoc and were never evaluated by the optimization. The archive comparison is:

| mode | kind | flag | n | independently_sensitive_fraction |
| --- | --- | --- | --- | --- |
| baseline | evaluated | non-robust | 620 | 1 |
| drop | evaluated | non-robust | 620 | 1 |
| drop | rejected | unevaluated | 620 | 1 |
| resample | evaluated | non-robust | 620 | 1 |
| resample | rejected | unevaluated | 620 | 1 |

Independent full-box sensitivity is separate from archive-supported flags. Improved nominal metrics do not imply robustness, and removing sensitive nominal optima can reduce HV or narrow coverage. The median relative sensitivity and actual independently sensitive fractions are included in the statistical tables.

## Paired statistics

Means, medians, sample SD, quartiles and IQR for all metrics:

| mode | metric | n | mean | median | std | q25 | q75 | iqr |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | hypervolume | 20 | 0.43862 | 0.438652 | 0.000158432 | 0.438527 | 0.438691 | 0.000163767 |
| resample | hypervolume | 20 | 0.435319 | 0.436073 | 0.0033034 | 0.435095 | 0.437438 | 0.00234242 |
| drop | hypervolume | 20 | 0.435314 | 0.436063 | 0.00329739 | 0.435124 | 0.437451 | 0.00232713 |
| baseline | igd_plus | 20 | 0.00212209 | 0.00210513 | 7.59946e-05 | 0.00208515 | 0.00213221 | 4.7065e-05 |
| resample | igd_plus | 20 | 0.0031284 | 0.00280788 | 0.00103842 | 0.00239251 | 0.00349737 | 0.00110486 |
| drop | igd_plus | 20 | 0.00313127 | 0.00281196 | 0.00103735 | 0.00240536 | 0.0034903 | 0.00108495 |
| baseline | actual_model_calls | 20 | 20000 | 20000 | 0 | 20000 | 20000 | 0 |
| resample | actual_model_calls | 20 | 20000 | 20000 | 0 | 20000 | 20000 | 0 |
| drop | actual_model_calls | 20 | 15243.2 | 15275.5 | 131.119 | 15192 | 15318.5 | 126.5 |
| baseline | saved_percent | 20 | 0 | 0 | 0 | 0 | 0 | 0 |
| resample | saved_percent | 20 | 0 | 0 | 0 | 0 | 0 | 0 |
| drop | saved_percent | 20 | 23.7837 | 23.6225 | 0.655593 | 23.4075 | 24.04 | 0.6325 |
| baseline | mean_reference_distance | 20 | 0.000728972 | 0.000723951 | 0.00011556 | 0.000633948 | 0.000798364 | 0.000164417 |
| resample | mean_reference_distance | 20 | 0.00107184 | 0.00106351 | 0.000218172 | 0.000950455 | 0.0012282 | 0.000277747 |
| drop | mean_reference_distance | 20 | 0.00107559 | 0.00109303 | 0.000241567 | 0.000898991 | 0.00127779 | 0.000378798 |
| baseline | narrow_coverage | 20 | 0.884032 | 0.878244 | 0.0306626 | 0.867764 | 0.902196 | 0.0344311 |
| resample | narrow_coverage | 20 | 0.602295 | 0.611776 | 0.137067 | 0.463573 | 0.71008 | 0.246507 |
| drop | narrow_coverage | 20 | 0.602196 | 0.611776 | 0.136958 | 0.463573 | 0.71008 | 0.246507 |
| baseline | narrow_mean_distance | 20 | 0.00547229 | 0.00551742 | 0.000358822 | 0.00523071 | 0.00566276 | 0.000432055 |
| resample | narrow_mean_distance | 20 | 0.0121683 | 0.0117798 | 0.00486163 | 0.00832259 | 0.0156099 | 0.00728736 |
| drop | narrow_mean_distance | 20 | 0.0121705 | 0.0117798 | 0.00485882 | 0.00832259 | 0.0156099 | 0.00728736 |
| baseline | sensitive_audit_fraction | 20 | 0.8805 | 0.88 | 0.0143178 | 0.87 | 0.89 | 0.02 |
| resample | sensitive_audit_fraction | 20 | 0.8285 | 0.83 | 0.018432 | 0.82 | 0.84 | 0.02 |
| drop | sensitive_audit_fraction | 20 | 0.8295 | 0.83 | 0.0139454 | 0.82 | 0.8325 | 0.0125 |
| baseline | R1_median | 20 | 0.0906976 | 0.0914133 | 0.00227836 | 0.0889698 | 0.0923474 | 0.00337763 |
| resample | R1_median | 20 | 0.0972383 | 0.0968264 | 0.0039352 | 0.0936014 | 0.100683 | 0.0070818 |
| drop | R1_median | 20 | 0.0972044 | 0.0970046 | 0.00402476 | 0.0937549 | 0.101037 | 0.00728165 |
| baseline | R2_median | 20 | 0.0711096 | 0.0683408 | 0.00920848 | 0.0677894 | 0.0700589 | 0.00226951 |
| resample | R2_median | 20 | 0.0656129 | 0.0657007 | 0.00183674 | 0.0640759 | 0.0670831 | 0.00300723 |
| drop | R2_median | 20 | 0.0655986 | 0.0656441 | 0.00183831 | 0.0638918 | 0.0670758 | 0.00318398 |
| baseline | run_seconds | 20 | 5.55137 | 5.44332 | 0.511725 | 5.19898 | 5.96754 | 0.768556 |
| resample | run_seconds | 20 | 6.02155 | 5.78379 | 0.661346 | 5.64102 | 6.37498 | 0.733966 |
| drop | run_seconds | 20 | 4.79489 | 4.64485 | 0.623284 | 4.29526 | 5.12849 | 0.833232 |

Paired differences are B minus A across the same 20 seeds. Confidence intervals are percentile 95% paired-bootstrap intervals from 10,000 fixed-seed resamples of seed pairs. Two-sided Wilcoxon signed-rank tests use SciPy's automatic method, retaining ties/zeros; Holm adjustment is across all 33 metric/comparison tests in this report. Bootstrap intervals are pointwise, not simultaneous. Failure to reject does not establish equivalence.

| metric | method_a | method_b | n_pairs | mean_difference | ci95_low | ci95_high | p_raw | p_holm |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| hypervolume | baseline | resample | 20 | -0.0033009 | -0.00487537 | -0.00202578 | 3.8147e-06 | 8.7738e-05 |
| hypervolume | baseline | drop | 20 | -0.00330614 | -0.00487353 | -0.00203385 | 3.8147e-06 | 8.7738e-05 |
| hypervolume | resample | drop | 20 | -5.24427e-06 | -3.40838e-05 | 2.19067e-05 | 0.985435 | 1 |
| igd_plus | baseline | resample | 20 | 0.00100631 | 0.000611074 | 0.00152107 | 3.8147e-06 | 8.7738e-05 |
| igd_plus | baseline | drop | 20 | 0.00100918 | 0.000617251 | 0.0015212 | 3.8147e-06 | 8.7738e-05 |
| igd_plus | resample | drop | 20 | 2.86838e-06 | -1.10598e-05 | 1.77194e-05 | 0.985435 | 1 |
| actual_model_calls | baseline | resample | 20 | 0 | 0 | 0 | 1 | 1 |
| actual_model_calls | baseline | drop | 20 | -4756.75 | -4817.2 | -4702.85 | 1.90735e-06 | 6.29425e-05 |
| actual_model_calls | resample | drop | 20 | -4756.75 | -4817.2 | -4702.85 | 1.90735e-06 | 6.29425e-05 |
| saved_percent | baseline | resample | 20 | 0 | 0 | 0 | 1 | 1 |
| saved_percent | baseline | drop | 20 | 23.7837 | 23.5142 | 24.086 | 1.90735e-06 | 6.29425e-05 |
| saved_percent | resample | drop | 20 | 23.7837 | 23.5142 | 24.086 | 1.90735e-06 | 6.29425e-05 |
| mean_reference_distance | baseline | resample | 20 | 0.000342865 | 0.000239756 | 0.000445339 | 9.53674e-06 | 0.000181198 |
| mean_reference_distance | baseline | drop | 20 | 0.00034662 | 0.000238659 | 0.000456902 | 1.90735e-05 | 0.000305176 |
| mean_reference_distance | resample | drop | 20 | 3.75517e-06 | -2.64927e-05 | 2.92841e-05 | 0.261099 | 1 |
| narrow_coverage | baseline | resample | 20 | -0.281737 | -0.343016 | -0.221756 | 1.90735e-06 | 6.29425e-05 |
| narrow_coverage | baseline | drop | 20 | -0.281836 | -0.343019 | -0.222053 | 8.84492e-05 | 0.00126941 |
| narrow_coverage | resample | drop | 20 | -9.98004e-05 | -0.000299401 | 0 | 0.317311 | 1 |
| narrow_mean_distance | baseline | resample | 20 | 0.00669598 | 0.00472363 | 0.00889605 | 1.90735e-06 | 6.29425e-05 |
| narrow_mean_distance | baseline | drop | 20 | 0.00669821 | 0.00473022 | 0.00889568 | 1.90735e-06 | 6.29425e-05 |
| narrow_mean_distance | resample | drop | 20 | 2.23721e-06 | -5.26114e-06 | 1.27389e-05 | 1 | 1 |
| sensitive_audit_fraction | baseline | resample | 20 | -0.052 | -0.063 | -0.042 | 8.46275e-05 | 0.00126941 |
| sensitive_audit_fraction | baseline | drop | 20 | -0.051 | -0.06 | -0.0415 | 0.000126535 | 0.00164495 |
| sensitive_audit_fraction | resample | drop | 20 | 0.001 | -0.0055 | 0.008 | 0.923441 | 1 |
| R1_median | baseline | resample | 20 | 0.00654072 | 0.00467794 | 0.00837244 | 1.33514e-05 | 0.000226974 |
| R1_median | baseline | drop | 20 | 0.00650676 | 0.00465189 | 0.00837427 | 1.90735e-06 | 6.29425e-05 |
| R1_median | resample | drop | 20 | -3.39571e-05 | -0.000771548 | 0.000628896 | 0.728506 | 1 |
| R2_median | baseline | resample | 20 | -0.00549674 | -0.0100573 | -0.00254165 | 9.53674e-06 | 0.000181198 |
| R2_median | baseline | drop | 20 | -0.00551098 | -0.00997003 | -0.00261311 | 1.90735e-06 | 6.29425e-05 |
| R2_median | resample | drop | 20 | -1.42445e-05 | -0.000371555 | 0.000361851 | 0.728506 | 1 |
| run_seconds | baseline | resample | 20 | 0.470176 | 0.219275 | 0.741978 | 0.00120926 | 0.0133018 |
| run_seconds | baseline | drop | 20 | -0.756479 | -1.00654 | -0.488862 | 0.000133514 | 0.00164495 |
| run_seconds | resample | drop | 20 | -1.22666 | -1.38638 | -1.0568 | 1.90735e-06 | 6.29425e-05 |

## Actual evaluation budgets, targets and overhead

`analysis/common_evaluation_budgets.csv` compares all 20 seeds at common budgets up to 14899 evaluations. Only the latest completed generation at or below each budget is used (step hold); unused budget is recorded. This does not claim interpolation or exact partial-generation trajectories. Figures show seed medians and IQR, retaining the full cohort over the shared domain. Fixed-generation final results are separately reported above.
Reference-derived HV targets (99%, 99.5%, 99.9%) were locked before optimization. Unattained targets are right-censored at each run's actual evaluation count; they are never replaced with a successful attainment time. Attainment counts:

| mode | target_fraction | reached | n |
| --- | --- | --- | --- |
| baseline | 0.99 | 20 | 20 |
| baseline | 0.995 | 0 | 20 |
| baseline | 0.999 | 0 | 20 |
| drop | 0.99 | 1 | 20 |
| drop | 0.995 | 0 | 20 |
| drop | 0.999 | 0 | 20 |
| resample | 0.99 | 2 | 20 |
| resample | 0.995 | 0 | 20 |
| resample | 0.999 | 0 | 20 |

Mean timing in seconds:

| mode | objective_seconds | classification_seconds | indicator_seconds | run_seconds | posthoc_seconds |
| --- | --- | --- | --- | --- | --- |
| baseline | 0.424135 | 3.61737 | 0.307777 | 5.55137 | 0.732677 |
| resample | 0.349049 | 3.65449 | 0.313202 | 6.02155 | 0.895349 |
| drop | 0.262137 | 2.77485 | 0.305863 | 4.79489 | 0.730132 |

Wall time includes instrumented optimization, archive classification from activation for all three algorithms, indicators, proposal logging and operators. Posthoc seconds include final archive classification and independent sensitivity/export work. Two worker processes share this machine; timings characterize this instrumented experiment, not isolated production speed. Fewer objective evaluations alone do not demonstrate a wall-clock advantage for this inexpensive analytical model.

## Figures and retained data

`figures/01` through `11` provide objective surfaces (full domain and peak detail), independent reference, all-seed final fronts, seed-42 archive classifications, all-seed peak zoom, independent sensitive points with actual rejected events, HV/evaluations, IGD+/evaluations, final indicator distributions, rejection histories and final-front sensitivity. Both PNG (320 dpi) and vector PDF with rasterized dense point clouds are supplied. Seed 42 is a predefined paired seed, used only for the full-domain classification illustration; metrics and the other comparisons use all 20 seeds.
Each run retains history.csv, process.csv, archive.csv, pareto.csv, final_population.csv, sensitivity.csv, evaluated.csv.gz, screening.csv.gz, front_history.npz, peak_classification_history.csv, settings.json and summary.json. `analysis/verification.csv` verifies all 60 counts, 200-generation completion and recomputed indicators. Software versions, source hashes and source base commit are recorded in environment.json.

## Scientific limits and conclusions

NSGA-II evaluated and retained radius-.03 peak designs in 20/20 runs. ALSF-Resample and ALSF-Drop did so in 17/20 and 17/20 runs. Drop recorded 958 actual radius-.03 pre-evaluation rejections. It saved 23.78% of optimization evaluations on average, while mean HV fell from 0.438620 to 0.435314 and mean narrow-segment coverage fell from 0.884 to 0.602. The independently sensitive final-solution fraction fell from 0.8805 to 0.8295, but remained high. The experiment therefore demonstrates discovery and actual sensitivity-driven rejection, together with a measurable loss of nominal quality; it does not establish an acceptable application tradeoff or universal robustness.

Discovery, sufficient archive evidence and actual pre-evaluation rejection are different questions. The tables quantify all three and permit negative findings. Neighborhood exploration alone is insufficient evidence of accurate peak approximation; archive robust labels are not certified robustness. This study supports only the fixed w=.01 problem and protocol. No Gaussian-width tuning or second-stage experiment was performed. A separate predeclared w∈{.01,.04,.10} study could investigate spatial extent, but must retain its own output and paired protocol. Nominal-quality losses and independent sensitivity fractions must be considered together when judging whether evaluation savings are acceptable; no application-specific acceptable-loss threshold was predefined, so acceptability cannot be claimed.

## Reproduction

From this branch, install requirements.txt (the exact tested package versions are in environment.json). Use a fresh directory; preparation refuses to overwrite a locked protocol:

```sh
python prepare_modified_gaussian.py --root results/test1_modified_gaussian_reproduction
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python run_test1_modified_gaussian.py --root results/test1_modified_gaussian_reproduction --workers 2
python analyze_modified_gaussian.py --root results/test1_modified_gaussian_reproduction
python -m pytest tests/test_modified_gaussian.py tests/test_offspring_drop.py tests/test_local_filter.py
```

Completed runs are resumed only when requested settings match. A partially written run directory is deliberately not silently overwritten; move that incomplete directory aside and rerun the identical protocol. Do not remove a completed unsuccessful run. The JCAM manuscript is unchanged; suggested_subsection.tex is a separate conservative insertion proposal.
