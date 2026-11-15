"""Pavel review variant: tolerance-normalized Chebyshev DBSCAN and LOCAL flags.

DBSCAN is used to visualize density-connected components, *not* to assert
stability across the entire chain. Local objective deviations are checked only
for archived pairs inside the same prescribed tolerance hyperrectangle.
Greedy medoid trimming and its optimistic boundary bias are removed entirely.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from src.local_variation import LocalVariationState, tolerance_vector


def decision_columns(df):
    cols = sorted((c for c in df if c.startswith("X") and c[1:].isdigit()),
                  key=lambda c: int(c[1:]))
    if not cols or cols != [f"X{i}" for i in range(1, len(cols) + 1)]:
        raise ValueError("Archive must contain X1..Xn consecutively")
    return cols


def classify_dataframe(archive: pd.DataFrame, tolerances, t1, t2,
                       min_samples=10, radius=1.0, floors=(1e-3, 1e-3),
                       state: LocalVariationState | None = None, *,
                       dbscan_eps=1.0, dbscan_min_samples=10,
                       partition_width=1.0):
    """Classify each archive point using observed L-infinity local evidence.

    robust: min_samples observations (including self) in its own tolerance neighborhood and NO
            observed pairwise objective-threshold violation at this point.
    non-robust: endpoint of at least one observed violating pair, regardless of
                whether DBSCAN considers the endpoint core, border or noise.
    NEI: not enough direct local observations to claim robustness.

    Importantly, these are archive-supported labels, not worst-case guarantees.
    A single DBSCAN cluster can contain robust and non-robust points.
    """
    archive = archive.reset_index(drop=True).copy()
    cols = decision_columns(archive)
    X = archive[cols].to_numpy(dtype=float)
    F = archive[['Obj1','Obj2']].to_numpy(dtype=float)
    delta = tolerance_vector(tolerances, len(cols))
    if not np.isfinite(min_samples) or int(min_samples) != min_samples or min_samples < 2:
        raise ValueError("min_samples must be at least 2")
    if state is None:
        state = LocalVariationState(delta, thresholds=(t1, t2), floors=floors,
                                    radius=radius)
    # A reused incremental cache must describe exactly the requested evidence.
    if (not np.array_equal(tolerance_vector(state.tolerances, len(cols)), delta)
            or state.radius != radius
            or not np.array_equal(state.thresholds, [t1, t2])
            or not np.array_equal(state.floors, floors)):
        raise ValueError("LocalVariationState parameters differ from requested sensitivity policy")
    if not np.isfinite(dbscan_eps) or dbscan_eps <= 0:
        raise ValueError("dbscan_eps must be finite and positive")
    if int(dbscan_min_samples) != dbscan_min_samples or dbscan_min_samples < 1:
        raise ValueError("dbscan_min_samples must be a positive integer")
    if not np.isfinite(partition_width) or partition_width <= 0:
        raise ValueError("partition_width must be finite and positive")
    counts = state.update(X, F)
    normalized = X / delta
    labels = (DBSCAN(eps=dbscan_eps + 1e-12, min_samples=dbscan_min_samples,
                     metric='chebyshev', algorithm='auto').fit_predict(normalized)
              if len(X) else np.zeros(0, dtype=int))
    bad = state.verified_violation
    flag = np.full(len(X), 'NEI', dtype=object)
    flag[(counts >= min_samples) & ~bad] = 'robust'
    flag[bad] = 'non-robust'
    archive['flag'] = flag
    archive['cluster_id'] = labels
    archive['is_medoid'] = False
    archive['cluster_pass'] = np.where(bad, 4, np.where(flag == 'robust', 1, 0))
    archive['verified_violation'] = bad
    archive['local_n'] = counts
    archive['R1_local'] = state.max_observed_variation[:, 0]
    archive['R2_local'] = state.max_observed_variation[:, 1]
    archive['cluster_n'] = 0

    # Connectivity summaries never propagate local sensitivity labels.
    flags_by_cluster = {}
    archive['cluster_status'] = 'noise'
    for category in ('robust', 'non-robust', 'NEI'):
        archive['cluster_fraction_' + category.replace('-', '_').lower()] = np.nan
    for cid in np.unique(labels[labels >= 0]):
        indices = np.flatnonzero(labels == cid)
        archive.loc[indices, 'cluster_n'] = len(indices)
        statuses = set(flag[indices])
        status = next(iter(statuses)) if len(statuses) == 1 else 'mixed'
        flags_by_cluster[int(cid)] = status
        archive.loc[indices, 'cluster_status'] = status
        for category in ('robust', 'non-robust', 'NEI'):
            archive.loc[indices, 'cluster_fraction_' + category.replace('-', '_').lower()] = np.mean(flag[indices] == category)
        if status == 'robust':
            members = normalized[indices]
            center = members.mean(axis=0)
            candidate = indices[np.argmin(np.max(np.abs(members-center), axis=1))]
            archive.at[candidate, 'is_medoid'] = True

    # Fixed, origin-anchored half-open cells give diameter <= partition_width
    # in normalized L-infinity distance, including DBSCAN noise. No merging,
    # target count, or objective evaluations. These are reporting regions only.
    cells = np.floor(normalized / partition_width).astype(np.int64)
    _, region_ids = np.unique(cells, axis=0, return_inverse=True)
    archive['region_id'] = region_ids
    region_status = np.full(len(X), 'NEI', dtype=object)
    representatives = np.zeros(len(X), dtype=bool)
    for indices in archive.groupby('region_id', sort=False).indices.values():
        statuses = set(flag[indices])
        region_status[indices] = next(iter(statuses)) if len(statuses) == 1 else 'mixed'
        stable = indices[flag[indices] == 'robust']
        if len(stable):
            center = normalized[stable].mean(axis=0)
            candidate = stable[np.argmin(np.max(np.abs(normalized[stable]-center), axis=1))]
            representatives[candidate] = True
    archive['region_status'] = region_status
    archive['is_stable_representative'] = representatives

    stats = {
        'n_clusters': len(flags_by_cluster),
        'n_robust_clusters': sum(v == 'robust' for v in flags_by_cluster.values()),
        'n_non_robust_clusters': sum(v == 'non-robust' for v in flags_by_cluster.values()),
        'n_nei_clusters': sum(v == 'NEI' for v in flags_by_cluster.values()),
        'n_mixed_clusters': sum(v == 'mixed' for v in flags_by_cluster.values()),
        'largest_cluster': int(archive.cluster_n.max()) if len(X) else 0,
        'n_regions': len(np.unique(region_ids)),
        'n_mixed_regions': int(archive.groupby('region_id').region_status.first().eq('mixed').sum()),
        'n_noise': int(np.count_nonzero(labels == -1)),
        'n_verified_violations': int(bad.sum()),
    }
    return archive, stats


def flag_by_clusters(archive_path, d_cluster, t1, t2, min_samples=10,
                     max_trim_fraction=None, tolerances=None, floors=(1e-3, 1e-3),
                     state=None, *, local_radius=1.0, local_min_samples=10,
                     partition_width=1.0):
    """Compatibility entry point; d_cluster is DBSCAN radius in tolerance units.

    min_samples controls DBSCAN only; local_min_samples controls evidence.
    Local radius defaults to 1 regardless of DBSCAN settings.

    max_trim_fraction is retained in the signature for old callers, but greedy
    trimming is NOT performed: it could create false-safe cluster boundaries.
    The tolerance vector is mandatory to avoid silently reverting to raw L2.
    """
    if tolerances is None:
        raise ValueError("tolerances must be provided explicitly in the L-infinity version")
    archive = pd.read_csv(archive_path)
    archive, stats = classify_dataframe(
        archive, tolerances, t1, t2, local_min_samples, radius=local_radius,
        floors=floors, state=state, dbscan_eps=float(d_cluster),
        dbscan_min_samples=min_samples, partition_width=partition_width,
    )
    archive.to_csv(archive_path, index=False)
    print(f"Local L-inf filter ({len(archive)} evaluations, {stats['n_clusters']} DBSCAN components): "
          f"{stats['n_verified_violations']} verified-violation endpoints, "
          f"{int((archive.flag == 'robust').sum())} locally stable samples, "
          f"{int((archive.flag == 'NEI').sum())} NEI")
    return archive, stats


def robust_cluster_medoids(archive_path):
    archive = pd.read_csv(archive_path)
    if 'flag' not in archive or 'is_medoid' not in archive:
        return pd.DataFrame()
    return archive[(archive.flag == 'robust') & archive.is_medoid][
        decision_columns(archive) + ['Obj1','Obj2']].reset_index(drop=True)
