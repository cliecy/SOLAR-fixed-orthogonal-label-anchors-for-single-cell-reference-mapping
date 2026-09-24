"""Original biological metric definitions; workspace discovery is intentionally omitted.

The public scorer supplies explicit arrays and frozen method-independent samples.
See source_file_mapping.json for the original file identity and hash.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors

RNG_SEED = 0
MAX_SILHOUETTE_CELLS = 8000


def within_label_silhouette(embedding, labels, groups, rng):
    scores, weights = [], []
    for label in pd.unique(labels):
        mask = labels == label
        n = int(mask.sum())
        if n < 10:
            continue
        sub_groups = groups[mask]
        if pd.unique(sub_groups).size < 2:
            continue
        sub_embedding = embedding[mask]
        if n > MAX_SILHOUETTE_CELLS:
            idx = rng.choice(n, size=MAX_SILHOUETTE_CELLS, replace=False)
            sub_embedding = sub_embedding[idx]
            sub_groups = sub_groups[idx]
            if pd.unique(sub_groups).size < 2:
                continue
            n = MAX_SILHOUETTE_CELLS
        try:
            score = silhouette_score(sub_embedding, sub_groups)
        except ValueError:
            continue
        scores.append(score)
        weights.append(n)
    if not scores:
        return None
    return float(np.average(scores, weights=weights)), len(scores)


def pseudotime_smoothness(embedding, pseudotime, k):
    valid = ~np.isnan(pseudotime)
    if valid.sum() < k + 1:
        return None
    embedding = embedding[valid]
    pseudotime = pseudotime[valid]
    k_eff = min(k, len(embedding) - 1)
    if k_eff < 1:
        return None
    nn = NearestNeighbors(n_neighbors=k_eff + 1).fit(embedding)
    _, indices = nn.kneighbors(embedding)
    neighbor_mean_pt = pseudotime[indices[:, 1:]].mean(axis=1)
    rho, _ = spearmanr(pseudotime, neighbor_mean_pt)
    return float(rho)
