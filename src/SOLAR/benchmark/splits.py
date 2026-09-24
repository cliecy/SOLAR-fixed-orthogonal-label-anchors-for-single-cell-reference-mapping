from __future__ import annotations

import anndata as ad
import numpy as np
from scipy import sparse
from sklearn.decomposition import PCA


def _strata(
    adata: ad.AnnData, labels_key: str, batch_key: str | None
) -> np.ndarray:
    if labels_key not in adata.obs:
        raise KeyError(f"labels_key {labels_key!r} not found in reference.obs")
    labels = adata.obs[labels_key].astype(str).to_numpy(dtype=str)
    if batch_key is None:
        return labels
    if batch_key not in adata.obs:
        raise KeyError(f"batch_key {batch_key!r} not found in reference.obs")
    batches = adata.obs[batch_key].astype(str).to_numpy(dtype=str)
    return np.char.add(np.char.add(batches, "||"), labels)


def stratified_holdout(
    adata: ad.AnnData,
    labels_key: str,
    seed: int,
    test_fraction: float = 0.2,
    batch_key: str | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Create a deterministic label(/batch)-stratified train/query split."""
    if not 0 < test_fraction < 1:
        raise ValueError("test_fraction must lie in (0, 1)")
    rng = np.random.default_rng(seed)
    strata = _strata(adata, labels_key, batch_key)
    train: list[int] = []
    query: list[int] = []
    for value in sorted(set(strata.tolist())):
        indices = np.flatnonzero(strata == value)
        rng.shuffle(indices)
        n_query = max(1, int(round(len(indices) * test_fraction)))
        if n_query >= len(indices):
            n_query = len(indices) - 1
        if n_query <= 0:
            raise ValueError(f"Stratum {value!r} has fewer than two cells")
        query.extend(indices[:n_query].tolist())
        train.extend(indices[n_query:].tolist())
    return np.asarray(sorted(train)), np.asarray(sorted(query))


def batch_holdout_split(
    adata: ad.AnnData,
    held_out_batch: str,
    batch_key: str = "batch",
) -> tuple[np.ndarray, np.ndarray]:
    """Split by batch without consulting any cell-type labels."""
    if batch_key not in adata.obs:
        raise KeyError(f"Batch holdout requires obs[{batch_key!r}]")
    batches = adata.obs[batch_key].astype(str).to_numpy()
    query = np.flatnonzero(batches == str(held_out_batch))
    train = np.flatnonzero(batches != str(held_out_batch))
    if not len(train) or not len(query):
        raise ValueError(f"Invalid held-out batch {held_out_batch!r}")
    return train, query


def lobo_split(
    adata: ad.AnnData,
    held_out_batch: str,
    labels_key: str,
    batch_key: str = "batch",
) -> tuple[np.ndarray, np.ndarray]:
    """Leave one batch out and enforce a closed-set label-transfer task."""
    if labels_key not in adata.obs:
        raise KeyError(f"LOBO requires obs[{labels_key!r}]")
    train, query = batch_holdout_split(adata, held_out_batch, batch_key)
    train_types = set(adata.obs.iloc[train][labels_key].astype(str))
    query_types = set(adata.obs.iloc[query][labels_key].astype(str))
    if not query_types <= train_types:
        raise ValueError(
            "LOBO is not closed-set; unseen query types: "
            f"{sorted(query_types - train_types)}"
        )
    return train, query


def subsample_labels(
    reference: ad.AnnData,
    labels_key: str,
    fraction: float,
    seed: int,
    batch_key: str | None = None,
) -> np.ndarray:
    """Select a deterministic, stratified label budget from reference cells."""
    if not 0 < fraction <= 1:
        raise ValueError("label fraction must lie in (0, 1]")
    if fraction == 1:
        return np.arange(reference.n_obs)
    rng = np.random.default_rng(seed)
    strata = _strata(reference, labels_key, batch_key)
    selected: list[int] = []
    for value in sorted(set(strata.tolist())):
        local = np.flatnonzero(strata == value)
        rng.shuffle(local)
        count = max(1, int(round(len(local) * fraction)))
        selected.extend(local[:count].tolist())
    return np.asarray(sorted(selected))


def expression_matrix(
    adata: ad.AnnData, source: str = "X", key: str | None = None
) -> np.ndarray:
    if source == "X":
        if key is not None:
            raise ValueError("source_key must be omitted when source='X'")
        matrix = adata.X
    elif source == "layer":
        if key is None or key not in adata.layers:
            raise KeyError(f"Layer {key!r} not found")
        matrix = adata.layers[key]
    elif source == "obsm":
        if key is None or key not in adata.obsm:
            raise KeyError(f"obsm representation {key!r} not found")
        matrix = adata.obsm[key]
    else:
        raise ValueError("source must be one of: X, layer, obsm")
    if sparse.issparse(matrix):
        matrix = matrix.toarray()
    matrix = np.asarray(matrix, dtype=np.float32)
    if matrix.ndim != 2:
        raise ValueError("Input expression representation must be two-dimensional")
    if not np.isfinite(matrix).all():
        raise ValueError("Input expression representation contains non-finite values")
    return matrix


def fit_reference_pca(
    reference: ad.AnnData,
    query: ad.AnnData,
    n_components: int,
    seed: int,
    source: str = "X",
    source_key: str | None = None,
) -> tuple[np.ndarray, np.ndarray, PCA]:
    """Fit PCA on reference only, then transform reference and held-out query."""
    reference_x = expression_matrix(reference, source, source_key)
    query_x = expression_matrix(query, source, source_key)
    if reference_x.shape[1] != query_x.shape[1]:
        raise ValueError(
            "Reference and query feature dimensions differ: "
            f"{reference_x.shape[1]} != {query_x.shape[1]}"
        )
    n_components = min(
        int(n_components), reference_x.shape[0] - 1, reference_x.shape[1]
    )
    if n_components <= 0:
        raise ValueError("Reference needs at least two cells and one feature")
    pca = PCA(n_components=n_components, random_state=seed)
    reference_z = pca.fit_transform(reference_x).astype(np.float32)
    query_z = pca.transform(query_x).astype(np.float32)
    return reference_z, query_z, pca
