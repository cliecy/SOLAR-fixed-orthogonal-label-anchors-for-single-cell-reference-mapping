from __future__ import annotations

from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd


def read_split(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype={"barcode": str, "batch": str, "label": str})
    required = {"barcode", "batch", "label", "role"}
    missing = required - set(frame)
    if missing:
        raise ValueError(f"Split file lacks columns: {', '.join(sorted(missing))}")
    if frame["barcode"].duplicated().any():
        raise ValueError("Split file contains duplicate barcodes")
    roles = set(frame["role"])
    if roles != {"reference", "query"}:
        raise ValueError(f"Split roles must be reference/query, got {sorted(roles)}")
    return frame


def read_gene_list(path: Path | None) -> list[str] | None:
    if path is None:
        return None
    values = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
    values = [value for value in values if value]
    if not values or len(values) != len(set(values)):
        raise ValueError(f"Gene list must be non-empty and unique: {path}")
    return values


def load_reference_query(
    dataset_path: Path,
    split: pd.DataFrame,
    genes: list[str] | None = None,
) -> tuple[ad.AnnData, ad.AnnData]:
    """Load one reference/query pair without loading unrelated cells twice."""
    backed = ad.read_h5ad(dataset_path, backed="r")
    try:
        obs_names = pd.Index(backed.obs_names.astype(str))
        if not obs_names.is_unique:
            raise ValueError("Dataset obs_names must be unique")
        var_names = pd.Index(backed.var_names.astype(str))
        if not var_names.is_unique:
            raise ValueError("Dataset var_names must be unique")
        reference_names = split.loc[split["role"] == "reference", "barcode"]
        query_names = split.loc[split["role"] == "query", "barcode"]
        reference_positions = obs_names.get_indexer(reference_names)
        query_positions = obs_names.get_indexer(query_names)
        if np.any(reference_positions < 0) or np.any(query_positions < 0):
            raise KeyError("Split contains barcodes absent from dataset")
        if genes is None:
            gene_positions: slice | np.ndarray = slice(None)
        else:
            gene_positions = var_names.get_indexer(genes)
            if np.any(gene_positions < 0):
                missing = [genes[i] for i in np.flatnonzero(gene_positions < 0)[:10]]
                raise KeyError(f"Gene list contains genes absent from dataset: {missing}")
            if np.any(np.diff(gene_positions) < 0):
                raise ValueError("Gene list must follow dataset var_names order")
        reference = backed[reference_positions, gene_positions].to_memory()
        query = backed[query_positions, gene_positions].to_memory()
    finally:
        backed.file.close()
    if not np.array_equal(reference.var_names.astype(str), query.var_names.astype(str)):
        raise RuntimeError("Reference/query genes differ after loading")
    if set(reference.obs_names.astype(str)) & set(query.obs_names.astype(str)):
        raise RuntimeError("Reference/query barcode overlap after loading")
    return reference, query


def load_role(
    dataset_path: Path,
    split: pd.DataFrame,
    role: str,
    genes: list[str] | None = None,
) -> ad.AnnData:
    """Load exactly one split role; used to keep query expression out of fitting."""
    if role not in {"reference", "query"}:
        raise ValueError("role must be reference or query")
    selected = split.loc[split["role"] == role, "barcode"]
    if selected.empty:
        raise ValueError(f"Split contains no {role} cells")
    backed = ad.read_h5ad(dataset_path, backed="r")
    try:
        obs_names = pd.Index(backed.obs_names.astype(str))
        var_names = pd.Index(backed.var_names.astype(str))
        positions = obs_names.get_indexer(selected)
        if np.any(positions < 0):
            raise KeyError(f"Split contains {role} barcodes absent from dataset")
        if genes is None:
            gene_positions: slice | np.ndarray = slice(None)
        else:
            gene_positions = var_names.get_indexer(genes)
            if np.any(gene_positions < 0):
                raise KeyError("Gene list contains genes absent from dataset")
            if np.any(np.diff(gene_positions) < 0):
                raise ValueError("Gene list must follow dataset var_names order")
        result = backed[positions, gene_positions].to_memory()
    finally:
        backed.file.close()
    return result


def select_reference_hvgs(
    dataset_path: Path,
    split: pd.DataFrame,
    batch_key: str,
    n_top_genes: int,
    flavor: str = "cell_ranger",
) -> list[str]:
    """Select batch-aware HVGs from reference cells only."""
    if n_top_genes <= 0:
        raise ValueError("n_top_genes must be positive")
    reference = load_role(dataset_path, split, "reference")
    if batch_key not in reference.obs:
        raise KeyError(f"batch_key {batch_key!r} missing from reference.obs")
    if reference.n_vars <= n_top_genes:
        return reference.var_names.astype(str).tolist()
    import scanpy as sc

    sc.pp.highly_variable_genes(
        reference,
        n_top_genes=n_top_genes,
        flavor=flavor,
        batch_key=batch_key,
        subset=False,
        inplace=True,
    )
    selected = reference.var_names[reference.var["highly_variable"].to_numpy()].astype(str)
    if not len(selected):
        raise RuntimeError("HVG selection returned no genes")
    return selected.tolist()
