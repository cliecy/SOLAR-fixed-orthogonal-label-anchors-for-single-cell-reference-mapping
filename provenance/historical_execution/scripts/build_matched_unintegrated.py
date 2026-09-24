#!/usr/bin/env python3
"""Rebuild the 'unintegrated' baseline so it uses EXACTLY SOLAR's own
preprocessing: the same reference-only batch-aware HVG2000 gene list and the
same reference-fitted 40D PCA transform (components/mean) already saved per
split as pca_preprocessor.npz — just skip the SOLAR encoder. This replaces
the earlier unintegrated_pca30/50 (raw whole-dataset PCA, no HVG, not
reference-fit), which did not match the reviewer's requirement.

For each dataset, loads the source h5ad ONCE (backed), then for every split
(core seeds 40-44, and the 15 selected heldout_* splits) subsets to that
split's HVG gene list and applies the ALREADY-FITTED PCA transform saved
alongside any core-track SOLAR job for that split (components/mean are
identical across SOLAR methods for a fixed dataset+split, verified).

Writes embedding.npz (reference+query, matching artifacts/*/embedding.npz
schema) under outputs/benchmark/solar_scib_rebuttal_v1/unintegrated_matched/
<dataset_id>/<split_id>/embedding.npz -- registration into the scib scoring
pipeline is a separate step (register_baseline_embedding.py).
"""
from __future__ import annotations

import glob
import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

RUN_ROOT = Path("outputs/benchmark/solar_scib_rebuttal_v1")
OUT_ROOT = RUN_ROOT / "unintegrated_matched"

DATASETS = {
    "pancreas": "data/scib/raw/human_pancreas_norm_complexBatch.h5ad",
    "immune_cell_human": "data/scib/raw/Immune_ALL_human.h5ad",
    "lung_atlas": "data/scib/raw/Lung_atlas_public.h5ad",
    "simulation_1": "data/scib/raw/sim1_1_norm.h5ad",
    "simulation_2": "data/scib/raw/sim2_norm.h5ad",
}


def find_pca_preprocessor(dataset_id: str, split_id: str) -> Path | None:
    pattern = str(
        RUN_ROOT / "artifacts" / "*" / dataset_id / "hvg2000_pca40" / "*"
        / split_id / "seed_*" / "anchor_0" / "labels_1p00" / "pca_preprocessor.npz"
    )
    matches = glob.glob(pattern)
    return Path(matches[0]) if matches else None


def process_split(adata: ad.AnnData, X_dense: np.ndarray, dataset_id: str, split_id: str) -> bool:
    gene_file = RUN_ROOT / "preprocessing" / dataset_id / split_id / "hvg2000_pca40.txt"
    pca_file = find_pca_preprocessor(dataset_id, split_id)
    split_file = RUN_ROOT / "splits" / dataset_id / split_id / "split.csv"
    if not (gene_file.is_file() and pca_file is not None and split_file.is_file()):
        print(f"SKIP {dataset_id}/{split_id}: missing gene list, pca preprocessor, or split")
        return False

    genes = [g.strip() for g in gene_file.read_text().splitlines() if g.strip()]
    split = pd.read_csv(split_file, dtype={"barcode": str}).set_index("barcode")
    with np.load(pca_file) as pca:
        components = np.asarray(pca["components"], dtype=np.float32)
        mean = np.asarray(pca["mean"], dtype=np.float32)

    var_index = pd.Index(adata.var_names.astype(str))
    gene_pos = var_index.get_indexer(genes)
    if np.any(gene_pos < 0):
        raise KeyError(f"{dataset_id}/{split_id}: HVG gene(s) absent from source file")

    out_dir = OUT_ROOT / dataset_id / split_id
    out_dir.mkdir(parents=True, exist_ok=True)

    reference_bc = split.index[split["role"] == "reference"].to_numpy()
    query_bc = split.index[split["role"] == "query"].to_numpy()
    obs_index = pd.Index(adata.obs_names.astype(str))
    ref_pos = obs_index.get_indexer(reference_bc)
    query_pos = obs_index.get_indexer(query_bc)
    if np.any(ref_pos < 0) or np.any(query_pos < 0):
        raise KeyError(f"{dataset_id}/{split_id}: split barcodes absent from source file")

    def transform(cell_pos: np.ndarray) -> np.ndarray:
        x = X_dense[cell_pos][:, gene_pos]
        return (x - mean) @ components.T

    reference_z = transform(ref_pos)
    query_z = transform(query_pos)

    np.savez_compressed(
        out_dir / "embedding.npz",
        reference=reference_z.astype(np.float32),
        query=query_z.astype(np.float32),
        reference_barcodes=reference_bc.astype(str),
        query_barcodes=query_bc.astype(str),
    )
    (out_dir / "provenance.json").write_text(
        json.dumps(
            {
                "dataset_id": dataset_id, "split_id": split_id,
                "gene_file": str(gene_file), "pca_preprocessor_source": str(pca_file),
                "n_genes": len(genes), "n_components": int(components.shape[0]),
                "note": "reference-fitted PCA components/mean reused verbatim from a "
                        "core-track SOLAR job for this split; no SOLAR encoder applied.",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"OK {dataset_id}/{split_id}: ref={len(reference_bc)} query={len(query_bc)} dim={components.shape[0]}")
    return True


def main() -> int:
    import sys
    only_dataset = sys.argv[1] if len(sys.argv) > 1 else None
    for dataset_id, h5ad_path in DATASETS.items():
        if only_dataset and dataset_id != only_dataset:
            continue
        adata = ad.read_h5ad(h5ad_path)
        x = adata.X
        X_dense = x.toarray() if hasattr(x, "toarray") else np.asarray(x)
        X_dense = X_dense.astype(np.float32, copy=False)
        splits_dir = RUN_ROOT / "splits" / dataset_id
        split_ids = sorted(p.name for p in splits_dir.iterdir() if p.is_dir())
        for split_id in split_ids:
            process_split(adata, X_dense, dataset_id, split_id)
        del adata, X_dense
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
