#!/usr/bin/env python3
"""Minimal smoke example for the core SOLAR implementation.

Generates a small synthetic reference/query pair (no real data required),
trains a tiny SOLAR model on the reference only (labels visible), and embeds
the query cells the model never saw. Runs in well under a minute on CPU.

This exercises the same code path used to produce the manuscript's held-out
results (`SOLAR.benchmark.run_inductive`, variant "solar_orthogonal"), just
at a toy scale, and is meant only to confirm the package installs and runs
end to end -- it is not a scientific result.
"""
from __future__ import annotations

import anndata as ad
import numpy as np

from SOLAR.benchmark import PreprocessConfig, TrainConfig, run_inductive


def make_synthetic_adata(n_cells: int, n_genes: int, seed: int, prefix: str) -> ad.AnnData:
    rng = np.random.default_rng(seed)
    label_centers = {"type_a": 0.0, "type_b": 4.0, "type_c": 8.0}
    labels = rng.choice(list(label_centers.keys()), size=n_cells)
    X = np.vstack([
        rng.normal(loc=label_centers[label], scale=1.0, size=n_genes) for label in labels
    ]).astype(np.float32)
    obs = {"cell_type": labels, "batch": ["batch_0"] * n_cells}
    var = {"gene_name": [f"gene_{i}" for i in range(n_genes)]}
    adata = ad.AnnData(X=X, obs=obs, var=var)
    adata.obs_names = [f"{prefix}_{i}" for i in range(n_cells)]
    adata.var_names = var["gene_name"]
    return adata


def main() -> int:
    n_genes = 60
    reference = make_synthetic_adata(n_cells=240, n_genes=n_genes, seed=1, prefix="ref")
    query = make_synthetic_adata(n_cells=80, n_genes=n_genes, seed=2, prefix="qry")

    result = run_inductive(
        reference=reference,
        query=query,
        variant="solar_orthogonal",
        labels_key="cell_type",
        batch_key="batch",
        seed=40,
        anchor_seed=0,
        preprocess=PreprocessConfig(n_components=10, source="X"),
        train=TrainConfig(
            embedding_dim=8,
            max_epochs=3,
            batch_size=32,
            early_stopping_patience=3,
        ),
        device="cpu",
    )

    # A non-default dimension is intentional: this assertion guards against
    # variant-level kwargs silently overriding TrainConfig.embedding_dim.
    assert result.reference_embedding.shape == (240, 8), result.reference_embedding.shape
    assert result.query_embedding.shape == (80, 8), result.query_embedding.shape
    assert not np.isnan(result.reference_embedding).any()
    assert not np.isnan(result.query_embedding).any()

    print(f"reference embedding: {result.reference_embedding.shape}")
    print(f"query embedding:     {result.query_embedding.shape}")
    print("SMOKE TEST PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
