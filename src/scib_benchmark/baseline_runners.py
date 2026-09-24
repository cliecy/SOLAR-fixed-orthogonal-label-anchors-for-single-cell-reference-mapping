from __future__ import annotations

import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
from typing import Any

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.io import mmwrite
from sklearn.decomposition import PCA

from .artifacts import begin_output, commit_output, write_json
from .data import read_gene_list, read_split


UNLABELED = "Unknown"


def _environment() -> dict[str, Any]:
    packages = {}
    for name in (
        "anndata",
        "bbknn",
        "harmonypy",
        "numpy",
        "scanorama",
        "scanpy",
        "scikit-learn",
        "scipy",
        "scvi-tools",
        "torch",
    ):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "packages": packages,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }


def _as_csr(matrix: Any) -> sparse.csr_matrix:
    if sparse.issparse(matrix):
        return matrix.tocsr()
    return sparse.csr_matrix(np.asarray(matrix))


def _to_dense(matrix: Any) -> np.ndarray:
    if sparse.issparse(matrix):
        return np.asarray(matrix.toarray(), dtype=np.float64)
    return np.asarray(matrix, dtype=np.float64)


def _save_embedding(path: Path, embedding: np.ndarray, barcodes: np.ndarray) -> None:
    if embedding.ndim != 2 or embedding.shape[0] != len(barcodes):
        raise ValueError(f"Invalid embedding shape {embedding.shape} for {len(barcodes)} barcodes")
    if not np.isfinite(embedding).all():
        raise ValueError("Embedding contains non-finite values")
    if len(set(barcodes.tolist())) != len(barcodes):
        raise ValueError("Embedding barcodes are not unique")
    np.savez_compressed(
        path,
        embedding=np.asarray(embedding, dtype=np.float32),
        barcodes=np.asarray(barcodes, dtype=str),
    )


def _write_bbknn_graph(
    staging: Path,
    connectivities: sparse.spmatrix,
    distances: sparse.spmatrix,
    *,
    n_neighbors: int,
    use_rep: str,
) -> None:
    connectivities = connectivities.tocsr()
    distances = distances.tocsr()
    if connectivities.shape != distances.shape:
        raise ValueError("BBKNN connectivities/distances shapes differ")
    if not np.isfinite(connectivities.data).all() or not np.isfinite(distances.data).all():
        raise ValueError("BBKNN graph contains non-finite values")
    sparse.save_npz(staging / "connectivities.npz", connectivities)
    sparse.save_npz(staging / "distances.npz", distances)
    write_json(
        staging / "graph.json",
        {
            "state": "completed",
            "n_obs": int(connectivities.shape[0]),
            "n_neighbors": int(n_neighbors),
            "use_rep": use_rep,
            "random_state": None,
            "params": {
                "method": "bbknn",
                "metric": "euclidean",
                "n_pcs": None,
                "n_neighbors": int(n_neighbors),
                "random_state": 0,
                "use_rep": use_rep,
            },
            "neighbors_uns": {
                "connectivities_key": "connectivities",
                "distances_key": "distances",
                "params": {
                    "method": "bbknn",
                    "metric": "euclidean",
                    "n_pcs": None,
                    "n_neighbors": int(n_neighbors),
                    "random_state": 0,
                    "use_rep": use_rep,
                },
            },
            "connectivities_nnz": int(connectivities.nnz),
            "distances_nnz": int(distances.nnz),
            "implementation": "bbknn.bbknn",
            "backend": "method_provided_knn",
        },
    )


def _select_genes_and_pca(
    data: ad.AnnData,
    *,
    fit_scope: str,
    split: pd.DataFrame,
    batch_key: str,
    n_hvg: int | None,
    hvg_flavor: str,
    batch_aware: bool,
    n_components: int,
    seed: int,
    gene_file: Path | None,
) -> tuple[ad.AnnData, np.ndarray, dict[str, Any]]:
    import scanpy as sc

    meta: dict[str, Any] = {
        "fit_scope": fit_scope,
        "n_hvg": n_hvg,
        "n_components": n_components,
        "query_expression_used_for_preprocessing_fit": fit_scope == "classic_full",
    }
    working = data.copy()
    if fit_scope == "classic_full":
        if n_hvg is not None:
            sc.pp.highly_variable_genes(
                working,
                n_top_genes=int(n_hvg),
                flavor=hvg_flavor,
                batch_key=batch_key if batch_aware else None,
                subset=True,
            )
            meta["gene_source"] = "classic_full_batch_aware_hvg"
        else:
            meta["gene_source"] = "classic_full_all_genes"
        sc.pp.pca(working, n_comps=int(n_components), zero_center=True, random_state=int(seed))
        pca = np.asarray(working.obsm["X_pca"], dtype=np.float32)
        return working, pca, meta

    if fit_scope != "reference_only":
        raise ValueError(f"Unknown fit_scope: {fit_scope!r}")
    if n_hvg is not None:
        if gene_file is None:
            raise ValueError("reference_only HVG profile requires gene_file")
        genes = read_gene_list(gene_file)
        positions = working.var_names.astype(str).get_indexer(genes)
        if np.any(positions < 0):
            raise KeyError("reference-only gene list is absent from the dataset")
        working = working[:, positions].copy()
        meta["gene_source"] = "reference_only_hvg_list"
        meta["gene_file"] = str(gene_file)
    else:
        meta["gene_source"] = "reference_only_all_genes"
    reference_names = set(split.loc[split["role"] == "reference", "barcode"].astype(str))
    reference_mask = np.asarray(working.obs_names.astype(str).isin(reference_names))
    if not reference_mask.any():
        raise ValueError("No reference cells available for reference_only PCA fit")
    dense = _to_dense(working.X)
    pca_model = PCA(n_components=int(n_components), random_state=int(seed))
    pca_model.fit(dense[reference_mask])
    pca = np.asarray(pca_model.transform(dense), dtype=np.float32)
    working.obsm["X_pca"] = pca
    return working, pca, meta


def _run_harmony(data: ad.AnnData, batch_key: str, seed: int) -> np.ndarray:
    import harmonypy

    X = np.ascontiguousarray(data.obsm["X_pca"], dtype=np.float32)
    harmony = harmonypy.run_harmony(
        X,
        data.obs,
        batch_key,
        random_state=int(seed),
        verbose=False,
    )
    embedding = np.ascontiguousarray(np.asarray(harmony.Z_corr))
    if embedding.shape[0] != data.n_obs and embedding.shape[1] == data.n_obs:
        embedding = embedding.T
    return np.ascontiguousarray(embedding, dtype=np.float32)


def _run_scanorama(data: ad.AnnData, batch_key: str, n_components: int) -> np.ndarray:
    import scanorama

    batch_values = data.obs[batch_key].astype("string")
    categories = sorted(batch_values.dropna().unique().tolist())
    batch_indices = [np.flatnonzero((batch_values == category).to_numpy()) for category in categories]
    scanorama_inputs = [data[idx].copy() for idx in batch_indices]
    scanorama.integrate_scanpy(scanorama_inputs, dimred=int(n_components), verbose=0)
    embedding = np.empty((data.n_obs, int(n_components)), dtype=np.float32)
    for idx, integrated_batch in zip(batch_indices, scanorama_inputs, strict=True):
        embedding[idx] = np.asarray(integrated_batch.obsm["X_scanorama"], dtype=np.float32)
    return embedding


def _run_bbknn(
    data: ad.AnnData,
    batch_key: str,
    n_components: int,
    neighbors_within_batch: int,
) -> tuple[sparse.csr_matrix, sparse.csr_matrix]:
    import bbknn

    graph_data = data.copy()
    bbknn.bbknn(
        graph_data,
        batch_key=batch_key,
        use_rep="X_pca",
        neighbors_within_batch=int(neighbors_within_batch),
        n_pcs=int(n_components),
    )
    connectivities = graph_data.obsp["connectivities"].tocsr()
    distances = graph_data.obsp["distances"].tocsr()
    return connectivities, distances


def _run_fastmnn(
    data: ad.AnnData,
    batch_key: str,
    *,
    n_components: int,
    k: int,
    staging: Path,
    repo_root: Path,
) -> np.ndarray:
    export_dir = staging / "fastmnn_exchange"
    export_dir.mkdir(exist_ok=True)
    expression = _as_csr(data.X).T.tocoo()
    mmwrite(export_dir / "expression.mtx", expression)
    data.obs[[batch_key]].to_csv(export_dir / "batches.tsv", sep="\t", header=False, index=False)
    output = export_dir / "embedding.tsv"
    command = [
        "pixi",
        "run",
        "Rscript",
        "scripts/run_fastmnn.R",
        str(export_dir / "expression.mtx"),
        str(export_dir / "batches.tsv"),
        str(output),
        str(int(n_components)),
        str(int(k)),
    ]
    subprocess.run(command, check=True, cwd=repo_root)
    embedding = np.loadtxt(output, delimiter="\t", ndmin=2)
    return np.asarray(embedding, dtype=np.float32)


def _run_scvi(
    data: ad.AnnData,
    batch_key: str,
    *,
    seed: int,
    defaults: dict[str, Any],
) -> Any:
    import scvi
    import torch

    if "counts" not in data.layers:
        raise KeyError("Dataset lacks layers['counts'] required by scVI/scANVI")
    scvi.settings.seed = int(seed)
    scvi.model.SCVI.setup_anndata(data, layer="counts", batch_key=batch_key)
    model = scvi.model.SCVI(
        data,
        n_latent=int(defaults.get("n_latent", 30)),
        n_layers=int(defaults.get("n_layers", 2)),
        n_hidden=int(defaults.get("n_hidden", 128)),
        gene_likelihood="nb",
    )
    accelerator = "gpu" if torch.cuda.is_available() else "cpu"
    model.train(
        max_epochs=int(defaults.get("max_epochs", 200)),
        early_stopping=bool(defaults.get("early_stopping", True)),
        accelerator=accelerator,
        devices=1,
        batch_size=int(defaults.get("batch_size", 512)),
        train_size=0.9,
        enable_progress_bar=False,
    )
    return model


def _run_scanvi(
    data: ad.AnnData,
    batch_key: str,
    label_key: str,
    *,
    supervision: str,
    split: pd.DataFrame,
    seed: int,
    scvi_defaults: dict[str, Any],
    scanvi_defaults: dict[str, Any],
) -> np.ndarray:
    labels = data.obs[label_key].astype(str).to_numpy(copy=True)
    if supervision == "reference_labels":
        query_names = set(split.loc[split["role"] == "query", "barcode"].astype(str))
        query_mask = np.asarray(data.obs_names.astype(str).isin(query_names))
        labels[query_mask] = UNLABELED
    elif supervision != "all_labels":
        raise ValueError(f"Unsupported scANVI supervision: {supervision!r}")
    data.obs["_scanvi_label"] = pd.Categorical(labels)
    scvi_model = _run_scvi(data, batch_key, seed=seed, defaults=scvi_defaults)
    import scvi
    import torch

    scanvi_model = scvi.model.SCANVI.from_scvi_model(
        scvi_model,
        labels_key="_scanvi_label",
        unlabeled_category=UNLABELED,
    )
    accelerator = "gpu" if torch.cuda.is_available() else "cpu"
    scanvi_model.train(
        max_epochs=int(scanvi_defaults.get("max_epochs", 50)),
        early_stopping=bool(scanvi_defaults.get("early_stopping", True)),
        accelerator=accelerator,
        devices=1,
        batch_size=int(scanvi_defaults.get("batch_size", 512)),
        train_size=0.9,
        enable_progress_bar=False,
    )
    return np.asarray(scanvi_model.get_latent_representation(), dtype=np.float32)


def _reference_mask(data: ad.AnnData, split: pd.DataFrame) -> np.ndarray:
    ref_names = set(split.loc[split["role"] == "reference", "barcode"].astype(str))
    return np.asarray(data.obs_names.astype(str).isin(ref_names))


def _reorder_embedding(
    data: ad.AnnData,
    ref_mask: np.ndarray,
    ref_latent: np.ndarray,
    qry_latent: np.ndarray,
) -> np.ndarray:
    embedding = np.empty((data.n_obs, ref_latent.shape[1]), dtype=np.float32)
    embedding[ref_mask] = ref_latent
    embedding[~ref_mask] = qry_latent
    return np.asarray(embedding, dtype=np.float32)


def _run_scvi_reference(
    data: ad.AnnData,
    split: pd.DataFrame,
    batch_key: str,
    *,
    seed: int,
    defaults: dict[str, Any],
) -> np.ndarray:
    """Inductive scVI: fit on reference cells only, then map the query."""
    import scvi
    import torch

    if "counts" not in data.layers:
        raise KeyError("Dataset lacks layers['counts'] required by scVI/scANVI")
    ref_mask = _reference_mask(data, split)
    if not ref_mask.any():
        raise ValueError("No reference cells available for reference-mapping fit")
    reference = data[ref_mask].copy()
    query = data[~ref_mask].copy()
    scvi.settings.seed = int(seed)
    scvi.model.SCVI.setup_anndata(reference, layer="counts", batch_key=batch_key)
    model = scvi.model.SCVI(
        reference,
        n_latent=int(defaults.get("n_latent", 30)),
        n_layers=int(defaults.get("n_layers", 2)),
        n_hidden=int(defaults.get("n_hidden", 128)),
        gene_likelihood="nb",
    )
    accelerator = "gpu" if torch.cuda.is_available() else "cpu"
    model.train(
        max_epochs=int(defaults.get("max_epochs", 200)),
        early_stopping=bool(defaults.get("early_stopping", True)),
        accelerator=accelerator,
        devices=1,
        batch_size=int(defaults.get("batch_size", 512)),
        train_size=0.9,
        enable_progress_bar=False,
    )
    # Register directly against this model instance (not via the setup_anndata
    # classmethod, which does not attach to model.adata_manager) so a query
    # batch category absent from the reference is transferred rather than
    # rejected by _validate_anndata's internal transfer_fields call.
    model._register_manager_for_instance(
        model.adata_manager.transfer_fields(query, extend_categories=True)
    )
    ref_latent = model.get_latent_representation(reference)
    qry_latent = model.get_latent_representation(query)
    return _reorder_embedding(data, ref_mask, ref_latent, qry_latent)


def _run_scanvi_reference(
    data: ad.AnnData,
    split: pd.DataFrame,
    batch_key: str,
    label_key: str,
    *,
    seed: int,
    scvi_defaults: dict[str, Any],
    scanvi_defaults: dict[str, Any],
) -> np.ndarray:
    """Inductive scANVI: fit on reference cells with reference labels, map query."""
    import scvi
    import torch

    if "counts" not in data.layers:
        raise KeyError("Dataset lacks layers['counts'] required by scVI/scANVI")
    ref_mask = _reference_mask(data, split)
    if not ref_mask.any():
        raise ValueError("No reference cells available for reference-mapping fit")
    reference = data[ref_mask].copy()
    query = data[~ref_mask].copy()
    reference.obs["_scanvi_label"] = pd.Categorical(reference.obs[label_key].astype(str))
    scvi.settings.seed = int(seed)
    scvi.model.SCVI.setup_anndata(reference, layer="counts", batch_key=batch_key)
    scvi_model = scvi.model.SCVI(
        reference,
        n_latent=int(scvi_defaults.get("n_latent", 30)),
        n_layers=int(scvi_defaults.get("n_layers", 2)),
        n_hidden=int(scvi_defaults.get("n_hidden", 128)),
        gene_likelihood="nb",
    )
    accelerator = "gpu" if torch.cuda.is_available() else "cpu"
    scvi_model.train(
        max_epochs=int(scvi_defaults.get("max_epochs", 200)),
        early_stopping=bool(scvi_defaults.get("early_stopping", True)),
        accelerator=accelerator,
        devices=1,
        batch_size=int(scvi_defaults.get("batch_size", 512)),
        train_size=0.9,
        enable_progress_bar=False,
    )
    scanvi_model = scvi.model.SCANVI.from_scvi_model(
        scvi_model,
        labels_key="_scanvi_label",
        unlabeled_category=UNLABELED,
    )
    scanvi_model.train(
        max_epochs=int(scanvi_defaults.get("max_epochs", 50)),
        early_stopping=bool(scanvi_defaults.get("early_stopping", True)),
        accelerator=accelerator,
        devices=1,
        batch_size=int(scanvi_defaults.get("batch_size", 512)),
        train_size=0.9,
        enable_progress_bar=False,
    )
    query.obs["_scanvi_label"] = UNLABELED
    # See _run_scvi_reference: transfer against this model instance directly
    # so a query batch category absent from the reference is extended rather
    # than rejected.
    scanvi_model._register_manager_for_instance(
        scanvi_model.adata_manager.transfer_fields(query, extend_categories=True)
    )
    ref_latent = scanvi_model.get_latent_representation(reference)
    qry_latent = scanvi_model.get_latent_representation(query)
    return _reorder_embedding(data, ref_mask, ref_latent, qry_latent)


def run_baseline_job(spec: dict[str, Any], *, repo_root: Path, force: bool = False) -> Path:
    if spec.get("engine") != "baseline":
        raise ValueError(f"Unsupported engine for baseline runner: {spec.get('engine')!r}")
    output = Path(spec["output_dir"])
    transaction = begin_output(output, force=force)
    if transaction.reused:
        return output
    assert transaction.staging is not None
    started = time.time()
    status: dict[str, Any] = {
        "state": "running",
        "job_id": spec["job_id"],
        "started_unix": started,
    }
    write_json(transaction.staging / "status.json", status)
    warnings: list[str] = []
    try:
        split = read_split(Path(spec["split_file"]))
        dataset_path = Path(spec["dataset_path"])
        before = (dataset_path.stat().st_size, dataset_path.stat().st_mtime_ns)
        data = ad.read_h5ad(dataset_path)
        after = (dataset_path.stat().st_size, dataset_path.stat().st_mtime_ns)
        if before != after:
            raise RuntimeError("Dataset file changed while loading")
        if not data.obs_names.is_unique:
            raise ValueError("Dataset obs_names must be unique")
        data.obs_names = data.obs_names.astype(str)
        batch_key = spec["batch_key"]
        label_key = spec["label_key"]
        method = spec["variant"]
        seed = int(spec["seed"])
        defaults = dict(spec.get("baseline_defaults") or {})
        gene_file = Path(spec["gene_file"]) if spec.get("gene_file") else None
        prepared, pca, prep_meta = _select_genes_and_pca(
            data,
            fit_scope=str(spec["fit_scope"]),
            split=split,
            batch_key=batch_key,
            n_hvg=spec.get("n_hvg"),
            hvg_flavor=str(spec.get("hvg_flavor", "cell_ranger")),
            batch_aware=bool(spec.get("batch_aware", True)),
            n_components=int(spec["n_components"]),
            seed=seed,
            gene_file=gene_file,
        )
        barcodes = prepared.obs_names.astype(str).to_numpy()
        if set(barcodes) != set(data.obs_names.astype(str)):
            # Gene subset keeps all cells; barcode set must still cover the full dataset.
            if len(barcodes) != data.n_obs:
                raise RuntimeError("Baseline preprocessing dropped cells")
        if spec["dataset_id"] == "pancreas" and method in {
            "scvi",
            "scanvi_all_labels",
            "scanvi_reference_labels",
        }:
            warnings.append(
                "Pancreas layers/counts is not a uniform raw-UMI matrix; used only to reproduce "
                "the historical scVI/scANVI input path."
            )

        method_meta: dict[str, Any] = {"method": method}
        if method == "harmony":
            embedding = _run_harmony(prepared, batch_key, seed)
            _save_embedding(transaction.staging / "embedding.npz", embedding, barcodes)
        elif method == "scanorama":
            embedding = _run_scanorama(prepared, batch_key, int(spec["n_components"]))
            _save_embedding(transaction.staging / "embedding.npz", embedding, barcodes)
        elif method == "fastmnn":
            fastmnn_defaults = dict(defaults.get("fastmnn") or {})
            embedding = _run_fastmnn(
                prepared,
                batch_key,
                n_components=int(spec["n_components"]),
                k=int(fastmnn_defaults.get("k", 20)),
                staging=transaction.staging,
                repo_root=repo_root,
            )
            _save_embedding(transaction.staging / "embedding.npz", embedding, barcodes)
        elif method == "bbknn":
            bbknn_defaults = dict(defaults.get("bbknn") or {})
            connectivities, distances = _run_bbknn(
                prepared,
                batch_key,
                int(spec["n_components"]),
                int(bbknn_defaults.get("neighbors_within_batch", 3)),
            )
            _save_embedding(transaction.staging / "embedding.npz", pca, barcodes)
            _write_bbknn_graph(
                transaction.staging,
                connectivities,
                distances,
                n_neighbors=int(bbknn_defaults.get("neighbors_within_batch", 3)),
                use_rep="X_pca",
            )
            method_meta["graph_source"] = "bbknn"
        elif method == "scvi":
            model = _run_scvi(
                prepared,
                batch_key,
                seed=seed,
                defaults=dict(defaults.get("scvi") or {}),
            )
            embedding = np.asarray(model.get_latent_representation(), dtype=np.float32)
            _save_embedding(transaction.staging / "embedding.npz", embedding, barcodes)
            method_meta["accelerator"] = "gpu" if os.environ.get("CUDA_VISIBLE_DEVICES") else "auto"
        elif method == "scanvi_all_labels":
            embedding = _run_scanvi(
                prepared,
                batch_key,
                label_key,
                supervision="all_labels",
                split=split,
                seed=seed,
                scvi_defaults=dict(defaults.get("scvi") or {}),
                scanvi_defaults=dict(defaults.get("scanvi") or {}),
            )
            _save_embedding(transaction.staging / "embedding.npz", embedding, barcodes)
        elif method == "scanvi_reference_labels":
            embedding = _run_scanvi(
                prepared,
                batch_key,
                label_key,
                supervision="reference_labels",
                split=split,
                seed=seed,
                scvi_defaults=dict(defaults.get("scvi") or {}),
                scanvi_defaults=dict(defaults.get("scanvi") or {}),
            )
            _save_embedding(transaction.staging / "embedding.npz", embedding, barcodes)
            method_meta["unlabeled_category"] = UNLABELED
            method_meta["unlabeled_from"] = "query_split"
        elif method in ("frozen_reference_scvi", "scvi_reference_mapping"):
            # Trains scVI on the reference only, then encodes the query with
            # the frozen trained encoder (transfer_fields + extend_categories
            # for any batch category absent from the reference). This is NOT
            # scArches-style query adaptation: there is no query-side
            # fine-tuning/architecture surgery. Name it accordingly.
            embedding = _run_scvi_reference(
                prepared,
                split,
                batch_key,
                seed=seed,
                defaults=dict(defaults.get("scvi") or {}),
            )
            _save_embedding(transaction.staging / "embedding.npz", embedding, barcodes)
            method_meta["training_scope"] = "reference_only"
            method_meta["inductive_reference_mapping"] = True
            method_meta["adaptation_type"] = "frozen_encoder_no_query_finetuning"
            method_meta["accelerator"] = "gpu" if os.environ.get("CUDA_VISIBLE_DEVICES") else "auto"
        elif method in ("frozen_reference_scanvi", "scanvi_reference_mapping"):
            # See frozen_reference_scvi above: frozen encoder, no query-side
            # fine-tuning/surgery.
            embedding = _run_scanvi_reference(
                prepared,
                split,
                batch_key,
                label_key,
                seed=seed,
                scvi_defaults=dict(defaults.get("scvi") or {}),
                scanvi_defaults=dict(defaults.get("scanvi") or {}),
            )
            _save_embedding(transaction.staging / "embedding.npz", embedding, barcodes)
            method_meta["training_scope"] = "reference_only"
            method_meta["inductive_reference_mapping"] = True
            method_meta["adaptation_type"] = "frozen_encoder_no_query_finetuning"
            method_meta["unlabeled_category"] = UNLABELED
        else:
            raise KeyError(f"Unsupported baseline method: {method}")

        run_info = {
            "schema_version": 1,
            "engine": "baseline",
            "job_id": spec["job_id"],
            "job_index": spec["job_index"],
            "dataset_id": spec["dataset_id"],
            "dataset_sha256": spec["dataset_sha256"],
            "batch_key": batch_key,
            "label_key": label_key,
            "split_id": spec["split_id"],
            "protocol": spec["protocol"],
            "supervision": spec["supervision"],
            "preprocessing_profile": spec["preprocessing_profile"],
            "fit_scope": spec["fit_scope"],
            "seed": seed,
            "anchor_seed": None,
            "label_fraction": None,
            "variant": method,
            "output_kind": spec["output_kind"],
            "scib_output_type": spec["scib_output_type"],
            "execution_profile": spec.get("execution_profile", "formal"),
            "query_expression_used_for_training": method not in {
                "frozen_reference_scvi",
                "frozen_reference_scanvi",
                "scvi_reference_mapping",
                "scanvi_reference_mapping",
            },
            "query_expression_used_for_preprocessing_fit": prep_meta[
                "query_expression_used_for_preprocessing_fit"
            ],
            "query_labels_visible_to_adapter": spec["supervision"] == "all_labels",
            "query_labels_visible_to_model": spec["supervision"] == "all_labels",
            "preprocessing": prep_meta,
            "method": method_meta,
            "warnings": warnings,
            "n_cells": int(prepared.n_obs),
            "n_genes": int(prepared.n_vars),
            "seconds": time.time() - started,
        }
        write_json(transaction.staging / "run_info.json", run_info)
        write_json(transaction.staging / "environment.json", _environment())
        status = {
            "state": "completed",
            "job_id": spec["job_id"],
            "execution_profile": spec.get("execution_profile", "formal"),
            "n_cells": int(prepared.n_obs),
            "output_kind": spec["output_kind"],
            "seconds": time.time() - started,
        }
        write_json(transaction.staging / "status.json", status)
        return commit_output(transaction)
    except Exception as exc:
        status = {
            "state": "failed",
            "job_id": spec["job_id"],
            "error": f"{type(exc).__name__}: {exc}",
            "seconds": time.time() - started,
        }
        write_json(transaction.staging / "status.json", status)
        raise
