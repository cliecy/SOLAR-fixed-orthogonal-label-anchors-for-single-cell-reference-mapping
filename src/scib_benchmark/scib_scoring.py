from __future__ import annotations

import json
import hashlib
import os
from pathlib import Path
import platform
import subprocess
import sys
from typing import Any

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse

from .artifacts import begin_output, commit_output, write_json
from .scib_contract import OFFICIAL_METRICS, metric_record, official_applicability


EMBEDDING_KEY = "X_emb"
PAPER_BACKEND = "paper_scib_0_2_0"
PAPER_SCIB_COMMIT = "e2a37e0ed63dc34b60aa535cc656400552af757a"
PAPER_KBET_COMMIT = "afc5f431bcbefd73267acc066a0f2e4eaa10a355"
PYTHON_KBET_FALLBACK_BACKEND = "scib_metrics_python_0_5_10"
PYTHON_KBET_FALLBACK_VERSION = "0.5.10"
GRAPH_TASKS = frozenset({"clustering", "isolated_f1", "graph_conn", "lisi", "trajectory"})
TASK_METRICS: dict[str, tuple[str, ...]] = {
    "clustering": ("NMI_cluster/label", "ARI_cluster/label"),
    "silhouette": ("ASW_label", "ASW_label/batch"),
    "pcr": ("PCR_batch",),
    "cell_cycle": ("cell_cycle_conservation",),
    "isolated_f1": ("isolated_label_F1",),
    "isolated_silhouette": ("isolated_label_silhouette",),
    "graph_conn": ("graph_conn",),
    "kbet": ("kBET",),
    "lisi": ("iLISI", "cLISI"),
    "trajectory": ("trajectory",),
}
TASK_ORDER: tuple[str, ...] = tuple(TASK_METRICS)


def scoring_context(config_data: dict[str, Any], dataset_id: str) -> dict[str, Any]:
    official = config_data.get("evaluation", {}).get("official_scib", {})
    if official.get("enabled") is not True:
        raise ValueError("evaluation.official_scib.enabled must be true")
    contexts = official.get("datasets", {})
    if dataset_id not in contexts:
        raise KeyError(f"No official scIB scoring context for dataset {dataset_id!r}")
    context = dict(contexts[dataset_id])
    context["backend"] = str(official.get("backend", PAPER_BACKEND))
    if context["backend"] != PAPER_BACKEND:
        raise ValueError(
            f"Formal scoring requires evaluation.official_scib.backend={PAPER_BACKEND!r}; "
            f"observed {context['backend']!r}"
        )
    configured_commit = str(official.get("scib_commit", ""))
    if configured_commit != PAPER_SCIB_COMMIT:
        raise ValueError(
            "evaluation.official_scib.scib_commit must pin the paper source commit "
            f"{PAPER_SCIB_COMMIT}; observed {configured_commit!r}"
        )
    configured_kbet = str(official.get("kbet_commit", ""))
    if configured_kbet != PAPER_KBET_COMMIT:
        raise ValueError(
            "evaluation.official_scib.kbet_commit must pin the paper kBET source commit "
            f"{PAPER_KBET_COMMIT}; observed {configured_kbet!r}"
        )
    context["output_type"] = official.get("output_type", "embed")
    context["lisi_subsample_percent"] = int(official.get("lisi_subsample_percent", 50))
    context["lisi_cores"] = int(official.get("lisi_cores", 1))
    context["neighbors"] = int(official.get("neighbors", 15))
    context["task_timeout_seconds"] = int(official.get("task_timeout_seconds", 7200))
    fallback = dict(official.get("kbet_python_fallback", {}))
    fallback["enabled"] = fallback.get("enabled") is True
    fallback["policy"] = str(fallback.get("policy", "disabled"))
    if fallback["enabled"] and fallback["policy"] != "on_legacy_failure":
        raise ValueError(
            "evaluation.official_scib.kbet_python_fallback.policy must be "
            "'on_legacy_failure' when enabled"
        )
    fallback["package_version"] = str(
        fallback.get("package_version", PYTHON_KBET_FALLBACK_VERSION)
    )
    if fallback["enabled"] and fallback["package_version"] != PYTHON_KBET_FALLBACK_VERSION:
        raise ValueError(
            "The approved Python kBET fallback is pinned to scib-metrics "
            f"{PYTHON_KBET_FALLBACK_VERSION}; observed {fallback['package_version']!r}"
        )
    fallback["neighbors"] = int(fallback.get("neighbors", 50))
    fallback["random_state"] = int(fallback.get("random_state", 0))
    fallback["n_jobs"] = int(fallback.get("n_jobs", 1))
    fallback["alpha"] = float(fallback.get("alpha", 0.05))
    fallback["diffusion_n_comps"] = int(fallback.get("diffusion_n_comps", 100))
    if fallback["neighbors"] < 2 or fallback["n_jobs"] < 1:
        raise ValueError("Invalid Python kBET fallback neighbor or worker count")
    if not 0.0 < fallback["alpha"] < 1.0 or fallback["diffusion_n_comps"] < 1:
        raise ValueError("Invalid Python kBET fallback alpha or diffusion component count")
    context["kbet_python_fallback"] = fallback
    return context


def source_cache_path(
    run_root: Path,
    spec: dict[str, Any],
    *,
    backend: str = PAPER_BACKEND,
) -> Path:
    token = str(spec["dataset_sha256"])[:16]
    return (
        run_root
        / "scib_scores"
        / "source_cache"
        / backend
        / f"{spec['dataset_id']}__{token}"
    )


def _package_versions() -> dict[str, str]:
    from importlib.metadata import PackageNotFoundError, version

    versions: dict[str, str] = {"python": platform.python_version()}
    for package in (
        "anndata",
        "numpy",
        "pandas",
        "scanpy",
        "scib",
        "scipy",
        "igraph",
        "leidenalg",
        "rpy2",
    ):
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            versions[package] = "not_installed"
    return versions


def _load_source_matrix(path: Path) -> tuple[Any, pd.DataFrame, pd.DataFrame]:
    backed = ad.read_h5ad(path, backed="r")
    try:
        matrix = backed.X[:]
        obs = backed.obs.copy()
        var = backed.var.copy()
    finally:
        backed.file.close()
    if sparse.issparse(matrix):
        matrix = matrix.tocsr()
    else:
        matrix = np.asarray(matrix)
    return matrix, obs, var


def _atomic_npz(path: Path, **arrays: Any) -> None:
    temporary = path.with_name(path.name + ".tmp.npz")
    np.savez_compressed(temporary, **arrays)
    temporary.replace(path)


def prepare_source_cache(
    *,
    cache_dir: Path,
    dataset_id: str,
    dataset_path: Path,
    dataset_sha256: str,
    batch_key: str,
    assay: str,
    organism: str | None,
    force: bool = False,
) -> dict[str, Any]:
    """Cache source-only invariants without changing the official input file."""
    transaction = begin_output(cache_dir, force=force)
    if transaction.reused:
        return json.loads((cache_dir / "status.json").read_text(encoding="utf-8"))
    assert transaction.staging is not None
    staging = transaction.staging
    status: dict[str, Any] = {
        "schema_version": 1,
        "state": "running",
        "dataset_id": dataset_id,
        "dataset_path": str(dataset_path),
        "dataset_sha256": dataset_sha256,
        "batch_key": batch_key,
        "assay": assay,
        "organism": organism,
        "package_versions": _package_versions(),
        "errors": {},
    }
    write_json(staging / "status.json", status)
    try:
        import scanpy as sc

        matrix, obs, var = _load_source_matrix(dataset_path)
        if batch_key not in obs:
            raise KeyError(f"Source AnnData lacks batch key {batch_key!r}")
        n_comps = min(50, min(matrix.shape))
        solver = "arpack"
        pca_input = matrix
        if n_comps == min(matrix.shape):
            solver = "full"
            if sparse.issparse(pca_input):
                pca_input = pca_input.toarray()
        scores, _, variance, variance_ratio = sc.tl.pca(
            pca_input,
            n_comps=n_comps,
            use_highly_variable=False,
            return_info=True,
            svd_solver=solver,
            copy=True,
        )
        _atomic_npz(
            staging / "source_pca.npz",
            barcodes=np.asarray(obs.index.astype(str), dtype=str),
            scores=np.asarray(scores, dtype=np.float32),
            variance=np.asarray(variance, dtype=np.float64),
            variance_ratio=np.asarray(variance_ratio, dtype=np.float64),
        )
        status["pcr_source_cache"] = {
            "state": "completed",
            "n_components": int(n_comps),
            "equivalence": (
                "Same Scanpy PCA call used by scib.metrics.pcr.pc_regression; cached once "
                "because the unintegrated matrix is invariant across integration jobs."
            ),
        }

        if assay == "simulation":
            status["cell_cycle_source_cache"] = {
                "state": "not_applicable",
                "reason": "Official scIB disables cell-cycle conservation for simulations.",
            }
        elif assay == "atac":
            status["cell_cycle_source_cache"] = {
                "state": "not_applicable",
                "reason": "Official scIB disables cell-cycle conservation for scATAC-seq.",
            }
        else:
            if organism not in {"human", "mouse"}:
                raise ValueError(f"Expression dataset has unsupported organism {organism!r}")
            import scib

            s_score = np.full(len(obs), np.nan, dtype=np.float64)
            g2m_score = np.full(len(obs), np.nan, dtype=np.float64)
            before_by_batch: dict[str, float] = {}
            batches = obs[batch_key].astype(str).to_numpy()
            for batch in pd.unique(batches):
                positions = np.flatnonzero(batches == batch)
                source = ad.AnnData(
                    X=matrix[positions].copy(),
                    obs=pd.DataFrame(index=obs.index[positions].astype(str)),
                    var=var.copy(),
                )
                scib.pp.score_cell_cycle(source, organism=organism)
                s_score[positions] = source.obs["S_score"].to_numpy(dtype=float)
                g2m_score[positions] = source.obs["G2M_score"].to_numpy(dtype=float)
                before = scib.me.pc_regression(
                    source.X,
                    source.obs[["S_score", "G2M_score"]],
                    n_comps=50,
                    linreg_method="numpy",
                )
                before_by_batch[str(batch)] = float(before)
            if not np.isfinite(s_score).all() or not np.isfinite(g2m_score).all():
                raise ValueError("Cell-cycle source cache contains non-finite scores")
            if not all(np.isfinite(value) for value in before_by_batch.values()):
                raise ValueError("Cell-cycle source PCR cache contains non-finite values")
            _atomic_npz(
                staging / "cell_cycle_scores.npz",
                barcodes=np.asarray(obs.index.astype(str), dtype=str),
                S_score=s_score,
                G2M_score=g2m_score,
            )
            write_json(
                staging / "cell_cycle_before.json",
                {"before_by_batch": before_by_batch},
            )
            status["cell_cycle_source_cache"] = {
                "state": "completed",
                "batches": len(before_by_batch),
                "equivalence": (
                    "Uses scib.pp.score_cell_cycle and scib.me.pc_regression exactly as "
                    "scib.me.cell_cycle, caching only source-invariant quantities."
                ),
            }
        status["state"] = "completed"
    except Exception as exc:
        status["state"] = "failed"
        status["errors"]["source_cache"] = f"{type(exc).__name__}: {exc}"
    write_json(staging / "status.json", status)
    commit_output(transaction)
    return status


def load_joint_embedding(spec: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    output = Path(spec["output_dir"])
    with np.load(output / "embedding.npz") as archive:
        keys = set(archive.files)
        if {"embedding", "barcodes"} <= keys:
            embedding = np.asarray(archive["embedding"], dtype=np.float32)
            barcodes = archive["barcodes"].astype(str)
        elif {"reference", "query", "reference_barcodes", "query_barcodes"} <= keys:
            reference = np.asarray(archive["reference"], dtype=np.float32)
            query = np.asarray(archive["query"], dtype=np.float32)
            reference_names = archive["reference_barcodes"].astype(str)
            query_names = archive["query_barcodes"].astype(str)
            if reference.ndim != 2 or query.ndim != 2 or reference.shape[1] != query.shape[1]:
                raise ValueError("Reference/query embeddings are not compatible matrices")
            embedding = np.concatenate([reference, query], axis=0)
            barcodes = np.concatenate([reference_names, query_names])
        else:
            raise ValueError(
                "embedding.npz must contain either embedding/barcodes or "
                "reference/query with barcode arrays"
            )
    if embedding.ndim != 2:
        raise ValueError("Embedding must be two-dimensional")
    if embedding.shape[0] != len(barcodes):
        raise ValueError("Embedding rows and barcode count differ")
    if len(set(barcodes.tolist())) != len(barcodes):
        raise ValueError("Joint embedding contains duplicate barcodes")
    if not np.isfinite(embedding).all():
        raise ValueError("Joint embedding contains non-finite values")
    return embedding, barcodes


def method_provided_graph_dir(spec: dict[str, Any]) -> Path | None:
    """Return a method-produced kNN graph directory when present (BBKNN)."""
    output = Path(spec["output_dir"])
    if (
        (output / "connectivities.npz").is_file()
        and (output / "distances.npz").is_file()
        and (output / "graph.json").is_file()
    ):
        return output
    return None


def install_method_provided_graph(source_dir: Path, graph_dir: Path, *, backend: str) -> dict[str, Any]:
    graph_dir.mkdir(parents=True, exist_ok=True)
    payload = json.loads((source_dir / "graph.json").read_text(encoding="utf-8"))
    connectivities = sparse.load_npz(source_dir / "connectivities.npz").tocsr()
    distances = sparse.load_npz(source_dir / "distances.npz").tocsr()
    if connectivities.shape != distances.shape:
        raise ValueError("Method-provided graph matrices have mismatched shapes")
    temporary_conn = graph_dir / "connectivities.tmp.npz"
    temporary_dist = graph_dir / "distances.tmp.npz"
    sparse.save_npz(temporary_conn, connectivities)
    sparse.save_npz(temporary_dist, distances)
    temporary_conn.replace(graph_dir / "connectivities.npz")
    temporary_dist.replace(graph_dir / "distances.npz")
    payload = dict(payload)
    payload["backend"] = backend
    payload["source"] = "method_provided_knn"
    payload["source_dir"] = str(source_dir)
    write_json(graph_dir / "graph.json", payload)
    return payload


def load_joint_obs(spec: dict[str, Any], barcodes: np.ndarray) -> pd.DataFrame:
    dataset_path = Path(spec["dataset_path"])
    backed = ad.read_h5ad(dataset_path, backed="r")
    try:
        source_names = pd.Index(backed.obs_names.astype(str))
        if len(barcodes) != backed.n_obs or set(barcodes.tolist()) != set(source_names):
            raise ValueError(
                "Official scIB scoring requires an embedding covering every dataset cell exactly once"
            )
        positions = source_names.get_indexer(barcodes)
        if np.any(positions < 0):
            raise KeyError("Embedding barcodes do not align to the official AnnData")
        obs = backed.obs.iloc[positions].copy()
    finally:
        backed.file.close()
    obs.index = pd.Index(barcodes.astype(str))
    for key in (spec["batch_key"], spec["label_key"]):
        if key not in obs:
            raise KeyError(f"Official AnnData lacks metadata key {key!r}")
        if obs[key].isna().any():
            raise ValueError(f"Official scoring key {key!r} contains missing values")
        obs[key] = obs[key].astype("category")
    return obs


def build_integrated(spec: dict[str, Any], *, with_graph: Path | None = None) -> ad.AnnData:
    embedding, barcodes = load_joint_embedding(spec)
    obs = load_joint_obs(spec, barcodes)
    integrated = ad.AnnData(
        X=sparse.csr_matrix((len(obs), 1), dtype=np.float32),
        obs=obs,
        var=pd.DataFrame(index=["__scib_embedding_placeholder__"]),
    )
    integrated.obsm[EMBEDDING_KEY] = embedding
    if with_graph is not None:
        integrated.obsp["connectivities"] = sparse.load_npz(with_graph / "connectivities.npz")
        integrated.obsp["distances"] = sparse.load_npz(with_graph / "distances.npz")
        graph_meta = json.loads((with_graph / "graph.json").read_text(encoding="utf-8"))
        neighbors_uns = graph_meta.get("neighbors_uns")
        if neighbors_uns is None:
            neighbors_uns = {
                "connectivities_key": "connectivities",
                "distances_key": "distances",
                "params": graph_meta.get("params", {}),
            }
        integrated.uns["neighbors"] = neighbors_uns
        expected = (integrated.n_obs, integrated.n_obs)
        if integrated.obsp["connectivities"].shape != expected:
            raise ValueError("Cached graph shape does not match the integrated embedding")
    return integrated


def prepare_neighbor_graph(
    spec: dict[str, Any],
    graph_dir: Path,
    n_neighbors: int,
    *,
    backend: str = PAPER_BACKEND,
) -> dict[str, Any]:
    import scanpy as sc

    provided = method_provided_graph_dir(spec)
    if provided is not None:
        return install_method_provided_graph(provided, graph_dir, backend=backend)

    graph_dir.mkdir(parents=True, exist_ok=True)
    integrated = build_integrated(spec)
    sc.pp.neighbors(
        integrated,
        n_neighbors=int(n_neighbors),
        use_rep=EMBEDDING_KEY,
        random_state=0,
    )
    temporary_conn = graph_dir / "connectivities.tmp.npz"
    temporary_dist = graph_dir / "distances.tmp.npz"
    sparse.save_npz(temporary_conn, integrated.obsp["connectivities"].tocsr())
    sparse.save_npz(temporary_dist, integrated.obsp["distances"].tocsr())
    temporary_conn.replace(graph_dir / "connectivities.npz")
    temporary_dist.replace(graph_dir / "distances.npz")
    payload = {
        "state": "completed",
        "n_obs": int(integrated.n_obs),
        "n_neighbors": int(n_neighbors),
        "use_rep": EMBEDDING_KEY,
        "random_state": 0,
        "neighbors_uns": dict(integrated.uns["neighbors"]),
        "connectivities_nnz": int(integrated.obsp["connectivities"].nnz),
        "distances_nnz": int(integrated.obsp["distances"].nnz),
        "implementation": "scanpy.pp.neighbors (official embed-output preprocessing)",
        "backend": backend,
    }
    write_json(graph_dir / "graph.json", payload)
    return payload


def compute_python_kbet_fallback(
    *,
    spec: dict[str, Any],
    primary_error: str,
    n_neighbors: int = 50,
    random_state: int = 0,
    n_jobs: int = 1,
    alpha: float = 0.05,
    diffusion_n_comps: int = 100,
) -> dict[str, dict[str, Any]]:
    """Compute the explicitly approved Python fallback for a failed legacy kBET call.

    This is intentionally separate from the paper backend. ``scib-metrics`` documents
    ``kbet_per_label`` as an approximation of original scIB, so the returned record
    carries a metric-level backend override and must not be represented as the R value.
    """
    from importlib.metadata import version

    observed_version = version("scib-metrics")
    if observed_version != PYTHON_KBET_FALLBACK_VERSION:
        raise RuntimeError(
            "Python kBET fallback requires scib-metrics "
            f"{PYTHON_KBET_FALLBACK_VERSION}, observed {observed_version}"
        )

    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    import scib_metrics
    from scib_metrics.nearest_neighbors import pynndescent

    embedding, barcodes = load_joint_embedding(spec)
    obs = load_joint_obs(spec, barcodes)
    neighbors = pynndescent(
        embedding,
        n_neighbors=int(n_neighbors),
        random_state=int(random_state),
        n_jobs=int(n_jobs),
    )
    value, per_label = scib_metrics.kbet_per_label(
        neighbors,
        batches=obs[spec["batch_key"]].astype(str).to_numpy(),
        labels=obs[spec["label_key"]].astype(str).to_numpy(),
        alpha=float(alpha),
        diffusion_n_comps=int(diffusion_n_comps),
        return_df=True,
    )
    numeric = float(value)
    if not np.isfinite(numeric) or not 0.0 <= numeric <= 1.0:
        raise ValueError(f"Python kBET fallback returned invalid value {numeric!r}")
    details = {
        "fallback": True,
        "fallback_policy": "on_legacy_failure",
        "primary_backend": PAPER_BACKEND,
        "primary_error": primary_error,
        "metric_backend": PYTHON_KBET_FALLBACK_BACKEND,
        "python_package": {
            "name": "scib-metrics",
            "version": observed_version,
        },
        "n_neighbors": int(n_neighbors),
        "random_state": int(random_state),
        "n_jobs": int(n_jobs),
        "alpha": float(alpha),
        "diffusion_n_comps": int(diffusion_n_comps),
        "per_label": per_label.astype(object).where(pd.notna(per_label), None).to_dict(
            orient="records"
        ),
        "comparability_warning": (
            "scib-metrics kbet_per_label is a Python approximation with an acceptance-rate "
            "statistic; it is not numerically equivalent to scIB 0.2.0 plus R kBET "
            "average.pval. The value is used directly with higher meaning better mixing."
        ),
    }
    return {
        "kBET": metric_record(
            "kBET",
            "computed",
            value=numeric,
            reason=(
                "Computed by the researcher-approved Python fallback after the locked "
                "legacy kBET worker failed."
            ),
            implementation=(
                f"scib-metrics {observed_version} kbet_per_label (Python fallback)"
            ),
            evidence=(
                "Persisted legacy-worker failure plus scib-metrics kbet_per_label output."
            ),
            details=details,
        )
    }


def _computed(name: str, value: Any, function: str, details: dict[str, Any] | None = None) -> dict[str, Any]:
    return metric_record(
        name,
        "computed",
        value=float(value),
        reason="Computed with the official scIB metric implementation.",
        implementation=function,
        evidence="scib/scib/metrics and scib-pipeline/scripts/metrics/metrics.py",
        details=details,
    )


def _undefined(
    name: str,
    reason: str,
    function: str,
    details: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return metric_record(
        name,
        "not_applicable",
        reason=reason,
        implementation=function,
        evidence="Observed metadata structure and official scIB metric definition.",
        details=details,
    )


def trajectory_root_support(integrated: ad.AnnData, label_key: str) -> dict[str, Any]:
    """Describe the root-cell domain used by scIB 0.2.0 trajectory conservation."""
    from scipy.sparse.csgraph import connected_components

    pseudotime_mask = integrated.obs["dpt_pseudotime"].notna().to_numpy()
    trajectory = integrated[pseudotime_mask]
    if trajectory.n_obs == 0:
        return {
            "pseudotime_cells": 0,
            "start_cluster": None,
            "start_cluster_cells": 0,
            "largest_component": None,
            "largest_component_cells": 0,
            "root_candidate_cells": 0,
        }

    _, neighborhoods = connected_components(
        csgraph=trajectory.obsp["connectivities"],
        directed=False,
        return_labels=True,
    )
    means = (
        trajectory.obs.groupby(label_key, observed=False)["dpt_pseudotime"]
        .mean()
        .dropna()
    )
    if means.empty:
        return {
            "pseudotime_cells": int(trajectory.n_obs),
            "start_cluster": None,
            "start_cluster_cells": 0,
            "largest_component": None,
            "largest_component_cells": 0,
            "root_candidate_cells": 0,
        }

    start_cluster = means.idxmin()
    start_mask = np.asarray(trajectory.obs[label_key] == start_cluster)
    component_counts = pd.Series(neighborhoods).value_counts()
    largest_component = int(component_counts.idxmax())
    largest_mask = neighborhoods == largest_component
    return {
        "pseudotime_cells": int(trajectory.n_obs),
        "start_cluster": str(start_cluster),
        "start_cluster_cells": int(start_mask.sum()),
        "largest_component": largest_component,
        "largest_component_cells": int(largest_mask.sum()),
        "root_candidate_cells": int(np.count_nonzero(start_mask & largest_mask)),
    }


def _load_source_pca(cache_dir: Path, integrated: ad.AnnData) -> ad.AnnData:
    with np.load(cache_dir / "source_pca.npz") as archive:
        source_barcodes = archive["barcodes"].astype(str)
        source_scores = np.asarray(archive["scores"], dtype=np.float32)
        variance = np.asarray(archive["variance"], dtype=np.float64)
    positions = pd.Index(source_barcodes).get_indexer(integrated.obs_names.astype(str))
    if np.any(positions < 0):
        raise KeyError("Source PCA cache does not cover the integrated barcodes")
    source = ad.AnnData(
        X=sparse.csr_matrix((integrated.n_obs, 1), dtype=np.float32),
        obs=integrated.obs.copy(),
        var=pd.DataFrame(index=["__scib_source_pca_placeholder__"]),
    )
    source.obsm["X_pca"] = source_scores[positions]
    source.uns["pca"] = {"variance": variance}
    return source


def _cell_cycle_from_cache(
    integrated: ad.AnnData,
    cache_dir: Path,
    batch_key: str,
) -> tuple[float, dict[str, Any]]:
    import scib

    with np.load(cache_dir / "cell_cycle_scores.npz") as archive:
        source_barcodes = archive["barcodes"].astype(str)
        s_score = np.asarray(archive["S_score"], dtype=float)
        g2m_score = np.asarray(archive["G2M_score"], dtype=float)
    before_by_batch = json.loads(
        (cache_dir / "cell_cycle_before.json").read_text(encoding="utf-8")
    )["before_by_batch"]
    positions = pd.Index(source_barcodes).get_indexer(integrated.obs_names.astype(str))
    if np.any(positions < 0):
        raise KeyError("Cell-cycle cache does not cover the integrated barcodes")
    aligned_s = s_score[positions]
    aligned_g2m = g2m_score[positions]
    batches = integrated.obs[batch_key].astype(str).to_numpy()
    per_batch: list[dict[str, Any]] = []
    scores: list[float] = []
    for batch in pd.unique(batches):
        mask = batches == batch
        covariate = pd.DataFrame(
            {"S_score": aligned_s[mask], "G2M_score": aligned_g2m[mask]}
        )
        after = float(
            scib.me.pc_regression(
                np.asarray(integrated.obsm[EMBEDDING_KEY][mask]),
                covariate,
                n_comps=50,
                linreg_method="numpy",
            )
        )
        before = float(before_by_batch[str(batch)])
        score = 1 - abs(after - before) / before
        if score < 0:
            score = 0.0
        scores.append(float(score))
        per_batch.append(
            {"batch": str(batch), "before": before, "after": after, "score": float(score)}
        )
    value = float(np.mean(scores))
    return value, {"per_batch": per_batch, "aggregation": "numpy.mean (official pipeline)"}


def compute_metric_task(
    *,
    task: str,
    spec: dict[str, Any],
    context: dict[str, Any],
    graph_dir: Path,
    cache_dir: Path,
) -> dict[str, dict[str, Any]]:
    if task not in TASK_METRICS:
        raise KeyError(f"Unknown official scIB metric task: {task}")
    import scib

    integrated = build_integrated(spec, with_graph=graph_dir if task in GRAPH_TASKS else None)
    batch_key = spec["batch_key"]
    label_key = spec["label_key"]
    if task == "clustering":
        resolution, _, profile = scib.me.cluster_optimal_resolution(
            integrated,
            label_key=label_key,
            cluster_key="cluster",
            metric=scib.me.nmi,
            use_rep=EMBEDDING_KEY,
            force=True,
            verbose=False,
            return_all=True,
        )
        details = {
            "optimized_resolution": float(resolution),
            "optimization_metric": "NMI arithmetic",
            "resolution_profile": profile.to_dict(orient="records"),
        }
        return {
            "NMI_cluster/label": _computed(
                "NMI_cluster/label",
                scib.me.nmi(
                    integrated,
                    cluster_key="cluster",
                    label_key=label_key,
                    implementation="arithmetic",
                ),
                "scib.me.cluster_optimal_resolution + scib.me.nmi",
                details,
            ),
            "ARI_cluster/label": _computed(
                "ARI_cluster/label",
                scib.me.ari(integrated, cluster_key="cluster", label_key=label_key),
                "scib.me.cluster_optimal_resolution + scib.me.ari",
                details,
            ),
        }
    if task == "silhouette":
        return {
            "ASW_label": _computed(
                "ASW_label",
                scib.me.silhouette(integrated, label_key=label_key, embed=EMBEDDING_KEY),
                "scib.me.silhouette",
            ),
            "ASW_label/batch": _computed(
                "ASW_label/batch",
                scib.me.silhouette_batch(
                    integrated,
                    batch_key=batch_key,
                    label_key=label_key,
                    embed=EMBEDDING_KEY,
                    return_all=False,
                    verbose=False,
                ),
                "scib.me.silhouette_batch",
            ),
        }
    if task == "pcr":
        source = _load_source_pca(cache_dir, integrated)
        return {
            "PCR_batch": _computed(
                "PCR_batch",
                scib.me.pcr_comparison(
                    source,
                    integrated,
                    embed=EMBEDDING_KEY,
                    covariate=batch_key,
                ),
                "scib.me.pcr_comparison with source-invariant PCA cache",
            )
        }
    if task == "cell_cycle":
        value, details = _cell_cycle_from_cache(integrated, cache_dir, batch_key)
        return {
            "cell_cycle_conservation": _computed(
                "cell_cycle_conservation",
                value,
                "scib.me.cell_cycle equivalent with source-invariant cache",
                details,
            )
        }
    if task in {"isolated_f1", "isolated_silhouette"}:
        from scib.metrics.isolated_labels import get_isolated_labels

        isolated = get_isolated_labels(integrated, label_key, batch_key, None, False)
        name = "isolated_label_F1" if task == "isolated_f1" else "isolated_label_silhouette"
        function = "scib.me.isolated_labels(cluster=True)" if task == "isolated_f1" else "scib.me.isolated_labels(cluster=False)"
        if not isolated:
            return {
                name: _undefined(
                    name,
                    "No labels satisfy the official isolated-label definition for this dataset.",
                    function,
                )
            }
        value = scib.me.isolated_labels(
            integrated,
            label_key=label_key,
            batch_key=batch_key,
            embed=EMBEDDING_KEY,
            cluster=task == "isolated_f1",
            iso_threshold=None,
            verbose=False,
        )
        return {name: _computed(name, value, function, {"isolated_labels": list(map(str, isolated))})}
    if task == "graph_conn":
        return {
            "graph_conn": _computed(
                "graph_conn",
                scib.me.graph_connectivity(integrated, label_key=label_key),
                "scib.me.graph_connectivity",
            )
        }
    if task == "kbet":
        return {
            "kBET": _computed(
                "kBET",
                scib.me.kBET(
                    integrated,
                    batch_key=batch_key,
                    label_key=label_key,
                    type_=context["output_type"],
                    embed=EMBEDDING_KEY,
                    scaled=True,
                    verbose=False,
                ),
                "scib.me.kBET backed by the official R kBET package",
            )
        }
    if task == "lisi":
        return {
            "iLISI": _computed(
                "iLISI",
                scib.me.ilisi_graph(
                    integrated,
                    batch_key=batch_key,
                    type_=context["output_type"],
                    use_rep=EMBEDDING_KEY,
                    subsample=context["lisi_subsample_percent"],
                    scale=True,
                    n_cores=context["lisi_cores"],
                    verbose=False,
                ),
                "scib.me.ilisi_graph",
            ),
            "cLISI": _computed(
                "cLISI",
                scib.me.clisi_graph(
                    integrated,
                    label_key=label_key,
                    type_=context["output_type"],
                    use_rep=EMBEDDING_KEY,
                    subsample=context["lisi_subsample_percent"],
                    scale=True,
                    n_cores=context["lisi_cores"],
                    verbose=False,
                ),
                "scib.me.clisi_graph",
            ),
        }
    if task == "trajectory":
        source = ad.AnnData(
            X=sparse.csr_matrix((integrated.n_obs, 1), dtype=np.float32),
            obs=integrated.obs.copy(),
            var=pd.DataFrame(index=["__scib_trajectory_placeholder__"]),
        )
        support = trajectory_root_support(integrated, label_key)
        if support["root_candidate_cells"] == 0:
            return {
                "trajectory": _undefined(
                    "trajectory",
                    (
                        "Official scIB 0.2.0 trajectory_conservation is undefined: "
                        "its canonical start cluster has no cell in the largest "
                        "connected component, so no root cell can be selected."
                    ),
                    "scib.me.trajectory_conservation root-selection domain",
                    details=support,
                )
            }
        return {
            "trajectory": _computed(
                "trajectory",
                scib.me.trajectory_conservation(source, integrated, label_key=label_key),
                "scib.me.trajectory_conservation",
                details={"root_support": support},
            )
        }
    raise AssertionError(f"Unhandled metric task {task}")


def paper_environment_root(repo_root: Path) -> Path:
    return repo_root / "environments" / "scib-paper" / ".pixi" / "envs" / "default"


def paper_worker_command(repo_root: Path, *arguments: str) -> list[str]:
    environment = paper_environment_root(repo_root)
    python = environment / "bin" / "python"
    worker = repo_root / "scripts" / "paper_scib_worker.py"
    if not python.is_file():
        raise FileNotFoundError(
            "Locked paper-scIB Python is missing; run "
            "PIXI_CACHE_DIR=$PWD/.pixi-cache pixi install --manifest-path "
            "environments/scib-paper/pixi.toml"
        )
    return [str(python), str(worker), *arguments]


def worker_environment(repo_root: Path, *, backend: str | None = None) -> dict[str, str]:
    environment = os.environ.copy()
    source = str(repo_root / "src")
    environment["PYTHONPATH"] = source + (
        os.pathsep + environment["PYTHONPATH"] if environment.get("PYTHONPATH") else ""
    )
    environment.setdefault("MPLCONFIGDIR", str(repo_root / ".cache" / "matplotlib"))
    environment.setdefault("XDG_CACHE_HOME", str(repo_root / ".cache"))
    pixi = (
        paper_environment_root(repo_root)
        if backend == PAPER_BACKEND
        else repo_root / ".pixi" / "envs" / "default"
    )
    if pixi.is_dir():
        environment["CONDA_PREFIX"] = str(pixi)
        environment["R_HOME"] = str(pixi / "lib" / "R")
        environment["R_LIBS"] = str(pixi / "lib" / "R" / "library")
        environment["PATH"] = str(pixi / "bin") + os.pathsep + environment.get("PATH", "")
        library_paths = [str(pixi / "lib" / "R" / "lib"), str(pixi / "lib")]
        if environment.get("LD_LIBRARY_PATH"):
            library_paths.append(environment["LD_LIBRARY_PATH"])
        environment["LD_LIBRARY_PATH"] = os.pathsep.join(library_paths)
    return environment


def run_isolated_task(
    command: list[str],
    *,
    repo_root: Path,
    log_path: Path,
    timeout_seconds: int,
    backend: str | None = None,
) -> dict[str, Any]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        completed = subprocess.run(
            command,
            cwd=repo_root,
            env=worker_environment(repo_root, backend=backend),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=timeout_seconds,
            check=False,
        )
        output = completed.stdout or ""
        log_path.write_text(output, encoding="utf-8")
        return {"returncode": int(completed.returncode), "timed_out": False, "output": output}
    except subprocess.TimeoutExpired as exc:
        output = (exc.stdout or "") if isinstance(exc.stdout, str) else ""
        log_path.write_text(output + "\nTASK TIMED OUT\n", encoding="utf-8")
        return {"returncode": None, "timed_out": True, "output": output}


def prepare_paper_source_cache(
    *,
    cache_dir: Path,
    repo_root: Path,
    dataset_id: str,
    dataset_path: Path,
    dataset_sha256: str,
    batch_key: str,
    assay: str,
    organism: str | None,
    timeout_seconds: int,
    force: bool = False,
) -> dict[str, Any]:
    """Prepare source invariants with the locked paper backend, atomically."""
    transaction = begin_output(cache_dir, force=force)
    if transaction.reused:
        status = json.loads((cache_dir / "status.json").read_text(encoding="utf-8"))
        if status.get("backend") != PAPER_BACKEND:
            raise ValueError(
                f"Refusing source cache from backend {status.get('backend')!r}; "
                f"expected {PAPER_BACKEND!r}"
            )
        return status
    assert transaction.staging is not None
    staging = transaction.staging
    command = paper_worker_command(
        repo_root,
        "--task",
        "source_cache",
        "--graph-dir",
        str(staging / "unused_graph"),
        "--source-cache",
        str(staging),
        "--output",
        str(staging / "status.json"),
        "--dataset-id",
        dataset_id,
        "--dataset-path",
        str(dataset_path),
        "--dataset-sha256",
        dataset_sha256,
        "--batch-key",
        batch_key,
        "--assay",
        assay,
        *(tuple(("--organism", organism)) if organism is not None else ()),
    )
    execution = run_isolated_task(
        command,
        repo_root=repo_root,
        log_path=staging / "source_cache.log",
        timeout_seconds=timeout_seconds,
        backend=PAPER_BACKEND,
    )
    if (staging / "status.json").is_file():
        status = json.loads((staging / "status.json").read_text(encoding="utf-8"))
    else:
        status = {
            "schema_version": 2,
            "state": "failed",
            "backend": PAPER_BACKEND,
            "scib_commit": PAPER_SCIB_COMMIT,
            "dataset_id": dataset_id,
            "errors": {
                "source_cache": (
                    "Paper source-cache worker timed out"
                    if execution["timed_out"]
                    else f"Paper source-cache worker exited with code {execution['returncode']}"
                )
            },
        }
        write_json(staging / "status.json", status)
    commit_output(transaction)
    return status


def verify_paper_environment(
    repo_root: Path,
    output_path: Path,
    *,
    timeout_seconds: int = 120,
) -> dict[str, Any]:
    """Execute and persist the locked paper-backend environment check."""
    command = paper_worker_command(
        repo_root,
    )[:1] + [str(repo_root / "scripts" / "check_paper_scib_environment.py")]
    execution = run_isolated_task(
        command,
        repo_root=repo_root,
        log_path=output_path.with_suffix(".log"),
        timeout_seconds=timeout_seconds,
        backend=PAPER_BACKEND,
    )
    if execution["returncode"] != 0:
        raise RuntimeError(
            "Locked paper-scIB environment validation failed; see "
            f"{output_path.with_suffix('.log')}"
        )
    payload = json.loads(execution["output"])
    lock_path = repo_root / "environments" / "scib-paper" / "pixi.lock"
    digest = hashlib.sha256(lock_path.read_bytes()).hexdigest()
    payload["pixi_manifest"] = "environments/scib-paper/pixi.toml"
    payload["pixi_lock"] = "environments/scib-paper/pixi.lock"
    payload["pixi_lock_sha256"] = digest
    write_json(output_path, payload)
    return payload


def failed_task_records(task: str, reason: str) -> dict[str, dict[str, Any]]:
    return {
        name: metric_record(
            name,
            "failed",
            reason=reason,
            implementation="isolated official scIB metric worker",
            evidence="Worker exit status and task log.",
        )
        for name in TASK_METRICS[task]
    }


def flatten_scib_result(result: dict[str, Any]) -> dict[str, Any]:
    kbet_details = result["official_scib_metrics"]["kBET"].get("details", {})
    kbet_fallback = kbet_details.get("fallback") is True
    row = {
        "job_index": result["job_index"],
        "job_id": result["job_id"],
        "dataset_id": result["dataset_id"],
        "method": result["method"],
        "track": result["track"],
        "protocol": result["protocol"],
        "supervision": result["supervision"],
        "label_fraction": result["label_fraction"],
        "preprocessing_profile": result["preprocessing_profile"],
        "fit_scope": result.get("fit_scope"),
        "seed": result["seed"],
        "anchor_seed": result["anchor_seed"],
        "scoring_backend": result["scoring_backend"],
        "scib_commit": result["scib_commit"],
        "kbet_commit": result["kbet_commit"],
        "kbet_backend": kbet_details.get(
            "metric_backend",
            "r_kbet_0_99_6_afc5f431",
        ),
        "kbet_fallback": kbet_fallback,
        "paper_environment_lock_sha256": result["paper_environment"]["pixi_lock_sha256"],
        "state": result["state"],
    }
    for name in OFFICIAL_METRICS:
        record = result["official_scib_metrics"][name]
        row[name] = record["value"]
        row[f"status::{name}"] = record["status"]
    return row
