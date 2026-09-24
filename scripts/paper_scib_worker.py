#!/usr/bin/env python
"""Isolated metric worker executed inside the locked scIB paper environment.

This file intentionally remains Python 3.7 compatible. Metric calls are delegated
to the exact scIB source commit used by the paper environment rather than copied
or reimplemented here.
"""
from __future__ import print_function

import argparse
import json
import os
from pathlib import Path
import shutil
import sys
import traceback

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse

import scIB


BACKEND = "paper_scib_0_2_0"
SCIB_COMMIT = "e2a37e0ed63dc34b60aa535cc656400552af757a"
EMBEDDING_KEY = "X_emb"
TASK_METRICS = {
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


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(str(temporary), str(path))


def record(name, value, implementation, details=None):
    numeric = float(value)
    if not np.isfinite(numeric):
        raise ValueError("Paper scIB metric {} returned a non-finite value".format(name))
    return {
        "name": name,
        "status": "computed",
        "value": numeric,
        "reason": "Computed by the locked scIB paper backend.",
        "implementation": implementation,
        "evidence": "theislab/scib commit {} and scib-pipeline scripts/metrics.py".format(
            SCIB_COMMIT
        ),
        "details": details or {},
    }


def undefined(name, reason, implementation, details=None):
    return {
        "name": name,
        "status": "not_applicable",
        "value": None,
        "reason": reason,
        "implementation": implementation,
        "evidence": "Observed metadata structure and scIB 0.2.0 metric definition.",
        "details": details or {},
    }


def trajectory_root_support(integrated, label_key):
    """Describe the root-cell domain used by scIB 0.2.0 trajectory conservation."""
    from scipy.sparse.csgraph import connected_components

    pseudotime_mask = integrated.obs["dpt_pseudotime"].notnull().to_numpy()
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
        csgraph=trajectory.uns["neighbors"]["connectivities"],
        directed=False,
        return_labels=True,
    )
    means = trajectory.obs.groupby(label_key)["dpt_pseudotime"].mean().dropna()
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


def close_backed(adata):
    file_manager = getattr(adata, "file", None)
    if file_manager is not None:
        file_manager.close()


def load_joint_embedding(spec):
    output = Path(spec["output_dir"])
    with np.load(str(output / "embedding.npz")) as archive:
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


def load_joint_obs(spec, barcodes):
    backed = ad.read_h5ad(spec["dataset_path"], backed="r")
    try:
        source_names = pd.Index(backed.obs_names.astype(str))
        if len(barcodes) != backed.n_obs or set(barcodes.tolist()) != set(source_names):
            raise ValueError("Embedding must cover every official dataset cell exactly once")
        positions = source_names.get_indexer(barcodes)
        if np.any(positions < 0):
            raise KeyError("Embedding barcodes do not align to the official AnnData")
        obs = backed.obs.iloc[positions].copy()
    finally:
        close_backed(backed)
    obs.index = pd.Index(barcodes.astype(str))
    for key in (spec["batch_key"], spec["label_key"]):
        if key not in obs:
            raise KeyError("Official AnnData lacks metadata key {!r}".format(key))
        if obs[key].isna().any():
            raise ValueError("Official scoring key {!r} contains missing values".format(key))
        obs[key] = obs[key].astype("category")
    return obs


def build_integrated(spec, graph_dir=None, source_cache=None):
    embedding, barcodes = load_joint_embedding(spec)
    obs = load_joint_obs(spec, barcodes)
    integrated = ad.AnnData(
        X=sparse.csr_matrix((len(obs), 1), dtype=np.float32),
        obs=obs,
        var=pd.DataFrame(index=["__scib_embedding_placeholder__"]),
    )
    integrated.obsm[EMBEDDING_KEY] = embedding
    if source_cache is not None:
        with np.load(str(Path(source_cache) / "source_pca.npz")) as archive:
            source_barcodes = archive["barcodes"].astype(str)
            source_scores = np.asarray(archive["scores"], dtype=np.float32)
        positions = pd.Index(source_barcodes).get_indexer(integrated.obs_names.astype(str))
        if np.any(positions < 0):
            raise KeyError("Source PCA cache does not cover all embedded barcodes")
        integrated.obsm["X_pca"] = source_scores[positions]
    if graph_dir is not None:
        graph_dir = Path(graph_dir)
        connectivities = sparse.load_npz(str(graph_dir / "connectivities.npz"))
        distances = sparse.load_npz(str(graph_dir / "distances.npz"))
        expected = (integrated.n_obs, integrated.n_obs)
        if connectivities.shape != expected or distances.shape != expected:
            raise ValueError("Cached graph dimensions do not match the integrated embedding")
        graph_meta = json.loads((graph_dir / "graph.json").read_text())
        params = graph_meta.get("params")
        if params is None and isinstance(graph_meta.get("neighbors_uns"), dict):
            params = graph_meta["neighbors_uns"].get("params", {})
        if params is None:
            params = {}
        integrated.uns["neighbors"] = {
            "connectivities": connectivities,
            "distances": distances,
            "params": params,
        }
    return integrated


def method_provided_graph_dir(spec):
    output = Path(spec["output_dir"])
    if (
        (output / "connectivities.npz").is_file()
        and (output / "distances.npz").is_file()
        and (output / "graph.json").is_file()
    ):
        return output
    return None


def prepare_neighbors(spec, graph_dir, n_neighbors):
    graph_dir = Path(graph_dir)
    graph_dir.mkdir(parents=True, exist_ok=True)
    provided = method_provided_graph_dir(spec)
    if provided is not None:
        connectivities = sparse.load_npz(str(provided / "connectivities.npz")).tocsr()
        distances = sparse.load_npz(str(provided / "distances.npz")).tocsr()
        source_meta = json.loads((provided / "graph.json").read_text())
        params = source_meta.get("params")
        if params is None and isinstance(source_meta.get("neighbors_uns"), dict):
            params = source_meta["neighbors_uns"].get("params", {})
        if params is None:
            params = {}
        for key, value in list(params.items()):
            if isinstance(value, np.generic):
                params[key] = value.item()
        conn_tmp = graph_dir / "connectivities.tmp.npz"
        dist_tmp = graph_dir / "distances.tmp.npz"
        sparse.save_npz(str(conn_tmp), connectivities)
        sparse.save_npz(str(dist_tmp), distances)
        os.replace(str(conn_tmp), str(graph_dir / "connectivities.npz"))
        os.replace(str(dist_tmp), str(graph_dir / "distances.npz"))
        payload = {
            "state": "completed",
            "backend": BACKEND,
            "scib_commit": SCIB_COMMIT,
            "n_obs": int(connectivities.shape[0]),
            "n_neighbors": int(n_neighbors),
            "use_rep": source_meta.get("use_rep", EMBEDDING_KEY),
            "random_state": source_meta.get("random_state"),
            "params": params,
            "connectivities_nnz": int(connectivities.nnz),
            "distances_nnz": int(distances.nnz),
            "implementation": source_meta.get("implementation", "method_provided_knn"),
            "source": "method_provided_knn",
            "source_dir": str(provided),
        }
        write_json(graph_dir / "graph.json", payload)
        return payload
    integrated = build_integrated(spec)
    sc.pp.neighbors(
        integrated,
        n_neighbors=int(n_neighbors),
        use_rep=EMBEDDING_KEY,
        random_state=0,
    )
    neighbors = integrated.uns["neighbors"]
    connectivities = neighbors["connectivities"].tocsr()
    distances = neighbors["distances"].tocsr()
    conn_tmp = graph_dir / "connectivities.tmp.npz"
    dist_tmp = graph_dir / "distances.tmp.npz"
    sparse.save_npz(str(conn_tmp), connectivities)
    sparse.save_npz(str(dist_tmp), distances)
    os.replace(str(conn_tmp), str(graph_dir / "connectivities.npz"))
    os.replace(str(dist_tmp), str(graph_dir / "distances.npz"))
    params = dict(neighbors.get("params", {}))
    for key, value in list(params.items()):
        if isinstance(value, np.generic):
            params[key] = value.item()
    payload = {
        "state": "completed",
        "backend": BACKEND,
        "scib_commit": SCIB_COMMIT,
        "n_obs": int(integrated.n_obs),
        "n_neighbors": int(n_neighbors),
        "use_rep": EMBEDDING_KEY,
        "random_state": 0,
        "params": params,
        "connectivities_nnz": int(connectivities.nnz),
        "distances_nnz": int(distances.nnz),
        "implementation": "scanpy 1.4.6 sc.pp.neighbors",
    }
    write_json(graph_dir / "graph.json", payload)
    return payload


def atomic_npz(path, **arrays):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp.npz")
    np.savez_compressed(str(temporary), **arrays)
    os.replace(str(temporary), str(path))


def prepare_source_cache(args):
    cache_dir = Path(args.source_cache)
    cache_dir.mkdir(parents=True, exist_ok=True)
    status = {
        "schema_version": 2,
        "state": "running",
        "backend": BACKEND,
        "scib_commit": SCIB_COMMIT,
        "dataset_id": args.dataset_id,
        "dataset_path": args.dataset_path,
        "dataset_sha256": args.dataset_sha256,
        "batch_key": args.batch_key,
        "assay": args.assay,
        "organism": args.organism,
        "errors": {},
    }
    write_json(cache_dir / "status.json", status)
    source = ad.read_h5ad(args.dataset_path)
    try:
        if args.batch_key not in source.obs:
            raise KeyError("Source AnnData lacks batch key {!r}".format(args.batch_key))
        source.obs[args.batch_key] = source.obs[args.batch_key].astype("category")
        matrix = source.X
        n_comps = min(50, min(matrix.shape))
        solver = "arpack"
        if n_comps == min(matrix.shape):
            solver = "full"
        pca = sc.tl.pca(
            matrix,
            n_comps=n_comps,
            use_highly_variable=False,
            return_info=True,
            svd_solver=solver,
            copy=True,
        )
        source_scores = np.asarray(pca[0], dtype=np.float32)
        pca_variance = np.asarray(pca[3], dtype=np.float64)
        pcr_before = float(
            scIB.me.pc_regression(
                source_scores,
                source.obs[args.batch_key],
                pca_sd=pca_variance,
                n_comps=n_comps,
                verbose=False,
            )
        )
        if not np.isfinite(pcr_before):
            raise ValueError("Source PCR is non-finite")
        write_json(cache_dir / "pcr_before.json", {"pcr_before": pcr_before})
        atomic_npz(
            cache_dir / "source_pca.npz",
            barcodes=np.asarray(source.obs_names.astype(str), dtype=str),
            scores=source_scores,
            variance=pca_variance,
        )
        status["pcr_source_cache"] = {
            "state": "completed",
            "pcr_before": pcr_before,
            "n_components": int(n_comps),
            "implementation": "Scanpy 1.4.6 PCA + scIB 0.2.0 pc_regression",
        }

        if args.assay in ("simulation", "atac"):
            status["cell_cycle_source_cache"] = {
                "state": "not_applicable",
                "reason": "The original scIB pipeline disables cell cycle for this assay.",
            }
        else:
            if args.organism not in ("human", "mouse"):
                raise ValueError("Unsupported organism {!r}".format(args.organism))
            scIB.me.precompute_cc_score(
                source,
                batch_key=args.batch_key,
                organism=args.organism,
                n_comps=50,
                verbose=False,
            )
            s_score = source.obs["S_score"].to_numpy(dtype=float)
            g2m_score = source.obs["G2M_score"].to_numpy(dtype=float)
            before_by_batch = {
                str(key): float(value) for key, value in source.uns["scores_before"].items()
            }
            if not np.isfinite(s_score).all() or not np.isfinite(g2m_score).all():
                raise ValueError("Cell-cycle scores contain non-finite values")
            if not all(np.isfinite(value) for value in before_by_batch.values()):
                raise ValueError("Cell-cycle source PCR contains non-finite values")
            atomic_npz(
                cache_dir / "cell_cycle_scores.npz",
                barcodes=np.asarray(source.obs_names.astype(str), dtype=str),
                S_score=s_score,
                G2M_score=g2m_score,
            )
            write_json(
                cache_dir / "cell_cycle_before.json",
                {"before_by_batch": before_by_batch},
            )
            status["cell_cycle_source_cache"] = {
                "state": "completed",
                "batches": len(before_by_batch),
                "implementation": "scIB 0.2.0 precompute_cc_score and pc_regression",
            }
        status["state"] = "completed"
    except Exception as exc:
        status["state"] = "failed"
        status["errors"]["source_cache"] = "{}: {}".format(type(exc).__name__, exc)
        traceback.print_exc()
    finally:
        del source
    write_json(cache_dir / "status.json", status)
    return status


def load_pcr_before(cache_dir):
    payload = json.loads((Path(cache_dir) / "pcr_before.json").read_text())
    return float(payload["pcr_before"])


def cell_cycle_from_cache(integrated, cache_dir, batch_key):
    cache_dir = Path(cache_dir)
    with np.load(str(cache_dir / "cell_cycle_scores.npz")) as archive:
        source_barcodes = archive["barcodes"].astype(str)
        s_score = np.asarray(archive["S_score"], dtype=float)
        g2m_score = np.asarray(archive["G2M_score"], dtype=float)
    before_by_batch = json.loads(
        (cache_dir / "cell_cycle_before.json").read_text()
    )["before_by_batch"]
    positions = pd.Index(source_barcodes).get_indexer(integrated.obs_names.astype(str))
    if np.any(positions < 0):
        raise KeyError("Cell-cycle cache does not cover all embedded barcodes")
    batches = integrated.obs[batch_key].astype(str).to_numpy()
    scores = []
    per_batch = []
    for batch in pd.unique(batches):
        mask = batches == batch
        covariate = pd.DataFrame(
            {
                "S_score": s_score[positions][mask],
                "G2M_score": g2m_score[positions][mask],
            }
        )
        after = float(
            scIB.me.pc_regression(
                np.asarray(integrated.obsm[EMBEDDING_KEY][mask]),
                covariate,
                n_comps=50,
                verbose=False,
            )
        )
        before = float(before_by_batch[str(batch)])
        score = 1.0 - abs(after - before) / before
        if score < 0:
            score = 0.0
        scores.append(float(score))
        per_batch.append(
            {"batch": str(batch), "before": before, "after": after, "score": score}
        )
    return float(np.mean(scores)), per_batch


def compute_task(task, spec, args):
    graph_tasks = set(("clustering", "isolated_f1", "graph_conn", "lisi", "trajectory"))
    integrated = build_integrated(
        spec,
        args.graph_dir if task in graph_tasks else None,
        args.source_cache if task == "isolated_silhouette" else None,
    )
    batch_key = spec["batch_key"]
    label_key = spec["label_key"]

    if task == "clustering":
        resolution, score_max, profile = scIB.cl.opt_louvain(
            integrated,
            label_key=label_key,
            cluster_key="cluster",
            function=scIB.me.nmi,
            method="arithmetic",
            resolutions=None,
            inplace=True,
            plot=False,
            force=True,
            verbose=False,
        )
        details = {
            "reported_resolution": float(resolution),
            "reported_score_max": float(score_max),
            "resolution_profile": profile.to_dict(orient="records"),
            "algorithm": "Louvain 0.6.1; 20 resolutions from 0.1 through 2.0",
            "compatibility_note": (
                "Runs the released scIB 0.2.0 opt_louvain code verbatim, including its "
                "scor_max/score_max assignment behavior."
            ),
        }
        return {
            "NMI_cluster/label": record(
                "NMI_cluster/label",
                scIB.me.nmi(integrated, "cluster", label_key, method="arithmetic"),
                "scIB 0.2.0 opt_louvain + nmi(arithmetic)",
                details,
            ),
            "ARI_cluster/label": record(
                "ARI_cluster/label",
                scIB.me.ari(integrated, "cluster", label_key),
                "scIB 0.2.0 opt_louvain + ari",
                details,
            ),
        }
    if task == "silhouette":
        sil_all, sil_means = scIB.me.silhouette_batch(
            integrated,
            batch_key=batch_key,
            group_key=label_key,
            embed=EMBEDDING_KEY,
            verbose=False,
        )
        return {
            "ASW_label": record(
                "ASW_label",
                scIB.me.silhouette(integrated, group_key=label_key, embed=EMBEDDING_KEY),
                "scIB 0.2.0 silhouette",
            ),
            "ASW_label/batch": record(
                "ASW_label/batch",
                sil_means["silhouette_score"].mean(),
                "scIB 0.2.0 silhouette_batch then mean over labels",
                {"scored_cells": int(len(sil_all)), "scored_labels": int(len(sil_means))},
            ),
        }
    if task == "pcr":
        before = load_pcr_before(args.source_cache)
        after = float(
            scIB.me.pcr(
                integrated,
                covariate=batch_key,
                embed=EMBEDDING_KEY,
                recompute_pca=True,
                n_comps=50,
                verbose=False,
            )
        )
        value = (before - after) / before
        if value < 0:
            value = 0.0
        return {
            "PCR_batch": record(
                "PCR_batch",
                value,
                "scIB 0.2.0 pcr_comparison with source-invariant pcr_before cache",
                {"pcr_before": before, "pcr_after": after},
            )
        }
    if task == "cell_cycle":
        value, per_batch = cell_cycle_from_cache(integrated, args.source_cache, batch_key)
        return {
            "cell_cycle_conservation": record(
                "cell_cycle_conservation",
                value,
                "scIB 0.2.0 cell_cycle with source-invariant score cache",
                {"per_batch": per_batch, "aggregation": "numpy.mean"},
            )
        }
    if task in ("isolated_f1", "isolated_silhouette"):
        labels = scIB.me.get_isolated_labels(
            integrated, label_key, batch_key, "iso_cluster", None, False
        )
        name = (
            "isolated_label_F1" if task == "isolated_f1" else "isolated_label_silhouette"
        )
        implementation = (
            "scIB 0.2.0 isolated_labels(cluster=True)"
            if task == "isolated_f1"
            else "scIB 0.2.0 isolated_labels(cluster=False; binary label-vs-rest ASW)"
        )
        if not labels:
            return {
                name: undefined(
                    name,
                    "No labels satisfy the scIB 0.2.0 isolated-label definition.",
                    implementation,
                )
            }
        value = scIB.me.isolated_labels(
            integrated,
            label_key=label_key,
            batch_key=batch_key,
            cluster=task == "isolated_f1",
            n=None,
            all_=False,
            verbose=False,
        )
        return {name: record(name, value, implementation, {"isolated_labels": labels})}
    if task == "graph_conn":
        return {
            "graph_conn": record(
                "graph_conn",
                scIB.me.graph_connectivity(integrated, label_key=label_key),
                "scIB 0.2.0 graph_connectivity",
            )
        }
    if task == "kbet":
        per_label = scIB.me.kBET(
            integrated,
            batch_key=batch_key,
            label_key=label_key,
            embed=EMBEDDING_KEY,
            type_=args.output_type,
            subsample=0.5,
            heuristic=True,
            verbose=False,
        )
        value = 1.0 - np.nanmean(per_label["kBET"].to_numpy(dtype=float))
        return {
            "kBET": record(
                "kBET",
                value,
                "scIB 0.2.0 kBET: 1 - mean(per-label R average.pval)",
                {
                    "per_label": per_label.astype(object).where(pd.notna(per_label), None).to_dict(
                        orient="records"
                    ),
                    "r_package": "kBET 0.99.6",
                },
            )
        }
    if task == "lisi":
        token = 3000000000 + int(spec["job_index"]) * 100000 + os.getpid()
        temp_prefix = "/tmp/lisi_tmp{}".format(token)
        original_time = scIB.me.time
        scIB.me.time = lambda: token
        try:
            ilisi, clisi = scIB.me.lisi_graph(
                integrated,
                batch_key=batch_key,
                label_key=label_key,
                k0=90,
                type_=args.output_type,
                subsample=int(args.lisi_subsample_percent),
                scale=True,
                multiprocessing=True,
                nodes=int(args.lisi_cores),
                verbose=False,
            )
        finally:
            scIB.me.time = original_time
            for candidate in Path("/tmp").glob("lisi_tmp{}*".format(token)):
                if candidate.is_dir() and str(candidate).startswith(temp_prefix):
                    shutil.rmtree(str(candidate))
        details = {
            "k0": 90,
            "subsample_percent": int(args.lisi_subsample_percent),
            "paper_scaling": "iLISI - 1; 2 - cLISI",
            "reproducibility_warning": (
                "The released paper C++ helper seeds 50% subsampling from std::random_device."
            ),
        }
        return {
            "iLISI": record("iLISI", ilisi, "scIB 0.2.0 graph LISI", details),
            "cLISI": record("cLISI", clisi, "scIB 0.2.0 graph LISI", details),
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
                "trajectory": undefined(
                    "trajectory",
                    (
                        "Official scIB 0.2.0 trajectory_conservation is undefined: "
                        "its canonical start cluster has no cell in the largest "
                        "connected component, so no root cell can be selected."
                    ),
                    "scIB 0.2.0 trajectory_conservation root-selection domain",
                    details=support,
                )
            }
        return {
            "trajectory": record(
                "trajectory",
                scIB.me.trajectory_conservation(source, integrated, label_key=label_key),
                "scIB 0.2.0 trajectory_conservation",
                details={"root_support": support},
            )
        }
    raise KeyError("Unknown task {!r}".format(task))


def parse_args():
    parser = argparse.ArgumentParser(description="Locked scIB paper metric worker")
    parser.add_argument("--task", required=True, choices=("source_cache", "neighbors") + tuple(TASK_METRICS))
    parser.add_argument("--job-spec", type=Path)
    parser.add_argument("--graph-dir", type=Path, required=True)
    parser.add_argument("--source-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--output-type", default="embed")
    parser.add_argument("--neighbors", type=int, default=15)
    parser.add_argument("--lisi-subsample-percent", type=int, default=50)
    parser.add_argument("--lisi-cores", type=int, default=1)
    parser.add_argument("--dataset-id")
    parser.add_argument("--dataset-path")
    parser.add_argument("--dataset-sha256")
    parser.add_argument("--batch-key")
    parser.add_argument("--assay")
    parser.add_argument("--organism")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.task == "source_cache":
        status = prepare_source_cache(args)
        return 0 if status["state"] == "completed" else 1
    if args.job_spec is None:
        raise ValueError("--job-spec is required for scoring tasks")
    spec = json.loads(args.job_spec.read_text())
    payload = {
        "schema_version": 2,
        "backend": BACKEND,
        "scib_commit": SCIB_COMMIT,
        "task": args.task,
        "job_id": spec["job_id"],
        "job_index": spec["job_index"],
        "state": "running",
        "metrics": {},
        "error": None,
    }
    write_json(args.output, payload)
    try:
        if args.task == "neighbors":
            payload["graph"] = prepare_neighbors(spec, args.graph_dir, args.neighbors)
        else:
            payload["metrics"] = compute_task(args.task, spec, args)
        payload["state"] = "completed"
    except Exception as exc:
        payload["state"] = "failed"
        payload["error"] = "{}: {}".format(type(exc).__name__, exc)
        traceback.print_exc()
    write_json(args.output, payload)
    return 0 if payload["state"] == "completed" else 1


if __name__ == "__main__":
    sys.exit(main())
