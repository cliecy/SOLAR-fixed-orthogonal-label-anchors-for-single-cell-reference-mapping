#!/usr/bin/env python3
"""Score explicitly registered v1.2 embeddings in a Slurm allocation.

Historical official records are audited, never inferred from aggregate tables.
New official metrics run through the frozen paper worker, not installed scib.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import copy
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import sys
import tempfile
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import anndata as ad
import numpy as np
import pandas as pd
import yaml
from sklearn.manifold import trustworthiness
from sklearn.metrics import f1_score, recall_score
from sklearn.neighbors import KNeighborsClassifier

from biovalid_ref_query_joint import (
    MAX_SILHOUETTE_CELLS, RNG_SEED as BIOLOGICAL_SAMPLE_SEED,
    pseudotime_smoothness, within_label_silhouette,
)
from scib_benchmark.audit import audit_official_scib_score
from scib_benchmark.baseline_runners import _select_genes_and_pca
from scib_benchmark.config import BenchmarkConfig
from scib_benchmark.scib_contract import OFFICIAL_METRICS
from scib_benchmark.scib_runner import score_job
from scib_benchmark.scib_scoring import (
    PAPER_BACKEND, PAPER_KBET_COMMIT, PAPER_SCIB_COMMIT, scoring_context,
    verify_paper_environment,
)

COLUMNS = ["run_id", "method", "dimension", "dataset_id", "heldout_batch", "seed",
           "metric", "population", "value", "status", "reason", "implementation", "comparison_eligible", "n_labels_tested"]
ALIASES = {"ASW_label": "labelASW", "ASW_label/batch": "batchASW"}
METHOD_DIMENSIONS = {
    "solar_orthogonal": {30, 128}, "solar_orthogonal_uniq": {128}, "solar_none": {128},
    "sclsc_refonly": {16, 30}, "frozen_reference_scvi": {30},
    "frozen_reference_scanvi": {30}, "unintegrated_pca_matched": {30, 40},
}
COVARIATES = {"study": "immune_cell_human", "sample_ID": "immune_cell_human",
              "donor": "lung_atlas", "patientGroup": "lung_atlas"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def resolve(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def inside(path: Path, root: Path) -> Path:
    path = path.resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError(f"Scoring output escapes version root: {path}")
    return path


def token(value: Any) -> str:
    value = str(value)
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", value) or value in {".", ".."}:
        raise ValueError(f"Unsafe file identity: {value!r}")
    return value


@contextmanager
def lock(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def atomic_json(path: Path, payload: Any, *, identical: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = (json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    try:
        if identical:
            try:
                os.link(temporary, path)
            except FileExistsError:
                if path.read_bytes() != data:
                    raise ValueError(f"Frozen identity mismatch: {path}")
        else:
            os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_csv(path: Path, frame: pd.DataFrame, *, compressed: bool = False) -> None:
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
    try:
        frame.to_csv(temporary, index=False,
                     compression={"method": "gzip", "mtime": 0} if compressed else None)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def freeze_copy(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    with lock(target.with_name(target.name + ".lock")):
        if target.exists():
            if sha256(target) != sha256(source):
                raise ValueError(f"Frozen cache differs: {target}")
            return
        with tempfile.NamedTemporaryFile(dir=target.parent, delete=False) as handle:
            temporary = Path(handle.name)
        try:
            shutil.copyfile(source, temporary)
            os.replace(temporary, target)
        finally:
            temporary.unlink(missing_ok=True)


def metric_row(record: dict, metric: str, population: str, value: float | None,
               reason: str, status: str = "computed") -> dict:
    if status == "computed" and (value is None or not np.isfinite(value)):
        raise ValueError(f"Nonfinite computed {metric}/{population}")
    if status != "computed" and value is not None:
        raise ValueError("Missing metric cannot contain an imputed value")
    if not reason:
        raise ValueError("Every metric requires a provenance or missingness reason")
    return {**{key: record[key] for key in COLUMNS[:6]}, "metric": metric,
            "population": population, "value": value, "status": status, "reason": reason,
            "implementation": "", "comparison_eligible": True}


def read_split(record: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    split = pd.read_csv(record["split_path"], dtype=str, keep_default_na=False)
    if not {"barcode", "role"} <= set(split):
        raise ValueError("Split requires barcode and role")
    if split.barcode.eq("").any() or split.barcode.duplicated().any():
        raise ValueError("Split contains empty or duplicate cell IDs")
    if set(split.role) != {"reference", "query"}:
        raise ValueError("Split must contain only nonempty reference and query populations")
    split = pd.concat([split[split.role == "reference"], split[split.role == "query"]],
                      ignore_index=True)
    source = ad.read_h5ad(record["dataset_path"], backed="r")
    try:
        obs = source.obs.copy()
        obs.index = obs.index.astype(str)
    finally:
        source.file.close()
    if not obs.index.is_unique or set(obs.index) != set(split.barcode):
        raise ValueError("Split and source must contain identical unique cell-ID sets")
    obs = obs.loc[split.barcode].copy()
    for key in (record["label_key"], record["batch_key"]):
        if key not in obs or obs[key].isna().any():
            raise ValueError(f"Missing source annotation: {key}")
    labels = obs[record["label_key"]].astype(str).to_numpy()
    batches = obs[record["batch_key"]].astype(str).to_numpy()
    if "label" in split and not np.array_equal(split.label.to_numpy(), labels):
        raise ValueError("Frozen split labels disagree with source metadata")
    if "batch" in split and not np.array_equal(split.batch.to_numpy(), batches):
        raise ValueError("Frozen split batches disagree with source metadata")
    if not np.array_equal(split.role.eq("query").to_numpy(), batches == str(record["heldout_batch"])):
        raise ValueError("Query is not exactly the registered held-out batch")
    split["label"] = labels
    return split, obs


def matrix(values: np.ndarray, names: np.ndarray, expected: np.ndarray, dimension: int) -> np.ndarray:
    names = np.asarray(names).astype(str)
    values = np.asarray(values, dtype=np.float32)
    if names.ndim != 1 or values.shape != (len(names), dimension):
        raise ValueError("Embedding dimension/ID shape does not match manifest")
    index = pd.Index(names)
    if not index.is_unique or set(names) != set(expected):
        raise ValueError("Embedding and frozen split cell-ID sets differ or contain duplicates")
    if not np.isfinite(values).all():
        raise ValueError("Embedding contains nonfinite values")
    return values[index.get_indexer(expected)]


def read_embedding(record: dict, split: pd.DataFrame) -> np.ndarray:
    ids = split.barcode.to_numpy(dtype=str)
    dimension = int(record["dimension"])
    n_reference = int(split.role.eq("reference").sum())
    with np.load(record["embedding_path"], allow_pickle=False) as archive:
        keys = set(archive.files)
        if {"reference", "query", "reference_barcodes", "query_barcodes"} <= keys:
            reference = matrix(archive["reference"], archive["reference_barcodes"], ids[:n_reference], dimension)
            query = matrix(archive["query"], archive["query_barcodes"], ids[n_reference:], dimension)
            return np.concatenate([reference, query])
        if {"embedding", "barcodes"} <= keys:
            return matrix(archive["embedding"], archive["barcodes"], ids, dimension)
    raise ValueError("Unknown NPZ schema; expected split or joint embedding arrays")


def ensure_pca(record: dict, split: pd.DataFrame, output_root: Path, inputs: dict) -> None:
    path = Path(record["embedding_path"])
    if record["method"] != "unintegrated_pca_matched" or int(record["dimension"]) != 30:
        if not path.is_file():
            raise FileNotFoundError(path)
        return
    inside(path, output_root)
    provenance = path.with_name("pca_complete.json")
    identity = {key: inputs[key] for key in ("dataset", "split", "features", "config", "source_manifest")}
    identity.update({"dimension": 30, "seed": int(record["seed"]), "fit_scope": "reference_only"})
    with lock(path.with_name("pca.lock")):
        if path.exists():
            if not provenance.is_file() or load_json(provenance) != {"inputs": identity, "embedding_sha256": sha256(path)}:
                raise ValueError("PCA embedding lacks matching frozen-input provenance")
            return
        if provenance.exists():
            raise ValueError("PCA completion exists but embedding is missing")
        source = ad.read_h5ad(record["dataset_path"])
        working, embedding, metadata = _select_genes_and_pca(
            source, fit_scope="reference_only", split=split, batch_key=record["batch_key"],
            n_hvg=2000, hvg_flavor="seurat", batch_aware=False, n_components=30,
            seed=int(record["seed"]), gene_file=Path(record["features_path"]),
        )
        embedding = matrix(embedding, working.obs_names.to_numpy(), split.barcode.to_numpy(), 30)
        n_reference = int(split.role.eq("reference").sum())
        with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".npz", delete=False) as handle:
            temporary = Path(handle.name)
        try:
            np.savez_compressed(temporary, reference=embedding[:n_reference], query=embedding[n_reference:],
                                reference_barcodes=split.barcode.to_numpy(dtype=str)[:n_reference],
                                query_barcodes=split.barcode.to_numpy(dtype=str)[n_reference:])
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)
        atomic_json(path.with_name("pca_preprocessing.json"), metadata)
        atomic_json(provenance, {"inputs": identity, "embedding_sha256": sha256(path)})


def classify(record: dict, split: pd.DataFrame, embedding: np.ndarray) -> tuple[list[dict], pd.DataFrame]:
    n_reference = int(split.role.eq("reference").sum())
    if n_reference < 15:
        raise ValueError("Full-reference 15NN requires at least 15 reference cells")
    ref_labels = split.label.to_numpy(dtype=str)[:n_reference]
    truth = split.label.to_numpy(dtype=str)[n_reference:]
    classifier = KNeighborsClassifier(n_neighbors=15, weights="distance", metric="minkowski", p=2, n_jobs=1)
    classifier.fit(embedding[:n_reference], ref_labels)
    query = embedding[n_reference:]
    prediction = classifier.predict(query).astype(str)
    confidence = classifier.predict_proba(query).max(axis=1)
    distances, indices = classifier.kneighbors(query)
    labels = np.unique(truth)
    rows = [metric_row(record, "macro_f1", "query",
                       float(f1_score(truth, prediction, labels=labels, average="macro", zero_division=0)),
                       "Full-reference distance-weighted 15NN, Minkowski p=2; labels=unique(query truth); zero_division=0"),
            metric_row(record, "balanced_accuracy", "query",
                       float(recall_score(truth, prediction, labels=labels, average="macro", zero_division=0)),
                       "Mean recall over query true classes, including classes absent from reference")]
    ref_ids = split.barcode.to_numpy(dtype=str)[:n_reference]
    predictions = pd.DataFrame({**{key: record[key] for key in COLUMNS[:6]},
        "barcode": split.barcode.to_numpy(dtype=str)[n_reference:], "true_label": truth,
        "prediction": prediction, "confidence": confidence,
        "unseen_reference_label": ~np.isin(truth, np.unique(ref_labels)), "unassigned": False,
        "neighbor_ids": [json.dumps(ref_ids[index].tolist()) for index in indices],
        "neighbor_distances": [json.dumps(distance.tolist()) for distance in distances],
        "readout": "full_reference_distance_weighted_15nn_minkowski_p2"})
    return rows, predictions


def shared_sample(record: dict, population: str, ids: np.ndarray, output_root: Path) -> np.ndarray:
    ordered = np.sort(ids.astype(str))
    selected = np.random.default_rng(40).choice(len(ordered), size=min(6000, len(ordered)), replace=False)
    sample_ids = ordered[np.sort(selected)]
    payload = {"schema_version": 1, "dataset_id": record["dataset_id"],
               "heldout_batch": str(record["heldout_batch"]), "population": population,
               "seed": 40, "neighbors": 15, "maximum_cells": 6000,
               "population_ids_sha256": hashlib.sha256(json.dumps(ordered.tolist()).encode()).hexdigest(),
               "cell_ids": sample_ids.tolist()}
    name = f"{token(record['dataset_id'])}__{token(record['heldout_batch'])}__{population}.json"
    atomic_json(output_root / "evaluation_samples" / name, payload, identical=True)
    return pd.Index(ids).get_indexer(sample_ids)


def biological_metrics(record: dict, split: pd.DataFrame, obs: pd.DataFrame, embedding: np.ndarray,
                       source_path: Path, output_root: Path) -> list[dict]:
    source = None
    trust_applicable = record["dataset_id"] in {"immune_cell_human", "lung_atlas"}
    if trust_applicable and source_path.is_file():
        with np.load(source_path, allow_pickle=False) as archive:
            source = matrix(archive["scores"], archive["barcodes"], split.barcode.to_numpy(), 50)
    rows = []
    for population in ("reference", "query", "joint"):
        mask = np.ones(len(split), dtype=bool) if population == "joint" else split.role.eq(population).to_numpy()
        ids = split.barcode.to_numpy(dtype=str)[mask]
        emb = embedding[mask]
        if not trust_applicable:
            rows.append(metric_row(record, "trustworthiness", population, None,
                                   "Protocol restricts cached-source trustworthiness to Immune and Lung", "not_applicable"))
        else:
            sample = shared_sample(record, population, ids, output_root)
            if len(ids) < 31:
                rows.append(metric_row(record, "trustworthiness", population, None,
                                       "Population has fewer than 31 cells; fixed k=15 is undefined", "not_applicable"))
            elif source is None:
                rows.append(metric_row(record, "trustworthiness", population, None,
                                       f"Frozen 50-component source PCA artifact is missing: {source_path}", "artifact_missing"))
            else:
                value = float(trustworthiness(source[mask][sample], emb[sample], n_neighbors=15, metric="euclidean"))
                rows.append(metric_row(record, "trustworthiness", population, value,
                                       "Cached full-source PCA50; Euclidean k15; frozen shared IDs, seed40, maximum6000"))
        reason = None
        missing_status = "not_applicable"
        if record["dataset_id"] != "immune_cell_human":
            reason = "Original pseudotime smoothness protocol applies only to Immune"
        elif "dpt_pseudotime" not in obs:
            reason = "Source metadata has no dpt_pseudotime field"
            missing_status = "artifact_missing"
        elif len(emb) < 16:
            reason = "Population has fewer than 16 cells for pseudotime k15"
        else:
            pt = pd.to_numeric(obs.loc[ids, "dpt_pseudotime"], errors="coerce").to_numpy(dtype=float)
            annotated = ~np.isnan(pt)
            if annotated.sum() < 16:
                reason = "Original annotated-cell subset has fewer than 16 cells for fixed pseudotime k15"
            elif not np.isfinite(pt[annotated]).all():
                reason = "Annotated pseudotime input contains nonfinite values"
            elif np.unique(pt[annotated]).size < 2:
                reason = "Annotated pseudotime input is constant; Spearman correlation is undefined"
            else:
                value = pseudotime_smoothness(emb, pt, 15)
                if value is None or not np.isfinite(value):
                    reason = "Original neighbour-mean pseudotime correlation is undefined (constant neighbour mean)"
                else:
                    rows.append(metric_row(record, "pseudotime_smoothness", population, float(value),
                                           f"Original NaN-excluded annotated-cell population ({int(annotated.sum())}/{len(pt)} cells); biovalid_ref_query_joint.pseudotime_smoothness k15; original first-neighbor removal"))
        if reason:
            rows.append(metric_row(record, "pseudotime_smoothness", population, None, reason, missing_status))
    return rows


def within_label_selection(record, split, obs, population, field, output_root):
    """Freeze the original per-label cap, independently of method and dimension."""
    mask = np.ones(len(split), dtype=bool) if population == "joint" else split.role.eq(population).to_numpy()
    ids = split.barcode.to_numpy(dtype=str)[mask]
    labels = split.label.to_numpy(dtype=str)[mask]
    if field not in obs or obs.loc[ids, field].isna().any():
        raise ValueError(f"Required historical covariate annotation is missing: {field}")
    groups = obs.loc[ids, field].to_numpy()
    rng = np.random.default_rng(BIOLOGICAL_SAMPLE_SEED)
    selected, capped = [], {}
    for label in pd.unique(labels):
        positions = np.flatnonzero(labels == label)
        if len(positions) > MAX_SILHOUETTE_CELLS and pd.unique(groups[positions]).size >= 2:
            positions = positions[rng.choice(len(positions), size=MAX_SILHOUETTE_CELLS, replace=False)]
            capped[str(label)] = ids[positions].tolist()
        selected.extend(positions.tolist())
    payload = {"schema_version": 1, "dataset_id": record["dataset_id"], "heldout_batch": record["heldout_batch"],
               "population": population, "field": field, "sample_seed": BIOLOGICAL_SAMPLE_SEED,
               "maximum_cells_per_label": MAX_SILHOUETTE_CELLS, "capped_label_cell_ids": capped,
               "uncapped_labels_use_all_cells": True,
               "population_ids_sha256": hashlib.sha256(json.dumps(ids.tolist()).encode()).hexdigest()}
    filename = f"{token(record['dataset_id'])}__{token(record['heldout_batch'])}__{population}__{field}.json"
    path = output_root / "within_label_samples" / filename
    atomic_json(path, payload, identical=True)
    population_positions = np.flatnonzero(mask)
    return population_positions[np.asarray(selected, dtype=int)], path


def covariate_metrics(record, split, obs, embedding, selections):
    rows = []
    for population in ("reference", "query", "joint"):
        for field, dataset in COVARIATES.items():
            metric = "within_label_silhouette__" + field
            if record["dataset_id"] != dataset:
                rows.append(metric_row(record, metric, population, None,
                    f"Historical {field} panel is defined only for {dataset}", "not_applicable"))
                continue
            selected = selections[(population, field)]
            ids = split.barcode.to_numpy(dtype=str)[selected]
            labels = split.label.to_numpy(dtype=str)[selected]
            result = within_label_silhouette(embedding[selected], labels, obs.loc[ids, field].to_numpy(),
                                            np.random.default_rng(BIOLOGICAL_SAMPLE_SEED))
            if result is None:
                row = metric_row(record, metric, population, None,
                    "No cell type has at least 10 cells and a mathematically valid multi-group silhouette",
                    "not_applicable")
                row["n_labels_tested"] = 0
            else:
                value, count = result
                row = metric_row(record, metric, population, float(value),
                    f"Original within_label_silhouette helper; cell-count weighting; labels_tested={count}; "
                    f"shared original seed{BIOLOGICAL_SAMPLE_SEED} per-label cap{MAX_SILHOUETTE_CELLS}; "
                    "raw covariate association, not uniformly higher-is-better")
                row["n_labels_tested"] = int(count)
            row["implementation"] = "scripts/biovalid_ref_query_joint.py::within_label_silhouette"
            rows.append(row)
    return rows


def legacy_result(record: dict, inputs: dict, project_root: Path) -> dict:
    path = Path(record["legacy_scib_path"])
    for key in ("legacy_scib_spec_path", "source_run_id", "source_job_id"):
        if not record.get(key):
            raise ValueError(f"Historical official reuse requires record.{key}")
    spec_path = resolve(project_root, record["legacy_scib_spec_path"])
    spec = load_json(spec_path)
    if spec["job_id"] != record["source_job_id"]:
        raise ValueError("Historical source-qualified job identity mismatch")
    if record["source_run_id"] not in path.parts or record["source_run_id"] not in spec_path.parts:
        raise ValueError("Historical raw/spec paths do not belong to declared source run")
    errors = audit_official_scib_score(spec, path)
    if errors:
        raise ValueError("Historical official record failed audit: " + "; ".join(errors))
    expected = {"dataset_id": record["dataset_id"], "variant": record["method"],
                "seed": int(record["seed"]), "dataset_sha256": inputs["dataset"],
                "batch_key": record["batch_key"], "label_key": record["label_key"]}
    for key, value in expected.items():
        if spec.get(key) != value:
            raise ValueError(f"Historical scIB spec {key} differs from evaluation identity")
    split_id = f"heldout_{record['heldout_batch']}"
    matched_pca = record["method"] == "unintegrated_pca_matched"
    if matched_pca:
        if spec.get("preprocessing_profile") != f"hvg2000_pca40_matched__{split_id}":
            raise ValueError("Historical registered PCA spec has the wrong held-out profile")
    elif spec.get("split_id") != split_id:
        raise ValueError("Historical scIB spec has the wrong split ID")
    for key, expected_hash in (("split_file", inputs["split"]), ("gene_file", inputs["features"])):
        if matched_pca and key not in spec:
            # Registered matched PCA specs predate these fields. The frozen
            # manifest split and the original embedding are validated by IDs.
            continue
        if sha256(resolve(project_root, spec[key])) != expected_hash:
            raise ValueError(f"Historical scIB {key} differs from frozen evaluation input")
    original_embedding = resolve(project_root, spec["output_dir"]) / "embedding.npz"
    if sha256(original_embedding) != inputs["embedding"]:
        raise ValueError("Historical scIB embedding differs from evaluation input")
    return load_json(path)


def official_metrics(record: dict, inputs: dict, config: dict, snapshot_root: Path,
                     project_root: Path, output_root: Path, directory: Path, index: int) -> tuple[list[dict], dict]:
    if record.get("legacy_scib_path"):
        result = legacy_result(record, inputs, project_root)
        provenance = {"source_run_id": record["source_run_id"], "source_job_id": result["job_id"],
                      "raw_path": record["legacy_scib_path"], "raw_sha256": inputs["legacy_scib"]}
    else:
        benchmark_path = resolve(snapshot_root, config["solar_config"])
        benchmark_data = yaml.safe_load(benchmark_path.read_text())
        benchmark_data = copy.deepcopy(benchmark_data)
        benchmark_data["output_root"] = str(output_root)
        # Historical non-equivalent kBET values are preserved only when already audited.
        # New failures must be repaired, not silently replaced with a different algorithm.
        benchmark_data["evaluation"]["official_scib"]["kbet_python_fallback"]["enabled"] = False
        context = scoring_context(benchmark_data, record["dataset_id"])
        if context["neighbors"] != 15 or context["output_type"] != "embed":
            raise ValueError("Official scoring must retain embedding output and 15 neighbors")
        workspace = f"official_scib/{token(record['run_id'])}"
        score_root = inside(output_root / workspace / "scib_scores", output_root)
        original_root = resolve(project_root, config["historical_run_root"]) / "scib_scores"
        environment = load_json(output_root / "official_environment.json")
        if environment.get("state") != "passed" or environment.get("backend") != PAPER_BACKEND:
            raise ValueError("Historical paper environment has not passed")
        if environment.get("pixi_lock_sha256") != sha256(snapshot_root / "environments/scib-paper/pixi.lock"):
            raise ValueError("Frozen paper environment lock differs from audited environment")
        freeze_copy(output_root / "official_environment.json", score_root / "paper_environment.json")
        cache_name = f"{record['dataset_id']}__{inputs['dataset'][:16]}"
        original_cache = original_root / "source_cache" / PAPER_BACKEND / cache_name
        cache_status = load_json(original_cache / "status.json")
        if any(cache_status.get(key) != value for key, value in {
            "state": "completed", "backend": PAPER_BACKEND, "scib_commit": PAPER_SCIB_COMMIT,
            "dataset_sha256": inputs["dataset"], "dataset_id": record["dataset_id"],
            "batch_key": record["batch_key"],
        }.items()):
            raise ValueError("Paper source-invariant cache provenance mismatch")
        for name in ("status.json", "source_pca.npz", "pcr_before.json", "cell_cycle_scores.npz", "cell_cycle_before.json"):
            freeze_copy(original_cache / name, score_root / "source_cache" / PAPER_BACKEND / cache_name / name)
        staged = directory / "official_input"
        staged.mkdir(exist_ok=True)
        freeze_copy(Path(record["embedding_path"]), staged / "embedding.npz")
        spec = {"schema_version": 1, "job_index": index, "job_id": f"v1_2_score__{record['run_id']}",
            "run_id": record["run_id"], "engine": "external_v1_2", "dataset_id": record["dataset_id"],
            "dataset_path": record["dataset_path"], "dataset_sha256": inputs["dataset"],
            "batch_key": record["batch_key"], "label_key": record["label_key"], "variant": record["method"],
            "track": "heldout", "protocol": "inductive_held_out_batch", "supervision": "reference_labels",
            "label_fraction": 1.0, "preprocessing_profile": "hvg2000_pca40", "fit_scope": "reference_only",
            "seed": int(record["seed"]), "anchor_seed": None, "split_id": f"heldout_{record['heldout_batch']}",
            "split_file": record["split_path"], "gene_file": record["features_path"],
            "output_dir": str(staged), "scib_output_type": "embed", "dimension": int(record["dimension"]),
            "input_embedding_sha256": inputs["embedding"], "source_manifest_sha256": inputs["source_manifest"],
            "config_sha256": inputs["config"], "input_hashes": inputs}
        spec_path = directory / "official_job.json"
        atomic_json(spec_path, spec, identical=True)
        runtime_config = directory / "official_config.json"
        atomic_json(runtime_config, benchmark_data, identical=True)
        benchmark = BenchmarkConfig(runtime_config, snapshot_root, benchmark_data,
                                    resolve(snapshot_root, config["dataset_manifest"]), {})
        result = score_job(benchmark, workspace, spec_path)
        raw = score_root / "raw_jobs" / f"{index:06d}.json"
        errors = audit_official_scib_score(spec, raw)
        if errors:
            raise RuntimeError("Repairable official-scIB scoring failure: " + "; ".join(errors))
        provenance = {"source_run_id": workspace, "source_job_id": spec["job_id"],
                      "raw_path": str(raw), "raw_sha256": sha256(raw)}
    rows = []
    for name in OFFICIAL_METRICS:
        metric = result["official_scib_metrics"][name]
        reason = metric.get("reason", "")
        fallback = bool(metric.get("details", {}).get("fallback"))
        if fallback:
            reason += "; HISTORICAL NON-EQUIVALENT FALLBACK: " + json.dumps(metric["details"], sort_keys=True)
        for metric_name in (name, ALIASES[name]) if name in ALIASES else (name,):
            row = metric_row(record, metric_name, "joint", metric["value"], reason, metric["status"])
            row["implementation"] = metric.get("implementation", "")
            row["comparison_eligible"] = not fallback
            rows.append(row)
    provenance.update({"backend": result["scoring_backend"], "scib_commit": result["scib_commit"],
                       "kbet_commit": result["kbet_commit"], "paper_environment": result["paper_environment"]})
    return rows, provenance


def score_record(record: dict, index: int, config: dict, manifest: dict,
                 config_path: Path, manifest_path: Path, output_root: Path) -> dict:
    snapshot_root, project_root = Path(manifest["snapshot_root"]), Path(manifest["project_root"])
    directory = inside(output_root / "scores" / token(record["run_id"]), output_root)
    directory.mkdir(parents=True, exist_ok=True)
    with lock(directory / "score.lock"):
        paths = {"config": config_path, "manifest": manifest_path,
                 "source_manifest": Path(manifest["source_manifest"]),
                 "scorer": Path(__file__), "dataset": Path(record["dataset_path"]),
                 "split": Path(record["split_path"]), "features": Path(record["features_path"]),
                 "official_config": resolve(snapshot_root, config["solar_config"])}
        for name in ("scripts/biovalid_ref_query_joint.py", "scripts/paper_scib_worker.py",
                     "src/scib_benchmark/scib_runner.py", "src/scib_benchmark/scib_scoring.py",
                     "src/scib_benchmark/scib_contract.py", "src/scib_benchmark/baseline_runners.py"):
            paths[f"source::{name}"] = snapshot_root / name
        if record.get("legacy_scib_path"):
            paths["legacy_scib"] = Path(record["legacy_scib_path"])
            if not record.get("legacy_scib_spec_path"):
                raise ValueError("Historical record requires legacy_scib_spec_path")
            paths["legacy_scib_spec"] = resolve(project_root, record["legacy_scib_spec_path"])
        inputs = {key: sha256(path) for key, path in paths.items()}
        inputs["verified_paper_environment"] = sha256(output_root / "official_environment.json")
        for field, key in (("dataset_sha256", "dataset"), ("data_sha256", "dataset"),
                           ("split_sha256", "split"), ("features_sha256", "features")):
            if record.get(field) and record[field] != inputs[key]:
                raise ValueError(f"Frozen {field} no longer matches artifact")
        if not record.get("legacy_scib_path"):
            paper_root = resolve(project_root, config["historical_run_root"]) / "scib_scores"
            inputs["paper_environment"] = sha256(paper_root / "paper_environment.json")
            inputs["paper_lock"] = sha256(snapshot_root / "environments/scib-paper/pixi.lock")
            paper_cache = paper_root / "source_cache" / PAPER_BACKEND / f"{record['dataset_id']}__{inputs['dataset'][:16]}"
            for name in ("status.json", "source_pca.npz", "pcr_before.json",
                         "cell_cycle_scores.npz", "cell_cycle_before.json"):
                inputs[f"paper_cache::{name}"] = sha256(paper_cache / name)
        source_dir = resolve(project_root, config["evaluation"]["source_cache"]) / f"{record['dataset_id']}__{inputs['dataset'][:16]}"
        source_path = source_dir / "source_pca.npz"
        inputs["trust_source"] = sha256(source_path) if source_path.is_file() else None
        if source_path.is_file():
            status = load_json(source_dir / "status.json")
            if status.get("dataset_sha256") != inputs["dataset"] or status.get("pcr_source_cache", {}).get("n_components") != 50:
                raise ValueError("Trust source PCA cache identity or 50D definition differs")
            inputs["trust_source_status"] = sha256(source_dir / "status.json")
        split, obs = read_split(record)
        if record["dataset_id"] in {"immune_cell_human", "lung_atlas"}:
            for population in ("reference", "query", "joint"):
                selected_ids = split.barcode if population == "joint" else split.loc[split.role == population, "barcode"]
                shared_sample(record, population, selected_ids.to_numpy(dtype=str), output_root)
                name = f"{token(record['dataset_id'])}__{token(record['heldout_batch'])}__{population}.json"
                inputs[f"sample_{population}"] = sha256(output_root / "evaluation_samples" / name)
        covariate_selections = {}
        for field, dataset in COVARIATES.items():
            if record["dataset_id"] != dataset:
                continue
            for population in ("reference", "query", "joint"):
                selected, path = within_label_selection(record, split, obs, population, field, output_root)
                covariate_selections[(population, field)] = selected
                inputs[f"within_sample::{population}::{field}"] = sha256(path)
        ensure_pca(record, split, output_root, inputs)
        inputs["embedding"] = sha256(Path(record["embedding_path"]))
        if record.get("embedding_sha256") and record["embedding_sha256"] != inputs["embedding"]:
            raise ValueError("Embedding differs from frozen manifest hash")
        completion_path = directory / "scores_complete.json"
        if completion_path.exists():
            complete = load_json(completion_path)
            if complete.get("inputs") != inputs or complete.get("run_id") != record["run_id"]:
                raise ValueError("Completed scores have different frozen inputs; refusing stale reuse")
            for name, expected in complete["outputs"].items():
                if sha256(directory / name) != expected:
                    raise ValueError(f"Completed score artifact was modified: {name}")
            raw = load_json(directory / "backend.json")
            if sha256(Path(raw["raw_path"])) != raw["raw_sha256"]:
                raise ValueError("Completed official raw record was modified")
            return complete
        atomic_json(directory / "score_inputs.json", inputs, identical=True)
        embedding = read_embedding(record, split)
        rows, predictions = classify(record, split, embedding)
        rows.extend(biological_metrics(record, split, obs, embedding, source_path, output_root))
        rows.extend(covariate_metrics(record, split, obs, embedding, covariate_selections))
        official, backend = official_metrics(record, inputs, config, snapshot_root, project_root,
                                             output_root, directory, index)
        rows.extend(official)
        frame = pd.DataFrame(rows, columns=COLUMNS)
        if frame.duplicated(["run_id", "metric", "population"]).any():
            raise ValueError("Duplicate metric panels")
        atomic_csv(directory / "metrics.csv", frame)
        atomic_csv(directory / "predictions.csv.gz", predictions, compressed=True)
        atomic_json(directory / "backend.json", backend)
        complete = {"schema_version": 1, "state": "completed", "run_id": record["run_id"],
                    "inputs": inputs, "slurm_job_id": os.environ["SLURM_JOB_ID"],
                    "outputs": {name: sha256(directory / name) for name in
                                ("metrics.csv", "predictions.csv.gz", "backend.json")},
                    "n_reference": int(split.role.eq("reference").sum()),
                    "n_query": int(split.role.eq("query").sum()), "dimension": int(record["dimension"]),
                    "missing_metrics": frame.loc[frame.status != "computed", ["metric", "population", "status", "reason"]].to_dict("records")}
        atomic_json(completion_path, complete)
        return complete


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--run-id")
    selection.add_argument("--all", action="store_true", help="Explicitly score all registered records in this Slurm job")
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("All v1.2 scoring, including PCA fitting, requires a Slurm allocation")
    config_path, manifest_path = args.config.resolve(), args.manifest.resolve()
    config, manifest = yaml.safe_load(config_path.read_text()), load_json(manifest_path)
    if manifest.get("schema_version") != 1 or manifest.get("config_sha256") != sha256(config_path):
        raise ValueError("Manifest schema/config identity mismatch")
    if Path(manifest["snapshot_root"]).resolve() != ROOT:
        raise ValueError("Scoring must execute the registered immutable source snapshot")
    if not Path(manifest["project_root"]).is_absolute():
        raise ValueError("Manifest project_root must be absolute")
    evaluation = config["evaluation"]
    if (evaluation["official_scib_backend"], evaluation["official_scib_commit"], evaluation["kbet_commit"]) != (PAPER_BACKEND, PAPER_SCIB_COMMIT, PAPER_KBET_COMMIT):
        raise ValueError("Official scoring backend differs from frozen paper contract")
    if (evaluation["n_neighbors"], evaluation["metric"], evaluation["p"], evaluation["zero_division"]) != (15, "minkowski", 2, 0):
        raise ValueError("Classification contract differs from fixed distance-weighted 15NN p2")
    if evaluation["classifier"] != "distance_weighted_15nn" or evaluation["macro_f1_labels"] != "unique_query_true_labels":
        raise ValueError("Classifier or macro-F1 label-set contract differs")
    trust_contract = {"source_components": 50, "neighbors": 15, "max_cells": 6000,
                      "sample_seed": 40, "minimum_cells": 31,
                      "datasets": ["immune_cell_human", "lung_atlas"],
                      "populations": ["reference", "query", "joint"]}
    if any(evaluation["trustworthiness"].get(key) != value for key, value in trust_contract.items()):
        raise ValueError("Trustworthiness configuration differs from frozen shared-sample contract")
    pseudotime_contract = {"dataset": "immune_cell_human", "field": "dpt_pseudotime",
                           "implementation": "scripts/biovalid_ref_query_joint.py::pseudotime_smoothness",
                           "neighbors": 15}
    if any(evaluation["pseudotime"].get(key) != value for key, value in pseudotime_contract.items()):
        raise ValueError("Pseudotime configuration differs from original helper contract")
    covariate_contract = {"sample_seed": BIOLOGICAL_SAMPLE_SEED, "max_cells_per_label": MAX_SILHOUETTE_CELLS,
                          "fields": {"immune_cell_human": ["study", "sample_ID"], "lung_atlas": ["donor", "patientGroup"]}}
    if evaluation.get("within_label_silhouette") != covariate_contract:
        raise ValueError("Historical covariate panel or original sampling parameters differ")
    records = manifest["evaluation_runs"]
    required = set(COLUMNS[:6]) | {"stochastic", "embedding_path", "split_path", "dataset_path",
                                        "label_key", "batch_key", "features_path", "output_dir", "legacy_scib_path"}
    ids, keys, pca_keys = set(), set(), set()
    for record in records:
        if required - set(record):
            raise ValueError(f"Evaluation record missing fields: {sorted(required - set(record))}")
        run_id = token(record["run_id"])
        key = (record["method"], int(record["dimension"]), record["dataset_id"], str(record["heldout_batch"]), int(record["seed"]))
        if run_id in ids or key in keys:
            raise ValueError("Duplicate run ID or evaluation key")
        ids.add(run_id)
        keys.add(key)
        if int(record["dimension"]) not in METHOD_DIMENSIONS.get(record["method"], set()):
            raise ValueError(f"Unexpected method/dimension: {key}")
        if record["dataset_id"] not in config["datasets"] or str(record["heldout_batch"]) not in map(str, config["datasets"][record["dataset_id"]]):
            raise ValueError("Unregistered dataset/batch")
        is_pca = record["method"] == "unintegrated_pca_matched"
        if not isinstance(record["stochastic"], bool) or record["stochastic"] == is_pca:
            raise ValueError("PCA must be deterministic and all other methods stochastic")
        if is_pca:
            pca_key = key[:4]
            if pca_key in pca_keys or int(record["seed"]) not in {0, 40}:
                raise ValueError("PCA must occur once per dimension/split, not as replicated seeds")
            pca_keys.add(pca_key)
        elif int(record["seed"]) not in config["seeds"]:
            raise ValueError("Unregistered stochastic seed")
        for field in ("embedding_path", "split_path", "dataset_path", "features_path", "output_dir"):
            if not Path(record[field]).is_absolute():
                raise ValueError(f"Manifest {field} must be absolute")
    selected = [(index, record) for index, record in enumerate(records)
                if args.all or record["run_id"] == args.run_id]
    if not selected or (not args.all and len(selected) != 1):
        raise ValueError("--run-id must select exactly one registered evaluation record")
    output_root = resolve(Path(manifest["project_root"]), config["output_root"])
    environment_path = output_root / "official_environment.json"
    with lock(output_root / "official_environment.lock"):
        if not environment_path.is_file():
            verify_paper_environment(ROOT, environment_path)
        environment = load_json(environment_path)
        if (environment.get("state"), environment.get("backend"), environment.get("scib_commit"),
            environment.get("kbet_commit")) != ("passed", PAPER_BACKEND, PAPER_SCIB_COMMIT, PAPER_KBET_COMMIT):
            raise ValueError("Actual locked scoring environment has not passed the required backend audit")
    for index, record in selected:
        complete = score_record(record, index, config, manifest, config_path, manifest_path, output_root)
        print(json.dumps({"run_id": complete["run_id"], "state": complete["state"],
                          "scores_complete": str(output_root / "scores" / record["run_id"] / "scores_complete.json")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
