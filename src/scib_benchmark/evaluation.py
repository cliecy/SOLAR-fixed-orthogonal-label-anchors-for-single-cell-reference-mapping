from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
from sklearn.metrics import (
    adjusted_rand_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    normalized_mutual_info_score,
    recall_score,
)
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier

from .artifacts import write_json
from .data import read_split
from .splits import label_budget, smoke_subset


def _bool_series(values: pd.Series) -> pd.Series:
    if values.dtype == bool:
        return values
    normalized = values.astype(str).str.lower()
    if not normalized.isin({"true", "false"}).all():
        raise ValueError("labeled column must contain only true/false values")
    return normalized == "true"


def label_transfer_metrics(
    reference_embedding: np.ndarray,
    reference_labels: np.ndarray,
    query_embedding: np.ndarray,
    query_labels: np.ndarray,
    n_neighbors: int = 15,
) -> dict[str, Any]:
    if len(reference_embedding) != len(reference_labels):
        raise ValueError("Reference embedding/label lengths differ")
    if len(query_embedding) != len(query_labels):
        raise ValueError("Query embedding/label lengths differ")
    if len(reference_embedding) < 2 or len(query_embedding) < 1:
        raise ValueError("Label transfer requires at least two reference and one query cell")
    neighbors = min(int(n_neighbors), len(reference_embedding))
    classifier = KNeighborsClassifier(n_neighbors=neighbors, weights="distance")
    classifier.fit(reference_embedding, reference_labels)
    prediction = classifier.predict(query_embedding)
    labels = sorted(set(query_labels.astype(str)) | set(prediction.astype(str)))
    per_class = recall_score(
        query_labels, prediction, labels=labels, average=None, zero_division=0
    )
    return {
        "macro_f1": float(f1_score(query_labels, prediction, average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(query_labels, prediction)),
        "n_neighbors": int(neighbors),
        "classes": labels,
        "per_class_recall": {label: float(value) for label, value in zip(labels, per_class, strict=True)},
        "confusion_matrix": confusion_matrix(query_labels, prediction, labels=labels).tolist(),
        "predictions": prediction.astype(str).tolist(),
    }


def _run_metric(
    name: str,
    function: Callable[[], Any],
    values: dict[str, Any],
    errors: dict[str, str],
) -> None:
    try:
        result = function()
        if isinstance(result, dict):
            values.update({f"{name}_{key}": float(value) for key, value in result.items()})
        else:
            values[name] = float(result)
    except Exception as exc:
        errors[name] = f"{type(exc).__name__}: {exc}"


def scib_embedding_diagnostics(
    embedding: np.ndarray,
    labels: np.ndarray,
    batches: np.ndarray,
    n_neighbors: int = 15,
    seed: int = 42,
) -> tuple[dict[str, float], dict[str, str]]:
    from scib_metrics import (
        graph_connectivity,
        ilisi_knn,
        kbet,
        silhouette_batch,
        silhouette_label,
    )
    from scib_metrics.nearest_neighbors import pynndescent

    values: dict[str, float] = {}
    errors: dict[str, str] = {}
    encoded_labels = pd.factorize(labels.astype(str), sort=True)[0]
    encoded_batches = pd.factorize(batches.astype(str), sort=True)[0]
    clusters = KMeans(
        n_clusters=len(set(encoded_labels.tolist())),
        random_state=int(seed),
        n_init=10,
    ).fit_predict(embedding)
    values["kmeans_ari"] = float(adjusted_rand_score(encoded_labels, clusters))
    values["kmeans_nmi"] = float(normalized_mutual_info_score(encoded_labels, clusters))
    _run_metric(
        "label_asw",
        lambda: silhouette_label(embedding, encoded_labels),
        values,
        errors,
    )
    batch_count_by_label = [
        len(set(encoded_batches[encoded_labels == label].tolist()))
        for label in sorted(set(encoded_labels.tolist()))
    ]
    if any(count > 1 for count in batch_count_by_label):
        _run_metric(
            "batch_asw",
            lambda: silhouette_batch(embedding, encoded_labels, encoded_batches),
            values,
            errors,
        )
    graph = None
    try:
        graph = pynndescent(
            embedding,
            n_neighbors=min(int(n_neighbors), len(embedding) - 1),
            random_state=int(seed),
            n_jobs=1,
        )
    except Exception as exc:
        errors["neighbors"] = f"{type(exc).__name__}: {exc}"
    if graph is not None:
        _run_metric(
            "graph_connectivity",
            lambda: graph_connectivity(graph, encoded_labels),
            values,
            errors,
        )
        if len(set(encoded_batches.tolist())) > 1:
            _run_metric("ilisi", lambda: ilisi_knn(graph, encoded_batches), values, errors)
            _run_metric(
                "kbet",
                lambda: kbet(graph, encoded_batches)[0],
                values,
                errors,
            )
    return values, errors


def evaluate_job(
    spec: dict[str, Any],
    n_neighbors: int = 15,
    graph_neighbors: int = 15,
    metric_seed: int = 42,
    include_scib_metrics: bool = True,
) -> dict[str, Any]:
    output = Path(spec["output_dir"])
    status = json.loads((output / "status.json").read_text(encoding="utf-8"))
    if status.get("state") != "completed":
        raise RuntimeError(f"Job output is not completed: {output}")
    with np.load(output / "embedding.npz") as archive:
        reference = np.asarray(archive["reference"], dtype=np.float32)
        query = np.asarray(archive["query"], dtype=np.float32)
        reference_barcodes = archive["reference_barcodes"].astype(str)
        query_barcodes = archive["query_barcodes"].astype(str)
    if not np.isfinite(reference).all() or not np.isfinite(query).all():
        raise ValueError("Embedding contains non-finite values")
    split = read_split(Path(spec["split_file"])).set_index("barcode")
    smoke_cap = spec.get("smoke_cells_per_role_batch_label")
    if smoke_cap is not None:
        split = smoke_subset(
            split.reset_index(), int(smoke_cap), int(spec["seed"])
        ).set_index("barcode")
    try:
        reference_meta = split.loc[reference_barcodes]
        query_meta = split.loc[query_barcodes]
    except KeyError as exc:
        raise KeyError("Embedding barcodes do not align to split manifest") from exc
    if not (reference_meta["role"] == "reference").all() or not (
        query_meta["role"] == "query"
    ).all():
        raise ValueError("Embedding reference/query roles disagree with split")
    if smoke_cap is None:
        budget = pd.read_csv(spec["label_budget_file"], dtype={"barcode": str}).set_index("barcode")
    else:
        budget = label_budget(
            split.reset_index(), float(spec["label_fraction"]), int(spec["seed"])
        ).set_index("barcode")
    labeled_names = budget.index[_bool_series(budget["labeled"])]
    reference_index = pd.Index(reference_barcodes)
    labeled_positions = reference_index.get_indexer(labeled_names)
    if np.any(labeled_positions < 0):
        raise KeyError("Label budget contains cells absent from reference embedding")
    transfer = label_transfer_metrics(
        reference[labeled_positions],
        reference_meta.loc[labeled_names, "label"].astype(str).to_numpy(),
        query,
        query_meta["label"].astype(str).to_numpy(),
        n_neighbors=n_neighbors,
    )
    diagnostics: dict[str, float] = {}
    metric_errors: dict[str, str] = {}
    if include_scib_metrics:
        joint = np.concatenate([reference, query], axis=0)
        joint_meta = pd.concat([reference_meta, query_meta])
        diagnostics, metric_errors = scib_embedding_diagnostics(
            joint,
            joint_meta["label"].astype(str).to_numpy(),
            joint_meta["batch"].astype(str).to_numpy(),
            n_neighbors=graph_neighbors,
            seed=metric_seed,
        )
    return {
        "schema_version": 1,
        "metric_contract": "legacy_partial_embedding_diagnostics_not_official_scib",
        "paper_comparable_scib_score": False,
        "job_id": spec["job_id"],
        "job_index": spec["job_index"],
        "dataset_id": spec["dataset_id"],
        "method": spec["variant"],
        "track": spec["track"],
        "protocol": spec["protocol"],
        "supervision": spec["supervision"],
        "label_fraction": spec["label_fraction"],
        "preprocessing_profile": spec["preprocessing_profile"],
        "seed": spec["seed"],
        "anchor_seed": spec["anchor_seed"],
        "execution_profile": spec.get("execution_profile", "formal"),
        "query_expression_used_for_training": False,
        "query_labels_visible_to_model": False,
        "split_uses_query_labels": spec["split_uses_query_labels"],
        "label_transfer": transfer,
        "embedding_diagnostics": diagnostics,
        "metric_errors": metric_errors,
        "state": "completed" if not metric_errors else "completed_with_metric_errors",
    }


def write_metric_result(path: Path, result: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_json(path, result)
