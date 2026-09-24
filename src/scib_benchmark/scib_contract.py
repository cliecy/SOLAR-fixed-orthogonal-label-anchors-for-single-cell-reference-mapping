from __future__ import annotations

import hashlib
import json
from typing import Any, Iterable

import numpy as np
import pandas as pd


BATCH_METRICS: tuple[str, ...] = (
    "PCR_batch",
    "ASW_label/batch",
    "iLISI",
    "graph_conn",
    "kBET",
)

BIOLOGY_METRICS: tuple[str, ...] = (
    "NMI_cluster/label",
    "ARI_cluster/label",
    "ASW_label",
    "isolated_label_F1",
    "isolated_label_silhouette",
    "cLISI",
    "hvg_overlap",
    "cell_cycle_conservation",
    "trajectory",
)

OFFICIAL_METRICS: tuple[str, ...] = BATCH_METRICS + BIOLOGY_METRICS
METRIC_STATUSES = frozenset({"computed", "not_applicable", "failed"})


def official_applicability(
    *,
    output_type: str,
    assay: str,
    n_obs: int,
    obs_columns: Iterable[str],
) -> dict[str, dict[str, Any]]:
    """Return the applicability contract used by the original scIB pipeline."""
    if output_type not in {"full", "embed", "knn"}:
        raise ValueError(f"Unsupported scIB output type: {output_type!r}")
    if assay not in {"expression", "atac", "simulation"}:
        raise ValueError(f"Unsupported scIB assay: {assay!r}")
    result = {
        name: {
            "applicable": True,
            "reason": "Enabled by the official scIB metric-selection rules.",
            "evidence": "scib-pipeline/scripts/metrics/metrics.py",
        }
        for name in OFFICIAL_METRICS
    }

    def disable(name: str, reason: str) -> None:
        result[name] = {
            "applicable": False,
            "reason": reason,
            "evidence": "scib-pipeline/scripts/metrics/metrics.py",
        }

    if output_type == "embed":
        disable("hvg_overlap", "The official scIB pipeline disables HVG overlap for embedding outputs.")
    elif output_type == "knn":
        for name in (
            "ASW_label/batch",
            "ASW_label",
            "PCR_batch",
            "cell_cycle_conservation",
            "isolated_label_silhouette",
            "hvg_overlap",
        ):
            disable(name, f"The official scIB pipeline disables {name} for kNN-only outputs.")

    if assay == "atac":
        disable("cell_cycle_conservation", "The official scIB pipeline disables cell cycle for scATAC-seq.")
        disable("hvg_overlap", "The official scIB pipeline disables HVG overlap for scATAC-seq.")
    elif assay == "simulation":
        disable("cell_cycle_conservation", "The official scIB pipeline disables cell cycle for simulations.")

    if "dpt_pseudotime" not in set(obs_columns):
        disable("trajectory", "The original AnnData has no obs['dpt_pseudotime'] column.")
    if int(n_obs) > 300_000:
        disable("kBET", "The official scIB pipeline disables kBET above 300,000 cells.")
    return result


def metric_record(
    name: str,
    status: str,
    *,
    value: float | None = None,
    reason: str,
    implementation: str,
    evidence: str,
    details: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if name not in OFFICIAL_METRICS:
        raise KeyError(f"Not an official scIB metric: {name}")
    if status not in METRIC_STATUSES:
        raise ValueError(f"Invalid metric status: {status}")
    numeric: float | None = None
    if value is not None:
        numeric = float(value)
    if status == "computed":
        if numeric is None or not np.isfinite(numeric):
            raise ValueError(f"Computed metric {name} must have a finite value")
    elif numeric is not None:
        raise ValueError(f"Metric {name} with status {status} cannot have a value")
    return {
        "name": name,
        "status": status,
        "value": numeric,
        "reason": reason,
        "implementation": implementation,
        "evidence": evidence,
        "details": details or {},
    }


def validate_metric_records(records: dict[str, dict[str, Any]]) -> None:
    missing = set(OFFICIAL_METRICS) - set(records)
    extra = set(records) - set(OFFICIAL_METRICS)
    if missing or extra:
        raise ValueError(
            f"Official metric schema mismatch; missing={sorted(missing)}, extra={sorted(extra)}"
        )
    for name in OFFICIAL_METRICS:
        record = records[name]
        if record.get("name") != name:
            raise ValueError(f"Metric record key/name mismatch for {name}")
        status = record.get("status")
        if status not in METRIC_STATUSES:
            raise ValueError(f"Metric {name} has invalid status {status!r}")
        value = record.get("value")
        if status == "computed":
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise ValueError(f"Computed metric {name} lacks a numeric value")
            if not np.isfinite(float(value)):
                raise ValueError(f"Computed metric {name} is non-finite")
        elif value is not None:
            raise ValueError(f"Non-computed metric {name} unexpectedly has a value")


def _rescale_series_like_scales(values: pd.Series) -> pd.Series:
    """Match scales::rescale, including its 0.5 result for a zero range."""
    result = pd.Series(np.nan, index=values.index, dtype=float)
    finite = values.notna() & np.isfinite(values.astype(float))
    if not finite.any():
        return result
    observed = values.loc[finite].astype(float)
    lower = float(observed.min())
    upper = float(observed.max())
    if np.isclose(lower, upper, rtol=0.0, atol=0.0):
        result.loc[finite] = 0.5
    else:
        result.loc[finite] = (observed - lower) / (upper - lower)
    return result


def minmax_scale_by_dataset(raw: pd.DataFrame) -> pd.DataFrame:
    required = {"dataset_id", *OFFICIAL_METRICS}
    missing = required - set(raw.columns)
    if missing:
        raise ValueError(f"Raw metric table lacks columns: {', '.join(sorted(missing))}")
    scaled = raw.copy()
    for metric in OFFICIAL_METRICS:
        scaled[metric] = raw.groupby("dataset_id", sort=False, group_keys=False)[metric].apply(
            _rescale_series_like_scales
        )
    return scaled


def _row_mean(frame: pd.DataFrame) -> pd.Series:
    result = frame.mean(axis=1, skipna=True)
    result.loc[frame.notna().sum(axis=1) == 0] = np.nan
    return result


def aggregate_official_scores(raw: pd.DataFrame) -> pd.DataFrame:
    """Apply the paper's dataset-wise min-max and 0.4/0.6 aggregation."""
    scaled = minmax_scale_by_dataset(raw)
    scaled["batch_correction"] = _row_mean(scaled.loc[:, list(BATCH_METRICS)])
    scaled["bio_conservation"] = _row_mean(scaled.loc[:, list(BIOLOGY_METRICS)])
    scaled["overall"] = (
        0.4 * scaled["batch_correction"] + 0.6 * scaled["bio_conservation"]
    )
    return scaled


def cohort_hash(raw: pd.DataFrame) -> str:
    identity = [
        name
        for name in (
            "job_id",
            "dataset_id",
            "method",
            "track",
            "protocol",
            "supervision",
            "label_fraction",
            "preprocessing_profile",
            "fit_scope",
            "seed",
            "anchor_seed",
            "scoring_backend",
            "scib_commit",
            "kbet_commit",
            "kbet_backend",
            "kbet_fallback",
            "paper_environment_lock_sha256",
        )
        if name in raw.columns
    ]
    status_columns = [f"status::{name}" for name in OFFICIAL_METRICS if f"status::{name}" in raw]
    columns = identity + list(OFFICIAL_METRICS) + status_columns
    ordered = raw.loc[:, columns].sort_values(identity or columns, kind="stable")
    payload = ordered.where(pd.notna(ordered), None).to_dict(orient="records")
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
