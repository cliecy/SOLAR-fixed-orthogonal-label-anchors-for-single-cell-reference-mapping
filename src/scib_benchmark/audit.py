from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import anndata as ad

from scib_data.validate import checksum

from .artifacts import write_json
from .config import BenchmarkConfig
from .data import read_split
from .scib_contract import validate_metric_records


def audit_preprocessing(config: BenchmarkConfig, run_id: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    run_root = config.output_root / run_id
    summary: dict[str, Any] = {}
    issues: list[dict[str, Any]] = []
    fractions = sorted(set(float(value) for value in config.data["protocol"]["label_fractions"]))
    for dataset_id in config.data["datasets"]:
        item = config.dataset(dataset_id)
        backed = ad.read_h5ad(config.dataset_path(dataset_id), backed="r")
        try:
            dataset_barcodes = pd.Index(backed.obs_names.astype(str))
            dataset_genes = pd.Index(backed.var_names.astype(str))
        finally:
            backed.file.close()
        dataset_result = {"splits": 0, "hvg_lists": 0, "singleton_reference_cells": 0}
        for seed in config.data["protocol"]["split_seeds"]:
            split_id = f"random_seed_{int(seed)}"
            split_dir = run_root / "splits" / dataset_id / split_id
            local_errors: list[str] = []
            try:
                manifest = json.loads((split_dir / "manifest.json").read_text(encoding="utf-8"))
                split_file = split_dir / "split.csv"
                split = read_split(split_file)
                if manifest.get("dataset_sha256") != item["checksum_sha256"]:
                    local_errors.append("split manifest dataset checksum mismatch")
                if checksum(split_file) != manifest.get("split_sha256"):
                    local_errors.append("split CSV checksum mismatch")
                if set(split["barcode"]) != set(dataset_barcodes):
                    local_errors.append("split does not cover exactly all dataset barcodes")
                reference_labels = set(split.loc[split["role"] == "reference", "label"].astype(str))
                query_labels = set(split.loc[split["role"] == "query", "label"].astype(str))
                if not query_labels <= reference_labels:
                    local_errors.append("query contains labels absent from reference")
                singleton_count = int(
                    split.get(
                        "singleton_stratum_assigned_to_reference",
                        pd.Series(False, index=split.index),
                    ).astype(bool).sum()
                )
                if singleton_count != manifest.get("singleton_strata_cells_assigned_to_reference"):
                    local_errors.append("singleton-stratum count mismatch")
                dataset_result["singleton_reference_cells"] += singleton_count
                reference_barcodes = set(split.loc[split["role"] == "reference", "barcode"])
                for fraction in fractions:
                    token = f"{fraction:.2f}"
                    budget_file = split_dir / f"label_budget_{token}.csv"
                    budget = pd.read_csv(budget_file, dtype={"barcode": str})
                    if checksum(budget_file) != manifest["label_budget_sha256"].get(token):
                        local_errors.append(f"label-budget {token} checksum mismatch")
                    if set(budget["barcode"]) != reference_barcodes:
                        local_errors.append(f"label-budget {token} does not cover reference exactly")
                    labeled = budget["labeled"] if budget["labeled"].dtype == bool else budget["labeled"].astype(str).str.lower() == "true"
                    if not labeled.any():
                        local_errors.append(f"label-budget {token} selects no cells")
                    selected = budget.loc[labeled]
                    if selected.groupby(["batch", "label"]).size().min() < 1:
                        local_errors.append(f"label-budget {token} leaves an empty reference stratum")
                hvg_file = run_root / "preprocessing" / dataset_id / split_id / "hvg2000_pca40.txt"
                hvg_meta_file = hvg_file.with_suffix(".json")
                genes = [line.strip() for line in hvg_file.read_text(encoding="utf-8").splitlines() if line.strip()]
                hvg_meta = json.loads(hvg_meta_file.read_text(encoding="utf-8"))
                expected_genes = min(2000, len(dataset_genes))
                if len(genes) != expected_genes or len(genes) != len(set(genes)):
                    local_errors.append("HVG list has incorrect length or duplicate genes")
                positions = dataset_genes.get_indexer(genes)
                if np.any(positions < 0) or np.any(np.diff(positions) < 0):
                    local_errors.append("HVG list is absent from or out of dataset gene order")
                if hvg_meta.get("reference_only") is not True:
                    local_errors.append("HVG metadata is not reference-only")
                if hvg_meta.get("dataset_sha256") != item["checksum_sha256"]:
                    local_errors.append("HVG metadata dataset checksum mismatch")
                dataset_result["splits"] += 1
                dataset_result["hvg_lists"] += 1
            except Exception as exc:
                local_errors.append(f"preprocessing audit exception: {type(exc).__name__}: {exc}")
            if local_errors:
                issues.append({"dataset_id": dataset_id, "split_id": split_id, "errors": local_errors})
        summary[dataset_id] = dataset_result
    return summary, issues


REQUIRED_OUTPUTS = ("embedding.npz", "pca_preprocessor.npz", "run_info.json", "environment.json", "status.json")
BASELINE_REQUIRED_OUTPUTS = ("embedding.npz", "run_info.json", "environment.json", "status.json")


def audit_baseline_job(spec: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    output = Path(spec["output_dir"])
    if not output.is_dir():
        return [f"missing output directory: {output}"]
    for filename in BASELINE_REQUIRED_OUTPUTS:
        if not (output / filename).is_file():
            errors.append(f"missing {filename}")
    if spec.get("output_kind") == "knn":
        for filename in ("connectivities.npz", "distances.npz", "graph.json"):
            if not (output / filename).is_file():
                errors.append(f"missing {filename}")
    if errors:
        return errors
    try:
        status = json.loads((output / "status.json").read_text(encoding="utf-8"))
        if status.get("state") != "completed":
            errors.append(f"status is {status.get('state')!r}")
        run_info = json.loads((output / "run_info.json").read_text(encoding="utf-8"))
        for key in (
            "dataset_sha256",
            "split_id",
            "protocol",
            "supervision",
            "fit_scope",
            "query_expression_used_for_preprocessing_fit",
            "query_labels_visible_to_model",
        ):
            if key not in run_info:
                errors.append(f"run_info lacks {key}")
        if run_info.get("protocol") != "transductive_scib":
            errors.append("baseline protocol is not transductive_scib")
        if run_info.get("dataset_sha256") != spec["dataset_sha256"]:
            errors.append("dataset checksum provenance mismatch")
        expected_query_fit = run_info.get("fit_scope") == "classic_full"
        if run_info.get("query_expression_used_for_preprocessing_fit") is not expected_query_fit:
            errors.append("preprocessing fit-scope provenance mismatch")
        expected_label_visibility = spec.get("supervision") == "all_labels"
        if run_info.get("query_labels_visible_to_model") is not expected_label_visibility:
            errors.append("label-visibility provenance mismatch")
        with np.load(output / "embedding.npz") as archive:
            if "embedding" not in archive.files or "barcodes" not in archive.files:
                errors.append("baseline embedding.npz lacks embedding/barcodes")
            else:
                embedding = archive["embedding"]
                barcodes = archive["barcodes"].astype(str)
                if embedding.ndim != 2:
                    errors.append("embeddings are not two-dimensional")
                if not np.isfinite(embedding).all():
                    errors.append("embedding contains non-finite values")
                if len(barcodes) != len(embedding) or len(set(barcodes.tolist())) != len(barcodes):
                    errors.append("baseline barcodes are invalid")
    except Exception as exc:
        errors.append(f"audit exception: {type(exc).__name__}: {exc}")
    return errors


def audit_job(spec: dict[str, Any]) -> list[str]:
    if spec.get("engine") == "baseline":
        return audit_baseline_job(spec)
    errors: list[str] = []
    output = Path(spec["output_dir"])
    if not output.is_dir():
        return [f"missing output directory: {output}"]
    for filename in REQUIRED_OUTPUTS:
        if not (output / filename).is_file():
            errors.append(f"missing {filename}")
    if errors:
        return errors
    try:
        status = json.loads((output / "status.json").read_text(encoding="utf-8"))
        if status.get("state") != "completed":
            errors.append(f"status is {status.get('state')!r}")
        run_info = json.loads((output / "run_info.json").read_text(encoding="utf-8"))
        for key in (
            "dataset_sha256",
            "split_id",
            "protocol",
            "supervision",
            "query_labels_visible_to_adapter",
            "query_expression_used_for_preprocessing_fit",
        ):
            if key not in run_info:
                errors.append(f"run_info lacks {key}")
        if run_info.get("query_labels_visible_to_adapter") is not False:
            errors.append("query labels were not recorded as hidden")
        if run_info.get("query_expression_used_for_preprocessing_fit") is not False:
            errors.append("query expression was not recorded as excluded from preprocessing fit")
        if run_info.get("dataset_sha256") != spec["dataset_sha256"]:
            errors.append("dataset checksum provenance mismatch")
        if spec["variant"] == "solar_orthogonal_uniq":
            if run_info.get("local_extension") is not True:
                errors.append("uniq result lacks local-extension provenance")
            if run_info.get("details", {}).get("model_kwargs", {}).get("anchor_dedup") is not True:
                errors.append("uniq result did not train with anchor_dedup=True")
        with np.load(output / "embedding.npz") as archive:
            reference = archive["reference"]
            query = archive["query"]
            reference_names = archive["reference_barcodes"].astype(str)
            query_names = archive["query_barcodes"].astype(str)
            if reference.ndim != 2 or query.ndim != 2:
                errors.append("embeddings are not two-dimensional")
            if reference.shape[1] != query.shape[1]:
                errors.append("reference/query embedding dimensions differ")
            if not np.isfinite(reference).all() or not np.isfinite(query).all():
                errors.append("embedding contains non-finite values")
            if len(set(reference_names)) != len(reference_names):
                errors.append("duplicate reference barcodes")
            if len(set(query_names)) != len(query_names):
                errors.append("duplicate query barcodes")
            if set(reference_names) & set(query_names):
                errors.append("reference/query barcode overlap")
    except Exception as exc:
        errors.append(f"audit exception: {type(exc).__name__}: {exc}")
    return errors


def audit_metric(spec: dict[str, Any], path: Path) -> list[str]:
    errors: list[str] = []
    if not path.is_file():
        return [f"missing metric result: {path}"]
    try:
        result = json.loads(path.read_text(encoding="utf-8"))
        expected = {
            "job_id": spec["job_id"],
            "job_index": spec["job_index"],
            "dataset_id": spec["dataset_id"],
            "method": spec["variant"],
            "track": spec["track"],
            "protocol": spec["protocol"],
            "supervision": spec["supervision"],
            "preprocessing_profile": spec["preprocessing_profile"],
            "seed": spec["seed"],
            "anchor_seed": spec["anchor_seed"],
        }
        for key, value in expected.items():
            if result.get(key) != value:
                errors.append(f"metric {key} does not match job spec")
        if spec.get("execution_profile") != "formal":
            errors.append("job spec is not a formal execution profile")
        if "smoke_cells_per_role_batch_label" in spec:
            errors.append("formal job spec contains a smoke-only cell cap")
        if result.get("execution_profile") != "formal":
            errors.append("metric result is not a formal execution profile")
        if result.get("query_expression_used_for_training") is not False:
            errors.append("metric result does not record query expression as excluded from training")
        if result.get("query_labels_visible_to_model") is not False:
            errors.append("metric result does not record query labels as hidden from the model")
        metric_errors = result.get("metric_errors")
        if not isinstance(metric_errors, dict):
            errors.append("metric_errors is not an object")
        elif metric_errors:
            errors.append(f"unresolved metric errors: {metric_errors}")
        if result.get("state") != "completed":
            errors.append(f"metric state is {result.get('state')!r}")
        transfer = result.get("label_transfer")
        diagnostics = result.get("embedding_diagnostics")
        if not isinstance(transfer, dict):
            errors.append("label_transfer is missing or not an object")
        else:
            for key in ("macro_f1", "balanced_accuracy"):
                if key not in transfer:
                    errors.append(f"label_transfer lacks {key}")
        if not isinstance(diagnostics, dict) or not diagnostics:
            errors.append("embedding_diagnostics is missing or empty")
        for section_name, section in (
            ("label_transfer", transfer),
            ("embedding_diagnostics", diagnostics),
        ):
            if not isinstance(section, dict):
                continue
            for key, value in section.items():
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    if not math.isfinite(float(value)):
                        errors.append(f"{section_name}.{key} is non-finite")
    except Exception as exc:
        errors.append(f"metric audit exception: {type(exc).__name__}: {exc}")
    return errors


def audit_official_scib_score(spec: dict[str, Any], path: Path) -> list[str]:
    errors: list[str] = []
    if not path.is_file():
        return [f"missing official scIB score result: {path}"]
    try:
        result = json.loads(path.read_text(encoding="utf-8"))
        expected = {
            "job_id": spec["job_id"],
            "job_index": spec["job_index"],
            "dataset_id": spec["dataset_id"],
            "dataset_sha256": spec["dataset_sha256"],
            "method": spec["variant"],
            "track": spec["track"],
            "protocol": spec["protocol"],
            "supervision": spec["supervision"],
            "preprocessing_profile": spec["preprocessing_profile"],
            "seed": spec["seed"],
            "anchor_seed": spec.get("anchor_seed"),
        }
        for key, value in expected.items():
            if result.get(key) != value:
                errors.append(f"official scIB score {key} does not match job spec")
        if spec.get("fit_scope") is not None and result.get("fit_scope") != spec.get("fit_scope"):
            errors.append("official scIB score fit_scope does not match job spec")
        if spec.get("scib_output_type") and result.get("output_type") != spec.get("scib_output_type"):
            errors.append("official scIB score output_type does not match job spec")
        if result.get("scib_metric_contract") != "paper_scib_0_2_0_14_metrics":
            errors.append("official scIB score has the wrong metric contract")
        if result.get("scoring_backend") != "paper_scib_0_2_0":
            errors.append("official scIB score was not produced by the paper backend")
        if result.get("scib_commit") != "e2a37e0ed63dc34b60aa535cc656400552af757a":
            errors.append("official scIB score has the wrong scIB source commit")
        if result.get("kbet_commit") != "afc5f431bcbefd73267acc066a0f2e4eaa10a355":
            errors.append("official scIB score has the wrong kBET source commit")
        environment = result.get("paper_environment", {})
        if environment.get("state") != "passed" or not environment.get("pixi_lock_sha256"):
            errors.append("official scIB score lacks a passed locked-environment record")
        try:
            validate_metric_records(result.get("official_scib_metrics", {}))
        except ValueError as exc:
            errors.append(str(exc))
        declared_fallbacks = result.get("fallback_metrics", [])
        observed_fallbacks = [
            name
            for name, record in result.get("official_scib_metrics", {}).items()
            if record.get("details", {}).get("fallback") is True
        ]
        if sorted(declared_fallbacks) != sorted(observed_fallbacks):
            errors.append("official scIB score fallback declaration does not match metric details")
        for name in observed_fallbacks:
            details = result["official_scib_metrics"][name]["details"]
            if name != "kBET":
                errors.append(f"unapproved fallback metric: {name}")
            if details.get("metric_backend") != "scib_metrics_python_0_5_10":
                errors.append("Python kBET fallback has the wrong metric backend")
            if details.get("python_package", {}).get("version") != "0.5.10":
                errors.append("Python kBET fallback has the wrong package version")
            if details.get("fallback_policy") != "on_legacy_failure":
                errors.append("Python kBET fallback has the wrong activation policy")
        if result.get("failed_metrics"):
            errors.append(f"official scIB score has failed metrics: {result['failed_metrics']}")
        if result.get("state") != "completed":
            errors.append(f"official scIB score state is {result.get('state')!r}")
    except Exception as exc:
        errors.append(f"official scIB score audit exception: {type(exc).__name__}: {exc}")
    return errors


def _audit_matrix(
    matrix: pd.DataFrame,
    *,
    run_root: Path,
    require_official_scib: bool,
    require_legacy_metrics: bool,
    raw_dirname: str,
) -> tuple[int, int, int, list[dict[str, Any]], list[dict[str, Any]]]:
    issues: list[dict[str, Any]] = []
    completed = 0
    completed_metrics = 0
    completed_official_scib_scores = 0
    approved_metric_fallbacks: list[dict[str, Any]] = []
    if matrix["job_index"].duplicated().any():
        issues.append({"scope": "job_matrix", "errors": ["duplicate job_index values"]})
    if matrix["job_id"].duplicated().any():
        issues.append({"scope": "job_matrix", "errors": ["duplicate job_id values"]})
    if matrix["output_dir"].duplicated().any():
        issues.append({"scope": "job_matrix", "errors": ["duplicate output_dir values"]})
    for row in matrix.itertuples(index=False):
        spec = json.loads(Path(row.spec_file).read_text(encoding="utf-8"))
        errors = audit_job(spec)
        if errors:
            issues.append({"job_index": spec["job_index"], "job_id": spec["job_id"], "errors": errors})
        else:
            completed += 1
        if require_legacy_metrics:
            metric_errors = audit_metric(
                spec,
                run_root / "metrics" / f"{int(spec['job_index']):06d}.json",
            )
            if metric_errors:
                issues.append(
                    {
                        "job_index": spec["job_index"],
                        "job_id": spec["job_id"],
                        "scope": "metrics",
                        "errors": metric_errors,
                    }
                )
            else:
                completed_metrics += 1
        if require_official_scib:
            official_errors = audit_official_scib_score(
                spec,
                run_root / "scib_scores" / raw_dirname / f"{int(spec['job_index']):06d}.json",
            )
            if official_errors:
                issues.append(
                    {
                        "job_index": spec["job_index"],
                        "job_id": spec["job_id"],
                        "scope": "official_scib_scores",
                        "errors": official_errors,
                    }
                )
            else:
                completed_official_scib_scores += 1
                official_result = json.loads(
                    (
                        run_root
                        / "scib_scores"
                        / raw_dirname
                        / f"{int(spec['job_index']):06d}.json"
                    ).read_text(encoding="utf-8")
                )
                for metric in official_result.get("fallback_metrics", []):
                    record = official_result["official_scib_metrics"][metric]
                    approved_metric_fallbacks.append(
                        {
                            "job_index": int(spec["job_index"]),
                            "job_id": spec["job_id"],
                            "dataset_id": spec["dataset_id"],
                            "metric": metric,
                            "backend": record["details"].get("metric_backend"),
                            "reason": record.get("reason"),
                        }
                    )
    return (
        completed,
        completed_metrics,
        completed_official_scib_scores,
        approved_metric_fallbacks,
        issues,
    )


def audit_run(
    config: BenchmarkConfig,
    run_id: str,
    verify_data_checksums: bool = True,
) -> dict[str, Any]:
    run_root = config.output_root / run_id
    issues: list[dict[str, Any]] = []
    preprocessing, preprocessing_issues = audit_preprocessing(config, run_id)
    issues.extend(preprocessing_issues)
    require_official_scib = (
        config.data.get("evaluation", {}).get("official_scib", {}).get("enabled") is True
    )
    sections: dict[str, Any] = {}
    approved_metric_fallbacks: list[dict[str, Any]] = []
    expected_jobs = 0
    completed = 0
    completed_metrics = 0
    completed_official_scib_scores = 0

    solar_matrix_path = run_root / "job_matrix.tsv"
    if solar_matrix_path.is_file():
        matrix = pd.read_csv(solar_matrix_path, sep="\t")
        (
            section_completed,
            section_metrics,
            section_official,
            section_fallbacks,
            section_issues,
        ) = _audit_matrix(
            matrix,
            run_root=run_root,
            require_official_scib=require_official_scib,
            require_legacy_metrics=True,
            raw_dirname="raw_jobs",
        )
        issues.extend(section_issues)
        approved_metric_fallbacks.extend(section_fallbacks)
        expected_jobs += int(len(matrix))
        completed += section_completed
        completed_metrics += section_metrics
        completed_official_scib_scores += section_official
        sections["solar"] = {
            "expected_jobs": int(len(matrix)),
            "completed_jobs": int(section_completed),
            "completed_metrics": int(section_metrics),
            "completed_official_scib_scores": int(section_official),
        }

    baseline_matrix_path = run_root / "baseline_job_matrix.tsv"
    if baseline_matrix_path.is_file():
        matrix = pd.read_csv(baseline_matrix_path, sep="\t")
        (
            section_completed,
            section_metrics,
            section_official,
            section_fallbacks,
            section_issues,
        ) = _audit_matrix(
            matrix,
            run_root=run_root,
            require_official_scib=require_official_scib,
            require_legacy_metrics=False,
            raw_dirname="baseline_raw_jobs",
        )
        issues.extend(section_issues)
        approved_metric_fallbacks.extend(section_fallbacks)
        expected_jobs += int(len(matrix))
        completed += section_completed
        completed_official_scib_scores += section_official
        sections["baselines"] = {
            "expected_jobs": int(len(matrix)),
            "completed_jobs": int(section_completed),
            "completed_official_scib_scores": int(section_official),
        }

    if not sections:
        raise FileNotFoundError("No job_matrix.tsv or baseline_job_matrix.tsv found for audit")

    checksum_results: dict[str, Any] = {}
    if verify_data_checksums:
        for dataset_id in config.data["datasets"]:
            expected = config.dataset(dataset_id)["checksum_sha256"]
            observed = checksum(config.dataset_path(dataset_id))
            checksum_results[dataset_id] = {
                "expected": expected,
                "observed": observed,
                "matches": observed == expected,
            }
            if observed != expected:
                issues.append({"dataset_id": dataset_id, "errors": ["dataset checksum mismatch"]})
    payload = {
        "schema_version": 2,
        "run_id": run_id,
        "expected_jobs": int(expected_jobs),
        "completed_jobs": int(completed),
        "completed_metrics": int(completed_metrics),
        "official_scib_scores_required": require_official_scib,
        "completed_official_scib_scores": int(completed_official_scib_scores),
        "approved_metric_fallbacks": approved_metric_fallbacks,
        "sections": sections,
        "issue_count": len(issues),
        "issues": issues,
        "preprocessing": preprocessing,
        "dataset_checksums": checksum_results,
        "state": "passed" if not issues else "failed",
    }
    write_json(run_root / "audit.json", payload)
    return payload
