from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any

import anndata as ad
import numpy as np
import pandas as pd

from .artifacts import write_json
from .config import BenchmarkConfig
from .scib_contract import (
    OFFICIAL_METRICS,
    aggregate_official_scores,
    cohort_hash,
    metric_record,
    official_applicability,
    validate_metric_records,
)
from .scib_scoring import (
    GRAPH_TASKS,
    PAPER_BACKEND,
    PAPER_KBET_COMMIT,
    PAPER_SCIB_COMMIT,
    PYTHON_KBET_FALLBACK_BACKEND,
    TASK_METRICS,
    TASK_ORDER,
    failed_task_records,
    flatten_scib_result,
    paper_worker_command,
    run_isolated_task,
    scoring_context,
    source_cache_path,
)


def _read_dataset_structure(spec: dict[str, Any]) -> tuple[int, list[str]]:
    backed = ad.read_h5ad(spec["dataset_path"], backed="r")
    try:
        return int(backed.n_obs), list(map(str, backed.obs.columns))
    finally:
        backed.file.close()


def _task_is_reusable(path: Path, expected: tuple[str, ...], backend: str) -> bool:
    if not path.is_file():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return (
            payload.get("state") == "completed"
            and payload.get("backend") == backend
            and set(payload.get("metrics", {})) == set(expected)
        )
    except (OSError, json.JSONDecodeError):
        return False


def _graph_is_reusable(task_file: Path, graph_dir: Path, backend: str) -> bool:
    graph_file = graph_dir / "graph.json"
    if not task_file.is_file() or not graph_file.is_file():
        return False
    try:
        payload = json.loads(task_file.read_text(encoding="utf-8"))
        graph = json.loads(graph_file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return (
        payload.get("state") == "completed"
        and payload.get("backend") == backend
        and graph.get("backend") == backend
        and (graph_dir / "connectivities.npz").is_file()
        and (graph_dir / "distances.npz").is_file()
    )


def _worker_command(
    *,
    config: BenchmarkConfig,
    spec_path: Path,
    task: str,
    graph_dir: Path,
    cache_dir: Path,
    output: Path,
    context: dict[str, Any],
) -> list[str]:
    if context["backend"] == PAPER_BACKEND:
        return paper_worker_command(
            config.repo_root,
            "--job-spec",
            str(spec_path),
            "--task",
            task,
            "--graph-dir",
            str(graph_dir),
            "--source-cache",
            str(cache_dir),
            "--output",
            str(output),
            "--output-type",
            str(context["output_type"]),
            "--neighbors",
            str(context["neighbors"]),
            "--lisi-subsample-percent",
            str(context["lisi_subsample_percent"]),
            "--lisi-cores",
            str(context["lisi_cores"]),
        )
    return [
        sys.executable,
        "-m",
        "scib_benchmark.scib_worker",
        "--config",
        str(config.path),
        "--job-spec",
        str(spec_path),
        "--task",
        task,
        "--graph-dir",
        str(graph_dir),
        "--source-cache",
        str(cache_dir),
        "--output",
        str(output),
    ]


def score_job(
    config: BenchmarkConfig,
    run_id: str,
    spec_path: Path,
    *,
    force: bool = False,
) -> dict[str, Any]:
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    context = scoring_context(config.data, spec["dataset_id"])
    if spec.get("scib_output_type"):
        context = dict(context)
        context["output_type"] = str(spec["scib_output_type"])
    run_root = config.output_root / run_id
    score_root = run_root / "scib_scores"
    environment_path = score_root / "paper_environment.json"
    if not environment_path.is_file():
        raise FileNotFoundError(
            f"Missing {environment_path}; run scripts/prepare_scib_scoring.py first"
        )
    paper_environment = json.loads(environment_path.read_text(encoding="utf-8"))
    if (
        paper_environment.get("state") != "passed"
        or paper_environment.get("scib_commit") != PAPER_SCIB_COMMIT
        or paper_environment.get("kbet_commit") != PAPER_KBET_COMMIT
    ):
        raise ValueError("The persisted paper-scIB environment validation is invalid or stale")
    is_baseline = spec.get("engine") == "baseline"
    raw_dirname = "baseline_raw_jobs" if is_baseline else "raw_jobs"
    work_dirname = "baseline_work" if is_baseline else "work"
    raw_path = score_root / raw_dirname / f"{int(spec['job_index']):06d}.json"
    existing_tasks_dir = score_root / work_dirname / f"{int(spec['job_index']):06d}" / "tasks"
    if raw_path.is_file() and not force:
        try:
            existing = json.loads(raw_path.read_text(encoding="utf-8"))
            task_files = list(existing_tasks_dir.glob("*.json"))
            raw_is_fresh = not task_files or raw_path.stat().st_mtime_ns >= max(
                path.stat().st_mtime_ns for path in task_files
            )
            if (
                existing.get("state") == "completed"
                and existing.get("job_id") == spec.get("job_id")
                and existing.get("scoring_backend") == context["backend"]
                and existing.get("scib_commit") == PAPER_SCIB_COMMIT
                and existing.get("kbet_commit") == PAPER_KBET_COMMIT
                and existing.get("paper_environment", {}).get("pixi_lock_sha256")
                == paper_environment.get("pixi_lock_sha256")
                and existing.get("output_type") == context["output_type"]
                and raw_is_fresh
            ):
                validate_metric_records(existing["official_scib_metrics"])
                return existing
        except (KeyError, ValueError, json.JSONDecodeError):
            pass

    n_obs, obs_columns = _read_dataset_structure(spec)
    applicability = official_applicability(
        output_type=context["output_type"],
        assay=context["assay"],
        n_obs=n_obs,
        obs_columns=obs_columns,
    )
    records: dict[str, dict[str, Any]] = {}
    for name, rule in applicability.items():
        if not rule["applicable"]:
            records[name] = metric_record(
                name,
                "not_applicable",
                reason=rule["reason"],
                implementation="official scIB applicability rules",
                evidence=rule["evidence"],
            )

    work_dir = score_root / work_dirname / f"{int(spec['job_index']):06d}"
    tasks_dir = work_dir / "tasks"
    logs_dir = work_dir / "logs"
    graph_dir = work_dir / "graph"
    for directory in (tasks_dir, logs_dir, graph_dir, raw_path.parent):
        directory.mkdir(parents=True, exist_ok=True)
    cache_dir = source_cache_path(run_root, spec, backend=context["backend"])
    cache_status_path = cache_dir / "status.json"
    cache_status: dict[str, Any] = {}
    if cache_status_path.is_file():
        cache_status = json.loads(cache_status_path.read_text(encoding="utf-8"))

    applicable_tasks = [
        task
        for task in TASK_ORDER
        if any(applicability[name]["applicable"] for name in TASK_METRICS[task])
    ]
    graph_required = any(task in GRAPH_TASKS for task in applicable_tasks)
    graph_ok = not graph_required
    if graph_required:
        neighbor_output = tasks_dir / "neighbors.json"
        if force or not _graph_is_reusable(neighbor_output, graph_dir, context["backend"]):
            execution = run_isolated_task(
                _worker_command(
                    config=config,
                    spec_path=spec_path,
                    task="neighbors",
                    graph_dir=graph_dir,
                    cache_dir=cache_dir,
                    output=neighbor_output,
                    context=context,
                ),
                repo_root=config.repo_root,
                log_path=logs_dir / "neighbors.log",
                timeout_seconds=context["task_timeout_seconds"],
                backend=context["backend"],
            )
            graph_ok = execution["returncode"] == 0 and _graph_is_reusable(
                neighbor_output, graph_dir, context["backend"]
            )
        else:
            graph_ok = True

    for task in applicable_tasks:
        expected = TASK_METRICS[task]
        fallback_output = tasks_dir / "kbet_python_fallback.json"
        fallback_config = context["kbet_python_fallback"]
        if (
            task == "kbet"
            and fallback_config["enabled"]
            and not force
            and _task_is_reusable(
                fallback_output,
                expected,
                PYTHON_KBET_FALLBACK_BACKEND,
            )
        ):
            payload = json.loads(fallback_output.read_text(encoding="utf-8"))
            records.update(payload["metrics"])
            continue
        if task in GRAPH_TASKS and not graph_ok:
            records.update(failed_task_records(task, "The official 15-neighbor graph prerequisite failed."))
            continue
        cache_component = {
            "pcr": "pcr_source_cache",
            "cell_cycle": "cell_cycle_source_cache",
        }.get(task)
        component_state = (
            cache_status.get(cache_component, {}).get("state") if cache_component else None
        )
        if cache_component and component_state != "completed":
            records.update(
                failed_task_records(
                    task,
                    "Source-invariant scoring cache component "
                    f"{cache_component!r} is unavailable or incomplete (state={component_state!r}).",
                )
            )
            continue
        task_output = tasks_dir / f"{task}.json"
        if force or not _task_is_reusable(task_output, expected, context["backend"]):
            execution = run_isolated_task(
                _worker_command(
                    config=config,
                    spec_path=spec_path,
                    task=task,
                    graph_dir=graph_dir,
                    cache_dir=cache_dir,
                    output=task_output,
                    context=context,
                ),
                repo_root=config.repo_root,
                log_path=logs_dir / f"{task}.log",
                timeout_seconds=context["task_timeout_seconds"],
                backend=context["backend"],
            )
            if execution["returncode"] != 0:
                detail = "timed out" if execution["timed_out"] else f"exited with code {execution['returncode']}"
                if task_output.is_file():
                    try:
                        worker_error = json.loads(task_output.read_text(encoding="utf-8")).get("error")
                        if worker_error:
                            detail += f": {worker_error}"
                    except json.JSONDecodeError:
                        pass
                if task == "kbet" and fallback_config["enabled"]:
                    command = [
                        sys.executable,
                        str(config.repo_root / "scripts" / "python_kbet_fallback.py"),
                        "--job-spec",
                        str(spec_path),
                        "--output",
                        str(fallback_output),
                        "--primary-error",
                        detail,
                        "--neighbors",
                        str(fallback_config["neighbors"]),
                        "--random-state",
                        str(fallback_config["random_state"]),
                        "--n-jobs",
                        str(fallback_config["n_jobs"]),
                        "--alpha",
                        str(fallback_config["alpha"]),
                        "--diffusion-n-comps",
                        str(fallback_config["diffusion_n_comps"]),
                    ]
                    fallback_execution = run_isolated_task(
                        command,
                        repo_root=config.repo_root,
                        log_path=logs_dir / "kbet_python_fallback.log",
                        timeout_seconds=context["task_timeout_seconds"],
                    )
                    try:
                        fallback_payload = json.loads(
                            fallback_output.read_text(encoding="utf-8")
                        )
                        if (
                            fallback_execution["returncode"] != 0
                            or fallback_payload.get("state") != "completed"
                            or set(fallback_payload.get("metrics", {})) != set(expected)
                        ):
                            raise ValueError(
                                fallback_payload.get("error")
                                or "Python fallback exited without a complete metric"
                            )
                        records.update(fallback_payload["metrics"])
                        continue
                    except Exception as exc:
                        detail += (
                            "; approved Python kBET fallback also failed: "
                            f"{type(exc).__name__}: {exc}"
                        )
                records.update(failed_task_records(task, f"Isolated metric worker {detail}."))
                continue
        try:
            payload = json.loads(task_output.read_text(encoding="utf-8"))
            if payload.get("state") != "completed" or set(payload.get("metrics", {})) != set(expected):
                raise ValueError("Worker output is incomplete or has the wrong metric schema")
            records.update(payload["metrics"])
        except Exception as exc:
            records.update(failed_task_records(task, f"Invalid worker output: {type(exc).__name__}: {exc}"))

    for name in OFFICIAL_METRICS:
        if name not in records:
            records[name] = metric_record(
                name,
                "failed",
                reason="The scoring orchestrator did not produce a terminal metric record.",
                implementation="official scIB scoring orchestrator",
                evidence="Internal completeness guard.",
            )
    validate_metric_records(records)
    failed = [name for name, record in records.items() if record["status"] == "failed"]
    fallback_metrics = [
        name
        for name, record in records.items()
        if record.get("details", {}).get("fallback") is True
    ]
    metric_backend_overrides = {
        name: records[name]["details"]["metric_backend"] for name in fallback_metrics
    }
    result = {
        "schema_version": 3,
        "scib_metric_contract": "paper_scib_0_2_0_14_metrics",
        "scoring_backend": context["backend"],
        "scib_commit": PAPER_SCIB_COMMIT,
        "kbet_commit": PAPER_KBET_COMMIT,
        "paper_environment": paper_environment,
        "job_index": spec["job_index"],
        "job_id": spec["job_id"],
        "dataset_id": spec["dataset_id"],
        "dataset_sha256": spec["dataset_sha256"],
        "method": spec["variant"],
        "track": spec["track"],
        "protocol": spec["protocol"],
        "supervision": spec["supervision"],
        "label_fraction": spec.get("label_fraction"),
        "preprocessing_profile": spec["preprocessing_profile"],
        "fit_scope": spec.get("fit_scope"),
        "seed": spec["seed"],
        "anchor_seed": spec.get("anchor_seed"),
        "output_type": context["output_type"],
        "assay": context["assay"],
        "organism": context.get("organism"),
        "scoring_population": "joint reference and query embedding; every official dataset cell exactly once",
        "official_scib_metrics": records,
        "failed_metrics": failed,
        "fallback_metrics": fallback_metrics,
        "metric_backend_overrides": metric_backend_overrides,
        "state": "completed" if not failed else "completed_with_metric_errors",
        "aggregation_note": (
            "Raw metrics use the locked paper scIB 0.2.0 backend where computed. Aggregate scores are "
            "protocol-scoped within this run and must not be mixed into a cross-protocol ranking."
            + (
                " This row contains a declared Python kBET fallback and is not numerically "
                "equivalent to the paper R kBET implementation."
                if fallback_metrics
                else ""
            )
        ),
    }
    write_json(raw_path, result)
    return result


def _json_rows(frame: pd.DataFrame) -> list[dict[str, Any]]:
    normalized = frame.astype(object).where(pd.notna(frame), None)
    return normalized.to_dict(orient="records")


def _load_matrix_scores(
    score_root: Path,
    matrix: pd.DataFrame,
    *,
    raw_dirname: str,
) -> tuple[pd.DataFrame, list[int], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    missing: list[int] = []
    invalid: list[dict[str, Any]] = []
    for matrix_row in matrix.itertuples(index=False):
        index = int(matrix_row.job_index)
        path = score_root / raw_dirname / f"{index:06d}.json"
        if not path.is_file():
            missing.append(index)
            continue
        try:
            result = json.loads(path.read_text(encoding="utf-8"))
            validate_metric_records(result["official_scib_metrics"])
            rows.append(flatten_scib_result(result))
        except Exception as exc:
            invalid.append({"job_index": index, "error": f"{type(exc).__name__}: {exc}"})
    return pd.DataFrame(rows), missing, invalid


def _ordered_raw(raw: pd.DataFrame) -> pd.DataFrame:
    if raw.empty:
        return raw
    ordered = [
        "job_index",
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
        *OFFICIAL_METRICS,
        *(f"status::{name}" for name in OFFICIAL_METRICS),
        "state",
    ]
    present = [name for name in ordered if name in raw.columns]
    return raw.loc[:, present].sort_values(["protocol", "job_index"], kind="stable")


def _aggregate_protocol_cohort(
    score_root: Path,
    protocol: str,
    raw: pd.DataFrame,
) -> dict[str, Any]:
    cohort = {
        "schema_version": 1,
        "protocol": protocol,
        "cohort_sha256": cohort_hash(raw),
        "rows": int(len(raw)),
        "datasets": sorted(raw["dataset_id"].unique().tolist()),
        "methods": sorted(raw["method"].unique().tolist()),
        "preprocessing_profiles": sorted(raw["preprocessing_profile"].unique().tolist()),
        "fit_scopes": sorted(
            [value for value in raw.get("fit_scope", pd.Series(dtype=object)).dropna().unique().tolist()]
        ),
        "scoring_backends": sorted(raw["scoring_backend"].unique().tolist()),
        "scib_commits": sorted(raw["scib_commit"].unique().tolist()),
        "kbet_commits": sorted(raw["kbet_commit"].unique().tolist()),
        "kbet_backends": sorted(raw["kbet_backend"].unique().tolist()),
        "python_kbet_fallback_jobs": raw.loc[
            raw["kbet_fallback"].astype(bool), "job_index"
        ].astype(int).tolist(),
        "paper_environment_lock_sha256": sorted(
            raw["paper_environment_lock_sha256"].unique().tolist()
        ),
        "paper_comparable": False,
        "paper_comparable_reason": (
            "Protocol-scoped aggregate only. Cross-protocol ranking is disabled; dataset-wise "
            "min-max is computed within this protocol cohort alone."
        ),
        "aggregation": (
            "Within each dataset and protocol cohort, scales::rescale-compatible min-max per "
            "metric; means over applicable non-NA metrics; overall=0.4*batch_correction+"
            "0.6*bio_conservation."
        ),
        "generate_cross_protocol_ranking": False,
    }
    protocol_dir = score_root / "by_protocol" / protocol
    protocol_dir.mkdir(parents=True, exist_ok=True)
    write_json(protocol_dir / "cohort.json", cohort)
    scaled = aggregate_official_scores(raw)
    scaled["aggregate_scope"] = f"protocol_scoped::{protocol}"
    scaled["paper_comparable"] = False
    scaled["protocol"] = protocol
    scaled.to_csv(protocol_dir / "scaled_metrics.csv", index=False)
    write_json(protocol_dir / "scaled_metrics.json", {"rows": _json_rows(scaled)})
    aggregate_columns = [
        "job_index",
        "job_id",
        "dataset_id",
        "method",
        "protocol",
        "preprocessing_profile",
        "fit_scope",
        "seed",
        "scoring_backend",
        "scib_commit",
        "kbet_commit",
        "kbet_backend",
        "kbet_fallback",
        "paper_environment_lock_sha256",
        "batch_correction",
        "bio_conservation",
        "overall",
        "aggregate_scope",
        "paper_comparable",
    ]
    present = [name for name in aggregate_columns if name in scaled.columns]
    aggregate = scaled.loc[:, present]
    aggregate.to_csv(protocol_dir / "aggregate_scores.csv", index=False)
    write_json(protocol_dir / "aggregate_scores.json", {"rows": _json_rows(aggregate)})
    return cohort


def aggregate_run(config: BenchmarkConfig, run_id: str) -> dict[str, Any]:
    run_root = config.output_root / run_id
    score_root = run_root / "scib_scores"
    protocol_sections: dict[str, Any] = {}
    all_rows: list[pd.DataFrame] = []

    solar_matrix_path = run_root / "job_matrix.tsv"
    if solar_matrix_path.is_file():
        solar_matrix = pd.read_csv(solar_matrix_path, sep="\t")
        solar_raw, solar_missing, solar_invalid = _load_matrix_scores(
            score_root, solar_matrix, raw_dirname="raw_jobs"
        )
        solar_raw = _ordered_raw(solar_raw)
        failed_rows = (
            []
            if solar_raw.empty
            else solar_raw.loc[solar_raw["state"] != "completed", "job_index"].astype(int).tolist()
        )
        complete = (
            len(solar_raw) == len(solar_matrix)
            and not solar_missing
            and not solar_invalid
            and not failed_rows
        )
        section = {
            "matrix": str(solar_matrix_path),
            "raw_dirname": "raw_jobs",
            "expected_jobs": int(len(solar_matrix)),
            "valid_result_files": int(len(solar_raw)),
            "missing_job_indices": solar_missing,
            "invalid_results": solar_invalid,
            "jobs_with_failed_metrics": failed_rows,
            "complete": complete,
            "state": "passed" if complete else "failed",
        }
        if complete and not solar_raw.empty:
            protocol = str(solar_raw["protocol"].iloc[0])
            if solar_raw["protocol"].nunique() != 1:
                raise ValueError("SOLAR job matrix spans multiple protocols")
            section["cohort"] = _aggregate_protocol_cohort(score_root, protocol, solar_raw)
            all_rows.append(solar_raw)
        protocol_sections["solar"] = section

    baseline_matrix_path = run_root / "baseline_job_matrix.tsv"
    if baseline_matrix_path.is_file():
        baseline_matrix = pd.read_csv(baseline_matrix_path, sep="\t")
        baseline_raw, baseline_missing, baseline_invalid = _load_matrix_scores(
            score_root, baseline_matrix, raw_dirname="baseline_raw_jobs"
        )
        baseline_raw = _ordered_raw(baseline_raw)
        failed_rows = (
            []
            if baseline_raw.empty
            else baseline_raw.loc[baseline_raw["state"] != "completed", "job_index"]
            .astype(int)
            .tolist()
        )
        complete = (
            len(baseline_raw) == len(baseline_matrix)
            and not baseline_missing
            and not baseline_invalid
            and not failed_rows
        )
        section = {
            "matrix": str(baseline_matrix_path),
            "raw_dirname": "baseline_raw_jobs",
            "expected_jobs": int(len(baseline_matrix)),
            "valid_result_files": int(len(baseline_raw)),
            "missing_job_indices": baseline_missing,
            "invalid_results": baseline_invalid,
            "jobs_with_failed_metrics": failed_rows,
            "complete": complete,
            "state": "passed" if complete else "failed",
        }
        if complete and not baseline_raw.empty:
            protocol = str(baseline_raw["protocol"].iloc[0])
            if baseline_raw["protocol"].nunique() != 1:
                raise ValueError("Baseline job matrix spans multiple protocols")
            section["cohort"] = _aggregate_protocol_cohort(score_root, protocol, baseline_raw)
            all_rows.append(baseline_raw)
        protocol_sections["baselines"] = section

    if not protocol_sections:
        raise FileNotFoundError("No job_matrix.tsv or baseline_job_matrix.tsv found for aggregation")

    raw = _ordered_raw(pd.concat(all_rows, ignore_index=True) if all_rows else pd.DataFrame())
    raw.to_csv(score_root / "raw_metrics.csv", index=False)
    write_json(score_root / "raw_metrics.json", {"rows": _json_rows(raw)})

    completeness = {
        "schema_version": 2,
        "run_id": run_id,
        "metric_contract": "paper_scib_0_2_0_14_metrics",
        "generate_cross_protocol_ranking": False,
        "sections": protocol_sections,
        "complete": all(section["complete"] for section in protocol_sections.values()),
        "state": (
            "passed"
            if all(section["state"] == "passed" for section in protocol_sections.values())
            else "failed"
        ),
    }
    write_json(score_root / "completeness.json", completeness)

    # Keep top-level aggregate files as the SOLAR inductive cohort when present for compatibility.
    solar_section = protocol_sections.get("solar")
    if solar_section and solar_section.get("complete") and "cohort" in solar_section:
        protocol = solar_section["cohort"]["protocol"]
        protocol_dir = score_root / "by_protocol" / protocol
        write_json(score_root / "cohort.json", solar_section["cohort"])
        for name in ("scaled_metrics.csv", "scaled_metrics.json", "aggregate_scores.csv", "aggregate_scores.json"):
            source = protocol_dir / name
            if source.is_file():
                target = score_root / name
                if name.endswith(".csv"):
                    pd.read_csv(source).to_csv(target, index=False)
                else:
                    target.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
        return {
            **completeness,
            "cohort_sha256": solar_section["cohort"]["cohort_sha256"],
        }
    return completeness
