from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from .config import BenchmarkConfig


BASELINE_METHODS: dict[str, dict[str, Any]] = {
    "harmony": {
        "supervision": "none",
        "needs_gpu": False,
        "output_kind": "embed",
        "scib_output_type": "embed",
    },
    "scanorama": {
        "supervision": "none",
        "needs_gpu": False,
        "output_kind": "embed",
        "scib_output_type": "embed",
    },
    "fastmnn": {
        "supervision": "none",
        "needs_gpu": False,
        "output_kind": "embed",
        "scib_output_type": "embed",
    },
    "bbknn": {
        "supervision": "none",
        "needs_gpu": False,
        "output_kind": "knn",
        "scib_output_type": "knn",
    },
    "scvi": {
        "supervision": "none",
        "needs_gpu": True,
        "output_kind": "embed",
        "scib_output_type": "embed",
    },
    "scanvi_all_labels": {
        "supervision": "all_labels",
        "needs_gpu": True,
        "output_kind": "embed",
        "scib_output_type": "embed",
    },
    "scanvi_reference_labels": {
        "supervision": "reference_labels",
        "needs_gpu": True,
        "output_kind": "embed",
        "scib_output_type": "embed",
    },
    "scvi_reference_mapping": {
        "supervision": "none",
        "needs_gpu": True,
        "output_kind": "embed",
        "scib_output_type": "embed",
        "inductive_reference_mapping": True,
    },
    "scanvi_reference_mapping": {
        "supervision": "reference_labels",
        "needs_gpu": True,
        "output_kind": "embed",
        "scib_output_type": "embed",
        "inductive_reference_mapping": True,
    },
}


def _fraction_token(value: float) -> str:
    return f"{value:.2f}".replace(".", "p")


def solar_package(config: BenchmarkConfig, method: str) -> dict[str, Any]:
    v2 = config.data.get("solar_v2") or {}
    v2_methods = set(v2.get("methods") or [])
    if method in v2_methods:
        if not v2.get("package_root") or not v2.get("python"):
            raise ValueError("solar_v2 block must define package_root and python")
        return v2
    return config.data["solar"]


def _job_output(
    run_root: Path,
    track: str,
    dataset_id: str,
    profile: str,
    method: str,
    seed: int,
    anchor_seed: int,
    fraction: float,
    split_id: str,
) -> Path:
    return (
        run_root
        / "artifacts"
        / track
        / dataset_id
        / profile
        / method
        / split_id
        / f"seed_{seed}"
        / f"anchor_{anchor_seed}"
        / f"labels_{_fraction_token(fraction)}"
    )


def _baseline_job_output(
    run_root: Path,
    track: str,
    dataset_id: str,
    profile: str,
    fit_scope: str,
    method: str,
    seed: int,
) -> Path:
    return (
        run_root
        / "artifacts"
        / track
        / dataset_id
        / profile
        / fit_scope
        / method
        / f"seed_{seed}"
    )


def build_jobs(
    config: BenchmarkConfig,
    run_id: str,
    track_names: list[str],
    dataset_ids: list[str] | None = None,
    methods: list[str] | None = None,
    preprocessing_profiles: list[str] | None = None,
    seeds: list[int] | None = None,
) -> list[dict[str, Any]]:
    run_root = config.output_root / run_id
    jobs: list[dict[str, Any]] = []
    selected_datasets = dataset_ids or config.data["datasets"]
    unknown_datasets = sorted(set(selected_datasets) - set(config.data["datasets"]))
    if unknown_datasets:
        raise KeyError(f"Unknown enabled datasets: {', '.join(unknown_datasets)}")
    for track_name in track_names:
        if track_name not in config.data["tracks"]:
            raise KeyError(f"Unknown track {track_name!r}")
        track = config.data["tracks"][track_name]
        selected_methods = [name for name in track["methods"] if methods is None or name in methods]
        selected_profiles = [
            name
            for name in track["preprocessing"]
            if preprocessing_profiles is None or name in preprocessing_profiles
        ]
        selected_seeds = [
            int(seed) for seed in track["train_seeds"] if seeds is None or int(seed) in seeds
        ]
        if not selected_methods or not selected_profiles or not selected_seeds:
            raise ValueError(f"Filters select no jobs from track {track_name!r}")
        for dataset_id in selected_datasets:
            dataset = config.dataset(dataset_id)
            for seed in selected_seeds:
                if track["protocol"] == "inductive_held_out_batch":
                    split_dirs = sorted(
                        path
                        for path in (run_root / "splits" / dataset_id).glob("heldout_*")
                        if (path / "split.csv").is_file()
                    )
                    if not split_dirs:
                        raise FileNotFoundError(
                            f"No held-out splits for {dataset_id}; prepare with --include-heldout"
                        )
                else:
                    split_dirs = [
                        run_root / "splits" / dataset_id / f"random_seed_{int(seed)}"
                    ]
                for split_dir in split_dirs:
                    split_id = split_dir.name
                    split_file = split_dir / "split.csv"
                    if not split_file.is_file():
                        raise FileNotFoundError(
                            f"Missing split {split_file}; run prepare_benchmark_splits.py first"
                        )
                    for profile_name in selected_profiles:
                        profile = config.data["preprocessing"][profile_name]
                        hvg_file = None
                        if profile.get("n_hvg") is not None:
                            hvg_file = (
                                run_root
                                / "preprocessing"
                                / dataset_id
                                / split_id
                                / f"{profile_name}.txt"
                            )
                        for method in selected_methods:
                            anchor_seeds = track["anchor_seeds"]
                            if method in {"solar_none", "cross_entropy", "solar_trainable"}:
                                anchor_seeds = [0]
                            for anchor_seed in anchor_seeds:
                                for fraction in track["label_fractions"]:
                                    fraction = float(fraction)
                                    budget = split_dir / f"label_budget_{fraction:.2f}.csv"
                                    if not budget.is_file():
                                        raise FileNotFoundError(budget)
                                    output = _job_output(
                                        run_root,
                                        track_name,
                                        dataset_id,
                                        profile_name,
                                        method,
                                        int(seed),
                                        int(anchor_seed),
                                        fraction,
                                        split_id,
                                    )
                                    pkg = solar_package(config, method)
                                    jobs.append(
                                        {
                                        "schema_version": 1,
                                        "engine": "solar",
                                        "track": track_name,
                                        "protocol": track["protocol"],
                                        "supervision": track["supervision"],
                                        "dataset_id": dataset_id,
                                        "dataset_path": str(config.dataset_path(dataset_id)),
                                        "dataset_sha256": dataset["checksum_sha256"],
                                        "batch_key": dataset["batch_key"],
                                        "label_key": dataset["label_key"],
                                        "split_id": split_id,
                                        "split_file": str(split_file),
                                        "label_budget_file": str(budget),
                                        "split_uses_query_labels": track["protocol"]
                                        != "inductive_held_out_batch",
                                        "preprocessing_profile": profile_name,
                                        "gene_file": str(hvg_file) if hvg_file else None,
                                        "n_components": int(profile["n_components"]),
                                        "variant": method,
                                        "seed": int(seed),
                                        "anchor_seed": int(anchor_seed),
                                        "label_fraction": fraction,
                                        "train": dict(pkg["train"]),
                                        "execution_profile": "formal",
                                        "solar_python": str(config.resolve(pkg["python"])),
                                        "solar_package_root": str(
                                            config.resolve(pkg["package_root"])
                                        ),
                                        "solar_upstream_archive": str(
                                            config.resolve(pkg["upstream_archive"])
                                        ),
                                        "solar_upstream_sha256": pkg["upstream_sha256"],
                                        "output_dir": str(output),
                                        }
                                    )
    jobs.sort(
        key=lambda item: (
            item["track"],
            item["dataset_id"],
            item["preprocessing_profile"],
            item["variant"],
            item["seed"],
            item["anchor_seed"],
            item["label_fraction"],
        )
    )
    for index, job in enumerate(jobs):
        job["job_index"] = index
        job["job_id"] = (
            f"{job['track']}__{job['dataset_id']}__{job['preprocessing_profile']}__"
            f"{job['variant']}__{job['split_id']}__s{job['seed']}__a{job['anchor_seed']}__"
            f"l{_fraction_token(job['label_fraction'])}"
        )
    return jobs


def build_baseline_jobs(
    config: BenchmarkConfig,
    run_id: str,
    track_names: list[str] | None = None,
    dataset_ids: list[str] | None = None,
    methods: list[str] | None = None,
    preprocessing_profiles: list[str] | None = None,
    seeds: list[int] | None = None,
    fit_scopes: list[str] | None = None,
) -> list[dict[str, Any]]:
    track_names = track_names or ["transductive_baselines"]
    run_root = config.output_root / run_id
    jobs: list[dict[str, Any]] = []
    selected_datasets = dataset_ids or config.data["datasets"]
    unknown_datasets = sorted(set(selected_datasets) - set(config.data["datasets"]))
    if unknown_datasets:
        raise KeyError(f"Unknown enabled datasets: {', '.join(unknown_datasets)}")
    baseline_defaults = dict(config.data.get("baselines", {}))
    for track_name in track_names:
        if track_name not in config.data["tracks"]:
            raise KeyError(f"Unknown track {track_name!r}")
        track = config.data["tracks"][track_name]
        if track.get("protocol") != "transductive_scib":
            raise ValueError(
                f"Track {track_name!r} is not a transductive baseline track "
                f"(protocol={track.get('protocol')!r})"
            )
        selected_methods = [name for name in track["methods"] if methods is None or name in methods]
        selected_profiles = [
            name
            for name in track["preprocessing"]
            if preprocessing_profiles is None or name in preprocessing_profiles
        ]
        selected_seeds = [
            int(seed) for seed in track["train_seeds"] if seeds is None or int(seed) in seeds
        ]
        selected_scopes = [
            name
            for name in track.get("fit_scopes", ["classic_full"])
            if fit_scopes is None or name in fit_scopes
        ]
        if not selected_methods or not selected_profiles or not selected_seeds or not selected_scopes:
            raise ValueError(f"Filters select no jobs from track {track_name!r}")
        unknown_methods = sorted(set(selected_methods) - set(BASELINE_METHODS))
        if unknown_methods:
            raise KeyError(f"Unknown baseline methods: {', '.join(unknown_methods)}")
        unknown_scopes = sorted(set(selected_scopes) - {"classic_full", "reference_only"})
        if unknown_scopes:
            raise ValueError(f"Unknown fit scopes: {', '.join(unknown_scopes)}")
        for dataset_id in selected_datasets:
            dataset = config.dataset(dataset_id)
            for seed in selected_seeds:
                split_id = f"random_seed_{int(seed)}"
                split_file = run_root / "splits" / dataset_id / split_id / "split.csv"
                if not split_file.is_file():
                    raise FileNotFoundError(
                        f"Missing split {split_file}; run prepare_benchmark_splits.py first"
                    )
                for profile_name in selected_profiles:
                    profile = config.data["preprocessing"][profile_name]
                    for fit_scope in selected_scopes:
                        gene_file = None
                        if profile.get("n_hvg") is not None and fit_scope == "reference_only":
                            gene_file = (
                                run_root
                                / "preprocessing"
                                / dataset_id
                                / split_id
                                / f"{profile_name}.txt"
                            )
                            if not gene_file.is_file():
                                raise FileNotFoundError(
                                    f"Missing reference-only HVG list {gene_file}"
                                )
                        for method in selected_methods:
                            meta = BASELINE_METHODS[method]
                            output = _baseline_job_output(
                                run_root,
                                track_name,
                                dataset_id,
                                profile_name,
                                fit_scope,
                                method,
                                int(seed),
                            )
                            jobs.append(
                                {
                                    "schema_version": 1,
                                    "engine": "baseline",
                                    "track": track_name,
                                    "protocol": track["protocol"],
                                    "supervision": meta["supervision"],
                                    "dataset_id": dataset_id,
                                    "dataset_path": str(config.dataset_path(dataset_id)),
                                    "dataset_sha256": dataset["checksum_sha256"],
                                    "batch_key": dataset["batch_key"],
                                    "label_key": dataset["label_key"],
                                    "split_id": split_id,
                                    "split_file": str(split_file),
                                    "label_budget_file": None,
                                    "split_uses_query_labels": False,
                                    "preprocessing_profile": profile_name,
                                    "fit_scope": fit_scope,
                                    "gene_file": str(gene_file) if gene_file else None,
                                    "n_hvg": profile.get("n_hvg"),
                                    "hvg_flavor": profile.get("hvg_flavor", "cell_ranger"),
                                    "batch_aware": bool(profile.get("batch_aware", True)),
                                    "n_components": int(profile["n_components"]),
                                    "variant": method,
                                    "method": method,
                                    "seed": int(seed),
                                    "anchor_seed": None,
                                    "label_fraction": None,
                                    "needs_gpu": bool(meta["needs_gpu"]),
                                    "output_kind": meta["output_kind"],
                                    "scib_output_type": meta["scib_output_type"],
                                    "baseline_defaults": baseline_defaults,
                                    "execution_profile": "formal",
                                    "query_expression_used_for_training": True,
                                    "query_labels_visible_to_model": meta["supervision"] != "none",
                                    "output_dir": str(output),
                                }
                            )
    jobs.sort(
        key=lambda item: (
            item["track"],
            item["dataset_id"],
            item["preprocessing_profile"],
            item["fit_scope"],
            item["variant"],
            item["seed"],
        )
    )
    for index, job in enumerate(jobs):
        job["job_index"] = index
        job["job_id"] = (
            f"{job['track']}__{job['dataset_id']}__{job['preprocessing_profile']}__"
            f"{job['fit_scope']}__{job['variant']}__s{job['seed']}"
        )
    return jobs


def write_job_matrix(
    jobs: list[dict[str, Any]],
    run_root: Path,
    force: bool = False,
    *,
    matrix_name: str = "job_matrix.tsv",
    jobs_dirname: str = "jobs",
) -> Path:
    jobs_dir = run_root / jobs_dirname
    matrix = run_root / matrix_name
    run_root.mkdir(parents=True, exist_ok=True)
    if jobs_dir.exists() or matrix.exists():
        if not force:
            raise FileExistsError(
                f"Job matrix already exists ({matrix_name}); use --force to archive it"
            )
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        if jobs_dir.exists():
            jobs_dir.replace(run_root / f"{jobs_dirname}.archived-{stamp}")
        if matrix.exists():
            matrix.replace(run_root / f"{Path(matrix_name).stem}.archived-{stamp}.tsv")
    jobs_dir.mkdir(parents=True, exist_ok=False)
    rows = []
    for job in jobs:
        spec_path = jobs_dir / f"{job['job_index']:06d}.json"
        spec_path.write_text(
            json.dumps(job, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        row = {
            "job_index": job["job_index"],
            "job_id": job["job_id"],
            "track": job["track"],
            "dataset_id": job["dataset_id"],
            "method": job["variant"],
            "preprocessing": job["preprocessing_profile"],
            "seed": job["seed"],
            "spec_file": str(spec_path),
            "output_dir": job["output_dir"],
            "needs_gpu": bool(job.get("needs_gpu", False)),
            "fit_scope": job.get("fit_scope"),
            "protocol": job.get("protocol"),
            "supervision": job.get("supervision"),
            "scib_output_type": job.get("scib_output_type", "embed"),
        }
        if "anchor_seed" in job:
            row["anchor_seed"] = job["anchor_seed"]
        if "label_fraction" in job:
            row["label_fraction"] = job["label_fraction"]
        rows.append(row)
    pd.DataFrame(rows).to_csv(matrix, sep="\t", index=False)
    return matrix
