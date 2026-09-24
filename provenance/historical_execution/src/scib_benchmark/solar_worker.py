from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import sys
import time
from typing import Any

import numpy as np
import pandas as pd
import torch

from .artifacts import begin_output, commit_output, write_json
from .data import load_reference_query, read_gene_list, read_split
from .solar_adapter import assert_only_dedup_diff, register_orthogonal_uniq
from .splits import label_budget, smoke_subset


def _environment() -> dict[str, Any]:
    packages = {}
    for name in (
        "anndata",
        "numpy",
        "scipy",
        "scikit-learn",
        "torch",
        "solar-label-anchor",
        "solar-scrna",
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
        "torch_cuda_available": torch.cuda.is_available(),
        "torch_cuda_devices": [
            torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())
        ],
    }


def run_job(spec: dict[str, Any], force: bool = False) -> Path:
    from SOLAR.benchmark import PreprocessConfig, TrainConfig, run_inductive
    from SOLAR.benchmark import variants as variants_module
    from SOLAR.benchmark.splits import subsample_labels

    derived = register_orthogonal_uniq(variants_module)
    assert_only_dedup_diff(variants_module.VARIANTS["solar_orthogonal"], derived)
    output = Path(spec["output_dir"])
    transaction = begin_output(output, force=force)
    if transaction.reused:
        return output
    assert transaction.staging is not None
    started = time.time()
    dataset_path = Path(spec["dataset_path"])
    before = (dataset_path.stat().st_size, dataset_path.stat().st_mtime_ns)
    status: dict[str, Any] = {
        "state": "running",
        "job_id": spec["job_id"],
        "started_unix": started,
    }
    write_json(transaction.staging / "status.json", status)
    try:
        split = read_split(Path(spec["split_file"]))
        smoke_cap = spec.get("smoke_cells_per_role_batch_label")
        if smoke_cap is not None:
            split = smoke_subset(split, int(smoke_cap), int(spec["seed"]))
        genes = read_gene_list(Path(spec["gene_file"])) if spec.get("gene_file") else None
        reference, query = load_reference_query(dataset_path, split, genes)
        if smoke_cap is None:
            budget = pd.read_csv(spec["label_budget_file"], dtype={"barcode": str})
        else:
            budget = label_budget(split, float(spec["label_fraction"]), int(spec["seed"]))
        expected_labeled = set(budget.loc[budget["labeled"].astype(bool), "barcode"])
        selected = subsample_labels(
            reference,
            spec["label_key"],
            float(spec["label_fraction"]),
            int(spec["seed"]),
            spec["batch_key"],
        )
        actual_labeled = set(reference.obs_names[selected].astype(str))
        if actual_labeled != expected_labeled:
            raise RuntimeError("SOLAR label subsampling differs from frozen label budget")
        train_config = TrainConfig(**spec["train"])
        result = run_inductive(
            reference=reference,
            query=query,
            variant=spec["variant"],
            labels_key=spec["label_key"],
            batch_key=spec["batch_key"],
            label_fraction=float(spec["label_fraction"]),
            seed=int(spec["seed"]),
            anchor_seed=int(spec["anchor_seed"]),
            preprocess=PreprocessConfig(
                n_components=int(spec["n_components"]), source="X"
            ),
            train=train_config,
            device="cuda" if torch.cuda.is_available() else "cpu",
        )
        if result.metadata["n_labeled_reference"] != len(expected_labeled):
            raise RuntimeError("SOLAR reported an unexpected labeled-reference count")
        environment = _environment()
        result.metadata.update(
            {
                "benchmark_job_id": spec["job_id"],
                "execution_profile": spec.get("execution_profile", "formal"),
                "dataset_id": spec["dataset_id"],
                "dataset_sha256": spec["dataset_sha256"],
                "n_genes_input": int(reference.n_vars),
                "split_id": spec["split_id"],
                "split_file": spec["split_file"],
                "label_budget_file": spec["label_budget_file"] if smoke_cap is None else None,
                "label_budget_source": (
                    "frozen_manifest" if smoke_cap is None else "smoke_subset_derived"
                ),
                "preprocessing_profile": spec["preprocessing_profile"],
                "smoke_cells_per_role_batch_label": smoke_cap,
                "hvg_file": spec.get("gene_file"),
                "protocol": spec["protocol"],
                "supervision": spec["supervision"],
                "split_uses_query_labels": spec["split_uses_query_labels"],
                "unlabeled_reference_expression_used_for_preprocessing": True,
                "local_extension": spec["variant"] == "solar_orthogonal_uniq",
                "derived_from": (
                    "solar_orthogonal"
                    if spec["variant"] == "solar_orthogonal_uniq"
                    else None
                ),
                "only_changed_parameter": (
                    "anchor_dedup"
                    if spec["variant"] == "solar_orthogonal_uniq"
                    else None
                ),
                "solar_upstream_archive": spec["solar_upstream_archive"],
                "solar_upstream_sha256": spec["solar_upstream_sha256"],
                "command": [sys.executable, *sys.argv],
            }
        )
        result.save(transaction.staging)
        write_json(transaction.staging / "environment.json", environment)
        after = (dataset_path.stat().st_size, dataset_path.stat().st_mtime_ns)
        if before != after:
            raise RuntimeError("Input dataset size or modification time changed during job")
        if not np.isfinite(result.reference_embedding).all() or not np.isfinite(
            result.query_embedding
        ).all():
            raise RuntimeError("SOLAR produced non-finite embedding values")
        status.update(
            {
                "state": "completed",
                "seconds": time.time() - started,
                "n_reference": int(reference.n_obs),
                "n_query": int(query.n_obs),
                "n_genes": int(reference.n_vars),
                "embedding_dimension": int(result.reference_embedding.shape[1]),
            }
        )
        write_json(transaction.staging / "status.json", status)
        return commit_output(transaction)
    except Exception as exc:
        status.update(
            {
                "state": "failed",
                "seconds": time.time() - started,
                "error": f"{type(exc).__name__}: {exc}",
            }
        )
        write_json(transaction.staging / "status.json", status)
        raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--job-spec", type=Path, required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    spec = json.loads(args.job_spec.read_text(encoding="utf-8"))
    output = run_job(spec, force=args.force)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
