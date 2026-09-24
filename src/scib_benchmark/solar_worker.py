from __future__ import annotations

import argparse
import hashlib
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


def _sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def _source_binding(spec: dict[str, Any]) -> dict[str, Any]:
    import SOLAR

    root = Path(SOLAR.__file__).resolve().parent
    expected = spec.get("solar_package_root")
    if expected is not None and root != Path(expected).resolve():
        raise RuntimeError(f"SOLAR import binding mismatch: expected {expected}, loaded {root}")
    modules = {}
    for name, module in sorted(sys.modules.items()):
        if name != "SOLAR" and not name.startswith("SOLAR."):
            continue
        filename = getattr(module, "__file__", None)
        if filename is None:
            continue
        path = Path(filename).resolve()
        if not path.is_relative_to(root):
            raise RuntimeError(f"SOLAR module {name} escaped package root: {path}")
        modules[name] = {"path": str(path), "sha256": _sha256(path)}
    return {
        "solar_package_root": str(root),
        "modules": modules,
        "worker": {"path": str(Path(__file__).resolve()), "sha256": _sha256(Path(__file__))},
        "variant_adapter": {
            "path": str(Path(sys.modules[register_orthogonal_uniq.__module__].__file__).resolve()),
            "sha256": _sha256(Path(sys.modules[register_orthogonal_uniq.__module__].__file__)),
        },
    }


def _input_hashes(spec: dict[str, Any]) -> dict[str, str]:
    hashes = {
        key: _sha256(Path(spec[key]))
        for key in ("dataset_path", "split_file", "label_budget_file", "gene_file")
        if spec.get(key) is not None
    }
    if hashes["dataset_path"] != spec["dataset_sha256"]:
        raise RuntimeError("Dataset checksum differs from frozen worker spec")
    for path_key, hash_key in (
        ("split_file", "split_sha256"),
        ("label_budget_file", "label_budget_sha256"),
        ("gene_file", "features_sha256"),
    ):
        if hash_key in spec and hashes.get(path_key) != spec[hash_key]:
            raise RuntimeError(f"{path_key} checksum differs from frozen worker spec")
    return hashes


def _validate_embeddings(result: Any, split: pd.DataFrame, dimension: int) -> None:
    for role in ("reference", "query"):
        expected = split.loc[split["role"] == role, "barcode"].to_numpy(dtype=str)
        barcodes = np.asarray(getattr(result, f"{role}_barcodes"), dtype=str)
        embedding = np.asarray(getattr(result, f"{role}_embedding"))
        if not np.array_equal(barcodes, expected):
            raise RuntimeError(f"SOLAR {role} embedding cell IDs/order differ from frozen split")
        if embedding.shape != (len(expected), dimension):
            raise RuntimeError(f"SOLAR {role} embedding has unexpected shape: {embedding.shape}")
        if not np.isfinite(embedding).all():
            raise RuntimeError(f"SOLAR {role} embedding contains non-finite values")


def _validate_reuse(
    output: Path, spec: dict[str, Any], source: dict[str, Any], inputs: dict[str, str]
) -> None:
    for filename, expected in (
        ("run_config.json", spec),
        ("source_binding.json", source),
        ("input_hashes.json", inputs),
    ):
        path = output / filename
        if not path.is_file() or json.loads(path.read_text()) != expected:
            raise RuntimeError(f"Completed SOLAR output identity mismatch: {filename}")
    status = json.loads((output / "status.json").read_text())
    hashes = status.get("artifact_sha256", {})
    required = {
        "embedding.npz", "model/solar_model.pt", "model/solar_metadata.json",
        "internal_reference_split.csv", "outer_split.csv", "training_history.json",
        "run_info.json", "pca_preprocessor.npz",
    }
    if not required.issubset(hashes):
        raise RuntimeError("Completed SOLAR output lacks required v1.2 artifact hashes")
    for filename in required:
        path = output / filename
        if not path.is_file() or _sha256(path) != hashes[filename]:
            raise RuntimeError(f"Completed SOLAR artifact missing or changed: {filename}")


def run_job(spec: dict[str, Any], force: bool = False) -> Path:
    from SOLAR.benchmark import PreprocessConfig, TrainConfig, run_inductive
    from SOLAR.benchmark import variants as variants_module
    from SOLAR.benchmark.splits import subsample_labels

    derived = register_orthogonal_uniq(variants_module)
    assert_only_dedup_diff(variants_module.VARIANTS["solar_orthogonal"], derived)
    formal_v1_2 = spec.get("contract_version") == "v1.2"
    save_model = spec.get("save_model", False)
    if formal_v1_2:
        if save_model is not True or int(spec["train"]["embedding_dim"]) != 30:
            raise ValueError("v1.2 requires save_model=true and requested embedding_dim=30")
        if spec.get("smoke_cells_per_role_batch_label") is not None:
            raise ValueError("v1.2 formal training does not permit smoke subsampling")
        if spec["variant"] != "solar_orthogonal" or float(spec["label_fraction"]) != 1.0:
            raise ValueError("v1.2 training requires solar_orthogonal with all reference labels")
        if not spec.get("solar_package_root"):
            raise ValueError("v1.2 requires an explicit solar_package_root binding")
    source = _source_binding(spec)
    inputs = _input_hashes(spec) if formal_v1_2 else {}
    output = Path(spec["output_dir"])
    transaction = begin_output(output, force=force)
    if transaction.reused:
        if formal_v1_2:
            _validate_reuse(output, spec, source, inputs)
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
        if formal_v1_2:
            write_json(transaction.staging / "run_config.json", spec)
            write_json(transaction.staging / "source_binding.json", source)
            write_json(transaction.staging / "input_hashes.json", inputs)
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
        if formal_v1_2 and reference.obs[spec["label_key"]].astype(str).nunique() > 30:
            raise ValueError("v1.2 embedding_dim=30 is smaller than the reference class count")
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
        if formal_v1_2:
            _validate_embeddings(result, split, 30)
            internal = pd.DataFrame(result.model.internal_reference_split)
            if (
                internal["barcode"].duplicated().any()
                or set(internal["barcode"]) != expected_labeled
                or set(internal["role"]) != {"train", "validation"}
            ):
                raise RuntimeError("Recorded DataLoader split does not partition labeled reference IDs")
            labeled_ids = reference.obs_names[selected].astype(str).to_numpy()
            if not np.array_equal(
                internal["barcode"].to_numpy(),
                labeled_ids[internal["dataset_index"].to_numpy(dtype=int)],
            ):
                raise RuntimeError("Recorded DataLoader indices do not map to reference cell IDs")
            selection = result.model.training_selection
            if not (
                selection["restored_best_checkpoint"]
                and 1 <= selection["selected_epoch"] <= selection["epochs_completed"]
            ):
                raise RuntimeError("v1.2 model lacks a valid restored validation-selected epoch")
            split.to_csv(transaction.staging / "outer_split.csv", index=False)
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
                "executable": sys.executable,
                "argv": list(sys.orig_argv),
                "source_binding": source,
                "contract_version": spec.get("contract_version"),
                "save_model": save_model,
            }
        )
        result.save(transaction.staging, save_model=save_model)
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
        if formal_v1_2:
            status["artifact_sha256"] = {
                name: _sha256(transaction.staging / name)
                for name in (
                    "embedding.npz", "model/solar_model.pt", "model/solar_metadata.json",
                    "internal_reference_split.csv", "outer_split.csv", "training_history.json",
                    "run_info.json", "pca_preprocessor.npz",
                )
            }
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
