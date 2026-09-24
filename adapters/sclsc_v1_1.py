#!/usr/bin/env python3
"""Run the pinned SCLSC core under the frozen-reference v1.1 or v1.2 contract.

The fixed whole-batch query is never used for fitting or model selection. The
remaining reference cells are split 9:1 with a seeded stratified split, as in
the SCLSC batch-split experiment. The official MLP, contrastive loss, type
representatives, projection helper, and early stopper are imported unchanged.
Both contracts write embeddings only: query labels and final metrics belong to
the separate scorer. v1.1 retains 16 dimensions; v1.2 requires 30 dimensions.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import os
import platform
import re
import resource
import subprocess
import sys
import time
import tomllib
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import anndata as ad
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import DataLoader, TensorDataset

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
OFFICIAL_REPOSITORY = REPOSITORY_ROOT / "external" / "SCLSC"
OFFICIAL_CODE = OFFICIAL_REPOSITORY / "code"
EXPECTED_COMMIT = "2e827dfebb793dffb548fd297e5c4e70fa40692f"
ENVIRONMENT_ROOT = REPOSITORY_ROOT / "environments" / "sclsc-v1_1"
ENVIRONMENT_MANIFEST = ENVIRONMENT_ROOT / "pyproject.toml"
ENVIRONMENT_LOCK = ENVIRONMENT_ROOT / "uv.lock"
EXPECTED_ENVIRONMENT_PREFIX = ENVIRONMENT_ROOT / ".venv"
CANONICAL_SEEDS = frozenset(range(40, 45))

if not (OFFICIAL_CODE / "models.py").is_file() or not (OFFICIAL_CODE / "util.py").is_file():
    raise RuntimeError(
        "Pinned SCLSC checkout is absent. Clone https://github.com/yaozhong/SCLSC.git "
        f"to {OFFICIAL_REPOSITORY} and checkout {EXPECTED_COMMIT}."
    )
if str(OFFICIAL_CODE) not in sys.path:
    sys.path.insert(0, str(OFFICIAL_CODE))

from models import ContrastiveLoss, EarlyStopper, MLP, project  # noqa: E402
from util import set_seed  # noqa: E402

DEFAULT_INPUT = REPOSITORY_ROOT / "data/scib/raw/human_pancreas_norm_complexBatch.h5ad"
DEFAULT_SPLIT = REPOSITORY_ROOT / (
    "outputs/benchmark/solar_scib_rebuttal_v1/splits/"
    "pancreas/heldout_fluidigmc1/split.csv"
)
DEFAULT_FEATURES = REPOSITORY_ROOT / (
    "outputs/benchmark/solar_scib_rebuttal_v1/preprocessing/"
    "pancreas/heldout_fluidigmc1/hvg2000_pca40.txt"
)
DEFAULT_OUTPUT = REPOSITORY_ROOT / (
    "outputs/baseline_extension_v1_1/sclsc_refonly/"
    "pancreas/heldout_fluidigmc1/seed_40"
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_ids(values: list[str] | np.ndarray) -> str:
    return hashlib.sha256(
        ("\n".join(np.asarray(values, dtype=str).tolist()) + "\n").encode("utf-8")
    ).hexdigest()


def write_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)


def write_internal_split(
    path: Path,
    reference_ids: list[str],
    reference_labels: np.ndarray,
    train_indices: np.ndarray,
    validation_indices: np.ndarray,
) -> None:
    membership = np.full(len(reference_ids), "", dtype=object)
    membership[train_indices] = "train"
    membership[validation_indices] = "validation"
    if set(membership.tolist()) != {"train", "validation"}:
        raise RuntimeError("internal split does not partition every reference cell")
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("barcode", "subset", "label"))
        writer.writerows(zip(reference_ids, membership.tolist(), reference_labels.tolist(), strict=True))
    os.replace(temporary, path)


def slurm_log_path(output_dir: Path) -> Path | None:
    override = os.environ.get("BASELINE_EXTENSION_SLURM_LOG")
    if override:
        return Path(override)
    job_id = os.environ.get("SLURM_JOB_ID")
    return output_dir / f"slurm-{job_id}.out" if job_id else None


def read_split(path: Path) -> tuple[list[str], list[str], dict[str, Any]]:
    """Read only cell IDs and split audit fields; never read the label column."""
    reference: list[str] = []
    query: list[str] = []
    held_out_batches: set[str] = set()
    split_uses_query_labels: set[str] = set()
    with path.open(newline="", encoding="utf-8") as handle:
        rows = csv.reader(handle)
        header = next(rows)
        positions = {name: header.index(name) for name in ("barcode", "role")}
        held_out_position = header.index("held_out_batch") if "held_out_batch" in header else None
        leakage_position = (
            header.index("split_uses_query_labels")
            if "split_uses_query_labels" in header
            else None
        )
        for row_number, row in enumerate(rows, start=2):
            barcode = row[positions["barcode"]]
            role = row[positions["role"]]
            if not barcode:
                raise ValueError(f"empty barcode at split row {row_number}")
            if role == "reference":
                reference.append(barcode)
            elif role == "query":
                query.append(barcode)
            else:
                raise ValueError(f"unknown role {role!r} at split row {row_number}")
            if held_out_position is not None:
                held_out_batches.add(row[held_out_position])
            if leakage_position is not None:
                split_uses_query_labels.add(row[leakage_position])
    return reference, query, {
        "held_out_batches": sorted(held_out_batches),
        "declared_split_uses_query_labels": sorted(split_uses_query_labels),
    }


def read_features(path: Path) -> list[str]:
    features = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not features:
        raise ValueError("feature list is empty")
    if len(features) != len(set(features)):
        raise ValueError("feature list contains duplicates")
    return features


def matrix_for(source: ad.AnnData, barcodes: list[str], features: list[str]) -> np.ndarray:
    subset = source[barcodes, :]
    matrix = subset.X
    if sp.issparse(matrix):
        matrix = matrix.toarray()
    feature_positions = source.var_names.get_indexer(features)
    if np.any(feature_positions < 0):
        raise ValueError("selected feature list contains genes absent from the dataset")
    matrix = np.asarray(matrix[:, feature_positions], dtype=np.float32)
    expected = (len(barcodes), len(features))
    if matrix.shape != expected:
        raise ValueError(f"unexpected expression shape {matrix.shape}, expected {expected}")
    if not np.isfinite(matrix).all():
        raise ValueError("expression matrix contains non-finite values")
    return np.ascontiguousarray(matrix)


def official_commit() -> tuple[str, bool]:
    commit = subprocess.run(
        ["git", "-C", str(OFFICIAL_REPOSITORY), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "-C", str(OFFICIAL_REPOSITORY), "status", "--porcelain", "--untracked-files=no"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return commit, not bool(status.strip())


def class_counts(labels: np.ndarray, classes: np.ndarray) -> dict[str, int]:
    return {
        str(label): int(np.sum(labels == index))
        for index, label in enumerate(classes)
    }


def make_internal_split(
    encoded_labels: np.ndarray,
    *,
    seed: int,
    validation_fraction: float,
) -> tuple[np.ndarray, np.ndarray]:
    splitter = StratifiedShuffleSplit(
        n_splits=1,
        test_size=validation_fraction,
        random_state=seed,
    )
    train_indices, validation_indices = next(
        splitter.split(np.zeros(len(encoded_labels), dtype=np.uint8), encoded_labels)
    )
    train_indices = np.asarray(train_indices, dtype=np.int64)
    validation_indices = np.asarray(validation_indices, dtype=np.int64)
    if set(train_indices.tolist()) & set(validation_indices.tolist()):
        raise RuntimeError("internal training and validation indices overlap")
    if len(train_indices) + len(validation_indices) != len(encoded_labels):
        raise RuntimeError("internal split does not cover every reference cell")
    expected_classes = set(np.unique(encoded_labels).tolist())
    if set(np.unique(encoded_labels[train_indices]).tolist()) != expected_classes:
        raise RuntimeError("internal training subset lost a reference class")
    if set(np.unique(encoded_labels[validation_indices]).tolist()) != expected_classes:
        raise RuntimeError("internal validation subset lost a reference class")
    return train_indices, validation_indices


def train_official_core(
    train_expression: np.ndarray,
    train_labels: np.ndarray,
    validation_expression: np.ndarray,
    validation_labels: np.ndarray,
    *,
    n_classes: int,
    device: torch.device,
    seed: int,
    margin: float,
    batch_size: int,
    epochs: int,
    learning_rate: float,
    output_dim: int,
    patience: int,
    min_delta: float,
    validation_step: int,
) -> tuple[torch.nn.Module, list[dict[str, Any]], np.ndarray, dict[str, Any]]:
    """Use the official optimizer loop semantics with reference-only validation."""
    if batch_size < 2 or len(train_expression) % batch_size == 1:
        raise ValueError("SCLSC BatchNorm forbids a singleton training batch; no cells were dropped")
    if len(train_expression) <= batch_size or len(validation_expression) <= batch_size:
        raise ValueError("official loss divisor requires at least two batches in each internal subset")
    expected_classes = set(range(n_classes))
    if n_classes < 2 or any(
        set(np.unique(labels).tolist()) != expected_classes
        for labels in (train_labels, validation_labels)
    ):
        raise ValueError("internal 90/10 subsets must both cover every reference class (at least two)")
    deterministic_algorithms = torch.use_deterministic_algorithms
    set_seed(seed)
    # Upstream util.set_seed assigns to this API instead of calling it. Restore
    # the function and enact the intended deterministic setting without editing
    # the pinned source checkout.
    torch.use_deterministic_algorithms = deterministic_algorithms
    deterministic_algorithms(True)

    train_label_tensor = torch.from_numpy(train_labels.astype(np.int64, copy=False))
    validation_label_tensor = torch.from_numpy(validation_labels.astype(np.int64, copy=False))
    train_one_hot = F.one_hot(train_label_tensor, num_classes=n_classes).to(torch.float32)
    validation_one_hot = F.one_hot(validation_label_tensor, num_classes=n_classes).to(torch.float32)
    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(train_expression), train_one_hot),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )
    validation_loader = DataLoader(
        TensorDataset(torch.from_numpy(validation_expression), validation_one_hot),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    class_representatives = np.stack(
        [train_expression[train_labels == index].mean(axis=0) for index in range(n_classes)]
    ).astype(np.float32, copy=False)
    representative_tensor = torch.from_numpy(np.ascontiguousarray(class_representatives)).to(device)

    model = MLP(train_expression.shape[1], 512, 128, output_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = ContrastiveLoss(margin=margin)
    early_stopper = EarlyStopper(patience=patience, min_delta=min_delta)
    history: list[dict[str, Any]] = []
    stopped_early = False

    for epoch_index in range(epochs):
        epoch_started = time.perf_counter()
        total_train_loss = 0.0
        train_batches = 0
        for input_x, label in train_loader:
            model.train()
            input_x = input_x.to(device)
            label = label.to(device)
            embed_x = model(input_x)
            embed_representative = model(representative_tensor)
            loss = criterion(embed_x[:, None, :], embed_representative, label)
            loss_value = float(loss.item())
            if not np.isfinite(loss_value):
                raise FloatingPointError("Non-finite SCLSC training loss")
            total_train_loss += loss_value
            train_batches += 1
            optimizer.zero_grad()
            loss.backward()
            if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
                raise FloatingPointError("Non-finite SCLSC training gradient")
            optimizer.step()
        if train_batches < 2:
            raise RuntimeError("official loss divisor requires at least two training batches")

        validation_due = epoch_index % validation_step == 0 or epoch_index == epochs - 1
        total_validation_loss = 0.0
        validation_batches = 0
        validation_loss: float | None = None
        if validation_due:
            with torch.no_grad():
                for input_x, label in validation_loader:
                    model.eval()
                    input_x = input_x.to(device)
                    label = label.to(device)
                    embed_x = model(input_x)
                    embed_representative = model(representative_tensor)
                    loss = criterion(embed_x[:, None, :], embed_representative, label)
                    loss_value = float(loss.item())
                    if not np.isfinite(loss_value):
                        raise FloatingPointError("Non-finite SCLSC validation loss")
                    total_validation_loss += loss_value
                    validation_batches += 1
            if validation_batches < 2:
                raise RuntimeError("official loss divisor requires at least two validation batches")
            # sc2l_main.py divides by the final zero-based batch index. Preserve
            # that executable behavior; also record the conventional batch mean.
            validation_loss = total_validation_loss / (validation_batches - 1)
            stopped_early = bool(early_stopper.early_stop(validation_loss))

        record = {
            "epoch": epoch_index + 1,
            "train_batches": train_batches,
            "finite_gradients_checked_before_every_update": True,
            "train_loss_official_divisor": total_train_loss / (train_batches - 1),
            "train_loss_batch_mean": total_train_loss / train_batches,
            "validation_due": validation_due,
            "validation_batches": validation_batches,
            "validation_loss_official_divisor": validation_loss,
            "validation_loss_batch_mean": (
                total_validation_loss / validation_batches if validation_batches else None
            ),
            "early_stopper_counter": int(early_stopper.counter),
            "early_stopper_min_validation_loss": float(early_stopper.min_validation_loss),
            "stopped_early": stopped_early,
            "seconds": time.perf_counter() - epoch_started,
        }
        history.append(record)
        print(
            f"Epoch-{epoch_index + 1:03d}/{epochs}, "
            f"Train loss={record['train_loss_official_divisor']:.8f}, "
            f"Validation loss={validation_loss}, stopped={stopped_early}, "
            f"seconds={record['seconds']:.3f}",
            flush=True,
        )
        if stopped_early:
            break

    return model, history, class_representatives, {
        "stopped_early": stopped_early,
        "stop_epoch": len(history),
        "minimum_validation_loss": float(early_stopper.min_validation_loss),
        "early_stopper_counter": int(early_stopper.counter),
    }


def package_versions() -> dict[str, str]:
    return {
        name: importlib.metadata.version(name)
        for name in ("anndata", "numpy", "scipy", "scikit-learn", "torch")
    }


def assert_locked_environment() -> dict[str, Any]:
    missing = [
        str(path)
        for path in (ENVIRONMENT_MANIFEST, ENVIRONMENT_LOCK)
        if not path.is_file()
    ]
    if missing:
        raise FileNotFoundError(f"SCLSC environment files are missing: {missing}")
    if Path(sys.prefix).resolve() != EXPECTED_ENVIRONMENT_PREFIX.resolve():
        raise RuntimeError(f"SCLSC must run from {EXPECTED_ENVIRONMENT_PREFIX}, got {sys.prefix}")
    if Path(sys.executable).resolve() != (EXPECTED_ENVIRONMENT_PREFIX / "bin/python").resolve():
        raise RuntimeError("SCLSC executable does not resolve to the locked interpreter")
    if sys.version_info[:2] != (3, 11):
        raise RuntimeError(f"SCLSC lock requires Python 3.11, got {platform.python_version()}")
    with ENVIRONMENT_LOCK.open("rb") as handle:
        lock = tomllib.load(handle)
    names = {"anndata", "numpy", "scipy", "scikit-learn", "torch"}
    locked_versions = {
        package["name"]: package["version"]
        for package in lock["package"]
        if package["name"] in names
    }
    installed_versions = package_versions()
    if installed_versions != locked_versions:
        raise RuntimeError(
            "Installed SCLSC packages differ from uv.lock: "
            f"installed={installed_versions}, locked={locked_versions}"
        )
    return {
        "prefix": str(Path(sys.prefix).resolve()),
        "executable": sys.executable,
        "resolved_executable": str(Path(sys.executable).resolve()),
        "python": platform.python_version(),
        "locked_top_level_versions": locked_versions,
        "manifest_sha256": sha256_file(ENVIRONMENT_MANIFEST),
        "lock_sha256": sha256_file(ENVIRONMENT_LOCK),
        "adapter_sha256": sha256_file(Path(__file__).resolve()),
        "torch_build": str(torch.__version__),
        "torch_cuda": torch.version.cuda,
    }


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract-version", choices=("v1.1", "v1.2"), default="v1.1")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--split", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument("--features", type=Path, default=DEFAULT_FEATURES)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--batch-key", default="tech")
    parser.add_argument("--label-key", default="celltype")
    parser.add_argument("--dataset-id", default="pancreas")
    parser.add_argument("--heldout-batch", default="fluidigmc1")
    parser.add_argument("--seed", type=int, default=40)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--margin", type=float, default=1.0)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--output-dim", type=int, default=16)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--min-delta", type=float, default=0.01)
    parser.add_argument("--validation-step", type=int, default=2)
    parser.add_argument("--expected-reference", type=int, default=15744)
    parser.add_argument("--expected-query", type=int, default=638)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--diagnostic", action="store_true")
    parser.add_argument(
        "--force", action="store_true",
        help="Explicitly replace incomplete legacy v1.1 output only; completed output is never overwritten.",
    )
    return parser.parse_args()


def canonical_parameters(args: argparse.Namespace) -> bool:
    return (
        args.seed in CANONICAL_SEEDS
        and args.epochs == 100
        and args.batch_size == 256
        and args.margin == 1.0
        and args.learning_rate == 1e-3
        and args.output_dim == (30 if args.contract_version == "v1.2" else 16)
        and args.validation_fraction == 0.1
        and args.patience == 10
        and args.min_delta == 0.01
        and args.validation_step == 2
    )


def parameter_contract(args: argparse.Namespace) -> dict[str, Any]:
    """Stable requested settings, excluding runtime outcomes and filesystem paths."""
    return {
        "output_dim": args.output_dim,
        "hidden_dim1": 512,
        "hidden_dim2": 128,
        "seed": args.seed,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "margin": args.margin,
        "learning_rate": args.learning_rate,
        "validation_fraction": args.validation_fraction,
        "patience": args.patience,
        "min_delta": args.min_delta,
        "validation_step": args.validation_step,
        "optimizer": "torch.optim.Adam",
        "data_loader_shuffle": False,
        "restore_best_checkpoint": False,
    }


def parameter_hash(parameters: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(parameters, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    ).hexdigest()


def recorded_parameters(info: dict[str, Any], contract_version: str) -> dict[str, Any]:
    """Recover legacy numeric architecture from JSON, never unpickle a checkpoint."""
    parameters = info["parameters"]
    encoder = re.fullmatch(
        r"official MLP\((\d+), (\d+), (\d+), (\d+)\)", parameters["encoder"]
    )
    if encoder is None:
        raise ValueError("unrecognized SCLSC encoder metadata")
    input_dim, hidden_dim1, hidden_dim2, encoder_dim = map(int, encoder.groups())
    if input_dim != info["preprocessing"]["feature_count"]:
        raise ValueError("SCLSC encoder input dimension disagrees with selected features")
    output_dim = parameters.get("output_dim")
    if output_dim is None and contract_version == "v1.1":
        output_dim = encoder_dim
    if type(output_dim) is not int or output_dim != encoder_dim:
        raise ValueError("missing or inconsistent numeric SCLSC output dimension")
    if contract_version == "v1.2" and info.get("output_dim") != output_dim:
        raise ValueError("v1.2 requires matching top-level numeric output_dim")
    stopping = parameters["early_stopping"]
    selection = "current final epoch; upstream does not restore the best epoch"
    if stopping["checkpoint_selection"] != selection:
        raise ValueError("SCLSC checkpoint selection contract changed")
    restore_best = stopping.get("restore_best_checkpoint")
    if restore_best is None and contract_version == "v1.1":
        restore_best = False
    internal = parameters["internal_validation"]
    if internal["random_state"] != info["model_seed"]:
        raise ValueError("internal split seed differs from model seed")
    if internal["implementation"] != "sklearn.model_selection.StratifiedShuffleSplit":
        raise ValueError("SCLSC internal split algorithm changed")
    return {
        "output_dim": output_dim,
        "hidden_dim1": hidden_dim1,
        "hidden_dim2": hidden_dim2,
        "seed": info["model_seed"],
        "epochs": parameters["epochs_requested"],
        "batch_size": parameters["batch_size"],
        "margin": parameters["margin"],
        "learning_rate": parameters["learning_rate"],
        "validation_fraction": internal["validation_fraction"],
        "patience": stopping["patience"],
        "min_delta": stopping["min_delta"],
        "validation_step": stopping["validation_every_epochs"],
        "optimizer": parameters["optimizer"],
        "data_loader_shuffle": parameters["data_loader_shuffle"],
        "restore_best_checkpoint": restore_best,
    }


def assert_reusable(
    output_dir: Path,
    args: argparse.Namespace,
    input_hashes: dict[str, str],
    parameters_sha256: str,
) -> None:
    info = json.loads((output_dir / "run_info.json").read_text(encoding="utf-8"))
    contract = info.get("contract_version")
    if contract is None and info.get("run_kind") == "formal_v1_1":
        contract = "v1.1"
    expected_kind = (
        f"formal_{args.contract_version.replace('.', '_')}" if not args.diagnostic else "diagnostic"
    )
    expected_identity = {
        "method": "sclsc_refonly",
        "run_kind": expected_kind,
        "model_seed": args.seed,
        "seed": args.seed,
        "dataset_id": args.dataset_id,
        "heldout_batch": str(args.heldout_batch),
        "label_key": args.label_key,
        "batch_key": args.batch_key,
        "n_reference": args.expected_reference,
        "n_query": args.expected_query,
        "source_repository_commit": EXPECTED_COMMIT,
    }
    if contract != args.contract_version:
        raise ValueError("existing SCLSC contract version does not match requested contract")
    for key, expected in expected_identity.items():
        if info.get(key) != expected:
            raise ValueError(f"existing SCLSC {key} does not match requested value")
    for key, expected in input_hashes.items():
        if info["input"].get(key) != expected:
            raise ValueError(f"existing SCLSC input {key} does not match current file")
    recovered_hash = parameter_hash(recorded_parameters(info, contract))
    if recovered_hash != parameters_sha256:
        raise ValueError("existing SCLSC parameter hash does not match requested settings")
    stored_hash = info.get("parameters_sha256")
    if stored_hash != parameters_sha256 and not (contract == "v1.1" and stored_hash is None):
        raise ValueError("missing or inconsistent SCLSC parameters_sha256")
    artifact_files = {
        "embedding": "embedding.npz",
        "checkpoint": "checkpoint.pt",
        "training_history": "training_history.json",
        "internal_reference_split": "internal_reference_split.csv",
    }
    artifacts = info["artifacts"]
    for name, filename in artifact_files.items():
        path = output_dir / filename
        if not path.is_file() or artifacts.get(f"{name}_sha256") != sha256_file(path):
            raise ValueError(f"existing SCLSC {name} artifact is missing or changed")
    if info.get("output_checksum") != artifacts["embedding_sha256"]:
        raise ValueError("existing SCLSC output checksum disagrees with embedding artifact")
    reference_ids, query_ids, _ = read_split(args.split.resolve())
    with np.load(output_dir / "embedding.npz", allow_pickle=False) as persisted:
        embedding = persisted["embedding"]
        if (
            embedding.shape != (args.expected_reference + args.expected_query, args.output_dim)
            or embedding.dtype != np.float32
            or not np.isfinite(embedding).all()
            or persisted["barcodes"].astype(str).tolist() != reference_ids + query_ids
        ):
            raise ValueError("existing SCLSC embedding dimension, values, or cell IDs do not match")


def main() -> None:
    args = parse_arguments()
    input_path = args.input.resolve()
    split_path = args.split.resolve()
    feature_path = args.features.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    status_path = output_dir / "status.json"
    runtime_path = output_dir / "runtime.json"

    if args.contract_version == "v1.2" and args.diagnostic:
        raise ValueError("v1.2 requires the formal contract; --diagnostic cannot bypass it")
    if not args.diagnostic and not canonical_parameters(args):
        raise ValueError(
            f"formal {args.contract_version} runs require the canonical seed/hyperparameter contract"
        )
    if args.seed < 0 or args.epochs <= 0 or args.validation_step <= 0:
        raise ValueError("seed, epochs, and validation-step must be valid positive settings")
    if not 0.0 < args.validation_fraction < 1.0:
        raise ValueError("validation-fraction must be between zero and one")
    input_hashes = {
        "sha256": sha256_file(input_path),
        "split_sha256": sha256_file(split_path),
        "features_sha256": sha256_file(feature_path),
    }
    parameters_sha256 = parameter_hash(parameter_contract(args))
    environment = assert_locked_environment()
    commit, source_clean = official_commit()
    if commit != EXPECTED_COMMIT:
        raise RuntimeError(f"official source commit {commit} does not match {EXPECTED_COMMIT}")
    if not source_clean:
        raise RuntimeError("official SCLSC checkout has tracked uncommitted changes")

    if status_path.is_file():
        existing = json.loads(status_path.read_text(encoding="utf-8"))
        if existing.get("state") == "completed":
            try:
                assert_reusable(output_dir, args, input_hashes, parameters_sha256)
            except (KeyError, TypeError, ValueError, OSError) as error:
                raise FileExistsError(f"non-reusable completed output at {output_dir}: {error}") from error
            print(json.dumps({"state": "reused", "output": str(output_dir)}), flush=True)
            return
        if not (args.contract_version == "v1.1" and args.force):
            raise FileExistsError(
                f"incomplete output exists at {output_dir}; preserve it and select a fresh attempt directory"
            )
    elif any(
        (output_dir / name).exists()
        for name in (
            "run_info.json", "embedding.npz", "checkpoint.pt",
            "training_history.json", "internal_reference_split.csv",
        )
    ):
        raise FileExistsError(f"artifacts without completion status exist at {output_dir}")

    started_wall = utc_now()
    started = time.perf_counter()
    phase_seconds: dict[str, float] = {}
    write_json(status_path, {"state": "running", "started_at": started_wall})

    try:

        phase_started = time.perf_counter()
        reference_ids, query_ids, split_audit = read_split(split_path)
        features = read_features(feature_path)
        if args.contract_version == "v1.2" and len(features) != 2000:
            raise ValueError("formal v1.2 requires exactly 2,000 frozen reference-selected HVGs")
        if len(reference_ids) != args.expected_reference:
            raise ValueError(f"expected {args.expected_reference} reference IDs, got {len(reference_ids)}")
        if len(query_ids) != args.expected_query:
            raise ValueError(f"expected {args.expected_query} query IDs, got {len(query_ids)}")
        if len(set(reference_ids)) != len(reference_ids) or len(set(query_ids)) != len(query_ids):
            raise ValueError("fixed split contains duplicate IDs")
        if set(reference_ids) & set(query_ids):
            raise ValueError("fixed reference and query IDs overlap")
        if split_audit["declared_split_uses_query_labels"] != ["False"]:
            raise ValueError("fixed split does not declare split_uses_query_labels=False")
        phase_seconds["split_and_feature_validation"] = time.perf_counter() - phase_started

        device = torch.device(args.device)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA device requested but torch.cuda.is_available() is false")

        phase_started = time.perf_counter()
        source = ad.read_h5ad(input_path, backed="r")
        try:
            required_obs = {args.batch_key, args.label_key}
            if missing_obs := required_obs - set(source.obs.columns):
                raise KeyError(f"dataset lacks obs columns {sorted(missing_obs)}")
            missing_cells = set(reference_ids).union(query_ids).difference(source.obs_names)
            if missing_cells:
                raise ValueError(f"dataset is missing {len(missing_cells)} fixed-split IDs")
            missing_features = set(features).difference(source.var_names)
            if missing_features:
                raise ValueError(f"dataset is missing {len(missing_features)} selected features")

            reference_positions = source.obs_names.get_indexer(reference_ids)
            reference_labels = (
                source.obs.iloc[reference_positions][args.label_key].astype(str).to_numpy()
            )
            classes, encoded_labels = np.unique(reference_labels, return_inverse=True)
            reference_expression = matrix_for(source, reference_ids, features)
            train_indices, validation_indices = make_internal_split(
                encoded_labels,
                seed=args.seed,
                validation_fraction=args.validation_fraction,
            )
            train_ids = np.asarray(reference_ids, dtype=str)[train_indices]
            validation_ids = np.asarray(reference_ids, dtype=str)[validation_indices]
            internal_split_path = output_dir / "internal_reference_split.csv"
            write_internal_split(
                internal_split_path,
                reference_ids,
                reference_labels,
                train_indices,
                validation_indices,
            )
            phase_seconds["load_and_split_reference"] = time.perf_counter() - phase_started

            phase_started = time.perf_counter()
            model, history, representatives, stopper = train_official_core(
                reference_expression[train_indices],
                encoded_labels[train_indices],
                reference_expression[validation_indices],
                encoded_labels[validation_indices],
                n_classes=len(classes),
                device=device,
                seed=args.seed,
                margin=args.margin,
                batch_size=args.batch_size,
                epochs=args.epochs,
                learning_rate=args.learning_rate,
                output_dim=args.output_dim,
                patience=args.patience,
                min_delta=args.min_delta,
                validation_step=args.validation_step,
            )
            phase_seconds["reference_only_training"] = time.perf_counter() - phase_started

            checkpoint_path = output_dir / "checkpoint.pt"
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "classes": classes.tolist(),
                    "features": features,
                    "class_representatives": (
                        torch.from_numpy(representatives)
                        if args.contract_version == "v1.2" else representatives
                    ),
                    "contract_version": args.contract_version,
                    "parameters_sha256": parameters_sha256,
                    "reference_ids": reference_ids,
                    "query_ids": query_ids,
                    "train_ids": train_ids.tolist(),
                    "validation_ids": validation_ids.tolist(),
                    "official_commit": commit,
                    "train_ids_sha256": sha256_ids(train_ids),
                    "validation_ids_sha256": sha256_ids(validation_ids),
                    "parameters": {
                        "input_dim": len(features),
                        "hidden_dim1": 512,
                        "hidden_dim2": 128,
                        "output_dim": args.output_dim,
                        "margin": args.margin,
                        "batch_size": args.batch_size,
                        "epochs": args.epochs,
                        "learning_rate": args.learning_rate,
                        "seed": args.seed,
                        "validation_fraction": args.validation_fraction,
                        "patience": args.patience,
                        "min_delta": args.min_delta,
                        "validation_step": args.validation_step,
                    },
                },
                checkpoint_path,
            )
            history_path = output_dir / "training_history.json"
            write_json(
                history_path,
                {
                    "history": history,
                    "epochs_requested": args.epochs,
                    "epochs_completed": len(history),
                    "early_stopping": stopper,
                    "external_query_used": False,
                },
            )

            phase_started = time.perf_counter()
            reference_embedding = project(model, reference_expression, device, "MLP")
            phase_seconds["reference_projection"] = time.perf_counter() - phase_started
            # Query expression is materialized only after all optimizer and
            # validation/early-stopping decisions have completed.
            phase_started = time.perf_counter()
            query_expression = matrix_for(source, query_ids, features)
            query_embedding = project(model, query_expression, device, "MLP")
            del query_expression
            phase_seconds["query_mapping"] = time.perf_counter() - phase_started
        finally:
            source.file.close()

        embedding = np.concatenate((reference_embedding, query_embedding), axis=0).astype(
            np.float32, copy=False
        )
        barcodes = np.asarray(reference_ids + query_ids, dtype=str)
        expected_rows = args.expected_reference + args.expected_query
        if embedding.shape != (expected_rows, args.output_dim):
            raise ValueError(f"unexpected embedding shape {embedding.shape}")
        if embedding.dtype != np.float32 or not np.isfinite(embedding).all():
            raise ValueError("embedding is not finite float32")

        embedding_path = output_dir / "embedding.npz"
        temporary_embedding = output_dir / "embedding.npz.tmp"
        with temporary_embedding.open("wb") as handle:
            np.savez_compressed(handle, embedding=embedding, barcodes=barcodes)
        os.replace(temporary_embedding, embedding_path)
        with np.load(embedding_path, allow_pickle=False) as persisted:
            persisted_embedding = persisted["embedding"]
            persisted_barcodes = persisted["barcodes"].astype(str)
            persisted_checks = {
                "row_count": int(persisted_embedding.shape[0]),
                "embedding_dim": int(persisted_embedding.shape[1]),
                "embedding_dtype_float32": persisted_embedding.dtype == np.float32,
                "all_finite": bool(np.isfinite(persisted_embedding).all()),
                "barcode_order_exact": persisted_barcodes.tolist() == reference_ids + query_ids,
                "reference_query_overlap_count": 0,
            }
        if not (
            persisted_checks["row_count"] == expected_rows
            and persisted_checks["embedding_dim"] == args.output_dim
            and persisted_checks["embedding_dtype_float32"]
            and persisted_checks["all_finite"]
            and persisted_checks["barcode_order_exact"]
            and persisted_checks["reference_query_overlap_count"] == 0
        ):
            raise RuntimeError(f"persisted embedding validation failed: {persisted_checks}")

        completed_wall = utc_now()
        elapsed = time.perf_counter() - started
        phase_seconds["total"] = elapsed
        log_path = slurm_log_path(output_dir)
        runtime = {
            "started_at": started_wall,
            "completed_at": completed_wall,
            "wall_seconds": elapsed,
            "total_wall_seconds": elapsed,
            "peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0,
            "reference_build_seconds": (
                phase_seconds["load_and_split_reference"]
                + phase_seconds["reference_only_training"]
            ),
            "query_mapping_seconds": phase_seconds["query_mapping"],
            "phase_seconds": phase_seconds,
            "hostname": platform.node(),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "slurm_array_job_id": os.environ.get("SLURM_ARRAY_JOB_ID"),
            "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
            "log_path": str(log_path) if log_path is not None else None,
            "device": str(device),
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda,
            "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        }
        write_json(runtime_path, runtime)

        model_source_path = OFFICIAL_CODE / "models.py"
        util_source_path = OFFICIAL_CODE / "util.py"
        native_train_path = OFFICIAL_CODE / "sc2l_main.py"
        internal_split_path = output_dir / "internal_reference_split.csv"
        run_kind = (
            f"formal_{args.contract_version.replace('.', '_')}"
            if canonical_parameters(args) and not args.diagnostic else "diagnostic"
        )
        run_info = {
            "schema_version": 3,
            "contract_version": args.contract_version,
            "output_dim": args.output_dim,
            "parameters_sha256": parameters_sha256,
            "executable": sys.executable,
            "argv": [sys.executable, *sys.argv],
            "method": "sclsc_refonly",
            "official_method": "SCLSC",
            "adapter": "sclsc_v1_1",
            "run_kind": run_kind,
            "dataset_id": args.dataset_id,
            "source_dataset_filename": input_path.name,
            "heldout_batch": str(args.heldout_batch),
            "split": f"heldout_{args.heldout_batch}",
            "protocol": "inductive_held_out_batch",
            "fit_scope": "reference_only_with_reference_internal_validation",
            "model_seed": args.seed,
            "seed": args.seed,
            "official_source": {
                "repository": "https://github.com/yaozhong/SCLSC.git",
                "commit": commit,
                "version": "unversioned Git repository",
                "tracked_files_clean": source_clean,
                "paper_doi": "10.1038/s41598-023-50185-2",
                "imported_symbols": {
                    "models.MLP": "external/SCLSC/code/models.py:32-56",
                    "models.ContrastiveLoss": "external/SCLSC/code/models.py:10-28",
                    "models.project": "external/SCLSC/code/models.py:134-146",
                    "models.EarlyStopper": "external/SCLSC/code/models.py:149-170",
                    "util.set_seed": "external/SCLSC/code/util.py:6-14",
                },
                "native_training_reference": "external/SCLSC/code/sc2l_main.py:130-250",
                "native_knn_reference": "external/SCLSC/code/sc2l_test_knn.py:49-59",
                "source_hashes_sha256": {
                    "models.py": sha256_file(model_source_path),
                    "util.py": sha256_file(util_source_path),
                    "sc2l_main.py": sha256_file(native_train_path),
                },
                "early_stopping_discrepancy": {
                    "repository_default_patience": 10,
                    "published_methods_patience": 5,
                    "selected": "repository_default_patience_10",
                    "reason": "The pinned executable repository and README invocation leave patience at its code default.",
                },
            },
            "source_repository_commit": commit,
            "environment": environment,
            "preprocessing": {
                "input_representation": "dataset-provided log-normalized X",
                "feature_selection": "fixed benchmark reference-only hvg2000_pca40.txt",
                "feature_count": len(features),
                "counts_used": False,
                "counts_reconstructed_with_expm1": False,
                "benchmark_adaptation": (
                    "Raw counts are not reliable across all three frozen benchmark datasets. "
                    "Therefore official count filtering, analytic-Pearson-residual HVG selection, "
                    "normalize_total, and log1p are not rerun; the identical reference-selected "
                    "2,000-HVG normalized-X input contract is used for every method."
                ),
            },
            "parameters": {
                "encoder": f"official MLP({len(features)}, 512, 128, {args.output_dim})",
                "output_dim": args.output_dim,
                "objective": "official instance-type ContrastiveLoss",
                "type_representation": "per-class mean expression over internal training cells only",
                "margin": args.margin,
                "batch_size": args.batch_size,
                "epochs_requested": args.epochs,
                "epochs_completed": len(history),
                "learning_rate": args.learning_rate,
                "optimizer": "torch.optim.Adam",
                "data_loader_shuffle": False,
                "internal_validation": {
                    "implementation": "sklearn.model_selection.StratifiedShuffleSplit",
                    "train_fraction": 1.0 - args.validation_fraction,
                    "validation_fraction": args.validation_fraction,
                    "random_state": args.seed,
                    "train_cells": int(len(train_indices)),
                    "validation_cells": int(len(validation_indices)),
                    "train_ids_sha256": sha256_ids(train_ids),
                    "validation_ids_sha256": sha256_ids(validation_ids),
                    "train_class_counts": class_counts(encoded_labels[train_indices], classes),
                    "validation_class_counts": class_counts(encoded_labels[validation_indices], classes),
                },
                "early_stopping": {
                    "implementation": "official models.EarlyStopper",
                    "patience": args.patience,
                    "min_delta": args.min_delta,
                    "validation_every_epochs": args.validation_step,
                    "checkpoint_selection": "current final epoch; upstream does not restore the best epoch",
                    "restore_best_checkpoint": False,
                    **stopper,
                },
            },
            "input": {
                "path": str(input_path),
                "split_path": str(split_path),
                "features_path": str(feature_path),
                **input_hashes,
            },
            "reference_cell_ids_hash": sha256_ids(reference_ids),
            "query_cell_ids_hash": sha256_ids(query_ids),
            "training_cell_ids_hash": sha256_ids(train_ids),
            "validation_cell_ids_hash": sha256_ids(validation_ids),
            "n_reference": len(reference_ids),
            "n_query": len(query_ids),
            "counts": {
                "reference": len(reference_ids),
                "internal_train": len(train_indices),
                "internal_validation": len(validation_indices),
                "query": len(query_ids),
                "total": len(barcodes),
                "classes_from_reference_only": len(classes),
            },
            "batch_key": args.batch_key,
            "label_key": args.label_key,
            "split_audit": split_audit,
            "query_labels_visible_to_model": False,
            "query_labels_read_by_adapter": False,
            "final_metrics_computed_by_adapter": False,
            "query_expression_used_for_reference_fit": False,
            "query_expression_used_for_parameter_update": False,
            "query_expression_used_for_model_selection": False,
            "mapping_mode": "frozen official MLP projection after reference-only training and model selection",
            "annotation_readouts": (
                {
                    "common": "distance-weighted 15-NN fit on all benchmark reference cells",
                    "native": "unweighted 10-NN fit on internal training cells only",
                }
                if args.contract_version == "v1.1"
                else {
                    "common": "external scorer: distance-weighted 15-NN fit on all benchmark reference cells",
                    "native": "not computed under the v1.2 contract",
                }
            ),
            "software_versions": {
                **package_versions(),
                "python": environment["python"],
                "torch_build": environment["torch_build"],
                "torch_cuda": environment["torch_cuda"],
            },
            "artifacts": {
                "embedding": str(embedding_path),
                "embedding_sha256": sha256_file(embedding_path),
                "checkpoint": str(checkpoint_path),
                "checkpoint_sha256": sha256_file(checkpoint_path),
                "training_history": str(history_path),
                "training_history_sha256": sha256_file(history_path),
                "internal_reference_split": str(internal_split_path),
                "internal_reference_split_sha256": sha256_file(internal_split_path),
                "slurm_log": str(log_path) if log_path is not None else None,
            },
            "output_checksum": sha256_file(embedding_path),
            "validation": persisted_checks,
        }
        write_json(output_dir / "run_info.json", run_info)
        write_json(
            status_path,
            {
                "state": "completed",
                "started_at": started_wall,
                "completed_at": completed_wall,
                "run_kind": run_kind,
                "checks": persisted_checks,
                "output_checksum": run_info["output_checksum"],
            },
        )
        print(
            json.dumps(
                {
                    "state": "completed",
                    "runtime_seconds": elapsed,
                    "epochs_completed": len(history),
                    **persisted_checks,
                }
            ),
            flush=True,
        )
    except Exception as error:
        completed_wall = utc_now()
        elapsed = time.perf_counter() - started
        log_path = slurm_log_path(output_dir)
        write_json(
            runtime_path,
            {
                "started_at": started_wall,
                "completed_at": completed_wall,
                "wall_seconds": elapsed,
                "total_wall_seconds": elapsed,
                "peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0,
                "phase_seconds": phase_seconds,
                "hostname": platform.node(),
                "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
                "error_type": type(error).__name__,
            },
        )
        write_json(
            status_path,
            {
                "state": "failed",
                "started_at": started_wall,
                "completed_at": completed_wall,
                "error_type": type(error).__name__,
                "error": str(error),
                "traceback": traceback.format_exc(),
                "log_path": str(log_path) if log_path is not None else None,
            },
        )
        raise


if __name__ == "__main__":
    main()
