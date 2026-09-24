from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import anndata as ad
import numpy as np
import torch
from sklearn.decomposition import PCA

from ..api import SOLARModel
from .controls import ControlConfig, _seed_everything, train_objective_control
from .splits import fit_reference_pca, subsample_labels
from .variants import VariantSpec, get_variant


@dataclass(frozen=True)
class PreprocessConfig:
    n_components: int = 40
    source: str = "X"
    source_key: str | None = None


@dataclass(frozen=True)
class TrainConfig:
    hidden_dim: int = 512
    embedding_dim: int = 128
    dropout: float = 0.1
    max_epochs: int = 50
    batch_size: int = 512
    learning_rate: float = 1e-4
    weight_decay: float = 0.0
    temperature: float = 0.07
    margin: float = 0.2
    train_size: float = 0.9
    early_stopping_patience: int = 10
    early_stopping_min_delta: float = 1e-4
    gradient_clip_norm: float | None = 1.0
    num_workers: int = 0


@dataclass
class BenchmarkResult:
    variant: VariantSpec
    reference_embedding: np.ndarray
    query_embedding: np.ndarray
    reference_barcodes: np.ndarray
    query_barcodes: np.ndarray
    metadata: dict[str, Any]
    model: Any
    pca: PCA
    anchors: np.ndarray | None = None

    def save(self, output_dir: str | Path, save_model: bool = False) -> Path:
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            output / "embedding.npz",
            reference=np.asarray(self.reference_embedding, dtype=np.float32),
            query=np.asarray(self.query_embedding, dtype=np.float32),
            reference_barcodes=np.asarray(self.reference_barcodes, dtype=str),
            query_barcodes=np.asarray(self.query_barcodes, dtype=str),
        )
        np.savez_compressed(
            output / "pca_preprocessor.npz",
            components=self.pca.components_,
            mean=self.pca.mean_,
            explained_variance=self.pca.explained_variance_,
        )
        if self.anchors is not None:
            np.save(output / "anchors.npy", self.anchors.astype(np.float32))
        (output / "run_info.json").write_text(
            json.dumps(self.metadata, indent=2, sort_keys=True) + "\n"
        )
        if save_model:
            checkpoint = output / "model"
            if isinstance(self.model, SOLARModel):
                self.model.save(checkpoint)
                with (output / "internal_reference_split.csv").open("w", newline="") as handle:
                    writer = csv.DictWriter(handle, fieldnames=["barcode", "role", "dataset_index"])
                    writer.writeheader()
                    writer.writerows(self.model.internal_reference_split)
                (output / "training_history.json").write_text(
                    json.dumps({
                        "schema_version": 1,
                        "split_source": "actual_dataloader_datasets",
                        "history": self.model.history,
                        **self.model.training_selection,
                        "training_config": self.model.training_config,
                    }, indent=2, sort_keys=True) + "\n"
                )
            else:
                checkpoint.mkdir(parents=True, exist_ok=True)
                torch.save(self.model.state_dict(), checkpoint / "control_model.pt")
        return output


def _validate_inputs(
    reference: ad.AnnData,
    query: ad.AnnData,
    labels_key: str,
    preprocess: PreprocessConfig,
) -> None:
    if reference.n_obs == 0 or query.n_obs == 0:
        raise ValueError("Reference and query must both contain cells")
    if labels_key not in reference.obs:
        raise KeyError(f"labels_key {labels_key!r} not found in reference.obs")
    if not reference.obs_names.is_unique or not query.obs_names.is_unique:
        raise ValueError("Reference and query barcodes must be unique within each set")
    overlap = set(reference.obs_names.astype(str)) & set(query.obs_names.astype(str))
    if overlap:
        raise ValueError(
            f"Reference/query barcode overlap detected ({len(overlap)} cells)"
        )
    if preprocess.source != "obsm" and not np.array_equal(
        reference.var_names.astype(str), query.var_names.astype(str)
    ):
        raise ValueError("Reference and query var_names must match in the same order")


def _anchor_diagnostics(model: SOLARModel, classes: list[str]) -> tuple[dict, np.ndarray]:
    anchors = np.asarray(model.encode_text(classes), dtype=np.float64)
    gram = anchors @ anchors.T
    offdiag = gram[~np.eye(len(classes), dtype=bool)]
    singular = np.linalg.svd(anchors, compute_uv=False)
    diagnostics = {
        "classes": classes,
        "mean_offdiag": float(offdiag.mean()) if len(offdiag) else 0.0,
        "mean_abs_coherence": float(np.abs(offdiag).mean()) if len(offdiag) else 0.0,
        "max_abs_coherence": float(np.abs(offdiag).max()) if len(offdiag) else 0.0,
        "min_angle_degrees": (
            float(np.degrees(np.arccos(np.clip(offdiag.max(), -1, 1))))
            if len(offdiag)
            else 90.0
        ),
        "condition_number": float(singular.max() / singular.min()),
    }
    return diagnostics, anchors.astype(np.float32)


def _representation_anndata(
    values: np.ndarray, source: ad.AnnData, obs_columns: list[str]
) -> ad.AnnData:
    obs = source.obs[obs_columns].copy() if obs_columns else source.obs.iloc[:, :0].copy()
    return ad.AnnData(X=np.asarray(values, dtype=np.float32), obs=obs)


def run_inductive(
    reference: ad.AnnData,
    query: ad.AnnData,
    variant: str,
    labels_key: str = "cell_type",
    batch_key: str | None = None,
    label_fraction: float = 1.0,
    seed: int = 42,
    anchor_seed: int = 0,
    preprocess: PreprocessConfig | None = None,
    train: TrainConfig | None = None,
    device: str | None = None,
) -> BenchmarkResult:
    """Train on labeled reference cells and embed untouched held-out query cells.

    PCA is fitted on reference expression only. Query ``obs`` columns are not
    copied into the model-facing AnnData, so query labels cannot be consumed by
    SOLAR or either objective control.
    """
    preprocess = preprocess or PreprocessConfig()
    train = train or TrainConfig()
    spec = get_variant(variant)
    _validate_inputs(reference, query, labels_key, preprocess)
    reference_z, query_z, pca = fit_reference_pca(
        reference,
        query,
        preprocess.n_components,
        seed,
        preprocess.source,
        preprocess.source_key,
    )
    labeled_indices = subsample_labels(
        reference, labels_key, label_fraction, seed, batch_key
    )
    obs_columns = [labels_key]
    if batch_key is not None and batch_key != labels_key:
        obs_columns.append(batch_key)
    reference_repr = _representation_anndata(reference_z, reference, obs_columns)
    # Deliberately expose no query metadata to the model adapter.
    query_repr = _representation_anndata(query_z, query, [])
    labeled = reference_repr[labeled_indices].copy()
    labels = labeled.obs[labels_key].astype(str).to_numpy()
    if len(set(labels.tolist())) < 2:
        raise ValueError("Selected label budget contains fewer than two classes")
    resolved_device = torch.device(
        device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    anchors = None
    if spec.objective == "solar_supcon":
        model_kwargs = {
            "hidden_dim": train.hidden_dim,
            "embedding_dim": train.embedding_dim,
            "dropout": train.dropout,
            "temperature": train.temperature,
            "margin": train.margin,
        }
        model_kwargs.update(spec.resolved_model_kwargs(anchor_seed))
        SOLARModel.setup_anndata(labeled, labels_key=labels_key)
        # Weight initialization happens inside SOLARModel.__init__, before
        # SOLARModel.train() sets any seed. Seed here so `seed` controls init.
        _seed_everything(seed)
        model = SOLARModel(labeled, device=str(resolved_device), **model_kwargs)
        history = model.train(
            max_epochs=train.max_epochs,
            batch_size=train.batch_size,
            learning_rate=train.learning_rate,
            weight_decay=train.weight_decay,
            train_size=train.train_size,
            random_seed=seed,
            early_stopping_patience=train.early_stopping_patience,
            early_stopping_min_delta=train.early_stopping_min_delta,
            gradient_clip_norm=train.gradient_clip_norm,
            num_workers=train.num_workers,
        )
        reference_embedding = model.transform(reference_repr)
        query_embedding = model.transform(query_repr)
        diagnostics = None
        if model.model_config.semantic_mode != "none":
            classes = sorted(set(labels.tolist()))
            diagnostics, anchors = _anchor_diagnostics(model, classes)
        details = {
            "objective": "anchor_augmented_supcon",
            "semantic_mode": model.model_config.semantic_mode,
            "model_kwargs": model_kwargs,
            "history": history,
            "training_selection": model.training_selection,
            "internal_split_source": "actual_dataloader_datasets",
            "anchor_diagnostics": diagnostics,
            "classifier_head_used_for_evaluation": False,
        }
    else:
        control_config = ControlConfig(
            hidden_dim=int(spec.model_kwargs.get("hidden_dim", train.hidden_dim)),
            embedding_dim=int(
                spec.model_kwargs.get("embedding_dim", train.embedding_dim)
            ),
            dropout=train.dropout,
            max_epochs=train.max_epochs,
            batch_size=train.batch_size,
            learning_rate=train.learning_rate,
            weight_decay=train.weight_decay,
            temperature=train.temperature,
            train_size=train.train_size,
            early_stopping_patience=train.early_stopping_patience,
            early_stopping_min_delta=train.early_stopping_min_delta,
            gradient_clip_norm=train.gradient_clip_norm,
        )
        model, reference_embedding, query_embedding, details, anchors = (
            train_objective_control(
                train_values=reference_z[labeled_indices],
                train_labels=labels,
                reference_values=reference_z,
                query_values=query_z,
                objective=spec.objective,
                config=control_config,
                seed=seed,
                anchor_seed=anchor_seed,
                device=resolved_device,
            )
        )
        details["control_config"] = asdict(control_config)

    metadata = {
        "schema_version": 1,
        "protocol": "inductive_reference_only_preprocessing",
        "variant": spec.name,
        "variant_group": spec.group,
        "objective": spec.objective,
        "evidence_scope": spec.evidence_scope,
        "description": spec.description,
        "labels_key": labels_key,
        "batch_key": batch_key,
        "label_fraction": float(label_fraction),
        "seed": int(seed),
        "anchor_seed": int(anchor_seed),
        "n_reference": int(reference.n_obs),
        "n_labeled_reference": int(len(labeled_indices)),
        "n_query": int(query.n_obs),
        "query_labels_visible_to_adapter": False,
        "query_expression_used_for_preprocessing_fit": False,
        "query_expression_adaptation": False,
        "preprocess_config": asdict(preprocess),
        "train_config": asdict(train),
        "embedding_dimension": int(reference_embedding.shape[1]),
        "details": details,
    }
    return BenchmarkResult(
        variant=spec,
        reference_embedding=np.asarray(reference_embedding, dtype=np.float32),
        query_embedding=np.asarray(query_embedding, dtype=np.float32),
        reference_barcodes=reference.obs_names.astype(str).to_numpy(),
        query_barcodes=query.obs_names.astype(str).to_numpy(),
        metadata=metadata,
        model=model,
        pca=pca,
        anchors=anchors,
    )
