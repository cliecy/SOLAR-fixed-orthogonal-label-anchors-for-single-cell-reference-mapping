from __future__ import annotations

import copy
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional

import anndata as ad
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from ._checkpoint import load_metadata, load_weights, save_checkpoint
from ._data import (
    build_dataloaders,
    build_dataset,
    compute_class_centroids,
    resolve_expression_matrix,
    resolve_text_and_labels,
)
from ._training import run_epochs
from .src.anchor_model import (
    CLASS_ANCHOR_MODES,
    IMBALANCE_MODES,
    SEMANTIC_MODES,
    MIModule,
)
from .src.loss import SupConLoss, inverse_frequency_weights


_SETUP_KEY = "_solar"


@dataclass
class AnnDataSetup:
    labels_key: str
    text_key: Optional[str] = None
    batch_key: Optional[str] = None
    layer: Optional[str] = None
    obsm_key: Optional[str] = None


@dataclass
class ModelConfig:
    hidden_dim: int = 512
    embedding_dim: int = 128
    dropout: float = 0.1
    temperature: float = 0.07
    margin: float = 0.2
    base_temperature: float = 0.07
    semantic_mode: str = "onehot"
    shuffle_labels: bool = False
    text_template: Optional[dict] = None
    anchor_seed: int = 0
    anchor_geometry: Optional[str] = None
    anchor_coherence: Optional[float] = None
    anchor_dedup: bool = False
    imbalance_mode: str = "natural"


class SOLARModel:
    """SOLAR label-anchor model without a pretrained language branch."""

    @classmethod
    def setup_anndata(
        cls,
        adata: ad.AnnData,
        labels_key: str,
        text_key: Optional[str] = None,
        batch_key: Optional[str] = None,
        layer: Optional[str] = None,
        obsm_key: Optional[str] = None,
    ) -> None:
        if layer and obsm_key:
            raise ValueError("Only one of layer or obsm_key can be set")
        for key, container, name in (
            (labels_key, adata.obs, "labels_key"),
            (batch_key, adata.obs, "batch_key"),
            (text_key, adata.obs, "text_key"),
            (layer, adata.layers, "layer"),
            (obsm_key, adata.obsm, "obsm_key"),
        ):
            if key is not None and key not in container:
                raise KeyError(f"{name} {key!r} not found in AnnData")
        adata.uns[_SETUP_KEY] = asdict(
            AnnDataSetup(labels_key, text_key, batch_key, layer, obsm_key)
        )

    @classmethod
    def view_anndata_setup(cls, adata: ad.AnnData) -> dict[str, Any]:
        if _SETUP_KEY not in adata.uns:
            raise RuntimeError("Call SOLARModel.setup_anndata before initialization")
        return copy.deepcopy(adata.uns[_SETUP_KEY])

    def __init__(
        self,
        adata: ad.AnnData,
        hidden_dim: int = 512,
        embedding_dim: int = 128,
        dropout: float = 0.1,
        temperature: float = 0.07,
        margin: float = 0.2,
        base_temperature: float = 0.07,
        semantic_mode: str = "onehot",
        shuffle_labels: bool = False,
        text_template: Optional[dict] = None,
        anchor_seed: int = 0,
        anchor_geometry: Optional[str] = None,
        anchor_coherence: Optional[float] = None,
        anchor_dedup: bool = False,
        imbalance_mode: str = "natural",
        device: Optional[str] = None,
    ) -> None:
        if semantic_mode not in SEMANTIC_MODES:
            raise ValueError(
                f"Unknown semantic_mode {semantic_mode!r}. Options: {SEMANTIC_MODES}"
            )
        if imbalance_mode not in IMBALANCE_MODES:
            raise ValueError(
                f"Unknown imbalance_mode {imbalance_mode!r}. Options: {IMBALANCE_MODES}"
            )
        self.adata = adata
        self.setup = self._get_setup(adata)
        self.device = self._resolve_device(device)
        self.model_config = ModelConfig(
            hidden_dim=hidden_dim,
            embedding_dim=embedding_dim,
            dropout=dropout,
            temperature=temperature,
            margin=margin,
            base_temperature=base_temperature,
            semantic_mode=semantic_mode,
            shuffle_labels=shuffle_labels,
            text_template=text_template,
            anchor_seed=anchor_seed,
            anchor_geometry=anchor_geometry,
            anchor_coherence=anchor_coherence,
            anchor_dedup=bool(anchor_dedup),
            imbalance_mode=imbalance_mode,
        )
        self.num_classes: int | None = None
        self.class_to_idx: dict[str, int] | None = None
        self.class_centroids: np.ndarray | None = None
        self._build_class_vocab(adata)
        self.input_dim = int(resolve_expression_matrix(adata, self.setup).shape[1])
        self.model = self._build_model().to(self.device)
        self.history: dict[str, list[float]] = {"train_loss": [], "val_loss": []}
        self.training_config: dict[str, Any] = {}
        self.internal_reference_split: list[dict[str, Any]] = []
        self.training_selection: dict[str, Any] = {}
        self.is_trained = False

    def _build_class_vocab(self, adata: ad.AnnData) -> None:
        if self.model_config.semantic_mode not in CLASS_ANCHOR_MODES:
            return
        anchors, _ = resolve_text_and_labels(
            adata,
            self.setup,
            self.model_config.text_template,
            self.model_config.shuffle_labels,
            self.model_config.anchor_seed,
        )
        classes = sorted(set(anchors))
        self.class_to_idx = {name: index for index, name in enumerate(classes)}
        self.num_classes = len(classes)
        if self.model_config.semantic_mode == "centroid":
            values = resolve_expression_matrix(adata, self.setup)
            self.class_centroids = compute_class_centroids(
                values, anchors, self.class_to_idx
            )

    def _build_model(self) -> MIModule:
        return MIModule(
            expression_input_dim=self.input_dim,
            expression_hidden_dim=self.model_config.hidden_dim,
            embedding_dim=self.model_config.embedding_dim,
            dropout=self.model_config.dropout,
            semantic_mode=self.model_config.semantic_mode,
            num_classes=self.num_classes,
            class_to_idx=self.class_to_idx,
            anchor_seed=self.model_config.anchor_seed,
            class_centroids=self.class_centroids,
            anchor_geometry=self.model_config.anchor_geometry,
            anchor_coherence=self.model_config.anchor_coherence,
        )

    def train(
        self,
        max_epochs: int = 50,
        batch_size: int = 512,
        learning_rate: float = 1e-4,
        weight_decay: float = 0.0,
        train_size: float = 0.9,
        shuffle: bool = True,
        random_seed: int = 42,
        early_stopping_patience: int = 10,
        early_stopping_min_delta: float = 1e-4,
        gradient_clip_norm: Optional[float] = 1.0,
        num_workers: int = 0,
    ) -> dict[str, list[float]]:
        if max_epochs <= 0 or batch_size <= 0:
            raise ValueError("max_epochs and batch_size must be positive")
        if learning_rate <= 0 or weight_decay < 0:
            raise ValueError("Invalid optimizer configuration")
        if not 0 < train_size <= 1:
            raise ValueError("train_size must lie in (0, 1]")
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(random_seed)
        dataset = build_dataset(
            self.adata,
            self.setup,
            self.model_config.text_template,
            self.model_config.shuffle_labels,
            self.model_config.anchor_seed,
        )
        train_loader, val_loader = build_dataloaders(
            dataset,
            batch_size,
            train_size,
            shuffle,
            random_seed,
            num_workers,
            self.model_config.imbalance_mode,
        )
        # Read the actual loader subsets without iterating a sampler or drawing RNG.
        cell_ids = self.adata.obs_names.astype(str).to_numpy()
        self.internal_reference_split = []
        for role, loader in (("train", train_loader), ("validation", val_loader)):
            if loader is None:
                continue
            loader_dataset = loader.dataset
            if loader_dataset is dataset:
                indices = range(len(dataset))
            elif isinstance(loader_dataset, Subset) and loader_dataset.dataset is dataset:
                indices = loader_dataset.indices
            else:
                raise RuntimeError("Cannot map DataLoader dataset to training cell IDs")
            self.internal_reference_split.extend(
                {"barcode": str(cell_ids[int(index)]), "role": role, "dataset_index": int(index)}
                for index in indices
            )
        class_weights = None
        if self.model_config.imbalance_mode == "weighted":
            class_weights = inverse_frequency_weights(dataset.labels)
        criterion = SupConLoss(
            temperature=self.model_config.temperature,
            base_temperature=self.model_config.base_temperature,
            margin=self.model_config.margin,
            class_weights=class_weights,
        )
        optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
        self.training_config = {
            "max_epochs": max_epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "train_size": train_size,
            "shuffle": shuffle,
            "random_seed": random_seed,
            "early_stopping_patience": early_stopping_patience,
            "early_stopping_min_delta": early_stopping_min_delta,
            "gradient_clip_norm": gradient_clip_norm,
            "num_workers": num_workers,
            "anchor_dedup": self.model_config.anchor_dedup,
            "imbalance_mode": self.model_config.imbalance_mode,
        }
        self.history = {"train_loss": [], "val_loss": []}
        self.training_selection = run_epochs(
            self.model,
            self.device,
            criterion,
            optimizer,
            train_loader,
            val_loader,
            self.history,
            max_epochs,
            early_stopping_patience,
            early_stopping_min_delta,
            gradient_clip_norm,
            self.model_config.anchor_dedup,
        )
        self.is_trained = True
        return copy.deepcopy(self.history)

    fit = train

    def get_latent_representation(
        self,
        adata: Optional[ad.AnnData] = None,
        batch_size: int = 1024,
        layer: Optional[str] = None,
        obsm_key: Optional[str] = None,
        store_key: Optional[str] = None,
        as_numpy: bool = True,
    ) -> np.ndarray | torch.Tensor:
        target = adata if adata is not None else self.adata
        setup = self._get_setup(target, allow_missing=True)
        if layer is not None or obsm_key is not None:
            setup = AnnDataSetup(
                setup.labels_key,
                setup.text_key,
                setup.batch_key,
                layer,
                obsm_key,
            )
        values = resolve_expression_matrix(target, setup)
        loader = DataLoader(
            torch.from_numpy(np.ascontiguousarray(values, dtype=np.float32)),
            batch_size=batch_size,
            shuffle=False,
        )
        outputs = []
        self.model.eval()
        with torch.no_grad():
            for batch in loader:
                outputs.append(
                    self.model.encode_expression(batch.to(self.device)).cpu()
                )
        latent = torch.cat(outputs)
        if store_key is not None:
            target.obsm[store_key] = latent.numpy()
        return latent.numpy() if as_numpy else latent

    transform = get_latent_representation

    def encode_text(
        self, anchors: list[str], batch_size: int = 256, as_numpy: bool = True
    ) -> np.ndarray | torch.Tensor:
        outputs = []
        self.model.eval()
        with torch.no_grad():
            for start in range(0, len(anchors), batch_size):
                outputs.append(
                    self.model.encode_text(
                        anchors[start : start + batch_size], self.device
                    ).cpu()
                )
        latent = torch.cat(outputs) if outputs else torch.empty((0, 0))
        return latent.numpy() if as_numpy else latent

    def save(self, dir_path: str | Path) -> None:
        metadata = {
            "input_dim": self.input_dim,
            "setup": asdict(self.setup),
            "model_config": asdict(self.model_config),
            "num_classes": self.num_classes,
            "class_to_idx": self.class_to_idx,
            "class_centroids": (
                self.class_centroids.tolist()
                if self.class_centroids is not None
                else None
            ),
            "training_config": copy.deepcopy(self.training_config),
            "history": copy.deepcopy(self.history),
            "internal_reference_split": copy.deepcopy(self.internal_reference_split),
            "training_selection": copy.deepcopy(self.training_selection),
            "is_trained": self.is_trained,
        }
        save_checkpoint(Path(dir_path), self.model.state_dict(), metadata)

    @classmethod
    def load(
        cls,
        dir_path: str | Path,
        adata: Optional[ad.AnnData] = None,
        device: Optional[str] = None,
    ) -> "SOLARModel":
        metadata = load_metadata(Path(dir_path))
        obj = cls.__new__(cls)
        obj.adata = adata
        obj.setup = AnnDataSetup(**metadata["setup"])
        obj.device = obj._resolve_device(device)
        obj.model_config = ModelConfig(**metadata["model_config"])
        obj.num_classes = metadata.get("num_classes")
        obj.class_to_idx = metadata.get("class_to_idx")
        centroids = metadata.get("class_centroids")
        obj.class_centroids = (
            np.asarray(centroids, dtype=np.float32) if centroids is not None else None
        )
        obj.input_dim = int(metadata["input_dim"])
        obj.model = obj._build_model().to(obj.device)
        obj.model.load_state_dict(load_weights(Path(dir_path), obj.device))
        obj.training_config = metadata.get("training_config", {})
        obj.history = metadata.get("history", {"train_loss": [], "val_loss": []})
        obj.internal_reference_split = metadata.get("internal_reference_split", [])
        obj.training_selection = metadata.get("training_selection", {})
        obj.is_trained = bool(metadata.get("is_trained", True))
        return obj

    @staticmethod
    def _resolve_device(device: Optional[str]) -> torch.device:
        return torch.device(
            device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        )

    @staticmethod
    def _get_setup(
        adata: ad.AnnData, allow_missing: bool = False
    ) -> AnnDataSetup:
        if _SETUP_KEY not in adata.uns:
            if allow_missing:
                return AnnDataSetup(labels_key="cell_type")
            raise RuntimeError("Call SOLARModel.setup_anndata before initialization")
        return AnnDataSetup(**adata.uns[_SETUP_KEY])
