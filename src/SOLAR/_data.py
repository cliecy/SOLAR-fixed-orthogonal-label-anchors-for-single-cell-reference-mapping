"""Private AnnData and DataLoader utilities for the SOLAR public facade."""
from __future__ import annotations

from collections import Counter
from typing import Any, Optional, Sequence

import anndata as ad
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler, random_split

from .src.anchor_model import IMBALANCE_MODES


class AnnDataPairDataset(Dataset):
    def __init__(
        self,
        expressions: np.ndarray,
        text_anchors: list[str],
        labels: list[str],
    ) -> None:
        self.expressions = torch.from_numpy(
            np.ascontiguousarray(expressions, dtype=np.float32)
        )
        self.text_anchors = text_anchors
        self.labels = labels

    def __len__(self) -> int:
        return self.expressions.shape[0]

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, str, str]:
        return self.expressions[idx], self.text_anchors[idx], self.labels[idx]


def collate_pairs(
    batch: list[tuple[torch.Tensor, str, str]],
) -> tuple[torch.Tensor, list[str], list[str]]:
    expressions = torch.stack([item[0] for item in batch])
    text_anchors = [item[1] for item in batch]
    labels = [item[2] for item in batch]
    return expressions, text_anchors, labels


def resolve_expression_matrix(adata: ad.AnnData, setup: Any) -> np.ndarray:
    if setup.obsm_key is not None:
        matrix = adata.obsm[setup.obsm_key]
    elif setup.layer is not None:
        matrix = adata.layers[setup.layer]
    else:
        matrix = adata.X

    if hasattr(matrix, "toarray"):
        matrix = matrix.toarray()
    matrix = np.asarray(matrix, dtype=np.float32)
    if matrix.ndim != 2:
        raise ValueError("Expression representation must be a 2D matrix.")
    if not np.all(np.isfinite(matrix)):
        matrix = np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)
    return np.ascontiguousarray(matrix, dtype=np.float32)


def resolve_text_and_labels(
    adata: ad.AnnData,
    setup: Any,
    text_template: Optional[dict],
    shuffle_labels: bool,
    anchor_seed: int,
) -> tuple[list[str], list[str]]:
    labels = adata.obs[setup.labels_key].astype(str).tolist()
    if setup.text_key is not None:
        text_anchors = adata.obs[setup.text_key].astype(str).tolist()
    else:
        text_anchors = list(labels)
    if len(labels) == 0:
        raise ValueError("AnnData contains no observations.")
    if len(set(labels)) < 2:
        raise ValueError(
            "SOLAR requires at least two distinct labels for supervised "
            "contrastive training."
        )

    if text_template:
        text_anchors = [text_template.get(text, text) for text in text_anchors]

    if shuffle_labels:
        rng = np.random.default_rng(anchor_seed)
        permutation = rng.permutation(len(labels))
        labels = [labels[index] for index in permutation]
        text_anchors = [text_anchors[index] for index in permutation]

    return text_anchors, labels


def compute_class_centroids(
    expressions: np.ndarray,
    text_anchors: Sequence[str],
    class_to_idx: dict[str, int],
) -> np.ndarray:
    """Mean expression/PCA vector per class, ordered by class_to_idx."""
    if expressions.ndim != 2:
        raise ValueError("expressions must be a 2D matrix.")
    if len(text_anchors) != expressions.shape[0]:
        raise ValueError("text_anchors length must match number of cells.")
    num_classes = len(class_to_idx)
    centroids = np.zeros((num_classes, expressions.shape[1]), dtype=np.float32)
    counts = np.zeros(num_classes, dtype=np.int64)
    for row, text in enumerate(text_anchors):
        if text not in class_to_idx:
            raise KeyError(f"Unknown class text for centroid: {text!r}")
        idx = class_to_idx[text]
        centroids[idx] += expressions[row]
        counts[idx] += 1
    if np.any(counts == 0):
        missing = [name for name, idx in class_to_idx.items() if counts[idx] == 0]
        raise ValueError(f"No cells found for classes: {missing}")
    centroids /= counts.astype(np.float32)[:, None]
    return np.ascontiguousarray(centroids, dtype=np.float32)


def inverse_frequency_sample_weights(labels: Sequence[str]) -> list[float]:
    """Per-example sampling weights proportional to inverse class frequency."""
    counts = Counter(str(label) for label in labels)
    return [1.0 / float(counts[str(label)]) for label in labels]


def build_dataset(
    adata: ad.AnnData,
    setup: Any,
    text_template: Optional[dict],
    shuffle_labels: bool,
    anchor_seed: int,
) -> AnnDataPairDataset:
    expressions = resolve_expression_matrix(adata, setup)
    text_anchors, labels = resolve_text_and_labels(
        adata, setup, text_template, shuffle_labels, anchor_seed
    )
    return AnnDataPairDataset(expressions, text_anchors, labels)


def build_dataloaders(
    dataset: AnnDataPairDataset,
    batch_size: int,
    train_size: float,
    shuffle: bool,
    random_seed: int,
    num_workers: int,
    imbalance_mode: str = "natural",
) -> tuple[DataLoader, Optional[DataLoader]]:
    if imbalance_mode not in IMBALANCE_MODES:
        raise ValueError(
            f"Unknown imbalance_mode '{imbalance_mode}'. Options: {IMBALANCE_MODES}"
        )

    total_size = len(dataset)
    loader_kwargs: dict[str, Any] = {
        "collate_fn": collate_pairs,
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
    }
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = True

    if total_size < 2 or train_size >= 1.0:
        train_loader = _make_train_loader(
            data_source=dataset,
            labels=dataset.labels,
            batch_size=batch_size,
            shuffle=shuffle,
            random_seed=random_seed,
            imbalance_mode=imbalance_mode,
            drop_last=(total_size % batch_size == 1),
            loader_kwargs=loader_kwargs,
        )
        return train_loader, None

    val_size = min(
        total_size - 1,
        max(1, int(round(total_size * (1.0 - train_size)))),
    )
    train_count = total_size - val_size
    train_dataset, val_dataset = random_split(
        dataset,
        [train_count, val_size],
        generator=torch.Generator().manual_seed(random_seed),
    )
    train_labels = [dataset.labels[int(i)] for i in train_dataset.indices]

    train_loader = _make_train_loader(
        data_source=train_dataset,
        labels=train_labels,
        batch_size=batch_size,
        shuffle=shuffle,
        random_seed=random_seed,
        imbalance_mode=imbalance_mode,
        drop_last=(train_count % batch_size == 1),
        loader_kwargs=loader_kwargs,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        **loader_kwargs,
    )
    return train_loader, val_loader


def _make_train_loader(
    data_source,
    labels: Sequence[str],
    batch_size: int,
    shuffle: bool,
    random_seed: int,
    imbalance_mode: str,
    drop_last: bool,
    loader_kwargs: dict[str, Any],
) -> DataLoader:
    if imbalance_mode == "balanced":
        weights = inverse_frequency_sample_weights(labels)
        sampler = WeightedRandomSampler(
            weights=torch.as_tensor(weights, dtype=torch.double),
            num_samples=len(weights),
            replacement=True,
            generator=torch.Generator().manual_seed(random_seed),
        )
        return DataLoader(
            data_source,
            batch_size=batch_size,
            sampler=sampler,
            shuffle=False,
            drop_last=drop_last,
            **loader_kwargs,
        )

    return DataLoader(
        data_source,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        **loader_kwargs,
    )
