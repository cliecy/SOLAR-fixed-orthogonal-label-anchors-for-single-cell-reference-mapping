from __future__ import annotations

import copy
import random
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split

from ..src.anchor_model import ExpressionEncoder, _make_controlled_anchors


CONTROL_OBJECTIVES = ("cross_entropy", "fixed_anchor_cosine")


@dataclass(frozen=True)
class ControlConfig:
    hidden_dim: int = 512
    embedding_dim: int = 128
    dropout: float = 0.1
    max_epochs: int = 50
    batch_size: int = 512
    learning_rate: float = 1e-4
    weight_decay: float = 0.0
    temperature: float = 0.07
    train_size: float = 0.9
    early_stopping_patience: int = 10
    early_stopping_min_delta: float = 1e-4
    gradient_clip_norm: float | None = 1.0


class ObjectiveControl(nn.Module):
    """Simple supervised controls sharing SOLAR's expression encoder."""

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        objective: str,
        config: ControlConfig,
        anchor_seed: int,
    ) -> None:
        super().__init__()
        if objective not in CONTROL_OBJECTIVES:
            raise ValueError(f"Unknown objective control: {objective}")
        self.objective = objective
        self.temperature = float(config.temperature)
        self.encoder = ExpressionEncoder(
            input_dim, config.hidden_dim, config.embedding_dim, config.dropout
        )
        if objective == "cross_entropy":
            self.classifier = nn.Linear(config.embedding_dim, num_classes)
        else:
            anchors = _make_controlled_anchors(
                "orthogonal", num_classes, config.embedding_dim, anchor_seed
            )
            self.register_buffer("anchors", anchors)

    def encode(self, values: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.encoder(values), dim=1)

    def forward(self, values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        embedding = self.encode(values)
        if self.objective == "cross_entropy":
            logits = self.classifier(embedding)
        else:
            logits = embedding @ self.anchors.T / self.temperature
        return embedding, logits


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _loaders(
    values: np.ndarray,
    labels: np.ndarray,
    config: ControlConfig,
    seed: int,
) -> tuple[DataLoader, DataLoader]:
    if len(values) < 4:
        raise ValueError("Objective controls require at least four labeled cells")
    dataset = TensorDataset(
        torch.as_tensor(values, dtype=torch.float32),
        torch.as_tensor(labels, dtype=torch.long),
    )
    val_size = min(
        len(dataset) - 2,
        max(1, round(len(dataset) * (1 - config.train_size))),
    )
    train_size = len(dataset) - val_size
    train, val = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(seed),
    )
    train_loader = DataLoader(
        train,
        batch_size=config.batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
        drop_last=(train_size % config.batch_size == 1),
    )
    return train_loader, DataLoader(val, batch_size=config.batch_size, shuffle=False)


def _encode(
    model: ObjectiveControl,
    values: np.ndarray,
    device: torch.device,
    batch_size: int = 1024,
) -> np.ndarray:
    loader = DataLoader(
        torch.as_tensor(values, dtype=torch.float32),
        batch_size=batch_size,
        shuffle=False,
    )
    embeddings = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            embeddings.append(model.encode(batch.to(device)).cpu().numpy())
    return np.concatenate(embeddings)


def train_objective_control(
    train_values: np.ndarray,
    train_labels: np.ndarray,
    reference_values: np.ndarray,
    query_values: np.ndarray,
    objective: str,
    config: ControlConfig,
    seed: int,
    anchor_seed: int,
    device: torch.device,
) -> tuple[ObjectiveControl, np.ndarray, np.ndarray, dict, np.ndarray | None]:
    """Train a simple control on labeled reference cells and encode both sets."""
    _seed_everything(seed)
    classes = sorted(set(train_labels.astype(str).tolist()))
    if len(classes) < 2:
        raise ValueError("Objective controls require at least two label classes")
    class_to_idx = {label: index for index, label in enumerate(classes)}
    encoded = np.asarray([class_to_idx[str(label)] for label in train_labels])
    model = ObjectiveControl(
        input_dim=train_values.shape[1],
        num_classes=len(classes),
        objective=objective,
        config=config,
        anchor_seed=anchor_seed,
    ).to(device)
    train_loader, val_loader = _loaders(train_values, encoded, config, seed)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    criterion = nn.CrossEntropyLoss()
    history = {"train_loss": [], "val_loss": []}
    best_loss = float("inf")
    best_state = None
    wait = 0
    for _ in range(config.max_epochs):
        model.train()
        train_losses = []
        for values, labels in train_loader:
            values, labels = values.to(device), labels.to(device)
            _, logits = model(values)
            loss = criterion(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            if config.gradient_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), config.gradient_clip_norm
                )
            optimizer.step()
            train_losses.append(float(loss.item()))
        history["train_loss"].append(float(np.mean(train_losses)))

        model.eval()
        val_losses = []
        with torch.no_grad():
            for values, labels in val_loader:
                _, logits = model(values.to(device))
                val_losses.append(float(criterion(logits, labels.to(device)).item()))
        val_loss = float(np.mean(val_losses))
        history["val_loss"].append(val_loss)
        if val_loss < best_loss - config.early_stopping_min_delta:
            best_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            wait = 0
        else:
            wait += 1
            if wait >= config.early_stopping_patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)

    anchors = (
        model.anchors.detach().cpu().numpy()
        if objective == "fixed_anchor_cosine"
        else None
    )
    details = {
        "objective": objective,
        "classes": classes,
        "history": history,
        "query_expression_adaptation": False,
        "query_labels_visible_to_adapter": False,
        "classifier_head_used_for_evaluation": False,
    }
    return (
        model,
        _encode(model, reference_values, device),
        _encode(model, query_values, device),
        details,
        anchors,
    )
