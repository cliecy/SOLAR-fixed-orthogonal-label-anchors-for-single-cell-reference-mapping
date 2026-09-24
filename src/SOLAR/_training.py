"""Private epoch loop for SOLARModel.train."""
from __future__ import annotations

import copy
from typing import Any, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader


def assemble_contrastive_features(
    z_x: torch.Tensor,
    z_t: Optional[torch.Tensor],
    labels: Sequence[str],
    text_anchors: Sequence[str],
    anchor_dedup: bool = False,
) -> tuple[torch.Tensor, list[str]]:
    """Build SupCon features/labels, optionally keeping one anchor per class.

    Default (anchor_dedup=False / rep): stack per-cell expression and text
    embeddings as two views [N, 2, D] — each class anchor is repeated once per
    cell of that class in the contrastive pool.

    With anchor_dedup=True / uniq: keep all expression embeddings and append
    one embedding per unique text anchor, yielding a single-view pool of size
    N + K rather than 2N.
    """
    if z_t is None:
        return z_x.unsqueeze(1), list(labels)

    if not anchor_dedup:
        return torch.stack([z_x, z_t], dim=1), list(labels)

    unique_texts = list(dict.fromkeys(text_anchors))
    first_index: dict[str, int] = {}
    for index, text in enumerate(text_anchors):
        if text not in first_index:
            first_index[text] = index
    z_t_unique = torch.stack(
        [z_t[first_index[text]] for text in unique_texts], dim=0
    )
    unique_labels = [labels[first_index[text]] for text in unique_texts]
    pooled = torch.cat([z_x, z_t_unique], dim=0)
    pooled_labels = list(labels) + unique_labels
    return pooled.unsqueeze(1), pooled_labels


def run_epochs(
    model: torch.nn.Module,
    device: torch.device,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader],
    history: dict[str, list[float]],
    max_epochs: int,
    early_stopping_patience: int,
    early_stopping_min_delta: float,
    gradient_clip_norm: Optional[float],
    anchor_dedup: bool = False,
) -> dict[str, Any]:
    best_val_loss = float("inf")
    best_model_state = None
    wait_counter = 0
    best_epoch = None
    stopped_early = False
    epochs_completed = 0
    history.setdefault("skipped_train_batches", [])
    history.setdefault("skipped_validation_batches", [])

    for epoch in range(1, max_epochs + 1):
        model.train()
        train_losses: list[float] = []
        skipped_train = 0
        for expressions, text_anchors, labels in train_loader:
            if not model.has_text_branch and len(set(labels)) == len(labels):
                skipped_train += 1
                optimizer.zero_grad(set_to_none=True)
                continue
            expressions = expressions.to(device, non_blocking=True)
            z_x, z_t = model(expressions, text_anchors, device)
            features, feature_labels = assemble_contrastive_features(
                z_x, z_t, labels, text_anchors, anchor_dedup=anchor_dedup
            )
            loss = criterion(features, labels=feature_labels)
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite training contrastive loss")

            optimizer.zero_grad()
            loss.backward()
            if gradient_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm, error_if_nonfinite=True)
            elif any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
                raise FloatingPointError("Non-finite training gradient")
            optimizer.step()
            train_losses.append(float(loss.item()))

        avg_train_loss = float(np.mean(train_losses)) if train_losses else 0.0
        history["train_loss"].append(avg_train_loss)
        history["skipped_train_batches"].append(skipped_train)
        epochs_completed = epoch

        if val_loader is None:
            continue

        model.eval()
        val_losses: list[float] = []
        skipped_validation = 0
        with torch.no_grad():
            for expressions, text_anchors, labels in val_loader:
                if not model.has_text_branch and len(set(labels)) == len(labels):
                    skipped_validation += 1
                    continue
                expressions = expressions.to(device, non_blocking=True)
                z_x, z_t = model(expressions, text_anchors, device)
                features, feature_labels = assemble_contrastive_features(
                    z_x, z_t, labels, text_anchors, anchor_dedup=anchor_dedup
                )
                val_loss = criterion(features, labels=feature_labels)
                if not torch.isfinite(val_loss):
                    raise FloatingPointError("Non-finite validation contrastive loss")
                val_losses.append(float(val_loss.item()))

        history["skipped_validation_batches"].append(skipped_validation)
        if not val_losses:
            raise ValueError("Validation epoch contains no valid positive contrastive pairs")
        avg_val_loss = float(np.mean(val_losses))
        history["val_loss"].append(avg_val_loss)

        if avg_val_loss < (best_val_loss - early_stopping_min_delta):
            best_val_loss = avg_val_loss
            wait_counter = 0
            best_model_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
        else:
            wait_counter += 1
            if wait_counter >= early_stopping_patience:
                stopped_early = True
                break

    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    return {
        "epoch_numbering": "one_based",
        "epochs_completed": epochs_completed,
        "selected_epoch": best_epoch if best_model_state is not None else epochs_completed,
        "best_validation_epoch": best_epoch,
        "best_validation_loss": best_val_loss if best_epoch is not None else None,
        "restored_best_checkpoint": best_model_state is not None,
        "stopped_early": stopped_early,
        "finite_gradients_checked_before_every_update": True,
        "selection_rule": (
            "validation_loss_improvement_exceeding_min_delta"
            if val_loader is not None else "last_epoch_no_validation"
        ),
    }

