from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


CLASS_ANCHOR_MODES = ("random", "onehot", "trainable", "centroid")
SEMANTIC_MODES = CLASS_ANCHOR_MODES + ("none",)
IMBALANCE_MODES = ("natural", "balanced", "weighted")
ANCHOR_GEOMETRIES = ("orthogonal", "simplex", "gaussian", "coherence")


class ExpressionEncoder(nn.Module):
    def __init__(
        self, input_dim: int, hidden_dim: int, output_dim: int, dropout: float = 0.1
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.net(values)


def _orthonormal_rows(
    num_rows: int, output_dim: int, generator: torch.Generator
) -> torch.Tensor:
    if output_dim < num_rows:
        raise ValueError(
            "orthogonal anchors require output_dim >= num_classes "
            f"(got {output_dim} < {num_rows})"
        )
    raw = torch.randn(output_dim, num_rows, generator=generator)
    q, _ = torch.linalg.qr(raw, mode="reduced")
    return q.T.contiguous()


def _make_controlled_anchors(
    geometry: str,
    num_classes: int,
    output_dim: int,
    anchor_seed: int,
    coherence: Optional[float] = None,
) -> torch.Tensor:
    if geometry not in ANCHOR_GEOMETRIES:
        raise ValueError(
            f"Unknown anchor_geometry {geometry!r}. Options: {ANCHOR_GEOMETRIES}"
        )
    generator = torch.Generator().manual_seed(int(anchor_seed))
    if geometry == "gaussian":
        return F.normalize(
            torch.randn(num_classes, output_dim, generator=generator), dim=1
        )
    if geometry == "orthogonal":
        return _orthonormal_rows(num_classes, output_dim, generator)
    if geometry == "simplex":
        if output_dim < num_classes - 1:
            raise ValueError(
                "simplex anchors require output_dim >= num_classes - 1"
            )
        base = torch.eye(num_classes) - torch.full(
            (num_classes, num_classes), 1.0 / num_classes
        )
        base = F.normalize(base, dim=1)
        rotation = _orthonormal_rows(num_classes, output_dim, generator)
        return F.normalize(base @ rotation, dim=1)
    if coherence is None or not 0.0 <= float(coherence) < 1.0:
        raise ValueError("coherence geometry requires 0 <= anchor_coherence < 1")
    if output_dim < num_classes + 1:
        raise ValueError(
            "coherence anchors require output_dim >= num_classes + 1"
        )
    basis = _orthonormal_rows(num_classes + 1, output_dim, generator)
    rho = float(coherence)
    anchors = (1.0 - rho) ** 0.5 * basis[:num_classes]
    anchors = anchors + rho**0.5 * basis[num_classes].unsqueeze(0)
    return F.normalize(anchors, dim=1)


def _project_centroids_to_embedding(
    centroids: torch.Tensor, output_dim: int, anchor_seed: int = 0
) -> torch.Tensor:
    input_dim = centroids.shape[1]
    if input_dim == output_dim:
        return F.normalize(centroids, dim=1)
    generator = torch.Generator().manual_seed(int(anchor_seed))
    projection = torch.randn(input_dim, output_dim, generator=generator)
    if input_dim >= output_dim:
        q, _ = torch.linalg.qr(projection)
        projection = q[:, :output_dim]
    else:
        q, _ = torch.linalg.qr(projection.T)
        projection = q[:, :input_dim].T
    return F.normalize(centroids @ projection, dim=1)


class ClassAnchorEncoder(nn.Module):
    def __init__(
        self,
        mode: str,
        output_dim: int,
        num_classes: int,
        class_to_idx: dict[str, int],
        anchor_seed: int = 0,
        class_centroids=None,
        anchor_geometry: str | None = None,
        anchor_coherence: float | None = None,
    ) -> None:
        super().__init__()
        if mode not in CLASS_ANCHOR_MODES:
            raise ValueError(f"Unknown class anchor mode: {mode}")
        self.mode = mode
        self.output_dim = output_dim
        self.num_classes = num_classes
        self.class_to_idx = dict(class_to_idx)
        if anchor_geometry is not None:
            if mode not in {"onehot", "random"}:
                raise ValueError(
                    "anchor_geometry is supported only for fixed onehot/random anchors"
                )
            anchors = _make_controlled_anchors(
                anchor_geometry,
                num_classes,
                output_dim,
                anchor_seed,
                anchor_coherence,
            )
            self.register_buffer("anchors", anchors)
        elif mode == "random":
            generator = torch.Generator().manual_seed(int(anchor_seed))
            anchors = torch.randn(num_classes, output_dim, generator=generator)
            self.register_buffer("anchors", F.normalize(anchors, dim=1))
        elif mode == "onehot":
            if output_dim != num_classes:
                raise ValueError(
                    "onehot anchors require output_dim == num_classes "
                    f"(got {output_dim} != {num_classes})"
                )
            self.register_buffer("anchors", torch.eye(num_classes))
        elif mode == "centroid":
            if class_centroids is None:
                raise ValueError("centroid anchors require class_centroids")
            centroids = torch.as_tensor(class_centroids, dtype=torch.float32)
            if centroids.ndim != 2 or centroids.shape[0] != num_classes:
                raise ValueError(
                    "class_centroids must have shape [num_classes, input_dim]"
                )
            anchors = _project_centroids_to_embedding(
                centroids, output_dim, anchor_seed
            )
            self.register_buffer("anchors", anchors)
        else:
            self.embedding = nn.Embedding(num_classes, output_dim)
            nn.init.normal_(self.embedding.weight, std=0.02)
            self.norm = nn.LayerNorm(output_dim)

    def _indices(self, values: list[str], device: torch.device) -> torch.Tensor:
        try:
            indices = [self.class_to_idx[value] for value in values]
        except KeyError as exc:
            raise KeyError(f"Unknown class anchor {exc}") from exc
        return torch.as_tensor(indices, dtype=torch.long, device=device)

    def forward(
        self, values: list[str], device: Optional[torch.device] = None
    ) -> torch.Tensor:
        if device is None:
            device = (
                self.embedding.weight.device
                if self.mode == "trainable"
                else self.anchors.device
            )
        if not values:
            return torch.empty((0, self.output_dim), device=device)
        indices = self._indices(values, device)
        if self.mode == "trainable":
            return self.norm(self.embedding(indices))
        return self.anchors.index_select(0, indices)


class MIModule(nn.Module):
    def __init__(
        self,
        expression_input_dim: int,
        expression_hidden_dim: int,
        embedding_dim: int,
        dropout: float = 0.1,
        semantic_mode: str = "onehot",
        num_classes: int | None = None,
        class_to_idx: dict[str, int] | None = None,
        anchor_seed: int = 0,
        class_centroids=None,
        anchor_geometry: str | None = None,
        anchor_coherence: float | None = None,
    ) -> None:
        super().__init__()
        if semantic_mode not in SEMANTIC_MODES:
            raise ValueError(
                f"Unknown semantic_mode {semantic_mode!r}. Options: {SEMANTIC_MODES}"
            )
        if semantic_mode in CLASS_ANCHOR_MODES and (
            num_classes is None or class_to_idx is None
        ):
            raise ValueError("Class-anchor modes require a class vocabulary")
        effective_dim = (
            int(num_classes)
            if semantic_mode == "onehot" and anchor_geometry is None
            else embedding_dim
        )
        self.semantic_mode = semantic_mode
        self.embedding_dim = effective_dim
        self.expression_encoder = ExpressionEncoder(
            expression_input_dim, expression_hidden_dim, effective_dim, dropout
        )
        self.anchor_encoder = None
        if semantic_mode in CLASS_ANCHOR_MODES:
            self.anchor_encoder = ClassAnchorEncoder(
                semantic_mode,
                effective_dim,
                int(num_classes),
                dict(class_to_idx),
                anchor_seed,
                class_centroids,
                anchor_geometry,
                anchor_coherence,
            )

    @property
    def has_text_branch(self) -> bool:
        return self.anchor_encoder is not None

    def forward(
        self,
        expression_vectors: torch.Tensor,
        text_anchors: list[str],
        device: Optional[torch.device] = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        expression = F.normalize(self.expression_encoder(expression_vectors), dim=1)
        if self.anchor_encoder is None:
            return expression, None
        anchors = F.normalize(self.anchor_encoder(text_anchors, device), dim=1)
        return expression, anchors

    def encode_expression(self, values: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.expression_encoder(values), dim=1)

    def encode_text(
        self, values: list[str], device: Optional[torch.device] = None
    ) -> torch.Tensor:
        if self.anchor_encoder is None:
            raise RuntimeError("semantic_mode='none' has no anchor branch")
        return F.normalize(self.anchor_encoder(values, device), dim=1)
