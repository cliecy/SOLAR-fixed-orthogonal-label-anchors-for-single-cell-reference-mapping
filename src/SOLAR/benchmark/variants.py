from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class VariantSpec:
    """One named SOLAR or objective-control configuration."""

    name: str
    group: str
    objective: str
    evidence_scope: str
    description: str
    model_kwargs: dict[str, Any]

    def resolved_model_kwargs(self, anchor_seed: int = 0) -> dict[str, Any]:
        kwargs = copy.deepcopy(self.model_kwargs)
        if self.objective == "solar_supcon":
            kwargs["anchor_seed"] = int(anchor_seed)
        return kwargs


def _solar(
    name: str,
    group: str,
    evidence_scope: str,
    description: str,
    **model_kwargs: Any,
) -> VariantSpec:
    return VariantSpec(
        name=name,
        group=group,
        objective="solar_supcon",
        evidence_scope=evidence_scope,
        description=description,
        model_kwargs=model_kwargs,
    )


def _control(name: str, description: str) -> VariantSpec:
    return VariantSpec(
        name=name,
        group="objective_control",
        objective=name,
        evidence_scope="reviewer_objective_control_60_job_matrix",
        description=description,
        model_kwargs={"hidden_dim": 512, "embedding_dim": 128},
    )


_SPECS = [
    # Phase 6 label-anchor family.
    _solar(
        "SOLAR-onehot-rep-natural",
        "phase6",
        "phase6_238_run_summary",
        "Literal K-dimensional one-hot anchors, repeated in the SupCon pool.",
        semantic_mode="onehot",
        anchor_dedup=False,
        imbalance_mode="natural",
    ),
    _solar(
        "SOLAR-random-rep-natural",
        "phase6",
        "phase6_238_run_summary",
        "Fixed Gaussian class anchors, repeated in the SupCon pool.",
        semantic_mode="random",
        anchor_dedup=False,
        imbalance_mode="natural",
    ),
    _solar(
        "SOLAR-trainable-rep-natural",
        "phase6",
        "phase6_238_run_summary",
        "Trainable class embeddings, repeated in the SupCon pool.",
        semantic_mode="trainable",
        anchor_dedup=False,
        imbalance_mode="natural",
    ),
    _solar(
        "SOLAR-centroid-rep-natural",
        "phase6",
        "phase6_238_run_summary",
        "Fixed class-centroid anchors, repeated in the SupCon pool.",
        semantic_mode="centroid",
        anchor_dedup=False,
        imbalance_mode="natural",
    ),
    # Reviewer main and mechanism family.
    _solar(
        "solar_orthogonal",
        "reviewer_main",
        "reviewer_fairness_and_lobo_matrix",
        "Controlled orthogonal anchors at the configured embedding dimension "
        "(reviewer main).",
        semantic_mode="onehot",
        anchor_geometry="orthogonal",
        anchor_dedup=False,
        imbalance_mode="natural",
    ),
    _solar(
        "solar_none",
        "reviewer_main",
        "reviewer_fairness_and_lobo_matrix",
        "Expression-only supervised contrastive learning without an anchor branch.",
        semantic_mode="none",
        embedding_dim=128,
        anchor_dedup=False,
        imbalance_mode="natural",
    ),
    _solar(
        "solar_trainable",
        "reviewer_mechanism",
        "reviewer_mechanism_333_job_matrix",
        "Trainable 128-dimensional class anchors.",
        semantic_mode="trainable",
        embedding_dim=128,
        anchor_dedup=False,
        imbalance_mode="natural",
    ),
    _solar(
        "solar_shuffled",
        "reviewer_mechanism",
        "reviewer_mechanism_333_job_matrix",
        "Shuffled-label negative control; destroys expression-label correspondence.",
        semantic_mode="onehot",
        embedding_dim=128,
        anchor_geometry="orthogonal",
        shuffle_labels=True,
        anchor_dedup=False,
        imbalance_mode="natural",
    ),
]

for _geometry, _coherence in (
    ("orthogonal", None),
    ("simplex", None),
    ("gaussian", None),
    ("coherence", 0.25),
    ("coherence", 0.50),
    ("coherence", 0.75),
    ("coherence", 0.90),
):
    _suffix = _geometry if _coherence is None else f"{_geometry}_{_coherence:.2f}"
    _SPECS.append(
        _solar(
            f"solar_geometry_{_suffix}",
            "reviewer_mechanism",
            "reviewer_mechanism_333_job_matrix",
            f"Controlled 128-dimensional {_suffix} anchor geometry.",
            semantic_mode="onehot",
            embedding_dim=128,
            anchor_geometry=_geometry,
            anchor_coherence=_coherence,
            anchor_dedup=False,
            imbalance_mode="natural",
        )
    )

_SPECS.extend(
    [
        _control(
            "cross_entropy",
            "Normalized SOLAR expression encoder with a trainable linear CE head.",
        ),
        _control(
            "fixed_anchor_cosine",
            "Normalized encoder trained against fixed orthogonal cosine prototypes.",
        ),
    ]
)

VARIANTS = {spec.name: spec for spec in _SPECS}


def get_variant(name: str) -> VariantSpec:
    try:
        return VARIANTS[name]
    except KeyError as exc:
        raise KeyError(
            f"Unknown SOLAR benchmark variant {name!r}. "
            f"Available: {', '.join(sorted(VARIANTS))}"
        ) from exc


def list_variants(group: str | None = None) -> list[VariantSpec]:
    variants = VARIANTS.values()
    if group is not None:
        variants = (spec for spec in variants if spec.group == group)
    return sorted(variants, key=lambda spec: (spec.group, spec.name))
