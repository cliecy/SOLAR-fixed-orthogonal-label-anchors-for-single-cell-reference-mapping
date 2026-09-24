"""Behavioural checks of the Methods description against the SOLAR source.

These complement the static checks in scripts/paper_static_checks.py by
executing the code: encoder shape, anchor geometry, pool sizes, the objective
of Eqs. (4)-(5), and query mapping without parameter updates.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
import torch.nn.functional as F  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from SOLAR._training import assemble_contrastive_features  # noqa: E402
from SOLAR.src.anchor_model import MIModule, _make_controlled_anchors  # noqa: E402
from SOLAR.src.loss import SupConLoss  # noqa: E402


def test_encoder_layers_and_output_width():
    for dim in (30, 128):
        model = MIModule(40, 512, dim, dropout=0.1, semantic_mode="onehot", num_classes=17,
                         class_to_idx={str(k): k for k in range(17)}, anchor_geometry="orthogonal")
        layers = [type(m).__name__ for m in model.expression_encoder.net]
        assert layers == ["Linear", "BatchNorm1d", "ReLU", "Dropout"] * 2 + ["Linear", "LayerNorm"]
        linears = [m for m in model.expression_encoder.net if isinstance(m, torch.nn.Linear)]
        assert [(l.in_features, l.out_features) for l in linears] == [(40, 512), (512, 512), (512, dim)]
        model.eval()
        z = model.encode_expression(torch.randn(8, 40))
        assert z.shape == (8, dim)
        torch.testing.assert_close(z.norm(dim=1), torch.ones(8))


def test_anchors_are_orthonormal_fixed_and_require_d_ge_k():
    anchors = _make_controlled_anchors("orthogonal", 16, 30, anchor_seed=0)
    torch.testing.assert_close(anchors @ anchors.T, torch.eye(16), atol=1e-5, rtol=0)
    model = MIModule(4, 8, 30, semantic_mode="onehot", num_classes=16,
                     class_to_idx={str(k): k for k in range(16)}, anchor_geometry="orthogonal")
    assert "anchor_encoder.anchors" not in dict(model.named_parameters())
    assert "anchor_encoder.anchors" in dict(model.named_buffers())
    with pytest.raises(ValueError):
        _make_controlled_anchors("orthogonal", 31, 30, anchor_seed=0)


@pytest.mark.parametrize("mode,expected", [("repeated", lambda b, k: 2 * b), ("unique", lambda b, k: b + k),
                                           ("none", lambda b, k: b)])
def test_view_pool_sizes(mode, expected):
    labels = ["a", "b", "a", "c", "b", "a"]
    z = F.normalize(torch.randn(len(labels), 8), dim=1)
    anchors = None if mode == "none" else F.normalize(torch.randn(len(labels), 8), dim=1)
    features, pooled = assemble_contrastive_features(z, anchors, labels, labels, anchor_dedup=mode == "unique")
    assert features.shape[0] * features.shape[1] == expected(len(labels), len(set(labels)))
    assert len(pooled) == features.shape[0]


def test_objective_matches_equations_4_and_5():
    torch.manual_seed(0)
    tau, m = 0.07, 0.2
    labels = ["a", "b", "a", "c"]           # 'c' has no expression positive without anchors
    z = F.normalize(torch.randn(4, 6, dtype=torch.float64), dim=1)
    anchors = F.normalize(torch.randn(3, 6, dtype=torch.float64), dim=1)
    u = torch.stack([anchors["abc".index(y)] for y in labels])
    for mode in ("repeated", "none"):
        features, pooled = assemble_contrastive_features(z, None if mode == "none" else u, labels, labels)
        views = torch.cat(torch.unbind(features, dim=1))
        names = pooled * features.shape[1]
        terms = []
        for a in range(len(names)):
            others = [b for b in range(len(names)) if b != a]
            pos = [b for b in others if names[b] == names[a]]
            if not pos:
                terms.append(torch.tensor(0.0, dtype=torch.float64))
                continue
            logit = {b: (views[a] @ views[b] - m * (names[b] == names[a])) / tau for b in others}
            log_den = torch.logsumexp(torch.stack(list(logit.values())), 0)
            terms.append(-torch.stack([logit[j] - log_den for j in pos]).mean())
        expected = torch.stack(terms).mean()
        observed = SupConLoss(temperature=tau, base_temperature=tau, margin=m)(features, labels=pooled)
        torch.testing.assert_close(observed, expected)
