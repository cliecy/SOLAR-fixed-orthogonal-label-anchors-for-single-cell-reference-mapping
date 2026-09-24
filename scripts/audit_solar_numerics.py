"""Numerical and update contracts for the historical all-row SupCon mean."""
import unittest
from pathlib import Path
import sys

import torch
import torch.nn.functional as F

from SOLAR._training import assemble_contrastive_features, run_epochs
from SOLAR.src.loss import SupConLoss
from SOLAR.src.anchor_model import MIModule


class TinyEncoder(torch.nn.Module):
    has_text_branch = False

    def __init__(self):
        super().__init__()
        self.encoder = torch.nn.Sequential(torch.nn.Linear(3, 3), torch.nn.BatchNorm1d(3))

    def forward(self, expressions, text_anchors, device):
        return self.encoder(expressions), None


class ContrastiveContract(unittest.TestCase):
    def test_fixed_anchors_survive_real_encoder_update(self):
        torch.manual_seed(40)
        model = MIModule(3, 8, 5, semantic_mode="onehot", num_classes=2,
                         class_to_idx={"a": 0, "b": 1}, anchor_geometry="orthogonal")
        labels = ["a", "a", "b", "b"]
        before = model.encode_text(["a", "b"]).detach().clone()
        initial = model.expression_encoder.net[0].weight.detach().clone()
        optimizer = torch.optim.AdamW(model.parameters(), lr=.01)
        run_epochs(model, torch.device("cpu"), SupConLoss(), optimizer,
                   [(torch.randn(4, 3), labels, labels)], None,
                   {"train_loss": [], "val_loss": []}, 1, 10, .0001, 1.)
        torch.testing.assert_close(model.encode_text(["a", "b"]), before, rtol=0, atol=0)
        self.assertFalse(torch.equal(model.expression_encoder.net[0].weight, initial))

    def test_identity_exclusion_and_all_row_mean(self):
        torch.manual_seed(40)
        for labels in (["a", "b", "c"], ["a", "a"], ["a", "a", "a", "b"]):
            for mode in ("none", "repeated", "unique"):
                with self.subTest(labels=labels, mode=mode):
                    x = torch.randn(len(labels), 5, dtype=torch.float64, requires_grad=True)
                    anchors = torch.stack([torch.eye(5, dtype=torch.float64)[ord(y)-ord("a")] for y in labels])
                    before = anchors.clone()
                    features, pooled_labels = assemble_contrastive_features(
                        x, None if mode == "none" else anchors, labels, labels,
                        anchor_dedup=mode == "unique",
                    )
                    criterion = SupConLoss(temperature=.7, base_temperature=.7, margin=.2)
                    observed = criterion(features, labels=pooled_labels)
                    flat = torch.cat(torch.unbind(F.normalize(features, dim=2), dim=1))
                    names = pooled_labels * features.shape[1]
                    contributions = []
                    for i in range(len(names)):
                        others = [j for j in range(len(names)) if j != i]
                        positives = [j for j in others if names[j] == names[i]]
                        if not positives:
                            contributions.append(flat[i].sum() * 0)
                            continue
                        logits = torch.stack([(flat[i] @ flat[j] - (.2 if names[j] == names[i] else 0)) / .7 for j in others])
                        contributions.append(torch.logsumexp(logits, 0) - torch.stack([(flat[i] @ flat[j] - .2) / .7 for j in positives]).mean())
                    expected = torch.stack(contributions).mean()
                    torch.testing.assert_close(observed, expected)
                    observed.backward()
                    self.assertTrue(torch.isfinite(x.grad).all().item())
                    torch.testing.assert_close(anchors, before, rtol=0, atol=0)
                    self.assertIsNone(anchors.grad)

    def test_singleton_has_differentiable_zero(self):
        x = torch.tensor([[[1., 2., 3.]]], requires_grad=True)
        loss = SupConLoss()(x, labels=["a"])
        loss.backward()
        self.assertEqual(loss.item(), 0)
        torch.testing.assert_close(x.grad, torch.zeros_like(x))

    def test_empty_batches_do_not_advance_optimizer(self):
        torch.manual_seed(40)
        model = TinyEncoder()
        empty = [(torch.randn(1, 3), ["a"], ["a"]),
                 (torch.randn(2, 3), ["a", "b"], ["a", "b"])]
        valid = [(torch.randn(2, 3), ["a", "a"], ["a", "a"])]
        optimizer = torch.optim.Adam(model.parameters(), lr=.01, weight_decay=.1)
        def execute(train, validation, history):
            run_epochs(model, torch.device("cpu"), SupConLoss(), optimizer,
                       train, validation, history, 1, 10, .0001, 1.)
        execute(valid, None, {"train_loss": [], "val_loss": []})
        before = {k: v.clone() for k, v in model.state_dict().items()}
        steps = [state["step"].clone() for state in optimizer.state.values()]
        history = {"train_loss": [], "val_loss": []}
        execute(empty, None, history)
        for k, v in model.state_dict().items():
            torch.testing.assert_close(v, before[k], rtol=0, atol=0)
        for state, step in zip(optimizer.state.values(), steps):
            torch.testing.assert_close(state["step"], step, rtol=0, atol=0)
        self.assertEqual(history["skipped_train_batches"], [2])
        with self.assertRaisesRegex(ValueError, "no valid positive"):
            execute(empty, empty, {"train_loss": [], "val_loss": []})
        history = {"train_loss": [], "val_loss": []}
        execute(empty, empty + valid, history)
        self.assertEqual(history["skipped_validation_batches"], [2])
        self.assertTrue(torch.isfinite(torch.tensor(history["val_loss"])).all().item())


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()
