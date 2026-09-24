from __future__ import annotations

import argparse
import json
from pathlib import Path

import anndata as ad

from .adapter import PreprocessConfig, TrainConfig, run_inductive
from .splits import batch_holdout_split, lobo_split
from .variants import list_variants


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a portable SOLAR variant on an inductive reference/query split."
    )
    parser.add_argument("--list-variants", action="store_true")
    parser.add_argument("--input", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--variant", default="solar_orthogonal")
    parser.add_argument("--labels-key", default="cell_type")
    parser.add_argument("--batch-key", default=None)
    parser.add_argument("--held-out-batch", default=None)
    parser.add_argument(
        "--validate-closed-set",
        action="store_true",
        help="Use query labels only during split construction to reject unseen query types.",
    )
    parser.add_argument("--split-key", default=None)
    parser.add_argument("--reference-value", default="reference")
    parser.add_argument("--query-value", default="query")
    parser.add_argument("--label-fraction", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--anchor-seed", type=int, default=0)
    parser.add_argument("--source", choices=["X", "layer", "obsm"], default="X")
    parser.add_argument("--source-key", default=None)
    parser.add_argument("--n-pcs", type=int, default=40)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--embedding-dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--max-epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--margin", type=float, default=0.2)
    parser.add_argument("--train-size", type=float, default=0.9)
    parser.add_argument("--early-stopping-patience", type=int, default=10)
    parser.add_argument("--early-stopping-min-delta", type=float, default=1e-4)
    parser.add_argument("--gradient-clip-norm", type=float, default=1.0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default=None)
    parser.add_argument("--save-model", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    if args.list_variants:
        payload = [
            {
                "name": spec.name,
                "group": spec.group,
                "objective": spec.objective,
                "evidence_scope": spec.evidence_scope,
                "description": spec.description,
            }
            for spec in list_variants()
        ]
        print(json.dumps(payload, indent=2))
        return
    if args.input is None or args.output is None:
        raise SystemExit("--input and --output are required unless --list-variants is used")

    data = ad.read_h5ad(args.input)
    if args.held_out_batch is not None:
        batch_key = args.batch_key or "batch"
        if args.validate_closed_set:
            reference_idx, query_idx = lobo_split(
                data, args.held_out_batch, args.labels_key, batch_key
            )
        else:
            reference_idx, query_idx = batch_holdout_split(
                data, args.held_out_batch, batch_key
            )
    else:
        if args.split_key is None or args.split_key not in data.obs:
            raise SystemExit(
                "Provide --held-out-batch or an obs column via --split-key"
            )
        split = data.obs[args.split_key].astype(str)
        reference_idx = (split == args.reference_value).to_numpy().nonzero()[0]
        query_idx = (split == args.query_value).to_numpy().nonzero()[0]
        batch_key = args.batch_key
    reference = data[reference_idx].copy()
    query = data[query_idx].copy()
    result = run_inductive(
        reference,
        query,
        variant=args.variant,
        labels_key=args.labels_key,
        batch_key=batch_key,
        label_fraction=args.label_fraction,
        seed=args.seed,
        anchor_seed=args.anchor_seed,
        preprocess=PreprocessConfig(
            n_components=args.n_pcs,
            source=args.source,
            source_key=args.source_key,
        ),
        train=TrainConfig(
            hidden_dim=args.hidden_dim,
            embedding_dim=args.embedding_dim,
            dropout=args.dropout,
            max_epochs=args.max_epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            temperature=args.temperature,
            margin=args.margin,
            train_size=args.train_size,
            early_stopping_patience=args.early_stopping_patience,
            early_stopping_min_delta=args.early_stopping_min_delta,
            gradient_clip_norm=args.gradient_clip_norm,
            num_workers=args.num_workers,
        ),
        device=args.device,
    )
    result.metadata["input"] = str(args.input)
    result.save(args.output, save_model=args.save_model)
    print(json.dumps(result.metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
