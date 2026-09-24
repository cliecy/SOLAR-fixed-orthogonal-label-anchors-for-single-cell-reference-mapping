#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from scib_benchmark.baseline_registry import register_embedding
from scib_benchmark.config import load_benchmark_config


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/benchmark_v1.yaml"))
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--method", required=True)
    parser.add_argument("--embedding", type=Path, required=True)
    parser.add_argument("--barcodes", type=Path, required=True)
    parser.add_argument("--preprocessing", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--supervision", default="none")
    parser.add_argument("--allow-subset-smoke", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    output = register_embedding(
        load_benchmark_config(args.config),
        args.run_id,
        args.dataset,
        args.method,
        args.embedding,
        args.barcodes,
        args.preprocessing,
        args.seed,
        supervision=args.supervision,
        allow_subset_smoke=args.allow_subset_smoke,
        force=args.force,
    )
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

