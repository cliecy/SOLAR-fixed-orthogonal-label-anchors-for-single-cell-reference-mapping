#!/usr/bin/env python3
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re

import anndata as ad

from scib_benchmark.config import load_benchmark_config
from scib_benchmark.data import read_split, select_reference_hvgs
from scib_benchmark.splits import (
    held_out_batch_split,
    label_budget,
    stratified_reference_query,
    write_split_bundle,
)


def _archive(path: Path) -> None:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    candidate = path.with_name(f"{path.name}.archived-{stamp}")
    counter = 1
    while candidate.exists():
        candidate = path.with_name(f"{path.name}.archived-{stamp}-{counter}")
        counter += 1
    path.replace(candidate)


def _slug(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip("-")
    return cleaned[:80] or "empty"


def _prepare_hvg_profiles(config, run_root: Path, dataset_id: str, split_dir: Path, force: bool) -> None:
    split = read_split(split_dir / "split.csv")
    dataset = config.dataset(dataset_id)
    for profile_name, profile in config.data["preprocessing"].items():
        n_hvg = profile.get("n_hvg")
        if n_hvg is None:
            continue
        output_dir = run_root / "preprocessing" / dataset_id / split_dir.name
        output = output_dir / f"{profile_name}.txt"
        if output.exists() and not force:
            continue
        if output.exists():
            _archive(output)
        output_dir.mkdir(parents=True, exist_ok=True)
        before = config.dataset_path(dataset_id).stat()
        genes = select_reference_hvgs(
            config.dataset_path(dataset_id),
            split,
            dataset["batch_key"],
            int(n_hvg),
            str(profile.get("hvg_flavor", "cell_ranger")),
        )
        after = config.dataset_path(dataset_id).stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise RuntimeError(f"Input changed during HVG selection: {dataset_id}")
        output.write_text("\n".join(genes) + "\n", encoding="utf-8")
        metadata = {
            "dataset_id": dataset_id,
            "dataset_sha256": dataset["checksum_sha256"],
            "split_id": split_dir.name,
            "profile": profile_name,
            "reference_only": True,
            "batch_key": dataset["batch_key"],
            "n_genes": len(genes),
            "gene_file": str(output),
        }
        (output.with_suffix(".json")).write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/benchmark_v1.yaml"))
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--datasets", nargs="*")
    parser.add_argument("--include-heldout", action="store_true")
    parser.add_argument("--prepare-hvg", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    config = load_benchmark_config(args.config)
    selected = args.datasets or config.data["datasets"]
    unknown = sorted(set(selected) - set(config.data["datasets"]))
    if unknown:
        raise ValueError(f"Unknown enabled datasets: {', '.join(unknown)}")
    run_root = config.output_root / args.run_id
    run_root.mkdir(parents=True, exist_ok=True)
    snapshot = run_root / "benchmark_config.json"
    if not snapshot.exists():
        snapshot.write_text(
            json.dumps(config.data, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    fractions = sorted(set(float(x) for x in config.data["protocol"]["label_fractions"]))
    for dataset_id in selected:
        item = config.dataset(dataset_id)
        dataset_path = config.dataset_path(dataset_id)
        if not dataset_path.is_file():
            raise FileNotFoundError(dataset_path)
        backed = ad.read_h5ad(dataset_path, backed="r")
        try:
            obs = backed.obs.copy()
        finally:
            backed.file.close()
        for seed in config.data["protocol"]["split_seeds"]:
            split_dir = run_root / "splits" / dataset_id / f"random_seed_{int(seed)}"
            if split_dir.exists():
                if not args.force:
                    if args.prepare_hvg:
                        _prepare_hvg_profiles(config, run_root, dataset_id, split_dir, False)
                    continue
                _archive(split_dir)
            split = stratified_reference_query(
                obs,
                item["batch_key"],
                item["label_key"],
                int(seed),
                float(config.data["protocol"]["reference_fraction"]),
            )
            budgets = {fraction: label_budget(split, fraction, int(seed)) for fraction in fractions}
            write_split_bundle(
                split,
                budgets,
                split_dir,
                {
                    "schema_version": 1,
                    "dataset_id": dataset_id,
                    "dataset_sha256": item["checksum_sha256"],
                    "split_id": split_dir.name,
                    "split_type": "batch_label_stratified_random",
                    "seed": int(seed),
                    "reference_fraction": float(config.data["protocol"]["reference_fraction"]),
                    "batch_key": item["batch_key"],
                    "label_key": item["label_key"],
                    "query_labels_used_for_split_construction": True,
                    "query_labels_visible_to_model": False,
                },
            )
            if args.prepare_hvg:
                _prepare_hvg_profiles(config, run_root, dataset_id, split_dir, args.force)
        if args.include_heldout:
            batches = sorted(obs[item["batch_key"]].astype(str).unique())
            for held_out in batches:
                split_dir = run_root / "splits" / dataset_id / f"heldout_{_slug(held_out)}"
                if split_dir.exists():
                    if not args.force:
                        if args.prepare_hvg:
                            _prepare_hvg_profiles(config, run_root, dataset_id, split_dir, False)
                        continue
                    _archive(split_dir)
                split = held_out_batch_split(
                    obs, item["batch_key"], item["label_key"], held_out
                )
                budgets = {1.0: label_budget(split, 1.0, 0)}
                write_split_bundle(
                    split,
                    budgets,
                    split_dir,
                    {
                        "schema_version": 1,
                        "dataset_id": dataset_id,
                        "dataset_sha256": item["checksum_sha256"],
                        "split_id": split_dir.name,
                        "split_type": "held_out_batch",
                        "held_out_batch": held_out,
                        "batch_key": item["batch_key"],
                        "label_key": item["label_key"],
                        "query_labels_used_for_split_construction": False,
                        "query_labels_visible_to_model": False,
                    },
                )
                if args.prepare_hvg:
                    _prepare_hvg_profiles(config, run_root, dataset_id, split_dir, args.force)
    print(run_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

