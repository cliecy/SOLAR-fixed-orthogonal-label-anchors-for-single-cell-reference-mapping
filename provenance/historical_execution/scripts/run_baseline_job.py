#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from scib_benchmark.baseline_runners import run_baseline_job
from scib_benchmark.config import load_benchmark_config


def main() -> int:
    parser = argparse.ArgumentParser(description="Run one formal transductive baseline job")
    parser.add_argument("--config", type=Path, default=Path("configs/benchmark_v1.yaml"))
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--job-index", type=int, required=True)
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--require-gpu",
        action="store_true",
        help="Fail if this job does not require a GPU (used by the GPU Slurm array).",
    )
    parser.add_argument(
        "--require-cpu",
        action="store_true",
        help="No-op succeed if this job requires a GPU (used by the CPU Slurm array).",
    )
    args = parser.parse_args()
    if args.require_gpu and args.require_cpu:
        raise ValueError("Choose at most one of --require-gpu / --require-cpu")
    config = load_benchmark_config(args.config)
    spec_path = config.output_root / args.run_id / "baseline_jobs" / f"{args.job_index:06d}.json"
    if not spec_path.is_file():
        raise FileNotFoundError(spec_path)
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    needs_gpu = bool(spec.get("needs_gpu"))
    if args.require_cpu and needs_gpu:
        print(json.dumps({"job_index": args.job_index, "skipped": True, "reason": "gpu_method"}))
        return 0
    if args.require_gpu and not needs_gpu:
        print(json.dumps({"job_index": args.job_index, "skipped": True, "reason": "cpu_method"}))
        return 0
    output = run_baseline_job(spec, repo_root=config.repo_root, force=args.force)
    print(json.dumps({"job_index": args.job_index, "output_dir": str(output), "state": "completed"}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
