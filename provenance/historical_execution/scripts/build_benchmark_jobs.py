#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from scib_benchmark.config import load_benchmark_config
from scib_benchmark.jobs import build_baseline_jobs, build_jobs, write_job_matrix

SOLAR_TRACKS = ("core", "budget", "geometry", "heldout")
BASELINE_TRACKS = ("transductive_baselines", "inductive_baselines")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/benchmark_v1.yaml"))
    parser.add_argument("--run-id", required=True)
    parser.add_argument(
        "--tracks",
        nargs="+",
        default=["core"],
        choices=[*SOLAR_TRACKS, *BASELINE_TRACKS],
    )
    parser.add_argument("--datasets", nargs="*")
    parser.add_argument("--methods", nargs="*")
    parser.add_argument("--preprocessing", nargs="*")
    parser.add_argument("--seeds", nargs="*", type=int)
    parser.add_argument("--fit-scopes", nargs="*")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Mark outputs smoke-only; shorten scVI/scANVI epochs. Use a dedicated run ID for SOLAR.",
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    config = load_benchmark_config(args.config)
    solar_tracks = [name for name in args.tracks if name in SOLAR_TRACKS]
    baseline_tracks = [name for name in args.tracks if name in BASELINE_TRACKS]
    if solar_tracks and baseline_tracks:
        raise ValueError(
            "Build SOLAR and baseline tracks separately so existing matrices are not archived together"
        )
    if solar_tracks:
        jobs = build_jobs(
            config,
            args.run_id,
            solar_tracks,
            dataset_ids=args.datasets,
            methods=args.methods,
            preprocessing_profiles=args.preprocessing,
            seeds=args.seeds,
        )
        if args.smoke:
            for job in jobs:
                job["execution_profile"] = "smoke_only_not_a_benchmark_result"
                job["train"]["max_epochs"] = 1
                job["smoke_cells_per_role_batch_label"] = 32
        matrix = write_job_matrix(
            jobs,
            config.output_root / args.run_id,
            force=args.force,
            matrix_name="job_matrix.tsv",
            jobs_dirname="jobs",
        )
    else:
        jobs = build_baseline_jobs(
            config,
            args.run_id,
            baseline_tracks,
            dataset_ids=args.datasets,
            methods=args.methods,
            preprocessing_profiles=args.preprocessing,
            seeds=args.seeds,
            fit_scopes=args.fit_scopes,
        )
        if args.smoke:
            for job in jobs:
                job["execution_profile"] = "smoke_only_not_a_benchmark_result"
                defaults = dict(job.get("baseline_defaults") or {})
                for key in ("scvi", "scanvi"):
                    section = dict(defaults.get(key) or {})
                    section["max_epochs"] = 1
                    defaults[key] = section
                job["baseline_defaults"] = defaults
        matrix = write_job_matrix(
            jobs,
            config.output_root / args.run_id,
            force=args.force,
            matrix_name="baseline_job_matrix.tsv",
            jobs_dirname="baseline_jobs",
        )
    print(f"{matrix}: {len(jobs)} jobs (indices 0..{len(jobs) - 1})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
