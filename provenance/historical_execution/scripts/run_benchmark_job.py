#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess

from scib_benchmark.config import load_benchmark_config


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/benchmark_v1.yaml"))
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--job-index", type=int, required=True)
    parser.add_argument("--solar-python", type=Path)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    config = load_benchmark_config(args.config)
    spec = config.output_root / args.run_id / "jobs" / f"{args.job_index:06d}.json"
    if not spec.is_file():
        raise FileNotFoundError(spec)
    payload = json.loads(spec.read_text(encoding="utf-8"))
    if payload["engine"] != "solar":
        raise NotImplementedError(f"Unsupported job engine: {payload['engine']}")
    solar_python = args.solar_python or Path(payload["solar_python"])
    if not solar_python.is_file():
        raise FileNotFoundError(
            f"SOLAR Python is missing: {solar_python}. Run: "
            f"uv sync --frozen --extra dev --project {payload['solar_package_root']}"
        )
    command = [
        str(solar_python),
        "-m",
        "scib_benchmark.solar_worker",
        "--job-spec",
        str(spec),
    ]
    if args.force:
        command.append("--force")
    environment = os.environ.copy()
    source = str(config.repo_root / "src")
    environment["PYTHONPATH"] = source + (
        os.pathsep + environment["PYTHONPATH"] if environment.get("PYTHONPATH") else ""
    )
    subprocess.run(command, check=True, cwd=config.repo_root, env=environment)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

