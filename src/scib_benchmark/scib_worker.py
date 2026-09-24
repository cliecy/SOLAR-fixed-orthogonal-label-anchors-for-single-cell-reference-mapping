from __future__ import annotations

import argparse
import json
from pathlib import Path

from .artifacts import write_json
from .config import load_benchmark_config
from .scib_scoring import (
    TASK_METRICS,
    compute_metric_task,
    prepare_neighbor_graph,
    scoring_context,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Isolated official scIB metric worker")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--job-spec", type=Path, required=True)
    parser.add_argument("--task", choices=("neighbors", *TASK_METRICS), required=True)
    parser.add_argument("--graph-dir", type=Path, required=True)
    parser.add_argument("--source-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    spec = json.loads(args.job_spec.read_text(encoding="utf-8"))
    config = load_benchmark_config(args.config)
    context = scoring_context(config.data, spec["dataset_id"])
    if spec.get("scib_output_type"):
        context = dict(context)
        context["output_type"] = str(spec["scib_output_type"])
    payload = {
        "schema_version": 1,
        "task": args.task,
        "job_id": spec["job_id"],
        "job_index": spec["job_index"],
        "state": "running",
        "metrics": {},
        "error": None,
    }
    write_json(args.output, payload)
    try:
        if args.task == "neighbors":
            payload["graph"] = prepare_neighbor_graph(
                spec,
                args.graph_dir,
                int(context["neighbors"]),
                backend=str(context["backend"]),
            )
        else:
            payload["metrics"] = compute_metric_task(
                task=args.task,
                spec=spec,
                context=context,
                graph_dir=args.graph_dir,
                cache_dir=args.source_cache,
            )
        payload["state"] = "completed"
    except Exception as exc:
        payload["state"] = "failed"
        payload["error"] = f"{type(exc).__name__}: {exc}"
    write_json(args.output, payload)
    return 0 if payload["state"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
