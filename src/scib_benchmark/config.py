from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from scib_data.manifest import load_manifest, resolve_local_path


REQUIRED_TOP_LEVEL = {
    "schema_version",
    "dataset_manifest",
    "output_root",
    "datasets",
    "protocol",
    "preprocessing",
    "solar",
    "tracks",
    "evaluation",
}


@dataclass(frozen=True)
class BenchmarkConfig:
    path: Path
    repo_root: Path
    data: dict[str, Any]
    dataset_manifest_path: Path
    dataset_manifest: dict[str, Any]

    def resolve(self, value: str | Path) -> Path:
        path = Path(value)
        # Do not call Path.resolve(): venv/bin/python is normally a symlink and
        # dereferencing it would bypass the virtual environment at execution.
        return path if path.is_absolute() else (self.repo_root / path).absolute()

    @property
    def output_root(self) -> Path:
        return self.resolve(self.data["output_root"])

    def dataset(self, dataset_id: str) -> dict[str, Any]:
        if dataset_id not in self.data["datasets"]:
            raise KeyError(f"Dataset {dataset_id!r} is not enabled in benchmark config")
        return self.dataset_manifest["datasets"][dataset_id]

    def dataset_path(self, dataset_id: str) -> Path:
        item = self.dataset(dataset_id)
        path = resolve_local_path(self.dataset_manifest_path, item.get("local_path"))
        if path is None:
            raise ValueError(f"Dataset {dataset_id!r} has no local_path")
        return path


def load_benchmark_config(path: str | Path) -> BenchmarkConfig:
    path = Path(path).resolve()
    with path.open(encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError("Benchmark config must be a mapping")
    missing = REQUIRED_TOP_LEVEL - set(data)
    if missing:
        raise ValueError(f"Benchmark config lacks fields: {', '.join(sorted(missing))}")
    if data["schema_version"] != 1:
        raise ValueError(f"Unsupported benchmark schema_version: {data['schema_version']}")
    if not isinstance(data["datasets"], list) or not data["datasets"]:
        raise ValueError("Benchmark config datasets must be a non-empty list")
    if len(data["datasets"]) != len(set(data["datasets"])):
        raise ValueError("Benchmark config contains duplicate dataset IDs")
    repo_root = path.parent.parent
    manifest_path = Path(data["dataset_manifest"])
    if not manifest_path.is_absolute():
        manifest_path = (repo_root / manifest_path).resolve()
    manifest = load_manifest(manifest_path)
    unknown = sorted(set(data["datasets"]) - set(manifest["datasets"]))
    if unknown:
        raise ValueError(f"Unknown datasets in benchmark config: {', '.join(unknown)}")
    for dataset_id in data["datasets"]:
        item = manifest["datasets"][dataset_id]
        if item.get("benchmark_inclusion") != "included":
            raise ValueError(f"Dataset {dataset_id!r} is not recorded as included")
        if not item.get("batch_key") or not item.get("label_key"):
            raise ValueError(f"Dataset {dataset_id!r} lacks selected batch/label keys")
        if not item.get("checksum_sha256"):
            raise ValueError(f"Dataset {dataset_id!r} lacks a SHA-256 checksum")
    profiles = data["preprocessing"]
    if not isinstance(profiles, dict) or not profiles:
        raise ValueError("preprocessing must be a non-empty mapping")
    for track, spec in data["tracks"].items():
        if not isinstance(spec, dict) or not spec.get("methods"):
            raise ValueError(f"Track {track!r} must define methods")
        missing_profiles = set(spec.get("preprocessing", [])) - set(profiles)
        if missing_profiles:
            raise ValueError(
                f"Track {track!r} references unknown preprocessing profiles: "
                f"{', '.join(sorted(missing_profiles))}"
            )
        if track == "transductive_baselines" or spec.get("protocol") == "transductive_scib":
            scopes = spec.get("fit_scopes") or []
            if not scopes:
                raise ValueError(f"Track {track!r} must define fit_scopes")
            unknown_scopes = set(scopes) - {"classic_full", "reference_only"}
            if unknown_scopes:
                raise ValueError(
                    f"Track {track!r} has unknown fit_scopes: {', '.join(sorted(unknown_scopes))}"
                )
    if data["protocol"].get("generate_cross_protocol_ranking") is not False:
        raise ValueError("Cross-protocol aggregate ranking must remain disabled")
    return BenchmarkConfig(path, repo_root, data, manifest_path, manifest)
