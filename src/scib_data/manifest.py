from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


DECISION_FIELDS = ("batch_key", "label_key", "counts_location", "normalized_location")


def load_manifest(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict) or not isinstance(data.get("datasets"), dict):
        raise ValueError("Manifest must contain a 'datasets' mapping")
    for dataset_id, item in data["datasets"].items():
        if not isinstance(item, dict) or not item.get("display_name"):
            raise ValueError(f"Invalid dataset entry: {dataset_id}")
        for field in DECISION_FIELDS:
            if field not in item:
                raise ValueError(f"Dataset {dataset_id!r} lacks decision field {field!r}")
    return data


def save_manifest(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(data, handle, sort_keys=False, allow_unicode=True)


def resolve_local_path(manifest_path: Path, value: str | None) -> Path | None:
    if value is None:
        return None
    path = Path(value)
    if path.is_absolute():
        return path
    # Config lives in configs/; repository-relative paths resolve from its parent.
    return (manifest_path.parent.parent / path).resolve()

