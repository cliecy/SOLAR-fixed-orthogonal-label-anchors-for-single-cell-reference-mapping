"""Private checkpoint I/O for the SOLAR public facade."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

METADATA_FILENAME = "solar_metadata.json"
WEIGHTS_FILENAME = "solar_model.pt"


def save_checkpoint(
    dir_path: Path,
    state_dict: dict[str, Any],
    metadata: dict[str, Any],
) -> None:
    dir_path.mkdir(parents=True, exist_ok=True)
    torch.save(state_dict, dir_path / WEIGHTS_FILENAME)
    with open(dir_path / METADATA_FILENAME, "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)


def load_metadata(dir_path: Path) -> dict[str, Any]:
    with open(dir_path / METADATA_FILENAME, "r", encoding="utf-8") as handle:
        return json.load(handle)


def load_weights(dir_path: Path, map_location: torch.device) -> dict[str, Any]:
    return torch.load(dir_path / WEIGHTS_FILENAME, map_location=map_location)
