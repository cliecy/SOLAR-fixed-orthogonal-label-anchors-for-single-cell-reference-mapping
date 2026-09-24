from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def _metadata(obs: pd.DataFrame, batch_key: str, label_key: str) -> pd.DataFrame:
    missing = [key for key in (batch_key, label_key) if key not in obs]
    if missing:
        raise KeyError(f"Missing obs columns: {', '.join(missing)}")
    if not obs.index.is_unique:
        raise ValueError("Cell barcodes must be unique")
    frame = pd.DataFrame(
        {
            "barcode": obs.index.astype(str),
            "batch": obs[batch_key].astype("string"),
            "label": obs[label_key].astype("string"),
        }
    )
    if frame[["batch", "label"]].isna().any().any():
        raise ValueError("Selected batch/label metadata contains missing values")
    return frame


def stratified_reference_query(
    obs: pd.DataFrame,
    batch_key: str,
    label_key: str,
    seed: int,
    reference_fraction: float = 0.8,
) -> pd.DataFrame:
    """Deterministic batch-by-label split; labels are used only to construct it."""
    if not 0 < reference_fraction < 1:
        raise ValueError("reference_fraction must lie in (0, 1)")
    frame = _metadata(obs, batch_key, label_key)
    strata = frame["batch"].astype(str) + "||" + frame["label"].astype(str)
    rng = np.random.default_rng(seed)
    roles = np.empty(len(frame), dtype=object)
    singleton = np.zeros(len(frame), dtype=bool)
    for value in sorted(strata.unique()):
        indices = np.flatnonzero((strata == value).to_numpy())
        if len(indices) < 2:
            # A 1-cell batch-label combination cannot be represented in both
            # roles. Keep it in reference so it cannot become query-only
            # biological information, and record the exception explicitly.
            roles[indices] = "reference"
            singleton[indices] = True
            continue
        rng.shuffle(indices)
        n_query = max(1, int(round(len(indices) * (1.0 - reference_fraction))))
        n_query = min(n_query, len(indices) - 1)
        roles[indices[:n_query]] = "query"
        roles[indices[n_query:]] = "reference"
    frame["role"] = roles
    if not np.any(roles == "query"):
        raise ValueError("Stratified split produced no query cells")
    frame["singleton_stratum_assigned_to_reference"] = singleton
    frame["split_seed"] = int(seed)
    frame["split_uses_query_labels"] = True
    return frame


def held_out_batch_split(
    obs: pd.DataFrame,
    batch_key: str,
    label_key: str,
    held_out_batch: str,
) -> pd.DataFrame:
    """Leave one batch out without consulting labels during role assignment."""
    frame = _metadata(obs, batch_key, label_key)
    query = frame["batch"].astype(str) == str(held_out_batch)
    if not query.any() or query.all():
        raise ValueError(f"Invalid held-out batch {held_out_batch!r}")
    frame["role"] = np.where(query, "query", "reference")
    frame["split_seed"] = pd.NA
    frame["split_uses_query_labels"] = False
    frame["held_out_batch"] = str(held_out_batch)
    return frame


def label_budget(
    split: pd.DataFrame,
    fraction: float,
    seed: int,
) -> pd.DataFrame:
    """Mirror SOLAR's sorted batch-by-label reference-label subsampling."""
    if not 0 < fraction <= 1:
        raise ValueError("label fraction must lie in (0, 1]")
    reference = split.loc[split["role"] == "reference"].copy()
    reference["labeled"] = False
    if fraction == 1:
        reference["labeled"] = True
    else:
        rng = np.random.default_rng(seed)
        strata = reference["batch"].astype(str) + "||" + reference["label"].astype(str)
        selected: list[int] = []
        for value in sorted(strata.unique()):
            positions = np.flatnonzero((strata == value).to_numpy())
            rng.shuffle(positions)
            count = max(1, int(round(len(positions) * fraction)))
            selected.extend(positions[:count].tolist())
        reference.iloc[selected, reference.columns.get_loc("labeled")] = True
    reference["label_fraction"] = float(fraction)
    reference["label_seed"] = int(seed)
    return reference[["barcode", "batch", "label", "labeled", "label_fraction", "label_seed"]]


def smoke_subset(split: pd.DataFrame, per_role_stratum: int, seed: int) -> pd.DataFrame:
    """Deterministically cap each role × batch × label group for smoke only."""
    if per_role_stratum <= 0:
        raise ValueError("per_role_stratum must be positive")
    rng = np.random.default_rng(seed)
    selected: list[int] = []
    strata = (
        split["role"].astype(str)
        + "||"
        + split["batch"].astype(str)
        + "||"
        + split["label"].astype(str)
    )
    for value in sorted(strata.unique()):
        positions = np.flatnonzero((strata == value).to_numpy())
        rng.shuffle(positions)
        selected.extend(positions[:per_role_stratum].tolist())
    return split.iloc[sorted(selected)].copy()


def frame_sha256(frame: pd.DataFrame) -> str:
    payload = frame.to_csv(index=False, lineterminator="\n").encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def write_split_bundle(
    split: pd.DataFrame,
    budgets: dict[float, pd.DataFrame],
    output_dir: Path,
    metadata: dict[str, object],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=False)
    split.to_csv(output_dir / "split.csv", index=False)
    budget_files: dict[str, str] = {}
    for fraction, frame in sorted(budgets.items()):
        name = f"label_budget_{fraction:.2f}.csv"
        frame.to_csv(output_dir / name, index=False)
        budget_files[f"{fraction:.2f}"] = name
    payload = {
        **metadata,
        "n_reference": int((split["role"] == "reference").sum()),
        "n_query": int((split["role"] == "query").sum()),
        "singleton_strata_cells_assigned_to_reference": int(
            split.get(
                "singleton_stratum_assigned_to_reference",
                pd.Series(False, index=split.index),
            ).sum()
        ),
        "split_sha256": frame_sha256(split),
        "label_budget_files": budget_files,
        "label_budget_sha256": {
            f"{fraction:.2f}": frame_sha256(frame)
            for fraction, frame in sorted(budgets.items())
        },
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
