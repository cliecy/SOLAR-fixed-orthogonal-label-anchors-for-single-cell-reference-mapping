"""Recompute every number reported in the SOLAR manuscript from per-run records.

The functions here implement the aggregation stated in Methods ("Aggregation")
and Additional file 1, Section 2.4:

* a stochastic method contributes a batch mean only when all five fitting
  seeds (40-44) are finite and comparison-eligible; Matched PCA needs its one
  deterministic fit;
* batch means are averaged with equal weight within a dataset, and datasets
  with equal weight;
* the reported SD is the sample SD (ddof=1) across contributing batch means;
* paired contrasts use batches complete for both representations; missing
  values are never replaced with zero.

Nothing in this module rounds. Rounding to the printed precision happens only
at comparison time in ``verify_paper_numbers.py``.
"""
from __future__ import annotations

import gzip
import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
DATASETS = ("immune_cell_human", "lung_atlas", "pancreas")
SEEDS = (40, 41, 42, 43, 44)
DETERMINISTIC = {"unintegrated_pca_matched"}

# Reasons that place a (dataset, metric, population) outside the endpoint's
# stated scope. Everything else that is not computed is an in-scope missing
# value and still counts in the applicable denominators.
OUT_OF_SCOPE = ("is defined only for", "applies only to Immune",
                "restricts cached-source trustworthiness", "has no obs['dpt_pseudotime'] column")


@lru_cache(maxsize=None)
def metrics() -> pd.DataFrame:
    frame = pd.read_csv(DATA / "heldout" / "metrics.csv", dtype={"heldout_batch": str})
    frame["in_scope"] = ~frame["reason"].fillna("").str.contains("|".join(map(re_escape, OUT_OF_SCOPE)))
    frame["valid"] = (frame["status"] == "computed") & frame["comparison_eligible"].astype(bool) & np.isfinite(frame["value"])
    return frame


def re_escape(text: str) -> str:
    import re
    return re.escape(text)


@lru_cache(maxsize=None)
def runs() -> pd.DataFrame:
    return pd.read_csv(DATA / "heldout" / "runs.csv", dtype={"heldout_batch": str})


def expected_fits(method: str) -> int:
    return 1 if method in DETERMINISTIC else len(SEEDS)


def _select(method: str, dim: int, metric: str, population: str) -> pd.DataFrame:
    m = metrics()
    return m[(m.method == method) & (m.dimension == int(dim)) & (m.metric == metric) & (m.population == population)]


@lru_cache(maxsize=None)
def batch_table(method: str, dim: int, metric: str, population: str) -> pd.DataFrame:
    """One row per (dataset, batch): fit counts, completeness, mean and seed SD."""
    rows = _select(method, dim, metric, population)
    expected = expected_fits(method)
    out = []
    for (dataset, batch), group in rows.groupby(["dataset_id", "heldout_batch"], sort=True):
        valid = group[group.valid]
        complete = len(valid) == expected and len(group) == expected
        out.append({
            "dataset_id": dataset, "heldout_batch": batch,
            "in_scope": bool(group.in_scope.all()), "n_valid": len(valid), "n_expected": expected,
            "complete": complete,
            "mean": float(valid.value.mean()) if complete else np.nan,
            "seed_sd": float(valid.value.std(ddof=1)) if complete and expected > 1 else np.nan,
            "reasons": tuple(sorted(set(group.loc[~group.valid, "reason"].fillna("")))),
        })
    columns = ["dataset_id", "heldout_batch", "in_scope", "n_valid", "n_expected", "complete",
               "mean", "seed_sd", "reasons"]
    return pd.DataFrame(out, columns=columns)


def dataset_weighted_mean(frame: pd.DataFrame, column: str = "mean") -> float:
    frame = frame.dropna(subset=[column])
    if frame.empty:
        return np.nan
    return float(frame.groupby("dataset_id")[column].mean().mean())


@dataclass
class Summary:
    mean: float
    sd: float
    datasets_valid: int
    datasets_applicable: int
    batches_valid: int
    batches_applicable: int
    fits_valid: int
    fits_expected: int
    codes: tuple[str, ...]


def na_codes(metric: str, population: str, table: pd.DataFrame) -> tuple[str, ...]:
    codes = set()
    if metric == "hvg_overlap":
        codes.add("H")
    if not table.in_scope.all() and metric != "trajectory":
        codes.add("S")
    scoped = table[table.in_scope]
    for reasons in scoped.loc[~scoped.complete, "reasons"]:
        text = " ".join(reasons)
        if metric == "trajectory":
            codes.add("T")
        elif "fewer than 16" in text:
            codes.add("P")
        elif "No cell type has at least 10 cells" in text:
            codes.add("C")
    if metric == "trajectory":
        codes.add("T")
    return tuple(sorted(codes))


@lru_cache(maxsize=None)
def summary(method: str, dim: int, metric: str, population: str) -> Summary:
    table = batch_table(method, dim, metric, population)
    scoped = table[table.in_scope]
    complete = scoped[scoped.complete]
    means = complete["mean"]
    return Summary(
        mean=dataset_weighted_mean(complete),
        sd=float(means.std(ddof=1)) if len(means) > 1 else np.nan,
        datasets_valid=complete.dataset_id.nunique(),
        datasets_applicable=scoped.dataset_id.nunique(),
        batches_valid=len(complete),
        batches_applicable=len(scoped),
        fits_valid=int(complete.n_valid.sum()),
        fits_expected=int(scoped.n_expected.sum()),
        codes=na_codes(metric, population, table),
    )


@dataclass
class Contrast:
    left: float
    right: float
    delta: float
    datasets_valid: int
    datasets_applicable: int
    batches_valid: int
    batches_applicable: int
    fits_left: int
    fits_left_expected: int
    fits_right: int
    fits_right_expected: int
    batch_differences: pd.DataFrame


@lru_cache(maxsize=None)
def contrast(left: tuple, right: tuple, metric: str, population: str) -> Contrast:
    a = batch_table(*left, metric, population)
    b = batch_table(*right, metric, population)
    keys = ["dataset_id", "heldout_batch"]
    merged = a.merge(b, on=keys, suffixes=("_l", "_r"))
    scoped = merged[merged.in_scope_l & merged.in_scope_r]
    paired = scoped[scoped.complete_l & scoped.complete_r].copy()
    paired["diff"] = paired["mean_l"] - paired["mean_r"]
    return Contrast(
        left=dataset_weighted_mean(paired, "mean_l"),
        right=dataset_weighted_mean(paired, "mean_r"),
        delta=dataset_weighted_mean(paired, "diff"),
        datasets_valid=paired.dataset_id.nunique(),
        datasets_applicable=scoped.dataset_id.nunique(),
        batches_valid=len(paired),
        batches_applicable=len(scoped),
        fits_left=int(paired.n_valid_l.sum()),
        fits_left_expected=int(scoped.n_expected_l.sum()),
        fits_right=int(paired.n_valid_r.sum()),
        fits_right_expected=int(scoped.n_expected_r.sum()),
        batch_differences=paired[keys + ["mean_l", "mean_r", "diff"]].reset_index(drop=True),
    )


def dataset_level(method: str, dim: int, dataset: str, metric: str, population: str) -> tuple[float, int, int]:
    """Tables S5/S6: mean of complete batch means within one dataset, with coverage."""
    table = batch_table(method, dim, metric, population)
    table = table[table.dataset_id == dataset]
    complete = table[table.complete]
    mean = float(complete["mean"].mean()) if len(complete) else np.nan
    return mean, len(complete), len(table)


@lru_cache(maxsize=None)
def within_batch_width() -> pd.DataFrame:
    return pd.read_csv(DATA / "within_batch" / "solar_width_runs.csv")


def width_mean(dataset: str, dim: int, metric: str) -> float:
    frame = within_batch_width()
    rows = frame[(frame.dataset_id == dataset) & (frame.dimension == int(dim))]
    column = f"metric__{metric}"
    status = rows[f"status__{metric}"]
    if len(rows) != len(SEEDS) or set(rows.split_seed.astype(int)) != set(SEEDS):
        raise ValueError(f"incomplete within-batch seeds for {dataset} {dim}")
    if (status != "computed").all():
        return np.nan
    if (status != "computed").any():
        raise ValueError(f"partially missing within-batch metric {dataset} {dim} {metric}")
    return float(rows[column].mean())


@lru_cache(maxsize=None)
def composition() -> pd.DataFrame:
    return pd.read_csv(DATA / "splits" / "heldout_batch_composition.csv", dtype={"heldout_batch": str})


@lru_cache(maxsize=None)
def villani_predictions() -> pd.DataFrame:
    frames = []
    for seed in SEEDS:
        path = DATA / "heldout" / "villani_predictions" / (
            f"historical__solar_orthogonal__d128__immune_cell_human__Villani__s{seed}.csv.gz")
        frames.append(pd.read_csv(path, usecols=["run_id", "seed", "true_label", "prediction"]))
    return pd.concat(frames, ignore_index=True)


def villani_confusion() -> pd.DataFrame:
    """Figure S2B: row-normalized confusion, averaged over the five fits."""
    frame = villani_predictions()
    labels = sorted(frame.true_label.unique())
    matrices = []
    for _, group in frame.groupby("seed"):
        counts = pd.crosstab(group.true_label, group.prediction).reindex(index=labels, columns=labels, fill_value=0)
        matrices.append(counts.to_numpy(float) / counts.to_numpy(float).sum(axis=1, keepdims=True))
    return pd.DataFrame(np.mean(matrices, axis=0), index=labels, columns=labels)


def villani_per_class_f1() -> pd.DataFrame:
    from sklearn.metrics import f1_score
    frame = villani_predictions()
    labels = sorted(frame.true_label.unique())
    rows = []
    for seed, group in frame.groupby("seed"):
        scores = f1_score(group.true_label, group.prediction, labels=labels, average=None, zero_division=0)
        rows.append(dict(zip(labels, scores), seed=seed,
                         macro_f1=f1_score(group.true_label, group.prediction, labels=labels,
                                           average="macro", zero_division=0)))
    return pd.DataFrame(rows)


def load_json(path: Path) -> dict:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        return json.load(handle)
