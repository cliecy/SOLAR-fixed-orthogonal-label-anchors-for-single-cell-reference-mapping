#!/usr/bin/env python3
"""Rebuild v1.2 tables using only portable runs.csv and metrics.csv.

Seed SD describes training variation, not biological replication. Differences
use complete five-seed means and common finite batches only. Dataset means
receive equal weight; no composite, significance test or imputation is used.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

DATASETS = {
    "immune_cell_human": ("10X", "Oetjen_A", "Villani"),
    "lung_atlas": ("3", "4", "B1"),
    "pancreas": ("fluidigmc1", "inDrop2", "inDrop3"),
}
SEEDS = frozenset(range(40, 45))
SOLAR = "solar_orthogonal"
SCLSC = "sclsc_refonly"
PCA = "unintegrated_pca_matched"
ORIGINAL = ((SOLAR, 128), (SCLSC, 16), ("frozen_reference_scvi", 30),
            ("frozen_reference_scanvi", 30), (PCA, 40))
COMMON = tuple((method, 30) for method, _ in ORIGINAL)
ABLATION = ((SOLAR, 128), ("solar_orthogonal_uniq", 128), ("solar_none", 128))
METHOD_DIMENSIONS = tuple(dict.fromkeys(ORIGINAL + COMMON + ABLATION))
OFFICIAL_METRICS = (
    "PCR_batch", "ASW_label/batch", "iLISI", "graph_conn", "kBET",
    "NMI_cluster/label", "ARI_cluster/label", "ASW_label", "isolated_label_F1",
    "isolated_label_silhouette", "cLISI", "hvg_overlap",
    "cell_cycle_conservation", "trajectory",
)
PANELS = (
    ("macro_f1", "query"), ("balanced_accuracy", "query"),
    ("labelASW", "joint"), ("batchASW", "joint"),
    *((metric, "joint") for metric in OFFICIAL_METRICS),
    *((metric, population) for metric in ("trustworthiness", "pseudotime_smoothness")
      for population in ("reference", "query", "joint")),
    *((f"within_label_silhouette__{field}", population)
      for field in ("study", "sample_ID", "donor", "patientGroup")
      for population in ("reference", "query", "joint")),
)
GROUP = ["method", "dimension", "dataset_id", "heldout_batch"]
IDENTITY = GROUP + ["seed"]
CONTRASTS = (
    ("solar30_minus_sclsc30", "E1", (SOLAR, 30), (SCLSC, 30)),
    ("solar30_minus_solar128", "E2", (SOLAR, 30), (SOLAR, 128)),
    ("sclsc30_minus_sclsc16", "E2", (SCLSC, 30), (SCLSC, 16)),
    ("repeated_minus_none128", "E3", (SOLAR, 128), ("solar_none", 128)),
    ("repeated_minus_unique128", "E3", (SOLAR, 128), ("solar_orthogonal_uniq", 128)),
)


def _require_columns(frame: pd.DataFrame, columns: list[str], name: str) -> None:
    missing = set(columns) - set(frame.columns)
    if missing:
        raise ValueError(f"{name} missing columns: {sorted(missing)}")


def _integers(frame: pd.DataFrame, columns: tuple[str, ...]) -> None:
    for column in columns:
        values = pd.to_numeric(frame[column], errors="raise")
        if not np.isfinite(values).all() or (values != np.floor(values)).any():
            raise ValueError(f"{column} must contain finite integers")
        frame[column] = values.astype(int)


def validate_inputs(runs: pd.DataFrame, metrics: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Reject an incomplete matrix rather than silently reporting fewer seeds."""
    runs, metrics = runs.copy(), metrics.copy()
    _require_columns(runs, ["run_id", *IDENTITY, "stochastic"], "runs.csv")
    _require_columns(metrics, ["run_id", *IDENTITY, "metric", "population", "value", "status", "reason", "comparison_eligible"], "metrics.csv")
    for frame in (runs, metrics):
        for column in ("run_id", "method", "dataset_id", "heldout_batch"):
            if frame[column].isna().any() or frame[column].astype(str).str.strip().eq("").any():
                raise ValueError(f"Empty identity field: {column}")
            frame[column] = frame[column].astype(str)
        _integers(frame, ("dimension", "seed"))
    if runs.run_id.duplicated().any() or runs.duplicated(IDENTITY).any():
        raise ValueError("Duplicate run_id or method/dimension/dataset/batch/seed in runs.csv")
    stochastic = runs.stochastic.astype(str).str.lower().map({"true": True, "false": False, "1": True, "0": False})
    if stochastic.isna().any():
        raise ValueError("stochastic must be a boolean")
    runs["stochastic"] = stochastic.astype(bool)
    expected_groups = {
        (method, dimension, dataset, batch)
        for method, dimension in METHOD_DIMENSIONS
        for dataset, batches in DATASETS.items() for batch in batches
    }
    observed_groups = set(runs[GROUP].itertuples(index=False, name=None))
    if observed_groups != expected_groups:
        raise ValueError("Wrong dimension or missing/unexpected method/dataset/batch group: "
                         f"missing={sorted(expected_groups - observed_groups)}, "
                         f"unexpected={sorted(observed_groups - expected_groups)}")
    for key, group in runs.groupby(GROUP, sort=True):
        deterministic = key[0] == PCA
        if (group.stochastic == deterministic).any():
            raise ValueError(f"Wrong stochastic flag for {key}")
        seeds = set(group.seed)
        if deterministic:
            if len(group) != 1 or seeds not in ({0}, {40}):
                raise ValueError(f"Deterministic PCA requires one seed 0 or 40 row: {key}")
        elif seeds != SEEDS:
            raise ValueError(f"Missing or unexpected seed for {key}: {sorted(seeds)}; require 40-44")
    if metrics.duplicated(["run_id", "metric", "population"]).any():
        raise ValueError("Duplicate run/metric/population in metrics.csv")
    unknown = set(metrics.run_id) - set(runs.run_id)
    if unknown:
        raise ValueError(f"Unknown metric run_id: {sorted(unknown)}")
    joined = metrics.merge(runs[["run_id", *IDENTITY]], on="run_id", suffixes=("", "_run"), validate="many_to_one")
    for column in IDENTITY:
        if (joined[column] != joined[f"{column}_run"]).any():
            raise ValueError(f"Metric identity disagrees with runs.csv: {column}")
    panel_set = set(PANELS)
    for run_id, group in metrics.groupby("run_id", sort=True):
        if set(group[["metric", "population"]].itertuples(index=False, name=None)) != panel_set:
            raise ValueError(f"Missing or unexpected metric/population panel for {run_id}")
    if set(metrics.run_id) != set(runs.run_id):
        raise ValueError("Missing all metric panels for one or more runs")
    metrics["value"] = pd.to_numeric(metrics.value.replace({"": np.nan, "NA": np.nan, "NaN": np.nan, "nan": np.nan}), errors="raise")
    if np.isinf(metrics.value).any():
        raise ValueError("Infinite metric values must be explicit NA with a reason")
    metrics["status"] = metrics.status.fillna("").astype(str)
    metrics["reason"] = metrics.reason.fillna("").astype(str)
    if metrics.status.str.strip().eq("").any():
        raise ValueError("Every metric requires a status")
    if metrics.status.isin(["artifact_missing", "missing_artifact", "failed", "calculation_failed"]).any():
        raise ValueError("Unresolved artifact or calculation failure cannot enter formal tables")
    eligible = metrics.comparison_eligible.astype(str).str.lower().map({"true": True, "false": False, "1": True, "0": False})
    if eligible.isna().any():
        raise ValueError("comparison_eligible must be a boolean")
    metrics["comparison_eligible"] = eligible.astype(bool)
    finite = np.isfinite(metrics.value)
    if (~finite & metrics.reason.str.strip().eq("")).any():
        raise ValueError("Every NA metric requires a reason")
    if (metrics.status.eq("computed") != finite).any():
        raise ValueError("Only computed status may carry a finite metric value")
    for metric, group in metrics.groupby("metric", sort=False):
        outside = ~group.dataset_id.isin(applicable_datasets(metric))
        if np.isfinite(group.loc[outside, "value"]).any():
            raise ValueError(f"Finite value outside predeclared applicability: {metric}")
    return runs, metrics


def _reasons(frame: pd.DataFrame) -> str:
    return "; ".join(sorted({f"{row.status}: {row.reason}" for row in frame.itertuples()
                             if str(row.reason).strip()}))


def seed_statistics(runs: pd.DataFrame, metrics: pd.DataFrame) -> pd.DataFrame:
    """Compute a formal mean only with all finite required seeds (PCA once)."""
    runs, metrics = validate_inputs(runs, metrics)
    rows = []
    for key, group in metrics.groupby(GROUP + ["metric", "population"], sort=True):
        expected = 1 if key[0] == PCA else 5
        finite = np.isfinite(group.value)
        count = int(finite.sum())
        comparable = finite & group.comparison_eligible
        comparable_count = int(comparable.sum())
        complete = comparable_count == expected
        statuses = "|".join(sorted(set(group.status)))
        rows.append(dict(zip(GROUP + ["metric", "population"], key),
                         mean=float(group.value.mean()) if complete else np.nan,
                         sd=float(group.value.std(ddof=1)) if complete and expected == 5 else np.nan,
                         sd_status="not_applicable" if expected == 1 else ("computed" if complete else "unavailable"),
                         n_seeds_expected=expected, n_seeds_present=len(group), n_seeds_finite=count,
                         n_seeds_comparable=comparable_count, numeric_seed_coverage=count / expected,
                         seed_coverage=comparable_count / expected, source_statuses=statuses,
                         status="computed" if complete else ("noncomparable_implementation" if (finite & ~group.comparison_eligible).any()
                                                              else ("partial_seeds" if count else "all_na")),
                         reason=_reasons(group),
                         run_ids="|".join(sorted(group.run_id))))
    return pd.DataFrame(rows)


def paired_differences(statistics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    join_columns = ["dataset_id", "heldout_batch", "metric", "population"]
    for comparison, table, left, right in CONTRASTS:
        lhs = statistics[(statistics.method == left[0]) & (statistics.dimension == left[1])]
        rhs = statistics[(statistics.method == right[0]) & (statistics.dimension == right[1])]
        pairs = lhs.merge(rhs, on=join_columns, suffixes=("_lhs", "_rhs"), validate="one_to_one")
        for pair in pairs.itertuples(index=False):
            valid = bool(np.isfinite(pair.mean_lhs) and np.isfinite(pair.mean_rhs))
            reason = "" if valid else "; ".join(
                f"{side}: {getattr(pair, 'status_' + side)} ({getattr(pair, 'reason_' + side)})"
                for side in ("lhs", "rhs") if not np.isfinite(getattr(pair, "mean_" + side)))
            rows.append({
                "comparison": comparison, "table": table,
                **{column: getattr(pair, column) for column in join_columns},
                "lhs_method": left[0], "lhs_dimension": left[1],
                "rhs_method": right[0], "rhs_dimension": right[1],
                "lhs_mean": pair.mean_lhs, "lhs_sd": pair.sd_lhs,
                "rhs_mean": pair.mean_rhs, "rhs_sd": pair.sd_rhs,
                "difference": pair.mean_lhs - pair.mean_rhs if valid else np.nan,
                "n_lhs_seeds_expected": pair.n_seeds_expected_lhs,
                "n_rhs_seeds_expected": pair.n_seeds_expected_rhs,
                "n_lhs_seeds_finite": pair.n_seeds_finite_lhs,
                "n_rhs_seeds_finite": pair.n_seeds_finite_rhs,
                "valid_batch": valid, "status": "computed" if valid else "unpaired_na",
                "reason": reason,
            })
    return pd.DataFrame(rows)


def applicable_datasets(metric: str) -> tuple[str, ...]:
    if metric in ("pseudotime_smoothness", "trajectory"):
        return ("immune_cell_human",)
    if metric == "trustworthiness":
        return ("immune_cell_human", "lung_atlas")
    if metric in ("within_label_silhouette__study", "within_label_silhouette__sample_ID"):
        return ("immune_cell_human",)
    if metric in ("within_label_silhouette__donor", "within_label_silhouette__patientGroup"):
        return ("lung_atlas",)
    return tuple(DATASETS)


def _coverage(group: pd.DataFrame, metric: str) -> dict:
    applicable = group.dataset_id.isin(applicable_datasets(metric))
    valid = np.isfinite(group.difference)
    n_expected = int(applicable.sum())
    n_valid = int(valid.sum())
    # Finite-seed counts describe available seeds; paired counts describe the
    # seeds actually contributing through complete, common-valid batch means.
    return {
        "n_batches_total": len(group), "n_batches_expected": n_expected,
        "n_batches_valid": n_valid,
        "batch_coverage": n_valid / n_expected if n_expected else np.nan,
        "n_lhs_seeds_expected": int(group.loc[applicable, "n_lhs_seeds_expected"].sum()),
        "n_rhs_seeds_expected": int(group.loc[applicable, "n_rhs_seeds_expected"].sum()),
        "n_lhs_seeds_finite": int(group.loc[applicable, "n_lhs_seeds_finite"].sum()),
        "n_rhs_seeds_finite": int(group.loc[applicable, "n_rhs_seeds_finite"].sum()),
        "n_lhs_seeds_paired": int(group.loc[valid, "n_lhs_seeds_finite"].sum()),
        "n_rhs_seeds_paired": int(group.loc[valid, "n_rhs_seeds_finite"].sum()),
        "valid_batches": "|".join(sorted(group.loc[valid, "dataset_id"] + "/" + group.loc[valid, "heldout_batch"])),
        "reason": _reasons(group.loc[~valid]),
    }


def summarize_pairs(pairs: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    dataset_rows, overall_rows, lodo_rows = [], [], []
    keys = ["comparison", "table", "metric", "population"]
    for key, group in pairs.groupby(keys, sort=True):
        identity = dict(zip(keys, key))
        metric = identity["metric"]
        scope = applicable_datasets(metric)
        if np.isfinite(group.loc[~group.dataset_id.isin(scope), "difference"]).any():
            raise ValueError(f"Finite value outside predeclared applicability: {metric}")
        for dataset, subset in group.groupby("dataset_id", sort=True):
            valid = subset[np.isfinite(subset.difference)]
            dataset_rows.append({**identity, "dataset_id": dataset,
                                 "lhs_mean": valid.lhs_mean.mean(), "rhs_mean": valid.rhs_mean.mean(),
                                 "difference": valid.difference.mean(),
                                 "status": "computed" if len(valid) else "all_na",
                                 **_coverage(subset, metric)})

        def summarize(subset: pd.DataFrame, included: tuple[str, ...]) -> dict:
            valid = subset[np.isfinite(subset.difference)]
            dataset_means = valid.groupby("dataset_id")[["lhs_mean", "rhs_mean", "difference"]].mean()
            return {**identity, **dataset_means.mean().to_dict(),
                    "n_datasets_expected": len(included), "n_datasets_valid": len(dataset_means),
                    "dataset_coverage": len(dataset_means) / len(included) if included else np.nan,
                    "included_datasets": "|".join(included),
                    "valid_datasets": "|".join(sorted(dataset_means.index)),
                    "status": "computed" if len(dataset_means) else "all_na",
                    **_coverage(subset, metric)}

        overall_rows.append(summarize(group, scope))
        # A single-dataset endpoint has no cross-dataset sensitivity analysis.
        if len(scope) > 1:
            for excluded in scope:
                included = tuple(dataset for dataset in scope if dataset != excluded)
                subset = group[group.dataset_id.isin(included)]
                lodo_rows.append({**summarize(subset, included), "excluded_dataset": excluded})
    return pd.DataFrame(dataset_rows), pd.DataFrame(overall_rows), pd.DataFrame(lodo_rows)


def build_tables(runs: pd.DataFrame, metrics: pd.DataFrame) -> dict[str, pd.DataFrame]:
    statistics = seed_statistics(runs, metrics)
    e1_parts = []
    for block, methods in (("original", ORIGINAL), ("common30", COMMON)):
        selection = statistics[["method", "dimension"]].apply(tuple, axis=1).isin(methods)
        e1_parts.append(statistics.loc[selection].assign(block=block))
    e1 = pd.concat(e1_parts, ignore_index=True)
    pairs = paired_differences(statistics)
    dataset, overall, lodo = summarize_pairs(pairs)
    index = ["dataset_id", "heldout_batch", "metric", "population"]
    e3_parts = []
    for name, (method, dimension) in zip(("repeated", "unique", "none"), ABLATION):
        selected = statistics[(statistics.method == method) & (statistics.dimension == dimension)]
        selected = selected.drop(columns=["method", "dimension"]).set_index(index).add_prefix(name + "_")
        e3_parts.append(selected)
    for comparison, name in (("repeated_minus_none128", "repeated_minus_none"),
                             ("repeated_minus_unique128", "repeated_minus_unique")):
        selected = pairs[pairs.comparison == comparison].set_index(index)
        e3_parts.append(selected[["difference", "valid_batch", "status", "reason"]].add_prefix(name + "_"))
    return {
        "E1_cross_method.csv": e1,
        "E2_dimension_sensitivity.csv": pairs[pairs.table == "E2"].copy(),
        "E3_anchor_ablation.csv": pd.concat(e3_parts, axis=1).reset_index().assign(dimension=128),
        "paired_differences.csv": pairs,
        "dataset_summary.csv": dataset,
        "overall_summary.csv": overall,
        "leave_one_dataset_out.csv": lodo,
    }


def load_tables(input_dir: Path) -> dict[str, pd.DataFrame]:
    # Do not dereference any artifact paths recorded in runs.csv.
    runs = pd.read_csv(input_dir / "runs.csv", dtype=str, keep_default_na=False)
    metrics = pd.read_csv(input_dir / "metrics.csv", dtype=str, keep_default_na=False)
    return build_tables(runs, metrics)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    tables = load_tables(args.input_dir)
    destination = args.output_dir / "tables"
    destination.mkdir(parents=True, exist_ok=True)
    for name, frame in tables.items():
        order = [column for column in ("block", "comparison", "method", "dimension", "metric", "population",
                                      "dataset_id", "heldout_batch", "excluded_dataset") if column in frame]
        frame.sort_values(order, kind="stable").to_csv(destination / name, index=False, na_rep="NA", float_format="%.17g")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
