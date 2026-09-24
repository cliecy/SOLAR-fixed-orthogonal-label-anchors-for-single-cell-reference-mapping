#!/usr/bin/env python3
"""heldout_batch_selection_audit.csv: for every one of the 57 possible
leave-one-batch-out configurations across all 5 datasets, record its
composition and whether it was selected for training/evaluation, with the
explicit selection rule: per dataset, the batch with the smallest query
cell count, one from the middle of the size-sorted list, and the batch with
the largest query cell count (3 per dataset x 5 datasets = 15 selected).
This was decided by cell-count percentile BEFORE inspecting any label
composition or downstream metric -- not chosen post hoc based on results.
"""
from __future__ import annotations

import glob
import re

import pandas as pd

SELECTED = {
    "pancreas": ["fluidigmc1", "inDrop2", "inDrop3"],
    "immune_cell_human": ["10X", "Oetjen_A", "Villani"],
    "lung_atlas": ["3", "B1", "4"],
    "simulation_1": ["Batch6", "Batch3", "Batch1"],
    "simulation_2": ["Batch3Sub4", "Batch1Sub3", "Batch3Sub1"],
}


def main() -> int:
    rows = []
    for dataset in SELECTED:
        for path in sorted(
            glob.glob(f"outputs/benchmark/solar_scib_rebuttal_v1/splits/{dataset}/*heldout_*")
        ):
            name = path.rsplit("/", 1)[-1]
            batch = re.sub(r"^zz_unused_heldout_|^heldout_", "", name)
            split = pd.read_csv(path + "/split.csv")
            ref = split[split["role"] == "reference"]
            query = split[split["role"] == "query"]
            ref_labels = set(ref["label"].astype(str))
            query_labels = set(query["label"].astype(str))
            supported = query["label"].astype(str).isin(ref_labels).mean()
            rows.append({
                "dataset_id": dataset,
                "heldout_batch": batch,
                "n_reference": len(ref),
                "n_query": len(query),
                "n_celltypes_reference": len(ref_labels),
                "n_celltypes_query": len(query_labels),
                "n_celltypes_query_only": len(query_labels - ref_labels),
                "fraction_query_cells_label_supported_by_reference": round(float(supported), 4),
                "selected_for_training": batch in SELECTED[dataset],
                "selection_rank_by_query_size": None,  # filled below
            })

    df = pd.DataFrame(rows)
    for dataset, group in df.groupby("dataset_id"):
        ranked = group.sort_values("n_query")
        for rank, idx in enumerate(ranked.index, start=1):
            df.loc[idx, "selection_rank_by_query_size"] = rank

    df["selection_rule"] = (
        "Per dataset: smallest, one middle, and largest n_query among all "
        "possible held-out batches, ranked before any label-composition or "
        "metric inspection. selection_rank_by_query_size=1 is smallest."
    )
    df = df.sort_values(["dataset_id", "n_query"])
    df.to_csv("rebuttal_response/heldout_batch_selection_audit.csv", index=False)
    print(f"wrote heldout_batch_selection_audit.csv: {len(df)} rows "
          f"({df['selected_for_training'].sum()} selected)")
    print(df.groupby("dataset_id")["selected_for_training"].sum().to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
