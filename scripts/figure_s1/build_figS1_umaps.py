#!/usr/bin/env python3
"""Supplementary Figure S1: UMAP of the joint reference+query embedding for
all 9 real held-out batches (solar_orthogonal, model_seed=40 only -- one
declared seed/params combination, not chosen per-panel for visual effect).
UMAP is descriptive only, not quantitative evidence (see caption). Source:
canonical read-only run directory
outputs/benchmark/solar_scib_rebuttal_v1/artifacts/heldout/... (referenced,
not copied, by rebuttal_response/FINAL_RESULTS_README.md) plus
outputs/benchmark/solar_scib_rebuttal_v1/splits/.../split.csv for labels.
Declared UMAP parameters: n_neighbors=15, min_dist=0.3, metric='euclidean',
random_state=0 -- fixed for every panel."""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import umap

sys.path.insert(0, str(Path(__file__).resolve().parent))
from style import FULL_WIDTH_IN, mm  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parents[1]
OUT_SUPP = HERE / "supplementary"
OUT_PD = HERE / "plot_data"
OUT_SUPP.mkdir(parents=True, exist_ok=True)

BATCHES = [
    ("immune_cell_human", "10X"), ("immune_cell_human", "Oetjen_A"), ("immune_cell_human", "Villani"),
    ("lung_atlas", "3"), ("lung_atlas", "4"), ("lung_atlas", "B1"),
    ("pancreas", "fluidigmc1"), ("pancreas", "inDrop2"), ("pancreas", "inDrop3"),
]
DATASET_SHORT = {"immune_cell_human": "Immune", "lung_atlas": "Lung", "pancreas": "Pancreas"}
BATCH_DISPLAY = {
    "10X": "10X", "Oetjen_A": "Oetjen A", "Villani": "Villani",
    "3": "3", "4": "4", "B1": "B1",
    "fluidigmc1": "Fluidigm C1", "inDrop2": "inDrop 2", "inDrop3": "inDrop 3",
}
SEED = 40
UMAP_KW = dict(n_neighbors=15, min_dist=0.3, metric="euclidean", random_state=0)


def load_panel(dataset_id: str, batch: str) -> pd.DataFrame:
    base = REPO / "outputs/benchmark/solar_scib_rebuttal_v1"
    emb_path = base / f"artifacts/heldout/{dataset_id}/hvg2000_pca40/solar_orthogonal/heldout_{batch}/seed_{SEED}/anchor_0/labels_1p00/embedding.npz"
    split_path = base / f"splits/{dataset_id}/heldout_{batch}/split.csv"
    d = np.load(emb_path, allow_pickle=True)
    split = pd.read_csv(split_path).set_index("barcode")

    ref_bc = d["reference_barcodes"]
    qry_bc = d["query_barcodes"]
    X = np.concatenate([d["reference"], d["query"]], axis=0)
    bc = np.concatenate([ref_bc, qry_bc])
    role = np.array(["reference"] * len(ref_bc) + ["query"] * len(qry_bc))

    reducer = umap.UMAP(**UMAP_KW)
    coords = reducer.fit_transform(X)

    labels = split.loc[bc, "label"].to_numpy()
    df = pd.DataFrame({
        "dataset_id": dataset_id, "heldout_batch": batch,
        "barcode": bc, "role": role, "label": labels,
        "umap1": coords[:, 0], "umap2": coords[:, 1],
    })
    return df


def main() -> int:
    all_rows = []
    fig, axes = plt.subplots(3, 3, figsize=(FULL_WIDTH_IN, mm(175)))
    for ax, (dataset_id, batch) in zip(axes.flat, BATCHES):
        print(f"UMAP: {dataset_id}/{batch}", flush=True)
        df = load_panel(dataset_id, batch)
        all_rows.append(df)

        labels_sorted = sorted(df["label"].unique())
        cmap = plt.get_cmap("tab20", max(len(labels_sorted), 3))
        color_map = {lab: cmap(i) for i, lab in enumerate(labels_sorted)}

        ref = df[df.role == "reference"]
        qry = df[df.role == "query"]
        ax.scatter(ref.umap1, ref.umap2, c=[color_map[l] for l in ref.label], s=1.2, alpha=0.35, linewidths=0, label=None)
        ax.scatter(qry.umap1, qry.umap2, c=[color_map[l] for l in qry.label], s=3.0, alpha=0.9,
                   linewidths=0.15, edgecolors="black", marker="D")
        ax.set_title(f"{DATASET_SHORT[dataset_id]} / {BATCH_DISPLAY.get(batch, batch)}", fontsize=8)
        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_linewidth(0.4)

    fig.text(0.01, 0.014, "Small dot = reference cell (alpha 0.35). Diamond = held-out query cell.", fontsize=7)
    fig.text(0.01, 0.001, "Colour = cell-type label (per-panel palette). Descriptive only, not quantitative evidence.", fontsize=7)
    fig.subplots_adjust(left=0.02, right=0.99, top=0.96, bottom=0.055, hspace=0.32, wspace=0.12)

    out_pdf = OUT_SUPP / "FigS1_heldout_umaps.pdf"
    out_png = OUT_SUPP / "FigS1_heldout_umaps.png"
    fig.savefig(out_pdf)
    fig.savefig(out_png, dpi=300)
    print(f"wrote {out_pdf}, {out_png}")

    full = pd.concat(all_rows, ignore_index=True)
    out_csv = OUT_PD / "figS1_heldout_umap_coords.csv"
    full.to_csv(out_csv, index=False)
    print(f"wrote {out_csv} ({len(full)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
