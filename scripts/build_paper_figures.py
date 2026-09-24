#!/usr/bin/env python3
"""Rebuild the quantitative manuscript figures from the per-run records.

    python scripts/build_paper_figures.py --output-dir rebuilt/figures

Writes Figures 2, 3 and 4 and Additional file 1 Figures S2-S5 as PDF and PNG,
plus one CSV per figure with the plotted values. Figure 1 is a schematic and
Figure S1 (UMAP) needs the per-cell embeddings from the model attachments;
neither is rebuilt here. Layout and styling are simplified; plotted values,
axis extents of the signed-difference panels (Figures S3 and S5) and printed
labels follow the manuscript and are checked by verify_paper_numbers.py.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import paper_numbers as pn  # noqa: E402

SOLAR30 = ("solar_orthogonal", 30)
PCA30 = ("unintegrated_pca_matched", 30)
SCVI30 = ("frozen_reference_scvi", 30)
SCANVI30 = ("frozen_reference_scanvi", 30)
SCLSC30 = ("sclsc_refonly", 30)
COMMON = {"Matched PCA": PCA30, "SOLAR": SOLAR30, "scVI": SCVI30, "scANVI": SCANVI30, "SCLSC": SCLSC30}
ARMS = {"No anchors": ("solar_none", 128), "Per label": ("solar_orthogonal_uniq", 128),
        "Per cell": ("solar_orthogonal", 128)}
DATASET_NAMES = {"immune_cell_human": "Immune", "lung_atlas": "Lung", "pancreas": "Pancreas"}
BATCH_ORDER = [("immune_cell_human", "10X"), ("immune_cell_human", "Oetjen_A"), ("immune_cell_human", "Villani"),
               ("lung_atlas", "3"), ("lung_atlas", "4"), ("lung_atlas", "B1"),
               ("pancreas", "fluidigmc1"), ("pancreas", "inDrop2"), ("pancreas", "inDrop3")]

# Signed-difference panels: (panel letter, title, metric, population).
FIG_S3_PANELS = [("A", "Macro-F1", "macro_f1", "query"), ("B", "Balanced accuracy", "balanced_accuracy", "query"),
                 ("C", "Label ASW", "ASW_label", "joint"), ("D", "NMI", "NMI_cluster/label", "joint"),
                 ("E", "ARI", "ARI_cluster/label", "joint"), ("F", "PCR batch correction", "PCR_batch", "joint"),
                 ("G", "iLISI", "iLISI", "joint"), ("H", "Batch ASW", "ASW_label/batch", "joint"),
                 ("I", "Cell-cycle conservation", "cell_cycle_conservation", "joint"),
                 ("J", "Trajectory conservation", "trajectory", "joint")]
FIG_S5_PANELS = [("A", "Query macro-F1", "macro_f1", "query"), ("B", "Balanced accuracy", "balanced_accuracy", "query"),
                 ("C", "Label ASW", "ASW_label", "joint"), ("D", "iLISI", "iLISI", "joint"),
                 ("E", "Batch ASW", "ASW_label/batch", "joint"), ("F", "Trustworthiness", "trustworthiness", "query"),
                 ("G", "Pseudotime smoothness", "pseudotime_smoothness", "joint")]
CONTRASTS = {
    "S3": [("Repeated minus no anchors", ("solar_orthogonal", 128), ("solar_none", 128)),
           ("Repeated minus unique anchors", ("solar_orthogonal", 128), ("solar_orthogonal_uniq", 128))],
    "S5": [("SOLAR30 minus SOLAR128", ("solar_orthogonal", 30), ("solar_orthogonal", 128)),
           ("SCLSC30 minus SCLSC16", ("sclsc_refonly", 30), ("sclsc_refonly", 16))],
}
# Fractional padding of the signed-difference axes beyond the zero-inclusive data range.
AXIS_PAD = {"S3": 0.16, "S5": 0.17}


def panels(figure: str):
    return FIG_S3_PANELS if figure == "S3" else FIG_S5_PANELS


def signed_differences(figure: str, metric: str, population: str) -> pd.DataFrame:
    frames = []
    for label, left, right in CONTRASTS[figure]:
        c = pn.contrast(left, right, metric, population)
        frames.append(c.batch_differences.assign(contrast=label, equal_dataset_mean=c.delta,
                                                 n_valid=c.batches_valid, n_applicable=c.batches_applicable))
    return pd.concat(frames, ignore_index=True)


def axis_extent(figure: str, panel: str) -> tuple[str, str, int, int]:
    """Printed axis end labels (3 significant digits) and the n=valid/applicable label."""
    _, _, metric, population = next(p for p in panels(figure) if p[0] == panel)
    diffs = signed_differences(figure, metric, population)
    low, high = min(diffs["diff"].min(), 0.0), max(diffs["diff"].max(), 0.0)
    pad = AXIS_PAD[figure] * (high - low)
    counts = diffs.groupby("contrast")[["n_valid", "n_applicable"]].first()
    if counts.nunique().max() != 1:
        raise ValueError("contrasts in one panel cover different batches")
    return f"{low - pad:.3g}", f"{high + pad:.3g}", int(counts.n_valid.iloc[0]), int(counts.n_applicable.iloc[0])


def figure2_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    diff = pn.contrast(SOLAR30, PCA30, "macro_f1", "query").batch_differences
    panel_a = diff.groupby("dataset_id", as_index=False)["diff"].mean().rename(columns={"diff": "mean_delta_macro_f1"})
    panel_b = pd.DataFrame([{"method": name, "mean_macro_f1": pn.summary(*rep, "macro_f1", "query").mean}
                            for name, rep in COMMON.items()])
    return panel_a, panel_b


def figure3_data() -> pd.DataFrame:
    rows = []
    for metric, population in (("ASW_label", "joint"), ("iLISI", "joint"), ("macro_f1", "query"),
                               ("balanced_accuracy", "query")):
        for arm, rep in ARMS.items():
            table = pn.batch_table(*rep, metric, population)
            for dataset, value in table[table.complete].groupby("dataset_id")["mean"].mean().items():
                rows.append({"metric": metric, "population": population, "condition": arm,
                             "dataset_id": dataset, "value": value})
    return pd.DataFrame(rows)


def figure4_data() -> pd.DataFrame:
    rows = []
    for panel, (metric, population, datasets) in {"A": ("trustworthiness", "query", ("immune_cell_human", "lung_atlas")),
                                                   "B": ("pseudotime_smoothness", "joint", ("immune_cell_human",))}.items():
        for name, rep in COMMON.items():
            y = pn.batch_table(*rep, metric, population)
            x = pn.batch_table(*rep, "ASW_label", "joint")
            y = y[y.complete & y.dataset_id.isin(datasets)]
            x = x[x.complete & x.set_index(["dataset_id", "heldout_batch"]).index.isin(
                y.set_index(["dataset_id", "heldout_batch"]).index)]
            rows.append({"panel": panel, "method": name, "label_asw_joint": pn.dataset_weighted_mean(x),
                         "y_metric": f"{metric}/{population}", "y_value": pn.dataset_weighted_mean(y),
                         "n_batches": len(y)})
    return pd.DataFrame(rows)


def figure_s4_data() -> pd.DataFrame:
    rows = []
    for name, rep in COMMON.items():
        for metric in ("macro_f1", "balanced_accuracy"):
            table = pn.batch_table(*rep, metric, "query")
            for r in table.itertuples():
                rows.append({"method": name, "metric": metric, "dataset_id": r.dataset_id,
                             "heldout_batch": r.heldout_batch, "batch_mean": r.mean, "seed_sd": r.seed_sd})
    return pd.DataFrame(rows)


def build(output: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output.mkdir(parents=True, exist_ok=True)

    def save(fig, name: str, data: pd.DataFrame) -> None:
        fig.savefig(output / f"{name}.pdf", bbox_inches="tight")
        fig.savefig(output / f"{name}.png", dpi=200, bbox_inches="tight")
        plt.close(fig)
        data.to_csv(output / f"{name}.csv", index=False)

    a, b = figure2_data()
    fig, axes = plt.subplots(1, 2, figsize=(9, 3))
    names = [DATASET_NAMES[d] for d in a.dataset_id]
    axes[0].barh(names, a.mean_delta_macro_f1, color="#4c72b0")
    for y, v in enumerate(a.mean_delta_macro_f1):
        axes[0].text(v, y, f" {v:+.3f}", va="center")
    axes[0].invert_yaxis()
    axes[0].set_xlabel("Mean Δ macro-F1\n(SOLAR − PCA)")
    b = b.sort_values("mean_macro_f1")
    axes[1].barh(b.method, b.mean_macro_f1, color="#8c8c8c")
    for y, v in enumerate(b.mean_macro_f1):
        axes[1].text(v, y, f" {v:.3f}", va="center")
    axes[1].set_xlim(0, 1.0)
    axes[1].set_xlabel("Mean macro-F1")
    save(fig, "Fig2_query_annotation", pd.concat([a.assign(panel="A"), b.assign(panel="B")]))

    f3 = figure3_data()
    fig, axes = plt.subplots(2, 2, figsize=(8, 6))
    for ax, (metric, title) in zip(axes.flat, [("ASW_label", "Label ASW"), ("iLISI", "iLISI"),
                                               ("macro_f1", "Query macro-F1"), ("balanced_accuracy", "Balanced accuracy")]):
        for dataset, group in f3[f3.metric == metric].groupby("dataset_id"):
            group = group.set_index("condition").loc[list(ARMS)]
            ax.plot(list(ARMS), group.value, marker="o", label=DATASET_NAMES[dataset])
        ax.set_ylabel(title)
    axes[0, 0].legend(frameon=False)
    save(fig, "Fig3_anchor_controls", f3)

    f4 = figure4_data()
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
    for ax, (panel, group) in zip(axes, f4.groupby("panel")):
        ax.scatter(group.label_asw_joint, group.y_value)
        for r in group.itertuples():
            ax.annotate(r.method, (r.label_asw_joint, r.y_value), fontsize=8)
        ax.set_xlim(0.5, 1.0)
        ax.set_xlabel("Label ASW")
        ax.set_ylabel(group.y_metric.iloc[0])
    save(fig, "Fig4_label_separation_vs_local_structure", f4)

    confusion = pn.villani_confusion()
    per_class = pn.villani_per_class_f1()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    classes = list(confusion.index)
    axes[0].bar(range(len(classes)), per_class[classes].mean(), yerr=per_class[classes].std(ddof=1))
    axes[0].set_xticks(range(len(classes)), classes, rotation=30, ha="right")
    axes[0].set_ylim(0, 1.0)
    axes[0].set_ylabel("Query F1")
    axes[1].imshow(confusion.to_numpy(), vmin=0, vmax=1, cmap="Blues")
    for (i, j), v in np.ndenumerate(confusion.to_numpy()):
        axes[1].text(j, i, f"{v:.2f}", ha="center", va="center")
    axes[1].set_xticks(range(len(classes)), classes, rotation=30, ha="right")
    axes[1].set_yticks(range(len(classes)), classes)
    save(fig, "FigS2_villani", confusion.reset_index().rename(columns={"index": "true_label"}))

    for figure in ("S3", "S5"):
        panel_list = panels(figure)
        fig, axes = plt.subplots(len(panel_list), 1, figsize=(6, 2.6 * len(panel_list)))
        frames = []
        for ax, (letter, title, metric, population) in zip(axes, panel_list):
            diffs = signed_differences(figure, metric, population)
            frames.append(diffs.assign(panel=letter, metric=metric, population=population))
            low, high, n_valid, n_applicable = axis_extent(figure, letter)
            labels = [f"{DATASET_NAMES[d]} / {b}" for d, b in BATCH_ORDER]
            for k, (name, group) in enumerate(diffs.groupby("contrast", sort=False)):
                y = [labels.index(f"{DATASET_NAMES[d]} / {b}") + (k - 0.5) * 0.3
                     for d, b in zip(group.dataset_id, group.heldout_batch)]
                ax.scatter(group["diff"], y, label=name, s=14)
                ax.axvline(group.equal_dataset_mean.iloc[0], ls="--", lw=0.8, color=f"C{k}")
            ax.axvline(0, color="k", lw=0.8)
            ax.set_xlim(float(low), float(high))
            ax.set_xticks([float(low), 0, float(high)], [low, "0", high])
            ax.set_yticks(range(len(labels)), labels, fontsize=7)
            ax.invert_yaxis()
            ax.set_title(f"{letter} {title}", loc="left")
            ax.text(1.0, 1.02, f"n={n_valid}/{n_applicable}", transform=ax.transAxes, ha="right")
        axes[0].legend(frameon=False, fontsize=7)
        fig.tight_layout()
        save(fig, f"Fig{figure}_signed_differences", pd.concat(frames, ignore_index=True))

    s4 = figure_s4_data()
    fig, axes = plt.subplots(2, 1, figsize=(10, 6))
    for ax, metric in zip(axes, ("macro_f1", "balanced_accuracy")):
        for k, (name, group) in enumerate(s4[s4.metric == metric].groupby("method", sort=False)):
            x = np.arange(len(group)) + (k - 2) * 0.14
            ax.errorbar(x, group.batch_mean, yerr=group.seed_sd.fillna(0), fmt="o", label=name, ms=3)
        ax.set_ylim(0, 1.0)
        ax.set_xticks(range(len(BATCH_ORDER)), [b for _, b in BATCH_ORDER])
        ax.set_ylabel(metric)
    axes[0].legend(frameon=False, ncol=5, fontsize=7)
    save(fig, "FigS4_query_recovery_by_batch", s4)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", type=Path, default=pn.ROOT / "rebuilt" / "figures")
    build(parser.parse_args().output_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
