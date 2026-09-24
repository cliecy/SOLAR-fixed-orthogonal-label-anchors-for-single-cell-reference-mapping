#!/usr/bin/env python3
"""Render v1.2 batch-paired differences from portable metrics.csv/runs.csv.

Each facet has its own metric-unit axis. NA is displayed, not placed at zero;
source reasons and coverage are available in the accompanying tables.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

from build_experiment_v1_2_tables import DATASETS, PANELS, load_tables

LABELS = {"immune_cell_human": "Immune", "lung_atlas": "Lung", "pancreas": "Pancreas"}
COLORS = ("#0072B2", "#D55E00")
MARKERS = ("o", "s")
BATCHES = tuple((dataset, batch) for dataset, batches in DATASETS.items() for batch in batches)
TITLES = {
    "macro_f1": "Query macro-F1", "balanced_accuracy": "Balanced accuracy",
    "labelASW": "Label ASW (canonical)", "batchASW": "Batch ASW (canonical)",
    "ASW_label": "Label ASW (official)", "ASW_label/batch": "Batch ASW (official)",
    "NMI_cluster/label": "NMI", "ARI_cluster/label": "ARI",
    "PCR_batch": "PCR batch", "graph_conn": "Graph connectivity",
    "isolated_label_F1": "Isolated-label F1",
    "isolated_label_silhouette": "Isolated-label silhouette",
    "hvg_overlap": "HVG overlap", "cell_cycle_conservation": "Cell-cycle conservation",
    "pseudotime_smoothness": "Pseudotime smoothness",
    "trustworthiness": "Trustworthiness", "trajectory": "Trajectory",
    "within_label_silhouette__study": "Within-label study association",
    "within_label_silhouette__sample_ID": "Within-label sample association",
    "within_label_silhouette__donor": "Within-label donor association",
    "within_label_silhouette__patientGroup": "Within-label patient-group association",
}


def draw_figure(pairs, comparisons, title: str, destination: Path) -> None:
    """Plot exact batch differences; never pool differently scaled metrics."""
    ncols = 4
    nrows = math.ceil(len(PANELS) / ncols)
    figure, axes = plt.subplots(nrows, ncols, figsize=(24, nrows * 3.9), squeeze=False)
    try:
        for axis, (metric, population) in zip(axes.flat, PANELS):
            panel = pairs[(pairs.metric == metric) & (pairs.population == population)]
            values = panel[panel.comparison.isin([name for name, _ in comparisons])].difference.to_numpy()
            finite = values[np.isfinite(values)]
            extent = float(np.max(np.abs(finite))) * 1.2 if finite.size else 1.0
            if extent == 0:
                extent = 0.01
            axis.set_xlim(-extent, extent)
            axis.axvline(0, color="0.3", linewidth=0.8, zorder=1)
            for start in (0, 6):
                axis.axhspan(start - 0.5, start + 2.5, color="0.94", zorder=0)
            for boundary in (2.5, 5.5):
                axis.axhline(boundary, color="0.7", linewidth=0.5)
            for number, (comparison, _) in enumerate(comparisons):
                selected = panel[panel.comparison == comparison].set_index(["dataset_id", "heldout_batch"])
                offset = 0 if len(comparisons) == 1 else (-0.15 if number == 0 else 0.15)
                for y, batch in enumerate(BATCHES):
                    # load_tables enforces the full matrix and all metric panels.
                    value = selected.loc[batch, "difference"]
                    if np.isfinite(value):
                        axis.scatter(value, y + offset, color=COLORS[number], marker=MARKERS[number],
                                     s=22, linewidths=0, zorder=3)
                    else:
                        axis.text(0.98, y + offset, "NA", transform=axis.get_yaxis_transform(),
                                  ha="right", va="center", fontsize=6.5, color=COLORS[number])
            axis.set_yticks(range(len(BATCHES)))
            axis.set_yticklabels([f"{LABELS[dataset]} / {batch}" for dataset, batch in BATCHES], fontsize=7)
            axis.set_ylim(len(BATCHES) - 0.5, -0.5)
            axis.set_title(f"{TITLES.get(metric, metric)} — {population}", fontsize=10)
            axis.set_xlabel("Difference (metric units; independent facet scale)", fontsize=7)
            axis.tick_params(axis="x", labelsize=7)
            axis.spines["top"].set_visible(False)
            axis.spines["right"].set_visible(False)
        for axis in list(axes.flat)[len(PANELS):]:
            axis.set_visible(False)
        handles = [Line2D([], [], color=COLORS[number], marker=MARKERS[number], linestyle="none",
                          label=label, markersize=6) for number, (_, label) in enumerate(comparisons)]
        handles.append(Line2D([], [], color="0.3", linewidth=0.8, label="Zero difference"))
        figure.suptitle(title, fontsize=16, y=0.995)
        figure.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.982),
                      ncol=len(handles), frameon=False, fontsize=11)
        figure.text(0.5, 0.008,
                    "Points: differences of complete five-seed means within each held-out batch. "
                    "NA: unavailable paired estimate (see tables for reasons).\n"
                    "Batch reference sets overlap; points are not independent biological replicates. "
                    "Metric scales differ; raw difference magnitudes are not comparable across facets.\n"
                    "Covariate silhouettes describe associations, not a uniformly higher-is-better quality score.",
                    ha="center", va="bottom", fontsize=10)
        figure.tight_layout(rect=(0, 0.03, 1, 0.963), h_pad=2.0, w_pad=2.0)
        figure.savefig(destination.with_suffix(".pdf"), metadata={"CreationDate": None, "ModDate": None})
        figure.savefig(destination.with_suffix(".png"), dpi=160)
    finally:
        plt.close(figure)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    pairs = load_tables(args.input_dir)["paired_differences.csv"]
    destination = args.output_dir / "figures"
    destination.mkdir(parents=True, exist_ok=True)
    draw_figure(pairs, (("solar30_minus_sclsc30", "SOLAR30 − SCLSC30"),),
                "E1: Common-dimension SOLAR minus SCLSC", destination / "E1_solar_minus_sclsc_30d")
    draw_figure(pairs, (("repeated_minus_none128", "Repeated − no anchor (128d)"),
                        ("repeated_minus_unique128", "Repeated − unique anchor (128d)")),
                "E2: Anchor effects at 128 dimensions", destination / "E2_anchor_effects_128d")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
