"""Shared style (Okabe-Ito colorblind-safe palette + marker/linestyle
mapping) so method identity is visually consistent across every figure in
this package. Import this before building any matplotlib figure."""
from __future__ import annotations

import matplotlib

matplotlib.rcParams.update({
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
    "font.family": "sans-serif",
    "font.sans-serif": ["Liberation Sans", "DejaVu Sans", "Arial"],
    "font.size": 8,
    "axes.linewidth": 0.5,
    "xtick.major.width": 0.5,
    "ytick.major.width": 0.5,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "figure.titlesize": 9,
})

MM_PER_INCH = 25.4
FULL_WIDTH_MM = 170.0
FULL_WIDTH_IN = FULL_WIDTH_MM / MM_PER_INCH


def mm(v: float) -> float:
    return v / MM_PER_INCH


# Okabe-Ito palette, colorblind-safe.
COLOR = {
    "unintegrated_pca_matched": "#999999",  # grey
    "solar_orthogonal": "#0072B2",          # blue
    "frozen_reference_scvi": "#E69F00",     # orange
    "frozen_reference_scanvi": "#D55E00",   # vermillion
    "solar_none": "#009E73",                # bluish green (ablation)
    "solar_orthogonal_uniq": "#CC79A7",     # reddish purple (ablation)
}

MARKER = {
    "unintegrated_pca_matched": "D",
    "solar_orthogonal": "o",
    "frozen_reference_scvi": "s",
    "frozen_reference_scanvi": "^",
    "solar_none": "v",
    "solar_orthogonal_uniq": "P",
}

LINESTYLE = {
    "unintegrated_pca_matched": "none",
    "solar_orthogonal": "-",
    "frozen_reference_scvi": "--",
    "frozen_reference_scanvi": ":",
    "solar_none": "-.",
    "solar_orthogonal_uniq": (0, (1, 1)),
}

LABEL = {
    "unintegrated_pca_matched": "Matched PCA (unintegrated)",
    "solar_orthogonal": "SOLAR (repeated anchor, primary)",
    "frozen_reference_scvi": "Frozen-reference scVI",
    "frozen_reference_scanvi": "Frozen-reference scANVI",
    "solar_none": "SOLAR, no anchor",
    "solar_orthogonal_uniq": "SOLAR, unique anchor",
}

METHOD_ORDER = [
    "unintegrated_pca_matched",
    "solar_orthogonal",
    "frozen_reference_scvi",
    "frozen_reference_scanvi",
]

ABLATION_METHOD_ORDER = ["solar_orthogonal", "solar_none", "solar_orthogonal_uniq"]

DATASET_LABEL = {
    "immune_cell_human": "Immune",
    "lung_atlas": "Lung",
    "pancreas": "Pancreas",
}

REAL_DATASET_ORDER = ["immune_cell_human", "lung_atlas", "pancreas"]
