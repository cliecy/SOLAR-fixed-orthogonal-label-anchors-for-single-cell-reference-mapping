#!/usr/bin/env python3
"""Ledger rows for numbers in prose, captions, Tables 1-3 and figures.

Tables S1-S9 are extracted mechanically by extract_si_tables.py; everything
else in the two PDFs is listed here by hand, section by section, in reading
order. Run from the repository root:

    python tools/manual_claims.py

Kinds: summary/contrast (per-run aggregation), derived (counts and figure
labels from the data), static (hyperparameters and protocol constants in
configs/ and the source), assertion (qualitative statements tested on the
data), non_result (axis ticks, equation and interval notation; recorded so the
completeness check can account for every printed number).
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOLAR30 = ("solar_orthogonal", 30)
PCA30 = ("unintegrated_pca_matched", 30)
SCVI30 = ("frozen_reference_scvi", 30)
SCANVI30 = ("frozen_reference_scanvi", 30)
SCLSC30 = ("sclsc_refonly", 30)
REPEATED = ("solar_orthogonal", 128)
UNIQUE = ("solar_orthogonal_uniq", 128)
NONE = ("solar_none", 128)
SEEDS = [40, 41, 42, 43, 44]


class Ledger:
    def __init__(self, doc: str, prefix: str):
        self.doc, self.prefix, self.rows = doc, prefix, []

    def add(self, location, printed, kind, **spec):
        self.rows.append({"claim_id": f"{self.prefix}-{len(self.rows) + 1:04d}", "doc": self.doc,
                          "location": location, "printed": printed, "kind": kind,
                          "spec": json.dumps(spec, sort_keys=True, ensure_ascii=False)})

    def mean(self, loc, printed, rep, metric, population, stat="mean"):
        self.add(loc, printed, "summary", method=rep[0], dim=rep[1], metric=metric, population=population, stat=stat)

    def static(self, loc, printed, check, value, note=""):
        self.add(loc, printed, "static", check=check, value=value, note=note)

    def derived(self, loc, printed, check, value=None, fmt=None, **extra):
        spec = dict(check=check, **extra)
        if fmt:
            spec["format"] = fmt
        else:
            spec["value"] = value
        self.add(loc, printed, "derived", **spec)

    def claim(self, loc, printed, check):
        self.add(loc, printed, "assertion", check=check)

    def tick(self, loc, *printed, note="axis tick label"):
        for p in printed:
            self.add(loc, p, "non_result", note=note)

    def write(self, name):
        with (ROOT / "paper" / name).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(self.rows[0]))
            writer.writeheader()
            writer.writerows(self.rows)


def main_text() -> Ledger:
    L = Ledger("main", "main")
    a = "Abstract"
    L.derived(a, "nine", "n_real_query_batches", 9)
    L.mean(a, "0.811", SOLAR30, "macro_f1", "query")
    L.mean(a, "0.774", PCA30, "macro_f1", "query")
    L.mean(a, "0.815", SCLSC30, "macro_f1", "query")
    L.mean(a, "0.838", SCANVI30, "macro_f1", "query")
    L.static(a, "30", "common_dim", 30)
    L.mean(a, "0.574", PCA30, "ASW_label", "joint")
    L.mean(a, "0.873", SOLAR30, "ASW_label", "joint")
    L.static(a, "128", "anchor_control_dim", 128)
    L.claim(a, "repetition mainly increased label separation and local batch diversity", "repeated_increases_separation")
    L.claim(a, "lower local-structure scores", "repeated_lowers_local_structure")

    s = "3.1 Study design"
    L.tick(s, "1", note="equation-style dataset notation i=1..n_D")
    L.derived(s, "five", "n_datasets", 5)
    L.static(s, "80%", "reference_fraction", 0.8)
    L.static(s, "20%", "query_fraction", 0.2)
    L.static(s, "40–44", "split_seeds", SEEDS)
    L.claim(s, "A one-cell stratum remains in the reference", "singleton_stratum_reference")
    L.claim(s, "Batch identity, not its biological labels, determines this split", "heldout_labels_unused")
    L.derived(s, "Three", "n_heldout_batches_per_dataset", 3)
    L.claim(s, "selected to span query sizes", "batch_selection_span")
    L.derived(s, "nine", "n_real_query_batches", 9)
    L.claim(s, "All evaluated query labels occur in the corresponding reference", "closed_label")
    L.static(s, "40–44", "model_seeds", SEEDS)
    L.tick(s, "1", note="equation number (1)")

    s = "3.2 Preprocessing"
    L.static(s, "2,000", "n_hvg", 2000)
    L.static(s, "Cell Ranger-style", "hvg_flavor", "cell_ranger")
    L.static(s, "using reference batch as the batch key", "hvg_batch_aware", True)
    t = "Table 1"
    for name, ds, cells, labels in (("Human immune cells", "immune_cell_human", "33,506", "16"),
                                    ("Lung atlas", "lung_atlas", "32,472", "17"),
                                    ("Human pancreas", "pancreas", "16,382", "14"),
                                    ("Simulation 1", "simulation_1", "12,097", "7"),
                                    ("Simulation 2", "simulation_2", "19,318", "4")):
        L.derived(t, cells, "dataset_cells", int(cells.replace(",", "")), dataset=ds)
        L.derived(t, labels, "dataset_labels", int(labels), dataset=ds)
        L.derived(t, "3", "n_heldout_batches_per_dataset", 3, dataset=ds)
        if ds.startswith("simulation"):
            L.tick(t, ds[-1], note="dataset name 'Simulation N'")
    L.tick(t, "1", "1", note="caption cross-references to Table S1 / Additional file 1")
    L.derived(t, "nine", "n_real_query_batches", 9)
    L.tick(t, "1", "1", note="caption cross-references to Additional file 1 / Table S1")
    L.static(s, "40", "solar_pca_components", 40)
    L.static(s, "30-dimensional", "embedding_dim_common", 30)
    L.static(s, "128-dimensional", "embedding_dim_anchor", 128)
    L.static(s, "30-component", "pca30_components", 30)
    L.static(s, "40-component", "pca_dims", [30, 40])

    s = "3.3 Encoder and anchors"
    L.static(s, "three-layer", "n_linear_layers", 3)
    L.static(s, "two 512-unit hidden layers", "hidden_dim", 512)
    L.static(s, "two hidden layers", "n_hidden_layers", 2)
    L.claim(s, "batch normalization, ReLU, dropout, linear projection, layer normalization, final l2 normalization", "encoder_layout")
    L.static(s, "0.1", "dropout", 0.1)
    L.static(s, "30-dimensional", "embedding_dim_common", 30)
    L.static(s, "128-dimensional", "embedding_dim_anchor", 128)
    L.claim(s, "no reconstruction decoder, batch-adversarial component, or classification head", "no_decoder_or_head")
    L.claim(s, "d >= K; Gaussian G, reduced QR; fixed anchor bank never updated", "orthogonal_anchors")
    L.tick(s, "2", "2", "0", note="equation number (2), U^T U = I_K, u_k^T u_l = 0")

    s = "3.4 Objective"
    L.claim(s, "V = 2B repeated, B + K_B unique, B none", "anchor_pools")
    L.tick(s, "1", "1", "2", "3", note="Eq. (3) index ranges 1<=a<=B, B<a<=2B and equation number")
    L.tick(s, "1", note="{1, ..., V} in C(a)")
    L.claim(s, "Eqs. (4)-(5): cosine offset before temperature, index self-exclusion, all-view mean", "loss_definition")
    L.tick(s, "4", "1", "1", "0", note="equation number (4); 1[.] equals 1 when true and 0 otherwise")
    L.tick(s, "4", note="Eq. (4) cross-reference")
    L.static(s, "τ = 0.07", "temperature", 0.07)
    L.static(s, "m = 0.2", "margin", 0.2)
    L.tick(s, "3.5", note="section cross-reference")
    L.tick(s, "1", "1", "0", "0", "5", "5", note="Eq. (5): 1/V, 1/|Pos(a)|, |Pos(a)|>0, 0 case, equation number and cross-reference")

    s = "3.5 Training"
    L.static(s, "90%", "train_size", 0.9)
    L.static(s, "10%", "validation_fraction", 0.1)
    L.static(s, "10−4", "learning_rate", 0.0001)
    L.static(s, "512", "batch_size", 512)
    L.static(s, "0.07", "temperature", 0.07)
    L.static(s, "0.2", "margin", 0.2)
    L.static(s, "1.0", "gradient_clip_norm", 1.0)
    L.static(s, "zero weight decay", "weight_decay", 0.0)
    L.static(s, "10 epochs", "patience", 10)
    L.static(s, "10−4", "min_delta", 0.0001)
    L.static(s, "50 epochs", "max_epochs", 50)
    L.static(s, "40–44", "model_seeds", SEEDS)
    L.claim(s, "The best validation checkpoint is restored", "best_checkpoint")
    L.tick("Figure 1", "1", note="figure number")

    s = "3.6 Comparators"
    L.claim(s, "SCLSC: stratified 90:10 split, class representatives, final epoch retained", "sclsc_final_epoch")
    L.static(s, "90:10", "sclsc_validation_fraction", 0.1)
    L.claim(s, "normalized expression matrix with reference-only selection of 2,000 genes", "sclsc_normalized_input")
    L.static(s, "2,000", "n_hvg", 2000)
    L.tick(s, "1", note="Additional file 1")
    L.static(s, "30-dimensional", "common_dim", 30)
    L.derived(s, "five fits per held-out batch (SOLAR)", "fits_per_batch", [5], method=SOLAR30[0], dim=30)
    L.derived(s, "five fits per held-out batch (SCLSC)", "fits_per_batch", [5], method=SCLSC30[0], dim=30)
    L.derived(s, "one deterministic fit", "fits_per_batch", [1], method=PCA30[0], dim=30)
    L.static(s, "30-dimensional frozen-reference encoders", "refmap_dims", 30)
    L.static(s, "15-nearest-neighbour", "knn_neighbors", 15)
    L.static(s, "40", "solar_pca_components", 40)
    L.tick(s, "1", "1", "1", "1", note="Additional file 1; F1 in macro-F1")
    L.static(s, "128-dimensional", "anchor_control_dim", 128)
    L.tick(s, "1", note="Additional file 1")

    s = "3.7 Metrics"
    L.tick(s, "1", "1", "2", "2", "1", "0", "1", note="macro-F1 formula 2TP/(2TP+FP+FN), |C_Q|^-1, [0, 1]")
    L.tick(s, "1", "2", "0", "1", "1", "1", "0", "1", note="(s+1)/2 in [0,1]; batch ASW 1-|s|; [0,1]")
    L.static(s, "15-neighbour graph", "graph_neighbors", 15)
    L.static(s, "90 cells", "lisi_k0", 90)
    L.static(s, "50%", "lisi_subsample_percent", 50)
    L.tick(s, "1", "2", "1", "0", "1", "1", note="median(1/sum p^2) - 1; nominal range [0, n_b - 1]; Additional file 1")
    L.static(s, "50-dimensional", "trust_source_components", 50)
    L.tick(s, "40", "30", note="target labels Matched PCA (40D) and (30D); checked via pca_dims")
    L.static(s, "k = 15", "trust_neighbors", 15)
    L.tick(s, "0", "1", note="range [0, 1]")
    L.static(s, "6,000", "trust_max_cells", 6000)
    L.tick(s, "1", note="Additional file 1")
    L.static(s, "15 neighbouring cells", "pseudotime_neighbors", 15)
    L.static(s, "At least 16 annotated cells", "pseudotime_min_cells", 16)
    L.tick(s, "1", "1", "1", "1", "0", "1", "1", "1", note="range [-1, 1]; Additional file 1; 0 to 1; [-1, 1]")
    L.derived(s, "three evaluated batches", "n_heldout_batches_per_dataset", 3)
    L.derived(s, "nine batch means", "n_real_query_batches", 9)
    L.derived(s, "nine means", "n_real_query_batches", 9)
    L.static(s, "five seeds", "required_seed_count", 5)

    s = "4.1 Results"
    L.static(s, "30-dimensional", "common_dim", 30)
    L.mean(s, "0.811", SOLAR30, "macro_f1", "query")
    L.mean(s, "0.837", SOLAR30, "balanced_accuracy", "query")
    L.derived(s, "nine", "n_real_query_batches", 9)
    L.tick(s, "2", note="Table 2")
    L.mean(s, "0.774", PCA30, "macro_f1", "query")
    L.mean(s, "0.803", PCA30, "balanced_accuracy", "query")
    L.claim(s, "The gain in macro-F1 was positive on average within each of the three datasets", "positive_within_each_dataset")
    L.tick(s, "2", "2", "4", "3", "1", note="Figure 2A/2B, Figure S4, Table S3, Additional file 1")
    L.mean(s, "0.815", SCLSC30, "macro_f1", "query")
    L.mean(s, "0.838", SCANVI30, "macro_f1", "query")
    L.mean(s, "0.882", SCANVI30, "balanced_accuracy", "query")
    L.claim(s, "scANVI produced higher macro-F1 and balanced accuracy than SOLAR", "scanvi_higher")
    L.claim(s, "Villani errors concentrated in CD16+ monocytes", "villani_cd16")
    L.claim(s, "despite complete reference-label coverage", "closed_label")
    L.tick(s, "16", "2", "1", note="CD16+ label text, Figure S2, Additional file 1")

    f = "Figure 2"
    for ds, printed in (("immune_cell_human", "+0.033"), ("lung_atlas", "+0.029"), ("pancreas", "+0.050")):
        L.derived(f + "A", printed, "fig2a", fmt="signed3", dataset=ds)
    L.tick(f + "A", "0", "0.02", "0.04", "0.06")
    for rep, printed in ((SCANVI30, "0.838"), (SCLSC30, "0.815"), (SOLAR30, "0.811"), (SCVI30, "0.807"), (PCA30, "0.774")):
        L.mean(f + "B", printed, rep, "macro_f1", "query")
    L.claim(f + "B", "method order scANVI > SCLSC > SOLAR > scVI > Matched PCA", "fig2b_order")
    L.tick(f + "B", "0", "0.5", "1.0")
    L.tick(f, "2", note="figure number")
    L.static(f, "30-dimensional", "common_dim", 30)
    L.static(f, "15-nearest-neighbour", "knn_neighbors", 15)
    L.derived(f, "five fits per batch", "fits_per_batch", [5], method=SOLAR30[0], dim=30)
    L.tick(f, "4", "3", "1", note="Figure S4, Table S3, Additional file 1")

    t = "Table 2"
    for rep, f1, f1sd, ba, basd in ((SOLAR30, "0.811", "0.114", "0.837", "0.082"),
                                    (PCA30, "0.774", "0.127", "0.803", "0.096"),
                                    (SCVI30, "0.807", "0.091", "0.855", "0.053"),
                                    (SCANVI30, "0.838", "0.087", "0.882", "0.061"),
                                    (SCLSC30, "0.815", "0.078", "0.845", "0.054")):
        L.mean(t, f1, rep, "macro_f1", "query")
        L.mean(t, f1sd, rep, "macro_f1", "query", stat="sd")
        L.mean(t, ba, rep, "balanced_accuracy", "query")
        L.mean(t, basd, rep, "balanced_accuracy", "query", stat="sd")
    L.tick(t, "2", note="table number; F1 in Macro-F1 header")
    L.tick(t, "1", note="F1 in Macro-F1 header")
    L.static(t, "30-dimensional", "common_dim", 30)
    L.static(t, "15-nearest-neighbour", "knn_neighbors", 15)
    L.derived(t, "nine batch means", "n_real_query_batches", 9)
    L.derived(t, "five fits", "fits_per_batch", [5], method=SOLAR30[0], dim=30)
    L.tick(t, "3", "1", note="Table S3, Additional file 1")

    s = "4.2 Results"
    L.mean(s, "0.873", SOLAR30, "ASW_label", "joint")
    L.mean(s, "0.574", PCA30, "ASW_label", "joint")
    L.mean(s, "0.578", SCVI30, "ASW_label", "joint")
    L.mean(s, "0.657", SCANVI30, "ASW_label", "joint")
    L.tick(s, "3", note="Table 3")
    L.mean(s, "0.948", SCLSC30, "ASW_label", "joint")
    L.claim(s, "SOLAR label ASW above Matched PCA, scVI, scANVI but not SCLSC", "label_asw_order")
    L.mean(s, "0.836", SOLAR30, "ASW_label/batch", "joint")
    L.mean(s, "0.743", SCLSC30, "ASW_label/batch", "joint")
    L.mean(s, "2.241", SCLSC30, "iLISI", "joint")
    L.mean(s, "1.740", SOLAR30, "iLISI", "joint")
    L.claim(s, "scVI and scANVI (and SCLSC) had higher iLISI than SOLAR", "ilisi_refmap_higher")

    t = "Table 3"
    L.tick(t, "3", note="table number")
    L.static(t, "30-dimensional", "common_dim", 30)
    L.derived(t, "nine equally weighted batch means", "n_real_query_batches", 9)
    L.derived(t, "five fits", "fits_per_batch", [5], method=SOLAR30[0], dim=30)
    for rep, asw, basw, ilisi in ((SOLAR30, "0.873", "0.836", "1.740"), (PCA30, "0.574", "0.831", "0.304"),
                                  (SCVI30, "0.578", "0.878", "2.125"), (SCANVI30, "0.657", "0.867", "1.902"),
                                  (SCLSC30, "0.948", "0.743", "2.241")):
        L.mean(t, asw, rep, "ASW_label", "joint")
        L.mean(t, basw, rep, "ASW_label/batch", "joint")
        L.mean(t, ilisi, rep, "iLISI", "joint")

    s = "4.3 Results"
    L.static(s, "128 dimensions", "anchor_control_dim", 128)
    L.claim(s, "repeated anchors increased held-out label ASW and iLISI relative to both controls", "repeated_increases_separation")
    L.tick(s, "3", note="Figure 3")
    L.add(s, "0.007", "contrast", left=list(REPEATED), right=list(NONE), metric="macro_f1", population="query", stat="delta")
    L.claim(s, "0.001", "repeated_f1_vs_unique_below_0001")
    L.claim(s, "balanced accuracy was slightly lower than with unique anchors", "repeated_ba_below_unique")
    L.claim(s, "reduced trustworthiness and immune-cell pseudotime smoothness relative to both controls",
            "repeated_lowers_local_structure")
    L.tick(s, "9", "3", "1", note="Table S9, Figure S3, Additional file 1")

    f = "Figure 3"
    L.tick(f + "A", "0.92", "0.88", "0.84", "0.80")
    L.tick(f + "B", "1.50", "1.25", "1.00", "0.75", "0.50")
    L.tick(f + "C", "0.90", "0.85", "0.80", "0.75")
    L.tick(f + "D", "0.90", "0.85", "0.80", "0.75")
    L.tick(f, "3", "1", note="figure number; F1 in macro-F1 axis title")
    L.static(f, "128 dimensions", "anchor_control_dim", 128)
    L.derived(f, "three held-out batch means", "n_heldout_batches_per_dataset", 3)
    L.derived(f, "five fits", "fits_per_batch", [5], method=REPEATED[0], dim=128)
    L.tick(f, "9", "1", "3", note="Table S9, Additional file 1, Figure S3")
    L.claim(f, "10 metrics", "fig_s3_ten_panels")

    s = "4.4 Results"
    L.derived(s, "six immune and lung held-out batches", "n_trust_batches", 6)
    L.mean(s, "0.936", SOLAR30, "trustworthiness", "query")
    L.mean(s, "0.903", SCLSC30, "trustworthiness", "query")
    L.claim(s, "Every batch mean favoured SOLAR", "trust_every_batch_favours_solar")
    L.tick(s, "5", "7", "1", note="Tables S5 and S7, Additional file 1")
    L.derived(s, "three immune batches", "n_pseudotime_batches", 3)
    L.mean(s, "0.861", SOLAR30, "pseudotime_smoothness", "joint")
    L.mean(s, "0.818", SCLSC30, "pseudotime_smoothness", "joint")
    L.claim(s, "same direction in all three batch means", "smoothness_every_batch_favours_solar")
    for rep, trust, smooth in ((PCA30, "0.969", "0.894"), (SCVI30, "0.946", "0.880"), (SCANVI30, "0.949", "0.886")):
        L.mean(s, trust, rep, "trustworthiness", "query")
        L.mean(s, smooth, rep, "pseudotime_smoothness", "joint")
    L.claim(s, "Matched PCA, scVI and scANVI retained higher query trustworthiness and joint smoothness than SOLAR",
            "refmap_higher_local_structure")
    L.claim(s, "Query-only pseudotime smoothness was evaluable in just one immune batch, Oetjen_A", "query_smoothness_only_oetjen")
    L.tick(s, "4", note="Figure 4")
    L.derived(s, "same six immune/lung batches", "n_trust_batches", 6)
    L.derived(s, "same three immune batches", "n_pseudotime_batches", 3)
    L.tick(s, "5", "5", "1", note="Figure S5, Table S5, Additional file 1")

    f = "Figure 4"
    L.tick(f + "A", "0.98", "0.96", "0.94", "0.92", "0.90", "0.5", "0.6", "0.7", "0.8", "0.9", "1.0")
    L.tick(f + "B", "0.90", "0.88", "0.86", "0.84", "0.82", "0.5", "0.6", "0.7", "0.8", "0.9", "1.0")
    L.tick(f, "4", note="figure number")
    L.static(f, "30-dimensional", "common_dim", 30)
    L.derived(f, "six immune and lung batches", "n_trust_batches", 6)
    L.derived(f, "three immune batches", "n_pseudotime_batches", 3)
    L.static(f, "50-dimensional", "trust_source_components", 50)
    L.tick(f, "5", "7", "1", note="Tables S5 and S7, Additional file 1")

    s = "5 Discussion"
    L.static(s, "30-dimensional", "common_dim", 30)
    L.tick(s, "3", "9", "1", note="Figure 3, Table S9, Additional file 1")
    L.derived(s, "nine held-out batches", "n_real_query_batches", 9)
    L.derived(s, "three datasets", "n_heldout_batches_per_dataset", 3)
    L.static(s, "50-dimensional", "trust_source_components", 50)
    return L


def si_text() -> Ledger:
    L = Ledger("si", "sitext")
    s = "SI 2.1"
    L.static(s, "40–44", "model_seeds", SEEDS)
    L.static(s, "80%", "reference_fraction", 0.8)
    L.static(s, "20%", "query_fraction", 0.2)
    L.static(s, "40–44", "split_seeds", SEEDS)
    L.claim(s, "a one-cell stratum remains in the reference", "singleton_stratum_reference")
    L.claim(s, "Every evaluated query label occurs in the corresponding reference", "closed_label")
    L.static(s, "2,000-gene", "n_hvg", 2000)
    L.static(s, "30-dimensional", "common_dim", 30)
    L.static(s, "128 / 40 / 30 / 30 / 16-dimensional method-specific outputs", "method_specific_dims",
             [[30, 128], [30, 40], [30], [30], [16, 30]])
    L.static(s, "40 reference-fitted principal components", "solar_pca_components", 40)
    L.static(s, "30-dimensional scVI and scANVI", "refmap_dims", 30)
    L.static(s, "PCA50", "trust_source_components", 50)

    s = "SI 2.2"
    for printed, check, value in (("512-unit", "hidden_dim", 512), ("0.1", "dropout", 0.1),
                                  ("10−4", "learning_rate", 0.0001), ("512", "batch_size", 512),
                                  ("0.07", "temperature", 0.07), ("0.2", "margin", 0.2),
                                  ("1.0", "gradient_clip_norm", 1.0), ("zero weight decay", "weight_decay", 0.0),
                                  ("10 epochs", "patience", 10), ("10−4", "min_delta", 0.0001),
                                  ("50 epochs", "max_epochs", 50)):
        L.static(s, printed, check, value)
    L.claim(s, "restores the best validation-loss checkpoint", "best_checkpoint")
    L.claim(s, "repeated / unique / no-anchor pools", "anchor_pools")
    L.static(s, "30-, 64-, and 128-dimensional", "within_batch_dims", [30, 64, 128])
    L.tick(s, "4", "8", "9", note="Tables S4, S8, S9")
    L.static(s, "128-dimensional anchor comparisons", "anchor_control_dim", 128)
    a = "Algorithm 1"
    L.tick(a, "1", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15", "16", "17",
           "18", "19", "20", "21", "22", note="algorithm number and line numbers")
    L.claim(a, "V = 2B, B + K_B, or B", "anchor_pools")
    L.tick(a, "2", "1", note="V = 2B; C(a) = {1, ..., V}")
    L.static(a, "10 epochs", "patience", 10)
    L.static(s, "τ/τbase is one because both temperatures are 0.07 (τ)", "temperature", 0.07)
    L.static(s, "τ/τbase is one because both temperatures are 0.07 (τbase)", "base_temperature", 0.07)

    s = "SI 2.3"
    L.static(s, "16-dimensional / separately trained 30-dimensional", "sclsc_dims", [16, 30])
    L.claim(s, "class representatives are arithmetic means; final stopped epoch retained", "sclsc_final_epoch")
    L.static(s, "90:10", "sclsc_validation_fraction", 0.1)
    L.derived(s, "five fits per held-out batch", "fits_per_batch", [5], method=SCLSC30[0], dim=30)
    L.derived(s, "five fits per held-out batch (16D)", "fits_per_batch", [5], method="sclsc_refonly", dim=16)
    L.static(s, "early-stopping patience of 10", "sclsc_patience", 10)
    L.claim(s, "normalized expression matrix and the shared reference-selected 2,000 genes", "sclsc_normalized_input")
    L.static(s, "2,000 genes", "n_hvg", 2000)
    L.static(s, "15-nearest-neighbour", "knn_neighbors", 15)
    L.static(s, "30-dimensional encoders", "refmap_dims", 30)
    L.tick(s, "2", note="Table S2")

    s = "SI 2.4"
    L.static(s, "15 nearest neighbours", "knn_neighbors", 15)
    L.static(s, "Minkowski p = 2", "knn_minkowski_p", 2)
    L.tick(s, "1", "0", "1", note="macro-F1; range 0 to 1")
    L.tick(s, "1", note="F1 in class-specific F1")
    L.static(s, "e2a37e0ed63dc34b60aa535cc656400552af757a", "scib_commit", "e2a37e0ed63dc34b60aa535cc656400552af757a")
    L.tick(s, "4", note="citation [4]")
    L.tick(s, "1", "2", "0", "1", "1", "0", "1", "1", "0", "1", "1", "1",
           note="(s+1)/2 in [0,1]; 1-|s_i|; 0 to 1; median minus 1; [0,1]; [0, B_joint - 1]")
    L.static(s, "50% graph-LISI subsampling", "lisi_subsample_percent", 50)
    L.tick(s, "5", note="citation [5]")
    L.static(s, "k = 15", "trust_neighbors", 15)
    L.tick(s, "0", "1", note="range [0, 1]")
    L.static(s, "50-component PCA", "trust_source_components", 50)
    L.static(s, "6,000 cell IDs", "trust_max_cells", 6000)
    L.static(s, "sampling seed 40", "trust_sample_seed", 40)
    L.static(s, "15 neighbours", "pseudotime_neighbors", 15)
    L.static(s, "requests 16 neighbours", "pseudotime_requested_neighbors", 16)
    L.tick(s, "1", "1", note="range [-1, 1]")
    L.static(s, "fewer than 16 annotated cells", "pseudotime_min_cells", 16)
    L.tick(s, "1", "1", "0", "1", note="(rho+1)/2 and [0,1]")
    L.static(s, "at least 10 cells", "silhouette_min_cells", 10)
    L.static(s, "8,000 cells", "silhouette_cap", 8000)
    L.static(s, "seed-0", "silhouette_seed", 0)
    L.tick(s, "1", "1", note="range -1 to 1")
    L.static(s, "five finite scores", "required_seed_count", 5)
    L.static(s, "seeds 40–44", "model_seeds", SEEDS)
    L.static(s, "divisor 5 − 1", "sd_ddof", 1)
    L.tick(s, "5", note="divisor 5 - 1 (the 5)")
    L.tick(s, "3", note="Table S3")
    L.derived(s, "nine batch means", "n_real_query_batches", 9)
    L.tick(s, "7", "9", note="Tables S7-S9")

    t = "Table S2"
    L.static(t, "Public SOLAR v1.2", "package_version", "1.2")
    L.static(t, "128 / 30 (SOLAR)", "solar_dims", [30, 128])
    L.static(t, "40 / 30 (Matched PCA)", "pca_dims", [30, 40])
    L.static(t, "30 (scVI, scANVI)", "refmap_dims", 30)
    L.static(t, "16 / 30 (SCLSC)", "sclsc_dims", [16, 30])
    for printed, rep in (("5", SOLAR30), ("1", PCA30), ("5", SCVI30), ("5", SCANVI30), ("5", SCLSC30)):
        L.derived(t, printed, "fits_per_batch", [int(printed)], method=rep[0], dim=rep[1])
    L.tick(t, "90", "10", note="internal 90:10 split (checked as sclsc_validation_fraction)")

    for t, notes in (("Table S1", ["table number"]), ("Table S3", []), ("Table S4", []), ("Table S5", []),
                     ("Table S6", []), ("Table S7", []), ("Table S8", []), ("Table S9", [])):
        L.tick(t, t[-1], note="table number")
    L.derived("Table S3", "All 90 held-out annotation rows", "n_table_s3_rows", 90)
    L.static("Table S3", "30-dimensional", "common_dim", 30)
    L.derived("Table S3", "five fits", "fits_per_batch", [5], method=SOLAR30[0], dim=30)
    L.static("Table S4", "five split seeds", "split_seeds", SEEDS)
    L.static("Table S5", "30-dimensional / 128-dimensional controls", "anchor_control_dim", 128)
    L.claim("Table S5", "P: fewer than 16 annotated query cells; only immune Oetjen_A contributes", "query_smoothness_only_oetjen")
    L.static("Table S6", "30-dimensional / controls are 128-dimensional", "anchor_control_dim", 128)
    L.claim("Table S7", "HVG overlap is disabled for embedding outputs", "hvg_overlap_disabled")
    L.static("Table S9", "128-dimensional", "anchor_control_dim", 128)
    L.static("Table S9", "Full paired 128-dimensional anchor contrasts", "embedding_dim_anchor", 128)
    return L


def figures() -> Ledger:
    L = Ledger("si", "fig")
    f = "Figure S1"
    L.tick(f, "1", note="figure number")
    L.tick(f, "0.35", note="plot styling (point alpha); not a result")
    L.tick(f, "2", "3", note="panel titles inDrop 2 / inDrop 3")
    L.tick(f, "1", note="panel title Fluidigm C1")
    L.tick(f, "10", "3", "4", "1", note="panel titles 10X, Lung / 3, Lung / 4, B1")
    L.tick(f, "128", note="128-dimensional repeated-anchor SOLAR; UMAP from model-attachment embeddings, not recomputed")
    L.tick(f, "40", note="seed 40 of the released fits shown; UMAP not recomputed from the released tables")
    L.claim(f, "nine real held-out batches; joint reference and query cells", "fig_s1_coordinates_match_splits")

    f = "Figure S2"
    labels = ["CD14+ Monocytes", "CD16+ Monocytes", "Monocyte-derived dendritic cells", "Plasmacytoid dendritic cells"]
    matrix = [["0.99", "0.01", "0.00", "0.00"], ["0.97", "0.03", "0.00", "0.00"],
              ["0.30", "0.18", "0.53", "0.00"], ["0.00", "0.00", "0.00", "1.00"]]
    for true, row in zip(labels, matrix):
        for pred, printed in zip(labels, row):
            L.add(f + "B", printed, "confusion", true=true, pred=pred)
    L.tick(f + "A", "1.0", "0.8", "0.6", "0.4", "0.2", "0.0")
    L.tick(f + "B", "1.0", "0.8", "0.6", "0.4", "0.2", "0.0", note="colour-bar tick")
    L.derived(f + "A", "5 seeds", "fits_per_batch", [5], method=REPEATED[0], dim=128)
    L.tick(f, "14", "16", "14", "16", "14", "16", note="CD14+/CD16+ label text in axes")
    L.tick(f, "2", note="figure number")
    L.tick(f, "1", note="F1 in per-class query F1")
    L.static(f, "128-dimensional repeated-anchor SOLAR", "anchor_control_dim", 128)
    L.static(f, "128-dimensional repeated-anchor SOLAR (caption)", "embedding_dim_anchor", 128)
    L.tick(f, "16", note="CD16+ in caption")
    L.claim(f, "errors concentrated in CD16+ monocytes", "villani_cd16")
    L.claim(f, "per-class F1 built from the same predictions as the reported Villani macro-F1", "villani_macro_f1_matches")

    for figure, panel_letters in (("S3", "ABCDEFGHIJ"), ("S5", "ABCDEFG")):
        f = f"Figure {figure}"
        for letter in panel_letters:
            L.derived(f + letter, None, "fig_axis", fmt="g3", figure=figure, panel=letter, end="low")
            L.derived(f + letter, None, "fig_axis", fmt="g3", figure=figure, panel=letter, end="high")
            L.derived(f + letter, None, "fig_axis", fmt="g3", figure=figure, panel=letter, end="n")
            L.tick(f + letter, "0", note="zero reference line label")
    # Printed axis end labels and coverage labels, in panel order.
    printed = {
        "S3": [("-0.0223", "0.0315", "n=9/9"), ("-0.0238", "0.021", "n=9/9"), ("-0.00528", "0.0383", "n=9/9"),
               ("-0.00811", "0.0204", "n=9/9"), ("-0.0149", "0.0463", "n=9/9"), ("-0.0129", "0.0728", "n=9/9"),
               ("-0.12", "0.868", "n=9/9"), ("-0.0424", "0.00584", "n=9/9"), ("-0.0974", "0.0162", "n=9/9"),
               ("-0.3", "0.0414", "n=3/3")],
        "S5": [("-0.0291", "0.0153", "n=9/9"), ("-0.0298", "0.0205", "n=9/9"), ("-0.00558", "0.00768", "n=9/9"),
               ("-0.365", "1.01", "n=9/9"), ("-0.0141", "0.0253", "n=9/9"), ("-0.0107", "0.00352", "n=6/6"),
               ("-0.0169", "0.0113", "n=3/3")],
    }
    k = {"S3": 0, "S5": 0}
    for row in L.rows:
        spec = json.loads(row["spec"])
        if spec.get("check") == "fig_axis":
            figure, idx = spec["figure"], "ABCDEFGHIJ".index(spec["panel"])
            row["printed"] = printed[figure][idx][("low", "high", "n").index(spec["end"])]
    f = "Figure S3"
    L.tick(f, "10", "3", "1", "1", "1", "1", "1", "3", "3", "4", "4",
           note="batch labels 10X, Lung: 3/4/B1, Oetjen_A etc.; repeated per panel")
    L.tick(f, "3", note="figure number")
    L.static(f, "128-dimensional", "anchor_control_dim", 128)
    L.static(f, "128-dimensional (caption)", "embedding_dim_anchor", 128)
    L.derived(f, "five fits", "fits_per_batch", [5], method=REPEATED[0], dim=128)
    L.tick(f, "1", note="F1 in macro-F1")
    L.derived(f, "three immune batches", "n_pseudotime_batches", 3)
    L.derived(f, "nine held-out batches", "n_real_query_batches", 9)
    L.tick(f, "9", note="Table S9")
    f = "Figure S4"
    L.tick(f + "A", "1.0", "0.8", "0.6", "0.4", "0.2", "0.0")
    L.tick(f + "B", "1.0", "0.8", "0.6", "0.4", "0.2", "0.0")
    L.tick(f, "10", "3", "4", "1", "1", "2", "3", note="x labels 10X, 3, 4, B1, C1, inDrop2/3")
    L.static(f, "30D (legend)", "common_dim", 30)
    L.derived(f, "5 seeds", "fits_per_batch", [5], method=SOLAR30[0], dim=30)
    L.tick(f, "4", "1", note="figure number; F1 in macro-F1")
    L.static(f, "30-dimensional", "common_dim", 30)
    L.static(f, "its 30-dimensional output", "embedding_dim_common", 30)
    L.derived(f, "nine held-out batches", "n_real_query_batches", 9)
    L.static(f, "15-nearest-neighbour", "knn_neighbors", 15)
    L.static(f, "40 reference-fitted principal components", "solar_pca_components", 40)
    L.tick(f, "3", "7", note="Tables S3 and S7")
    f = "Figure S5"
    L.static(f, "SOLAR30 / SOLAR128 / SCLSC30 / SCLSC16", "method_specific_dims",
             [[30, 128], [30, 40], [30], [30], [16, 30]])
    L.tick(f, "10", "1", note="batch labels; F1 in macro-F1; repeated per panel")
    L.tick(f, "5", note="figure number")
    L.static(f, "30-dimensional / 128-dimensional", "solar_dims", [30, 128])
    L.static(f, "SOLAR with 30-dimensional outputs (caption)", "embedding_dim_common", 30)
    L.static(f, "SCLSC with 30-dimensional outputs (caption)", "sclsc_dims", [16, 30])
    L.static(f, "SCLSC with 30-dimensional outputs (caption, second mention)", "sclsc_dims", [16, 30])
    L.static(f, "30-dimensional / 16-dimensional", "sclsc_dims", [16, 30])
    L.derived(f, "five fits", "fits_per_batch", [5], method=SOLAR30[0], dim=30)
    L.tick(f, "1", note="F1 in macro-F1")
    L.derived(f, "six immune/lung batches", "n_trust_batches", 6)
    L.derived(f, "three immune batches", "n_pseudotime_batches", 3)
    L.derived(f, "all nine", "n_real_query_batches", 9)
    L.tick(f, "8", note="Table S8")
    return L


if __name__ == "__main__":
    main_text().write("claims_main.csv")
    si_text().write("claims_si_text.csv")
    figures().write("claims_figures.csv")
