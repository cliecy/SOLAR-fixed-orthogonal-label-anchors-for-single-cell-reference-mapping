"""Checks for manuscript statements that are not table cells.

Three ledger kinds are handled here:

* ``static``: a protocol constant or hyperparameter. The ledger stores the
  canonical value represented by the printed text (``value``); the check reads
  the corresponding setting from ``configs/`` or the source code and compares.
* ``derived``: a count or number obtained from the per-run data (for example
  the number of held-out batches or a per-dataset mean difference in a figure).
* ``assertion``: a qualitative statement about the data ("every batch mean
  favoured SOLAR", "errors concentrated in CD16+ monocytes").

Each check returns the printed text when it holds, or a short description of
what was found instead.
"""
from __future__ import annotations

import ast
import math
import re
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

import paper_numbers as pn

ROOT = pn.ROOT
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
SOLAR30 = ("solar_orthogonal", 30)
PCA30 = ("unintegrated_pca_matched", 30)
SCVI30 = ("frozen_reference_scvi", 30)
SCANVI30 = ("frozen_reference_scanvi", 30)
SCLSC30 = ("sclsc_refonly", 30)
REPEATED = ("solar_orthogonal", 128)
UNIQUE = ("solar_orthogonal_uniq", 128)
NONE = ("solar_none", 128)


@lru_cache(maxsize=None)
def config(name: str) -> dict:
    return yaml.safe_load((ROOT / "configs" / name).read_text())


@lru_cache(maxsize=None)
def source(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def dataclass_default(path: str, cls: str, field: str):
    """Default value of `field` in dataclass `cls` defined in `path`."""
    tree = ast.parse(source(path))
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == cls:
            for item in node.body:
                if isinstance(item, ast.AnnAssign) and getattr(item.target, "id", None) == field:
                    return ast.literal_eval(item.value)
    raise KeyError(f"{cls}.{field} not found in {path}")


def function_default(path: str, function: str, argument: str):
    tree = ast.parse(source(path))
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == function:
            args = node.args.args + node.args.kwonlyargs
            defaults = [None] * (len(node.args.args) - len(node.args.defaults)) + node.args.defaults + node.args.kw_defaults
            for arg, default in zip(args, defaults):
                if arg.arg == argument and default is not None:
                    return ast.literal_eval(default)
    raise KeyError(f"{function}({argument}=...) not found in {path}")


def contains(path: str, *patterns: str) -> bool:
    text = source(path)
    return all(re.search(pattern, text) for pattern in patterns)


def train_setting(name: str) -> set:
    """A training setting in every SOLAR configuration that produced reported results."""
    values = {
        config("benchmark_rebuttal_v1.yaml")["solar"]["train"][name],       # 128D held-out and within-batch
        config("benchmark_rebuttal_dim30.yaml")["solar"]["train"][name] if name != "embedding_dim" else 30,
        config("benchmark_rebuttal_dim64.yaml")["solar"]["train"][name] if name != "embedding_dim" else 64,
        config("benchmark_v1_2_dim30.yaml")["solar"]["train"][name],        # 30D held-out comparison
    }
    if name != "embedding_dim":
        values.add(dataclass_default("src/SOLAR/benchmark/adapter.py", "TrainConfig", name))
    return values


def runs_frame() -> pd.DataFrame:
    return pn.runs()


def cohort_dims(method: str) -> set[int]:
    frame = runs_frame()
    return set(frame.loc[frame.method == method, "dimension"].astype(int))


def fits_per_batch(method: str, dim: int) -> set[int]:
    frame = runs_frame()
    frame = frame[(frame.method == method) & (frame.dimension == dim)]
    return set(frame.groupby(["dataset_id", "heldout_batch"]).size())


def seeds_of(method: str, dim: int) -> tuple[int, ...]:
    frame = runs_frame()
    return tuple(sorted(set(frame.loc[(frame.method == method) & (frame.dimension == dim), "seed"].astype(int))))


# ---------------------------------------------------------------- static ----
STATIC = {
    # Protocols and preprocessing.
    "reference_fraction": lambda: {config("benchmark_rebuttal_v1.yaml")["protocol"]["reference_fraction"],
                                   function_default("src/scib_benchmark/splits.py", "stratified_reference_query", "reference_fraction")},
    "query_fraction": lambda: {round(1 - config("benchmark_rebuttal_v1.yaml")["protocol"]["reference_fraction"], 10)},
    "split_seeds": lambda: {tuple(config("benchmark_rebuttal_v1.yaml")["protocol"]["split_seeds"]),
                            tuple(sorted(pn.within_batch_width().split_seed.astype(int).unique()))},
    "model_seeds": lambda: {tuple(config("experiment_v1_2.yaml")["seeds"]),
                            seeds_of(*SOLAR30), seeds_of(*REPEATED), seeds_of(*UNIQUE), seeds_of(*NONE),
                            seeds_of(*SCLSC30), seeds_of("sclsc_refonly", 16), seeds_of(*SCVI30), seeds_of(*SCANVI30)},
    "n_hvg": lambda: {config("experiment_v1_2.yaml")["preprocessing"]["hvg_count"],
                      config("benchmark_rebuttal_v1.yaml")["preprocessing"]["hvg2000_pca40"]["n_hvg"]},
    "hvg_flavor": lambda: {config("benchmark_rebuttal_v1.yaml")["preprocessing"]["hvg2000_pca40"]["hvg_flavor"],
                           function_default("src/scib_benchmark/data.py", "select_reference_hvgs", "flavor")},
    "hvg_batch_aware": lambda: {config("benchmark_rebuttal_v1.yaml")["preprocessing"]["hvg2000_pca40"]["batch_aware"]},
    "solar_pca_components": lambda: {config("experiment_v1_2.yaml")["preprocessing"]["solar_pca_components"],
                                     config("benchmark_rebuttal_v1.yaml")["preprocessing"]["hvg2000_pca40"]["n_components"],
                                     dataclass_default("src/SOLAR/benchmark/adapter.py", "PreprocessConfig", "n_components")},
    "common_dim": lambda: {d for m in ("solar_orthogonal", "unintegrated_pca_matched", "frozen_reference_scvi",
                                       "frozen_reference_scanvi", "sclsc_refonly") for d in cohort_dims(m)} & {30},
    "anchor_control_dim": lambda: cohort_dims("solar_orthogonal_uniq") | cohort_dims("solar_none"),
    "pca30_components": lambda: {30} if contains("scripts/score_experiment_v1_2.py", r"n_components=30") else set(),
    "method_specific_dims": lambda: {tuple(sorted(cohort_dims(m))) for m in (
        "solar_orthogonal", "unintegrated_pca_matched", "frozen_reference_scvi", "frozen_reference_scanvi", "sclsc_refonly")},
    "pca_dims": lambda: {tuple(sorted(cohort_dims("unintegrated_pca_matched")))},
    "solar_dims": lambda: {tuple(sorted(cohort_dims("solar_orthogonal")))},
    "refmap_dims": lambda: cohort_dims("frozen_reference_scvi") | cohort_dims("frozen_reference_scanvi"),
    "within_batch_dims": lambda: {tuple(sorted(pn.within_batch_width().dimension.astype(int).unique()))},
    # Encoder and anchors.
    "hidden_dim": lambda: train_setting("hidden_dim") | {dataclass_default("src/SOLAR/api.py", "ModelConfig", "hidden_dim")},
    "n_hidden_layers": lambda: {len(re.findall(r"nn\.Linear\(hidden_dim, hidden_dim\)|nn\.Linear\(input_dim, hidden_dim\)",
                                               source("src/SOLAR/src/anchor_model.py")))},
    "n_linear_layers": lambda: {len(re.findall(r"nn\.Linear\(", _encoder_source()))},
    "dropout": lambda: train_setting("dropout") | {dataclass_default("src/SOLAR/api.py", "ModelConfig", "dropout")},
    "embedding_dim_common": lambda: {config("benchmark_v1_2_dim30.yaml")["solar"]["train"]["embedding_dim"]} | cohort_dims("solar_orthogonal") & {30},
    "embedding_dim_anchor": lambda: {config("benchmark_rebuttal_v1.yaml")["solar"]["train"]["embedding_dim"]},
    "temperature": lambda: train_setting("temperature") | {dataclass_default("src/SOLAR/api.py", "ModelConfig", "temperature")},
    "base_temperature": lambda: {dataclass_default("src/SOLAR/api.py", "ModelConfig", "base_temperature")},
    "margin": lambda: train_setting("margin") | {dataclass_default("src/SOLAR/api.py", "ModelConfig", "margin")},
    # Optimization.
    "train_size": lambda: train_setting("train_size"),
    "validation_fraction": lambda: {round(1 - v, 10) for v in train_setting("train_size")},
    "learning_rate": lambda: train_setting("learning_rate"),
    "batch_size": lambda: train_setting("batch_size"),
    "gradient_clip_norm": lambda: train_setting("gradient_clip_norm"),
    "weight_decay": lambda: train_setting("weight_decay"),
    "patience": lambda: train_setting("early_stopping_patience"),
    "min_delta": lambda: train_setting("early_stopping_min_delta"),
    "max_epochs": lambda: train_setting("max_epochs"),
    # SCLSC adapter.
    "sclsc_validation_fraction": lambda: {_sclsc_default("--validation-fraction")},
    "sclsc_patience": lambda: {_sclsc_default("--patience")},
    "sclsc_dims": lambda: {tuple(sorted(cohort_dims("sclsc_refonly")))},
    # Readout and metrics.
    "knn_neighbors": lambda: {config("experiment_v1_2.yaml")["evaluation"]["n_neighbors"]}
    | ({15} if contains("scripts/score_experiment_v1_2.py",
                        r'KNeighborsClassifier\(n_neighbors=15, weights="distance", metric="minkowski", p=2') else set()),
    "knn_minkowski_p": lambda: {config("experiment_v1_2.yaml")["evaluation"]["p"]},
    "graph_neighbors": lambda: {config("benchmark_rebuttal_v1.yaml")["evaluation"]["official_scib"]["neighbors"]},
    "lisi_k0": lambda: {90} if contains("scripts/paper_scib_worker.py", r"k0=90") else set(),
    "lisi_subsample_percent": lambda: {config("benchmark_rebuttal_v1.yaml")["evaluation"]["official_scib"]["lisi_subsample_percent"]},
    "scib_commit": lambda: {config("experiment_v1_2.yaml")["evaluation"]["official_scib_commit"],
                            config("benchmark_rebuttal_v1.yaml")["evaluation"]["official_scib"]["scib_commit"]},
    "trust_source_components": lambda: {config("experiment_v1_2.yaml")["evaluation"]["trustworthiness"]["source_components"]},
    "trust_neighbors": lambda: {config("experiment_v1_2.yaml")["evaluation"]["trustworthiness"]["neighbors"]}
    | ({15} if contains("scripts/score_experiment_v1_2.py", r"trustworthiness\([^)]*n_neighbors=15, metric=\"euclidean\"") else set()),
    "trust_max_cells": lambda: {config("experiment_v1_2.yaml")["evaluation"]["trustworthiness"]["max_cells"]},
    "trust_sample_seed": lambda: {config("experiment_v1_2.yaml")["evaluation"]["trustworthiness"]["sample_seed"]},
    "pseudotime_neighbors": lambda: {config("experiment_v1_2.yaml")["evaluation"]["pseudotime"]["neighbors"]},
    "pseudotime_min_cells": lambda: {16} if contains("scripts/biovalid_ref_query_joint.py",
                                                     r"valid\.sum\(\) < k \+ 1", r"indices\[:, 1:\]") else set(),
    "pseudotime_requested_neighbors": lambda: {16} if contains("scripts/biovalid_ref_query_joint.py",
                                                               r"NearestNeighbors\(n_neighbors=k_eff \+ 1\)") else set(),
    "silhouette_min_cells": lambda: {10} if contains("scripts/biovalid_ref_query_joint.py", r"if n < 10:") else set(),
    "silhouette_cap": lambda: {config("experiment_v1_2.yaml")["evaluation"]["within_label_silhouette"]["max_cells_per_label"]},
    "silhouette_seed": lambda: {config("experiment_v1_2.yaml")["evaluation"]["within_label_silhouette"]["sample_seed"]},
    "sd_ddof": lambda: {config("experiment_v1_2.yaml")["aggregation"]["sample_sd_ddof"]},
    "required_seed_count": lambda: {config("experiment_v1_2.yaml")["aggregation"]["required_seed_count"]},
    "package_version": lambda: {".".join(_package_version().split(".")[:2])},
}


def _encoder_source() -> str:
    text = source("src/SOLAR/src/anchor_model.py")
    start = text.index("class ExpressionEncoder")
    return text[start:text.index("\ndef ", start)]


def _sclsc_default(flag: str):
    match = re.search(rf'add_argument\("{re.escape(flag)}", type=\w+, default=([^)]+)\)', source("adapters/sclsc_v1_1.py"))
    if not match:
        raise KeyError(flag)
    return ast.literal_eval(match.group(1))


def _package_version() -> str:
    match = re.search(r'^version = "([^"]+)"', source("pyproject.toml"), re.M)
    return match.group(1)


# ------------------------------------------------------------- assertion ----
def _batch_means(rep, metric, population) -> pd.DataFrame:
    t = pn.batch_table(*rep, metric, population)
    return t[t.complete].set_index(["dataset_id", "heldout_batch"])["mean"]


def _all_negative(left, rights, metric, populations) -> bool:
    return all(pn.contrast(left, r, metric, p).delta < 0 for r in rights for p in populations)


def assert_encoder_layout() -> bool:
    text = _encoder_source()
    order = [m.group(0) for m in re.finditer(r"nn\.(Linear|BatchNorm1d|ReLU|Dropout|LayerNorm)", text)]
    expected = ["nn.Linear", "nn.BatchNorm1d", "nn.ReLU", "nn.Dropout"] * 2 + ["nn.Linear", "nn.LayerNorm"]
    normalized = contains("src/SOLAR/src/anchor_model.py",
                          r"expression = F\.normalize\(self\.expression_encoder\(expression_vectors\), dim=1\)")
    return order == expected and normalized


def assert_no_decoder_or_head() -> bool:
    text = source("src/SOLAR/src/anchor_model.py")
    model = text[text.index("class MIModule"):]
    return not re.search(r"decoder|GradientReversal|adversar|classifier|nn\.Linear\(", model)


def assert_orthogonal_anchors() -> bool:
    return contains("src/SOLAR/src/anchor_model.py",
                    r"raw = torch\.randn\(output_dim, num_rows, generator=generator\)",
                    r'torch\.linalg\.qr\(raw, mode="reduced"\)',
                    r"orthogonal anchors require output_dim >= num_classes",
                    r'self\.register_buffer\("anchors", anchors\)')


def assert_anchor_pools() -> bool:
    text = source("src/SOLAR/_training.py")
    return all(p in text for p in (
        "return z_x.unsqueeze(1), list(labels)",                    # no anchors: B views
        "return torch.stack([z_x, z_t], dim=1), list(labels)",       # repeated: 2B views
        "pooled = torch.cat([z_x, z_t_unique], dim=0)",              # unique: B + K_B views
    ))


def assert_loss_definition() -> bool:
    text = source("src/SOLAR/src/loss.py")
    return all(p in text for p in (
        "logits = logits - (mask * self.margin)",                    # same-label cosine offset
        "logits = torch.div(logits, self.temperature)",              # then temperature
        "mask = mask * logits_mask",                                 # index-based self-exclusion
        "exp_logits = torch.exp(logits) * logits_mask",              # all non-self views in denominator
        "mean_log_prob_pos = (mask * log_prob).sum(1) / mask_sum.clamp_min(1)",
        "return loss.mean()",                                        # mean over all V views
    ))


def assert_best_checkpoint() -> bool:
    return contains("src/SOLAR/_training.py", r"model\.load_state_dict\(best_model_state\)",
                    r"avg_val_loss < \(best_val_loss - early_stopping_min_delta\)")


def assert_singleton_stratum_reference() -> bool:
    from scib_benchmark.splits import stratified_reference_query
    obs = pd.DataFrame({"b": ["x"] * 6 + ["y"], "l": ["p"] * 3 + ["q"] * 3 + ["p"]},
                       index=[f"c{i}" for i in range(7)])
    split = stratified_reference_query(obs, "b", "l", seed=40)
    return split.loc[split.barcode == "c6", "role"].item() == "reference"


def assert_heldout_labels_unused() -> bool:
    from scib_benchmark.splits import held_out_batch_split
    obs = pd.DataFrame({"b": ["x", "x", "y"], "l": ["p", "q", "p"]}, index=["a", "b", "c"])
    one = held_out_batch_split(obs, "b", "l", "y")
    obs["l"] = ["q", "p", "q"]
    two = held_out_batch_split(obs, "b", "l", "y")
    return one.role.tolist() == two.role.tolist() == ["reference", "reference", "query"]


def assert_batch_selection_span() -> bool:
    audit = pd.read_csv(pn.DATA / "splits" / "heldout_batch_selection_audit.csv", dtype={"heldout_batch": str})
    for _, group in audit.groupby("dataset_id"):
        chosen = group[group.selected_for_training]
        ranks = sorted(chosen.selection_rank_by_query_size.astype(int))
        if len(chosen) != 3 or ranks[0] != 1 or ranks[-1] != len(group) or not 1 < ranks[1] < len(group):
            return False
    return True


def assert_closed_label() -> bool:
    frame = pn.composition()
    return bool((frame.n_celltypes_query_only == 0).all()
                and (frame.fraction_query_cells_label_supported_by_reference == 1.0).all())


def assert_sclsc_final_epoch() -> bool:
    return contains("adapters/sclsc_v1_1.py", r'"restore_best_checkpoint": False',
                    r"StratifiedShuffleSplit", r"class_representatives = np\.stack")


def assert_sclsc_normalized_input() -> bool:
    return contains("adapters/sclsc_v1_1.py", r"matrix = subset\.X", r"2,000-HVG normalized-X")


def assert_positive_within_each_dataset() -> bool:
    c = pn.contrast(SOLAR30, PCA30, "macro_f1", "query").batch_differences
    return bool((c.groupby("dataset_id")["diff"].mean() > 0).all() and c.dataset_id.nunique() == 3)


def assert_fig2b_order() -> bool:
    order = [SCANVI30, SCLSC30, SOLAR30, SCVI30, PCA30]
    means = [pn.summary(*m, "macro_f1", "query").mean for m in order]
    return means == sorted(means, reverse=True)


def assert_scanvi_higher() -> bool:
    return all(pn.summary(*SCANVI30, met, "query").mean > pn.summary(*SOLAR30, met, "query").mean
               for met in ("macro_f1", "balanced_accuracy"))


def assert_villani_cd16() -> bool:
    f1 = pn.villani_per_class_f1().drop(columns=["seed", "macro_f1"]).mean()
    return f1.idxmin() == "CD16+ Monocytes"


def assert_villani_macro_f1_matches() -> bool:
    per_seed = pn.villani_per_class_f1().set_index("seed")["macro_f1"]
    table = pn.metrics()
    rows = table[(table.method == "solar_orthogonal") & (table.dimension == 128) & (table.heldout_batch == "Villani")
                 & (table.metric == "macro_f1")].set_index("seed")["value"]
    return bool(np.allclose(per_seed.sort_index().to_numpy(), rows.sort_index().to_numpy(), atol=1e-12, rtol=0))


def assert_label_asw_order() -> bool:
    solar = pn.summary(*SOLAR30, "ASW_label", "joint").mean
    return all(solar > pn.summary(*m, "ASW_label", "joint").mean for m in (PCA30, SCVI30, SCANVI30)) \
        and solar < pn.summary(*SCLSC30, "ASW_label", "joint").mean


def assert_ilisi_refmap_higher() -> bool:
    solar = pn.summary(*SOLAR30, "iLISI", "joint").mean
    return all(pn.summary(*m, "iLISI", "joint").mean > solar for m in (SCVI30, SCANVI30, SCLSC30))


def assert_repeated_increases_separation() -> bool:
    return all(pn.contrast(REPEATED, r, m, "joint").delta > 0 for r in (UNIQUE, NONE) for m in ("ASW_label", "iLISI"))


def assert_repeated_ba_below_unique() -> bool:
    return pn.contrast(REPEATED, UNIQUE, "balanced_accuracy", "query").delta < 0


def assert_repeated_f1_vs_unique_below_0001() -> bool:
    return abs(pn.contrast(REPEATED, UNIQUE, "macro_f1", "query").delta) < 0.001


def assert_repeated_lowers_local_structure() -> bool:
    return _all_negative(REPEATED, (UNIQUE, NONE), "trustworthiness", ("reference", "query", "joint")) and \
        _all_negative(REPEATED, (UNIQUE, NONE), "pseudotime_smoothness", ("reference", "query", "joint"))


def assert_trust_every_batch_favours_solar() -> bool:
    diff = pn.contrast(SOLAR30, SCLSC30, "trustworthiness", "query").batch_differences
    return len(diff) == 6 and bool((diff["diff"] > 0).all())


def assert_smoothness_every_batch_favours_solar() -> bool:
    diff = pn.contrast(SOLAR30, SCLSC30, "pseudotime_smoothness", "joint").batch_differences
    return len(diff) == 3 and bool((diff["diff"] > 0).all())


def assert_refmap_higher_local_structure() -> bool:
    return all(pn.summary(*m, met, pop).mean > pn.summary(*SOLAR30, met, pop).mean
               for m in (PCA30, SCVI30, SCANVI30)
               for met, pop in (("trustworthiness", "query"), ("pseudotime_smoothness", "joint")))


def assert_query_smoothness_only_oetjen() -> bool:
    ok = []
    for method, dim in (SOLAR30, PCA30, SCVI30, SCANVI30, SCLSC30, REPEATED, UNIQUE, NONE,
                        ("sclsc_refonly", 16), ("unintegrated_pca_matched", 40)):
        t = pn.batch_table(method, dim, "pseudotime_smoothness", "query")
        ok.append(t.loc[t.complete, "heldout_batch"].tolist() == ["Oetjen_A"])
    return all(ok)


def assert_hvg_overlap_disabled() -> bool:
    rows = pn.metrics()
    rows = rows[rows.metric == "hvg_overlap"]
    return bool((~rows.valid).all() and rows.reason.str.contains("disables HVG overlap for embedding outputs").all())


def assert_fig_s1_coordinates_match_splits() -> bool:
    coords = pd.read_csv(pn.DATA / "figures" / "figS1_heldout_umap_coords.csv.gz", dtype={"heldout_batch": str})
    counts = coords.groupby(["dataset_id", "heldout_batch", "role"]).size().unstack()
    comp = pn.composition().set_index(["dataset_id", "heldout_batch"])
    comp = comp.loc[counts.index]
    return len(counts) == 9 and bool((counts["reference"] == comp.n_reference).all()
                                     and (counts["query"] == comp.n_query).all())


def assert_fig_s3_ten_panels() -> bool:
    import build_paper_figures as figures
    return len(figures.FIG_S3_PANELS) == 10


ASSERTIONS = {name[len("assert_"):]: fn for name, fn in globals().items() if name.startswith("assert_")}


# --------------------------------------------------------------- derived ----
def BATCHES_FROM_DATA() -> set:
    return {(d, b) for d, b in runs_frame()[["dataset_id", "heldout_batch"]].itertuples(index=False)}


def derived(check: str, spec: dict):
    frame = pn.composition()
    if check == "n_datasets":
        return frame.dataset_id.nunique()
    if check == "n_heldout_batches_per_dataset":
        return set(frame.groupby("dataset_id").size())
    if check == "n_real_query_batches":
        return len({(d, b) for d, b in runs_frame()[["dataset_id", "heldout_batch"]].itertuples(index=False)})
    if check == "n_table_s3_rows":
        return 2 * 5 * len(BATCHES_FROM_DATA())
    if check == "n_trust_batches":
        return pn.summary(*SOLAR30, "trustworthiness", "query").batches_valid
    if check == "n_pseudotime_batches":
        return pn.summary(*SOLAR30, "pseudotime_smoothness", "joint").batches_valid
    if check == "dataset_cells":
        row = frame[frame.dataset_id == spec["dataset"]].iloc[0]
        return int(row.n_reference + row.n_query)
    if check == "dataset_labels":
        return set(frame.loc[frame.dataset_id == spec["dataset"], "n_celltypes_reference"])
    if check == "fits_per_batch":
        return fits_per_batch(spec["method"], spec["dim"])
    if check == "fig2a":
        diff = pn.contrast(SOLAR30, PCA30, "macro_f1", "query").batch_differences
        return float(diff.loc[diff.dataset_id == spec["dataset"], "diff"].mean())
    if check == "fig_axis":
        import build_paper_figures as figures
        low, high, n_valid, n_applicable = figures.axis_extent(spec["figure"], spec["panel"])
        return {"low": low, "high": high, "n": f"n={n_valid}/{n_applicable}"}[spec["end"]]
    raise KeyError(check)


def _canon(value):
    if isinstance(value, (list, tuple)):
        return tuple(_canon(v) for v in value)
    if isinstance(value, (set, frozenset)):
        return frozenset(_canon(v) for v in value)
    return value


def _matches(observed, expected) -> bool:
    """`expected` is the ledger value; a list may denote one tuple or a set of scalars/tuples."""
    observed = _canon(observed)
    if isinstance(observed, frozenset):
        candidates = [frozenset({_canon(expected)})]
        if isinstance(expected, list):
            candidates.append(frozenset(_canon(e) for e in expected))
        return observed in candidates
    return observed == _canon(expected)


def evaluate(kind: str, spec: dict, printed: str) -> str:
    if kind == "static":
        observed = STATIC[spec["check"]]()
        return printed if _matches(observed, spec["value"]) else f"observed {sorted(observed, key=str)}"
    if kind == "assertion":
        return printed if ASSERTIONS[spec["check"]]() else "assertion does not hold"
    if kind == "derived":
        value = derived(spec["check"], spec)
        if "format" in spec:                      # a printed decimal, e.g. a figure label
            return format_like(value, printed, spec["format"])
        return printed if _matches(value, spec["value"]) else f"observed {value}"
    raise KeyError(kind)


def format_like(value, printed: str, style: str) -> str:
    if isinstance(value, str):
        return value
    if style == "signed3":
        return f"{value:+.3f}"
    if style == "g3":
        text = f"{value:.3g}"
        return text
    if style == "f3":
        return f"{value:.3f}"
    raise KeyError(style)
