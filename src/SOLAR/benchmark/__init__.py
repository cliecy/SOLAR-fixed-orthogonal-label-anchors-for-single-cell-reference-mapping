"""Portable adapters for evaluating SOLAR in external benchmarks."""

from .adapter import BenchmarkResult, PreprocessConfig, TrainConfig, run_inductive
from .splits import batch_holdout_split, lobo_split, stratified_holdout, subsample_labels
from .variants import VARIANTS, VariantSpec, get_variant, list_variants

__all__ = [
    "BenchmarkResult",
    "PreprocessConfig",
    "TrainConfig",
    "VARIANTS",
    "VariantSpec",
    "batch_holdout_split",
    "get_variant",
    "list_variants",
    "lobo_split",
    "run_inductive",
    "stratified_holdout",
    "subsample_labels",
]
