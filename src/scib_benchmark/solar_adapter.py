from __future__ import annotations

import importlib
from dataclasses import replace
from types import ModuleType
from typing import Any


LOCAL_UNIQ_VARIANT = "solar_orthogonal_uniq"


def register_orthogonal_uniq(variants_module: ModuleType | None = None) -> Any:
    """Register the local uniq audit variant without altering the upstream archive."""
    module = variants_module or importlib.import_module("SOLAR.benchmark.variants")
    registry = module.VARIANTS
    base = registry["solar_orthogonal"]
    expected_kwargs = dict(base.model_kwargs)
    expected_kwargs["anchor_dedup"] = True
    if LOCAL_UNIQ_VARIANT in registry:
        existing = registry[LOCAL_UNIQ_VARIANT]
        if existing.model_kwargs != expected_kwargs:
            raise RuntimeError("Existing solar_orthogonal_uniq has unexpected parameters")
        return existing
    derived = replace(
        base,
        name=LOCAL_UNIQ_VARIANT,
        group="local_mechanism_audit",
        evidence_scope="scib_benchmark_local_anchor_pool_audit",
        description=(
            "One fixed orthogonal anchor per class; local audit variant derived "
            "from solar_orthogonal with anchor_dedup=True."
        ),
        model_kwargs=expected_kwargs,
    )
    registry[LOCAL_UNIQ_VARIANT] = derived
    return derived


def assert_only_dedup_diff(base: Any, derived: Any) -> None:
    base_kwargs = dict(base.model_kwargs)
    derived_kwargs = dict(derived.model_kwargs)
    if derived_kwargs.pop("anchor_dedup", None) is not True:
        raise AssertionError("Derived variant must set anchor_dedup=True")
    base_kwargs.pop("anchor_dedup", None)
    if base_kwargs != derived_kwargs:
        raise AssertionError("Derived uniq variant differs beyond anchor_dedup")

