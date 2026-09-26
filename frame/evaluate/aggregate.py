"""Aggregation and statistics over scored samples (Phase 5).

Unit of analysis: pixels are never independent samples and tiles of one scene are correlated, so every metric is first averaged WITHIN a scene unit
(the NEON acquisition, the source orthophoto, the region) and the statistics are taken over units. Sample-level (tile) descriptives are reported next to
them, with their own n. Paired comparisons take the difference of unit means for the same units. Everything is labelled ``descriptive_only`` when the number of
units is too small for an interval or a test (frame.evaluate.stats); nothing is corrected for multiple comparisons and no comparison is a ranking.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

from frame.evaluate.stats import bootstrap_ci, describe, group_means, paired_comparison

#: metrics shown in the README's headline tables (the full set is in aggregates.json)
def headline_metrics(primary_tau: float) -> List[str]:
    tau = str(primary_tau)
    return [
        "reference_accuracy.psnr_db", "reference_accuracy.ssim", "reference_accuracy.rmse", "reference_accuracy.mae", "reference_accuracy.sam_degrees", "reference_accuracy.ergas",
        "indices.NDVI.mae", "indices.NDVI.bias", "indices.NDVI.pearson_r", "indices.NDWI.mae", "band_ratios.B08/B04.bias_log_ratio",
        "spatial_detail.hf_relative_error", "spatial_detail.hf_correlation", "spatial_detail.hf_energy_ratio", "spatial_detail.gradient_correlation",
        "spatial_detail.phase_shift_px.magnitude", "seam.ratio",
        f"detail_analysis.{tau}.supported_synthesis", f"detail_analysis.{tau}.unsupported_detail", f"detail_analysis.{tau}.omission", f"detail_analysis.{tau}.mse_skill_vs_baseline",
        "self_consistency.overall.mae", "self_consistency.overall.rmse", "self_consistency.ndvi.mae",
    ] + [f"per_band.{b}.{m}" for b in ("B02", "B03", "B04", "B08") for m in ("rmse", "bias")]


Entry = Tuple[str, str, Optional[str], Dict[str, Optional[float]]]     # (sample_id, scene_unit, category, flat metrics)


def _unit_values(entries: Sequence[Entry], metric: str) -> Tuple[List[str], List[float]]:
    ids, means, _ = group_means([e[3].get(metric) for e in entries], [e[1] for e in entries])
    return ids, means


def _metric_names(entries_by_system: Dict[str, Sequence[Entry]]) -> List[str]:
    names = set()
    for entries in entries_by_system.values():
        for e in entries:
            names.update(e[3])
    return sorted(names)


def aggregate_dataset(
    entries_by_system: Dict[str, Sequence[Entry]],
    *,
    reference_system: str,
    pairs: Sequence[Tuple[str, str]],
    headline: Sequence[str],
    n_boot: int,
    alpha: float,
    seed: int,
) -> Dict[str, Any]:
    names = _metric_names(entries_by_system)
    out: Dict[str, Any] = {"systems": {}, "paired": {}}
    for system, entries in entries_by_system.items():
        metrics: Dict[str, Any] = {}
        for metric in names:
            sample_values = [e[3].get(metric) for e in entries]
            ids, unit_means = _unit_values(entries, metric)
            if not unit_means and not any(v is not None for v in sample_values):
                continue
            unit = describe(unit_means)
            unit["ci"] = bootstrap_ci(unit_means, n_boot=n_boot, alpha=alpha, seed=seed)
            metrics[metric] = {"sample": describe(sample_values), "unit": unit}
        out["systems"][system] = {"n_samples": len(entries), "n_units": len({e[1] for e in entries}), "metrics": metrics}

    wanted: List[Tuple[str, str]] = [(s, reference_system) for s in entries_by_system if s != reference_system]
    for pair in pairs:
        if pair not in wanted:
            wanted.append(pair)
    for a, b in wanted:
        if a not in entries_by_system or b not in entries_by_system:
            continue
        table: Dict[str, Any] = {}
        for metric in names:
            ida, va = _unit_values(entries_by_system[a], metric)
            idb, vb = _unit_values(entries_by_system[b], metric)
            shared = sorted(set(ida) & set(idb))
            if not shared:
                continue
            lookup_a, lookup_b = dict(zip(ida, va)), dict(zip(idb, vb))
            table[metric] = paired_comparison([lookup_a[u] for u in shared], [lookup_b[u] for u in shared], n_boot=n_boot, alpha=alpha, seed=seed)
        out["paired"][f"{a} - {b}"] = table

    categories = sorted({e[2] for entries in entries_by_system.values() for e in entries if e[2]})
    if categories:
        by_category: Dict[str, Any] = {}
        for cat in categories:
            block: Dict[str, Any] = {"systems": {}}
            for system, entries in entries_by_system.items():
                subset = [e for e in entries if e[2] == cat]
                if not subset:
                    continue
                block["n_samples"] = len(subset)
                block["n_units"] = len({e[1] for e in subset})
                block["systems"][system] = {m: describe([e[3].get(m) for e in subset]) for m in headline if any(e[3].get(m) is not None for e in subset)}
            by_category[cat] = block
        out["by_category"] = by_category
    return out


def seed_group_summary(group_systems: Sequence[str], aggregates: Dict[str, Any], headline: Sequence[str]) -> Dict[str, Any]:
    """Mean/std/min/max ACROSS the systems of one group (seeds of one model), per dataset, of each system's unit-level mean."""
    out: Dict[str, Any] = {"systems": list(group_systems), "n_systems": len(group_systems), "datasets": {}}
    for dataset, agg in aggregates.items():
        table: Dict[str, Any] = {}
        for metric in headline:
            values = [agg["systems"][s]["metrics"][metric]["unit"]["mean"] for s in group_systems if s in agg["systems"] and metric in agg["systems"][s]["metrics"]]
            if values:
                table[metric] = describe(values)
        out["datasets"][dataset] = table
    return out
