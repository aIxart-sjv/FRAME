"""Downstream metrics (Phase 7): NDVI fidelity and decision utility of a system against the HR reference, per region, per scene unit, and against the LR / bicubic baseline.

Everything is computed from a region table (frame.downstream.regions): the reference is called ``ref``. For a system ``s`` and a declared vegetation threshold:

* per region       signed / absolute / squared NDVI error of the region mean; the vegetation fraction (share of valid pixels with NDVI >= threshold) and its error; the pixel disagreement rate
                   ((false positives + false negatives) / valid pixels); the region decision (vegetated iff its fraction >= the declared majority fraction) and whether it disagrees with the reference's.
* per scene unit   the mean over the unit's regions of those (every region weighs the same inside its unit), the median absolute NDVI error, NDVI correlation across regions ONLY where it is
                   meaningful (>= 30 regions and a non-constant reference), valid coverage, and the pixel-level decision metrics from the unit's pooled confusion counts.
* across units     mean, standard deviation and bootstrap interval OVER SCENE UNITS (>= 5 units; fewer is descriptive only), and the paired difference against each baseline. A unit with many regions
                   never outweighs a unit with few; regions of one scene are not independent observations, so no statistic is taken over pooled regions.

There is no ranking anywhere: the outputs are per-system values and paired differences, with their uncertainty when the unit count allows it.
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from frame.downstream.config import DownstreamConfig
from frame.evaluate.stats import bootstrap_ci, describe, paired_comparison

REFERENCE = "ref"
MIN_REGIONS_FOR_CORRELATION = 30

UNIT_SCALARS = ("ndvi_mae", "ndvi_rmse", "ndvi_bias", "ndvi_median_abs_error", "ndvi_correlation", "veg_fraction_mae", "veg_fraction_bias", "disagreement_mean", "region_decision_error_rate",
                "valid_coverage")
DECISION_KEYS = ("accuracy", "balanced_accuracy", "false_positive_rate", "false_negative_rate", "disagreement_rate")


def region_metrics(table: Mapping[str, Any], system: str, threshold: str, *, majority_fraction: float) -> Dict[str, np.ndarray]:
    """Per-region downstream errors of ``system`` against the reference at one threshold (see the module docstring)."""
    n = np.asarray(table["n_valid"], dtype=np.float64)
    err = np.asarray(table[f"mean:{system}"], dtype=np.float64) - np.asarray(table[f"mean:{REFERENCE}"], dtype=np.float64)
    ref_veg = np.asarray(table[f"veg:{REFERENCE}@{threshold}"], dtype=np.float64)
    sys_veg = np.asarray(table[f"veg:{system}@{threshold}"], dtype=np.float64)
    fp, fn = np.asarray(table[f"fp:{system}@{threshold}"], dtype=np.float64), np.asarray(table[f"fn:{system}@{threshold}"], dtype=np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        f_ref, f_sys = ref_veg / n, sys_veg / n
        disagreement = (fp + fn) / n
    decision_error = ((f_sys >= majority_fraction) != (f_ref >= majority_fraction)).astype(np.int8)
    return {"ndvi_error": err, "ndvi_abs_error": np.abs(err), "ndvi_sq_error": err ** 2, "ref_veg_fraction": f_ref, "veg_fraction": f_sys, "veg_fraction_abs_error": np.abs(f_sys - f_ref),
            "veg_fraction_error": f_sys - f_ref, "disagreement": disagreement, "region_decision_error": decision_error}


def decision_from_counts(*, tp: int, fp: int, fn: int, tn: int) -> Dict[str, Any]:
    """Accuracy, balanced accuracy, false-positive / false-negative rate and disagreement rate of a binary decision; a rate that needs a class the reference does not have is None."""
    tp, fp, fn, tn = int(tp), int(fp), int(fn), int(tn)
    n = tp + fp + fn + tn
    if n == 0:
        return {"n": 0, "tp": 0, "fp": 0, "fn": 0, "tn": 0, "accuracy": None, "balanced_accuracy": None, "false_positive_rate": None, "false_negative_rate": None, "disagreement_rate": None,
                "reference_prevalence": None}
    pos, neg = tp + fn, fp + tn
    tpr = tp / pos if pos else None
    tnr = tn / neg if neg else None
    return {"n": n, "tp": tp, "fp": fp, "fn": fn, "tn": tn, "accuracy": (tp + tn) / n, "balanced_accuracy": (tpr + tnr) / 2 if (tpr is not None and tnr is not None) else None,
            "false_positive_rate": fp / neg if neg else None, "false_negative_rate": fn / pos if pos else None, "disagreement_rate": (fp + fn) / n, "reference_prevalence": pos / n}


def _correlation(ref: np.ndarray, sys: np.ndarray) -> Tuple[Optional[float], Optional[str]]:
    keep = np.isfinite(ref) & np.isfinite(sys)
    if int(keep.sum()) < MIN_REGIONS_FOR_CORRELATION:
        return None, f"fewer than {MIN_REGIONS_FOR_CORRELATION} regions"
    r, s = ref[keep], sys[keep]
    if float(np.ptp(r)) < 1e-6 or float(np.ptp(s)) < 1e-6:
        return None, "the reference (or the system) NDVI is constant across the unit's regions"
    return float(np.corrcoef(r, s)[0, 1]), None


def unit_metrics(table: Mapping[str, Any], systems: Sequence[str], thresholds: Sequence[str], *, majority_fraction: float) -> Dict[str, Dict[str, Dict[str, Dict[str, Any]]]]:
    """``{unit: {system: {threshold: metrics}}}`` from a region table whose regions carry their ``scene_unit``."""
    units = np.asarray(table["scene_unit"], dtype=object)
    size2 = float(table["size"]) ** 2
    out: Dict[str, Dict[str, Dict[str, Dict[str, Any]]]] = {}
    for unit in sorted({str(u) for u in units}):
        sel = np.asarray([str(u) == unit for u in units])
        out[unit] = {}
        for s in systems:
            out[unit][s] = {}
            for th in thresholds:
                m = region_metrics(table, s, th, majority_fraction=majority_fraction)
                m = {k: v[sel] for k, v in m.items()}
                n_valid = np.asarray(table["n_valid"], dtype=np.float64)[sel]
                ref_mean = np.asarray(table[f"mean:{REFERENCE}"], dtype=np.float64)[sel]
                sys_mean = np.asarray(table[f"mean:{s}"], dtype=np.float64)[sel]
                tp, fp, fn = (int(np.asarray(table[f"{k}:{s}@{th}"])[sel].sum()) for k in ("tp", "fp", "fn"))
                tn = int(n_valid.sum()) - tp - fp - fn
                corr, why = _correlation(ref_mean, sys_mean)
                dec = decision_from_counts(tp=tp, fp=fp, fn=fn, tn=tn)
                rec: Dict[str, Any] = {
                    "n_regions": int(sel.sum()), "n_valid_pixels": int(n_valid.sum()), "valid_coverage": float(n_valid.mean() / size2),
                    "ndvi_mae": float(np.mean(m["ndvi_abs_error"])), "ndvi_rmse": float(np.sqrt(np.mean(m["ndvi_sq_error"]))), "ndvi_bias": float(np.mean(m["ndvi_error"])),
                    "ndvi_median_abs_error": float(np.median(m["ndvi_abs_error"])), "ndvi_correlation": corr, "veg_fraction_mae": float(np.mean(m["veg_fraction_abs_error"])),
                    "veg_fraction_bias": float(np.mean(m["veg_fraction_error"])), "disagreement_mean": float(np.mean(m["disagreement"])),
                    "region_decision_error_rate": float(np.mean(m["region_decision_error"])), "reference_prevalence": dec["reference_prevalence"], "decision": dec,
                }
                if why is not None:
                    rec["correlation_reason"] = why
                out[unit][s][th] = rec
    return out


def _value(rec: Mapping[str, Any], key: str) -> Optional[float]:
    v = rec["decision"].get(key) if key in DECISION_KEYS else rec.get(key)
    return None if v is None else float(v)


def aggregate_units(units: Mapping[str, Any], systems: Sequence[str], thresholds: Sequence[str], config: DownstreamConfig, *, baselines: Sequence[str],
                    pairs: Sequence[Tuple[str, str]] = ()) -> Dict[str, Any]:
    """Mean / std / interval over scene units per system and metric, and paired differences (A - B over the same units) against each baseline and any extra ``pairs``."""
    ids = sorted(units)
    b = config.bootstrap
    out: Dict[str, Any] = {}
    wanted: List[Tuple[str, str]] = [(s, base) for s in systems for base in baselines if s != base and base in systems]
    wanted += [p for p in pairs if p[0] in systems and p[1] in systems and p not in wanted]
    for th in thresholds:
        block: Dict[str, Any] = {"systems": {}, "paired": {}, "n_units": len(ids)}
        for s in systems:
            block["systems"][s] = {}
            for key in (*UNIT_SCALARS, *DECISION_KEYS):
                vals = [_value(units[u][s][th], key) for u in ids]
                info = describe(vals)
                block["systems"][s][key] = {"unit": info, "ci": bootstrap_ci(vals, n_boot=b.n_boot, alpha=b.alpha, seed=b.seed), "n_units": info["n"]}
        for a_name, b_name in wanted:
            table: Dict[str, Any] = {}
            for key in (*UNIT_SCALARS, *DECISION_KEYS):
                pa = [_value(units[u][a_name][th], key) for u in ids]
                pb = [_value(units[u][b_name][th], key) for u in ids]
                table[key] = paired_comparison(pa, pb, n_boot=b.n_boot, alpha=b.alpha, seed=b.seed)
            block["paired"][f"{a_name} - {b_name}"] = table
        out[th] = block
    return out
