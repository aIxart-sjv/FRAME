"""From per-tile evidence to the answers (Phase 6): is the TTA stability informative about reconstruction error, on valid evidence only?

Input for one (dataset, system): the result rows of the ELIGIBLE tiles (frame.reliability.evidence) and their pooled arrays. Every level keeps its own unit of analysis:

* ``pixel_level``   a Spearman/Pearson correlation is computed WITHIN each tile (pixels of a tile are not independent) and summarised over scene units: mean of the per-unit means, the share
                    of tiles with a positive correlation, and a bootstrap interval that resamples whole units.
* ``cell_level``    the same at 10 m (4 HR px) and 40 m (16 HR px) cells, plus a POOLED cell correlation whose interval also resamples units. Cells average out sub-pixel noise, which is the scale at
                    which a small residual registration error matters least.
* ``tile_level`` / ``scene_level``  tile mean stability against tile RMSE / MAE / SAM / ERGAS across tiles, and across scene units (tiles of one unit averaged first).
* ``risk_coverage`` remove the most unstable tiles (or cells), measure the error that remains; the oracle ordering and random removal bracket it.
* ``high_error_detection``  the threshold that defines "high error" is a quantile of DEVELOPMENT units' error and is only APPLIED to held-out test units; AUROC / AUPRC / flag precision.
* ``calibration``   is the raw spread an interval (coverage, scale) and could a monotone map fitted on development units predict error on test units?
* ``detail_relationship``  stability against the Phase 5 element categories (unsupported detail, omission, supported synthesis).
* ``tta_cost``      single-pass and ensemble latency.

Every association is set beside two TRIVIAL predictors (image texture; how far the model moved from bicubic) and a partial correlation controlling for both: stability that only
re-encodes them adds nothing. With fewer than 5 scene units every result is descriptive only (no interval); nothing is corrected for multiple comparisons; nothing says "calibrated" unless
held-out calibration is actually shown, and even then only for that dataset and error scale. No causal claim is made anywhere.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from frame.evaluate.stats import MIN_UNITS_CI, describe, group_means, paired_comparison
from frame.reliability import calibration as CAL
from frame.reliability import detection as DET
from frame.reliability.association import (
    cluster_bootstrap_multi,
    correlation,
    dev_test_split,
    partial_spearman,
    risk_coverage,
)
from frame.reliability.config import ReliabilityConfig

MIN_UNITS_ACROSS_UNITS = 10               # a statistic whose value is a correlation / curve ACROSS units needs more units than a mean of per-unit values (MIN_UNITS_CI = 5)
POOLED_MAX_BOOT = 500                     # pooled analyses resample thousands of cells per replicate; capped (declared, recorded)
TARGETS = ("rmse", "mae", "sam_degrees", "ergas")
DETAIL_TARGETS = ("unsupported_detail", "omission", "supported_synthesis")
INTERPRETATION = ("A descriptive, non-causal association between the TTA model-stability signal (the spread of six geometric views of the same input) and reconstruction error against an "
                  "independent reference, on registration-checked, valid evidence only. The spread is a relative model-stability proxy: it is NOT a calibrated uncertainty, a confidence "
                  "or a probability of error, and nothing in this analysis changes that unless the calibration section says so on held-out evidence.")


# ---------------------------------------------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------------------------------------------

def _get(row: Dict[str, Any], path: str) -> Optional[float]:
    cur: Any = row
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    if cur is None or isinstance(cur, (dict, list, str, bool)):
        return None
    v = float(cur)
    return v if math.isfinite(v) else None


def _units(rows: Sequence[Dict[str, Any]]) -> List[str]:
    return sorted({r["scene_unit"] for r in rows})


def _n_boot(config: ReliabilityConfig, pooled: bool = False) -> int:
    return min(config.bootstrap.n_boot, POOLED_MAX_BOOT) if pooled else config.bootstrap.n_boot


def _boot(units: Dict[str, Any], statistic, config: ReliabilityConfig, *, pooled: bool = False, across_units: bool = False) -> Dict[str, Dict[str, Any]]:
    """Unit-clustered bootstrap. ``across_units``: the statistic is itself a correlation or a curve over units/tiles, so its bootstrap needs MIN_UNITS_ACROSS_UNITS units (with 6 units a
    rank correlation of a monotone relationship bootstraps to the single point [1, 1]: a degenerate interval that looks like certainty)."""
    return cluster_bootstrap_multi(units, statistic, n_boot=_n_boot(config, pooled), alpha=config.bootstrap.alpha, seed=config.bootstrap.seed,
                                   min_units=MIN_UNITS_ACROSS_UNITS if across_units else MIN_UNITS_CI)


def _aggregate(rows: Sequence[Dict[str, Any]], path: str, config: ReliabilityConfig) -> Dict[str, Any]:
    """One per-tile scalar (a within-tile correlation) summarised over scene units: unit-level mean of tile values, an interval that resamples units, the share of tiles above 0."""
    values = [_get(r, path) for r in rows]
    units = [r["scene_unit"] for r in rows]
    ids, means, _ = group_means(values, units)
    finite = [v for v in values if v is not None]
    payload: Dict[str, List[float]] = {}
    for v, u in zip(values, units):
        if v is not None:
            payload.setdefault(u, []).append(v)
    ci = _boot(payload, lambda ps: {"mean": float(np.mean([np.mean(p) for p in ps])) if ps else None}, config)["mean"] if payload else \
        {"status": "not_computable", "reason": "no defined tile values", "estimate": None, "ci_low": None, "ci_high": None}
    return {"n_tiles": len(finite), "n_undefined_tiles": len(values) - len(finite), "n_units": len(ids), "tile": describe(values), "unit": describe(means), "ci": ci,
            "share_positive": (float(np.mean([v > 0 for v in finite])) if finite else None)}


def _pair_payload(rows: Sequence[Dict[str, Any]], paths: Sequence[str]) -> Dict[str, List[Tuple[float, ...]]]:
    """Tiles with every requested value defined, grouped by scene unit."""
    out: Dict[str, List[Tuple[float, ...]]] = {}
    for r in rows:
        vals = [_get(r, p) for p in paths]
        if all(v is not None for v in vals):
            out.setdefault(r["scene_unit"], []).append(tuple(vals))
    return out


def _tile_association(rows: Sequence[Dict[str, Any]], x_path: str, y_path: str, config: ReliabilityConfig, controls: Sequence[str] = ()) -> Dict[str, Dict[str, Any]]:
    """Association of a tile-level x with a tile-level y across tiles (``tile`` block) and across scene units, tiles of a unit averaged first (``scene`` block)."""
    payload = _pair_payload(rows, [x_path, y_path, *controls])
    n_tiles = sum(len(v) for v in payload.values())

    def flat(ps: List[List[Tuple[float, ...]]]) -> np.ndarray:
        return np.asarray([t for p in ps for t in p], dtype=np.float64).reshape(-1, 2 + len(controls))

    def unit_table(ps: List[List[Tuple[float, ...]]]) -> np.ndarray:
        return np.asarray([np.mean(np.asarray(p, dtype=np.float64), axis=0) for p in ps], dtype=np.float64).reshape(-1, 2 + len(controls))

    def stat(ps: List[List[Tuple[float, ...]]]) -> Dict[str, Optional[float]]:
        f, u = flat(ps), unit_table(ps)
        out = {"tile_spearman": correlation(f[:, 0], f[:, 1])["value"], "scene_spearman": correlation(u[:, 0], u[:, 1])["value"]}
        if controls:
            out["tile_partial"] = partial_spearman(f[:, 0], f[:, 1], [f[:, 2 + i] for i in range(len(controls))])["value"]
            out["scene_partial"] = partial_spearman(u[:, 0], u[:, 1], [u[:, 2 + i] for i in range(len(controls))])["value"]
        return out

    if n_tiles < 3:
        empty = {"status": "not_computable", "reason": "fewer than 3 tiles with defined values", "n": n_tiles, "value": None}
        return {"tile": {"n_tiles": n_tiles, "n_units": len(payload), "spearman": empty, "pearson": empty, "spearman_ci": empty},
                "scene": {"n_units": len(payload), "spearman": empty, "spearman_ci": empty}}
    ci = _boot(payload, stat, config, across_units=True)
    f = flat(list(payload.values()))
    u = unit_table(list(payload.values()))
    tile = {"n_tiles": n_tiles, "n_units": len(payload), "spearman": correlation(f[:, 0], f[:, 1], "spearman"), "pearson": correlation(f[:, 0], f[:, 1], "pearson"), "spearman_ci": ci["tile_spearman"]}
    scene = {"n_units": len(payload), "spearman": correlation(u[:, 0], u[:, 1], "spearman"), "pearson": correlation(u[:, 0], u[:, 1], "pearson"), "spearman_ci": ci["scene_spearman"]}
    if controls:
        tile["partial_spearman"] = partial_spearman(f[:, 0], f[:, 1], [f[:, 2 + i] for i in range(len(controls))])
        tile["partial_ci"] = ci["tile_partial"]
        scene["partial_spearman"] = partial_spearman(u[:, 0], u[:, 1], [u[:, 2 + i] for i in range(len(controls))])
        scene["partial_ci"] = ci["scene_partial"]
    return {"tile": tile, "scene": scene}


def _arrays_by_unit(rows: Sequence[Dict[str, Any]], arrays: Dict[str, Dict[str, np.ndarray]], names: Sequence[str]) -> Dict[str, Dict[str, np.ndarray]]:
    """Per scene unit, the pooled arrays of all its tiles concatenated (a unit is resampled as a whole)."""
    out: Dict[str, Dict[str, List[np.ndarray]]] = {}
    for r in rows:
        a = arrays.get(r["sample_id"])
        if a is None or any(n not in a for n in names):
            continue
        bucket = out.setdefault(r["scene_unit"], {n: [] for n in names})
        for n in names:
            bucket[n].append(np.asarray(a[n], dtype=np.float64))
    return {u: {n: np.concatenate(v) for n, v in d.items()} for u, d in out.items()}


# ---------------------------------------------------------------------------------------------------------------
# risk-coverage
# ---------------------------------------------------------------------------------------------------------------

RC_NOTE = ("Items are ranked by instability and the most unstable are removed; the risk is the error of what remains. This describes an ordering. It is NOT calibrated selective prediction: "
           "nothing here is a calibrated probability, and the oracle (ranked by the true error) and random removal only bracket the result.")


def _rc(units: Dict[str, Dict[str, np.ndarray]], config: ReliabilityConfig, *, risk_fn=None, pooled: bool) -> Dict[str, Any]:
    grid = config.analysis.coverage_grid
    seed = config.bootstrap.seed

    def pool(ps: List[Dict[str, np.ndarray]], key: str) -> np.ndarray:
        return np.concatenate([p[key] for p in ps]) if ps else np.array([])

    def one(ps: List[Dict[str, np.ndarray]], score_key: str) -> Dict[str, Any]:
        return risk_coverage(pool(ps, score_key), pool(ps, "err"), grid, seed=seed, risk_fn=risk_fn)

    payloads = [units[u] for u in sorted(units)]
    point = one(payloads, "score")
    if point["status"] != "ok":
        return {"status": point["status"], "n_units": len(units), "note": RC_NOTE}

    def stat(ps: List[Dict[str, np.ndarray]]) -> Dict[str, Optional[float]]:
        rc = one(ps, "score")
        if rc["status"] != "ok":
            return {}
        out: Dict[str, Optional[float]] = {"selective_efficiency": rc["selective_efficiency"]}
        for c, v in rc["risk_reduction_at"].items():
            out[f"reduction_{c}"] = v
        return out

    ci = _boot(units, stat, config, pooled=pooled, across_units=not pooled)
    result = {k: point[k] for k in ("status", "n", "n_dropped", "coverage_grid", "curve", "oracle_curve", "random_risk", "aurc", "aurc_oracle", "aurc_random", "selective_efficiency",
                                    "risk_reduction_at", "definition")}
    result["n_units"] = len(units)
    result["ci"] = {"selective_efficiency": ci["selective_efficiency"], "risk_reduction_at": {c: ci[f"reduction_{c}"] for c in point["risk_reduction_at"]}}
    for name, key in (("texture_baseline", "texture"), ("added_detail_baseline", "added")):
        b = one(payloads, key)
        result[name] = {k: b[k] for k in ("curve", "selective_efficiency", "risk_reduction_at")} if b["status"] == "ok" else {"status": b["status"]}
    result["note"] = RC_NOTE
    return result


def _risk_coverage_section(rows: Sequence[Dict[str, Any]], arrays: Dict[str, Dict[str, np.ndarray]], config: ReliabilityConfig) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for tag, err_path, risk_fn, transform in (("tile_rmse", "targets.product.rmse", lambda e: float(math.sqrt(float(np.mean(e)))), lambda v: v * v),
                                              ("tile_sam", "targets.product.sam_degrees", None, lambda v: v)):
        units: Dict[str, Dict[str, List[float]]] = {}
        for r in rows:
            vals = [_get(r, p) for p in ("stability.mean", err_path, "baselines.texture_mean", "baselines.added_detail_mean")]
            if all(v is not None for v in vals):
                b = units.setdefault(r["scene_unit"], {"score": [], "err": [], "texture": [], "added": []})
                b["score"].append(vals[0]); b["err"].append(transform(vals[1])); b["texture"].append(vals[2]); b["added"].append(vals[3])
        out[tag] = _rc({u: {k: np.asarray(v) for k, v in d.items()} for u, d in units.items()}, config, risk_fn=risk_fn, pooled=False)
    for size in config.analysis.cell_sizes_hr_px[:1]:
        names = [f"c{size}_stability", f"c{size}_abs_error", f"c{size}_texture", f"c{size}_added_detail"]
        by_unit = _arrays_by_unit(rows, arrays, names)
        units = {u: {"score": d[names[0]], "err": d[names[1]], "texture": d[names[2]], "added": d[names[3]]} for u, d in by_unit.items()}
        out[f"cell_{size}_abs_error"] = _rc(units, config, pooled=True) if units else {"status": "not_computable", "n_units": 0, "note": RC_NOTE}
    return out


# ---------------------------------------------------------------------------------------------------------------
# high-error detection (development-only threshold)
# ---------------------------------------------------------------------------------------------------------------

def _detection_block(payloads: List[Dict[str, np.ndarray]], score_key: str, seed: int) -> Dict[str, Any]:
    s = np.concatenate([p[score_key] for p in payloads])
    y = np.concatenate([p["label"] for p in payloads])
    return {"auroc": DET.auroc(s, y), "auprc": DET.auprc(s, y), "flag": {str(f): DET.flag_metrics(s, y, fraction=f, seed=seed) for f in (0.1, 0.2)}}


def _high_error_section(rows: Sequence[Dict[str, Any]], arrays: Dict[str, Dict[str, np.ndarray]], config: ReliabilityConfig) -> Dict[str, Any]:
    a = config.analysis
    size = a.cell_sizes_hr_px[0]
    names = [f"c{size}_stability", f"c{size}_abs_error", f"c{size}_texture", f"c{size}_added_detail"]
    by_unit = _arrays_by_unit(rows, arrays, names)
    units = sorted(by_unit)
    base: Dict[str, Any] = {"cell_size_hr_px": int(size), "quantile": a.high_error_quantile, "error": f"cell mean absolute reflectance error at {size} HR px",
                            "definition": "high error = a cell whose error exceeds the quantile of DEVELOPMENT cells' error; score = ensemble spread (higher = less stable)",
                            "baselines_note": "texture_baseline and added_detail_baseline are trivial scores computed without an ensemble"}
    if not units:
        return {"status": "not_computable", "reason": "no pooled cell arrays", **base}
    split = dev_test_split(units, seed=a.split_seed, dev_fraction=a.dev_fraction, min_units=a.min_units_for_split)

    def payload_for(unit_ids: Sequence[str], theta: float) -> Dict[str, Dict[str, np.ndarray]]:
        return {u: {"stability": by_unit[u][names[0]], "texture": by_unit[u][names[2]], "added": by_unit[u][names[3]], "label": (by_unit[u][names[1]] > theta).astype(np.int8)} for u in unit_ids}

    def blocks(units_payload: Dict[str, Dict[str, np.ndarray]]) -> Dict[str, Any]:
        ps = [units_payload[u] for u in sorted(units_payload)]
        return {"stability": _detection_block(ps, "stability", config.bootstrap.seed), "texture_baseline": _detection_block(ps, "texture", config.bootstrap.seed),
                "added_detail_baseline": _detection_block(ps, "added", config.bootstrap.seed)}

    if split["status"] != "ok":
        theta = float(np.quantile(np.concatenate([by_unit[u][names[1]] for u in units]), a.high_error_quantile))
        p = payload_for(units, theta)
        labels = np.concatenate([p[u]["label"] for u in units])
        result = {**base, "status": "descriptive_only", "reason": split["reason"], "threshold_source": "same evidence (no held-out split): descriptive only", "threshold": theta,
                  "dev_units": [], "test_units": units, "n_test_cells": int(labels.size), "prevalence_test": float(labels.mean()), **blocks(p)}
        pending = {"status": "descriptive_only", "reason": "no held-out split: an interval would describe a threshold chosen on the same data", "estimate": None, "ci_low": None, "ci_high": None}
        for key in ("stability", "texture_baseline", "added_detail_baseline"):
            result[key]["auroc_ci"] = pending
            result[key]["auprc_ci"] = pending
        return result

    dev_err = np.concatenate([by_unit[u][names[1]] for u in split["dev"]])
    theta = float(np.quantile(dev_err, a.high_error_quantile))
    p = payload_for(split["test"], theta)
    test_labels = np.concatenate([p[u]["label"] for u in split["test"]])
    result = {**base, "status": "ok", "threshold_source": "development units only", "threshold": theta, "dev_units": split["dev"], "test_units": split["test"], "n_dev_cells": int(dev_err.size),
              "n_test_cells": int(test_labels.size), "prevalence_test": float(test_labels.mean()), "split_seed": a.split_seed, **blocks(p)}

    def stat(ps: List[Dict[str, np.ndarray]]) -> Dict[str, Optional[float]]:
        out: Dict[str, Optional[float]] = {}
        for tag, key in (("stability", "stability"), ("texture_baseline", "texture"), ("added_detail_baseline", "added")):
            s, y = np.concatenate([q[key] for q in ps]), np.concatenate([q["label"] for q in ps])
            out[f"{tag}_auroc"], out[f"{tag}_auprc"] = DET.auroc(s, y)["value"], DET.auprc(s, y)["value"]
        return out

    ci = _boot(p, stat, config, pooled=True)
    for tag in ("stability", "texture_baseline", "added_detail_baseline"):
        result[tag]["auroc_ci"], result[tag]["auprc_ci"] = ci[f"{tag}_auroc"], ci[f"{tag}_auprc"]
    return result


# ---------------------------------------------------------------------------------------------------------------
# calibration
# ---------------------------------------------------------------------------------------------------------------

CAL_STATEMENT = ("The raw TTA spread is not a calibrated uncertainty: it is far smaller than the error and covers a small fraction of it. Whether a monotone map from stability to expected "
                 "error, fitted on development units, is calibrated on held-out test units is reported below; even where it is, it is calibrated for this dataset and error scale only "
                 "and is not a probability of error, a confidence or a conformal guarantee.")


def _raw_calibration(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    cov: Dict[str, Any] = {}
    for k in ("k1", "k2"):
        vals = [(r["calibration_raw"]["coverage"][k]["coverage"], r["calibration_raw"]["n_elements"], r["calibration_raw"]["coverage"][k]["nominal"]) for r in rows if r["calibration_raw"]["coverage"][k]["coverage"] is not None]
        if vals:
            w = np.asarray([v[1] for v in vals], dtype=np.float64)
            cov[k] = {"coverage": float(np.average([v[0] for v in vals], weights=w)), "nominal": float(vals[0][2]), "n_tiles": len(vals), "definition": "share of (band, pixel) elements whose reference lies within mean +/- k*spread, weighted by elements"}
    scale = [r["calibration_raw"]["scale"] for r in rows if r["calibration_raw"]["scale"]["status"] == "ok"]
    return {"coverage": cov, "scale": {"n_tiles": len(scale), "median_error_over_median_spread": float(np.median([s["median_error_over_median_spread"] for s in scale])) if scale else None,
                                       "mean_error_over_mean_spread": float(np.median([s["mean_error_over_mean_spread"] for s in scale])) if scale else None,
                                       "definition": "median over tiles of (median |error| / median spread) over (band, pixel) elements"}}


def _calibration_section(rows: Sequence[Dict[str, Any]], arrays: Dict[str, Dict[str, np.ndarray]], config: ReliabilityConfig) -> Dict[str, Any]:
    a = config.analysis
    size = a.cell_sizes_hr_px[0]
    raw = _raw_calibration(rows)
    names = [f"c{size}_stability", f"c{size}_abs_error"]
    by_unit = _arrays_by_unit(rows, arrays, names)
    split = dev_test_split(sorted(by_unit), seed=a.split_seed, dev_fraction=a.dev_fraction, min_units=a.min_units_for_split)
    out: Dict[str, Any] = {"raw": raw, "verdict": "uncalibrated_stability_evidence", "statement": CAL_STATEMENT, "error": f"cell mean absolute reflectance error at {size} HR px"}
    if split["status"] != "ok":
        out["recalibration"] = {"status": "not_assessable", "reason": split["reason"] + "; recalibration needs held-out units", "verdict": "not_assessable"}
        return out
    dev_x = np.concatenate([by_unit[u][names[0]] for u in split["dev"]])
    dev_y = np.concatenate([by_unit[u][names[1]] for u in split["dev"]])
    test_x = np.concatenate([by_unit[u][names[0]] for u in split["test"]])
    test_y = np.concatenate([by_unit[u][names[1]] for u in split["test"]])
    assess = CAL.recalibration_assessment(dev_x, dev_y, test_x, test_y, n_bins=a.calibration_bins)
    if assess["status"] != "ok":
        out["recalibration"] = {"status": "not_assessable", "reason": assess["reason"], "verdict": "not_assessable"}
        return out
    model = CAL.isotonic_fit(dev_x, dev_y)
    payload = {u: {"pred": CAL.isotonic_predict(model, by_unit[u][names[0]]), "obs": by_unit[u][names[1]]} for u in split["test"]}

    def stat(ps: List[Dict[str, np.ndarray]]) -> Dict[str, Optional[float]]:
        line = CAL.calibration_line(np.concatenate([p["pred"] for p in ps]), np.concatenate([p["obs"] for p in ps]))
        return {"slope": line["slope"], "intercept": line["intercept"]}

    ci = _boot(payload, stat, config, pooled=True)
    slope_ok = ci["slope"]["status"] == "ok" and ci["slope"]["ci_low"] <= 1.0 <= ci["slope"]["ci_high"]
    icpt_ok = ci["intercept"]["status"] == "ok" and ci["intercept"]["ci_low"] <= 0.0 <= ci["intercept"]["ci_high"]
    supported = bool(slope_ok and icpt_ok and assess["skill_vs_constant"] is not None and assess["skill_vs_constant"] > 0)
    out["recalibration"] = {**assess, "fitted_on": "development units", "assessed_on": "test units", "dev_units": split["dev"], "test_units": split["test"], "line_ci": ci,
                            "verdict": "supported_for_this_dataset_and_error_scale" if supported else "not_supported",
                            "criteria": "held-out calibration slope interval contains 1, intercept interval contains 0 (both over test units), and skill over the development-mean constant is positive"}
    return out


# ---------------------------------------------------------------------------------------------------------------
# detail relationship, cost
# ---------------------------------------------------------------------------------------------------------------

def _detail_section(rows: Sequence[Dict[str, Any]], config: ReliabilityConfig) -> Dict[str, Any]:
    tile = {t: _tile_association(rows, "stability.mean", f"detail.{t}", config) for t in DETAIL_TARGETS}
    cells = {str(s): {name: _aggregate(rows, f"cells.{s}.spearman_{name}.value", config) for name in ("unsupported", "omission", "supported")} for s in config.analysis.cell_sizes_hr_px}
    return {"definition": ("Phase 5 element analysis relative to the bicubic baseline (tau = the primary hallucination threshold): unsupported detail = added but not confirmed by THIS reference, omission = reference "
                           "detail not added, supported synthesis = added and confirmed. Not semantic labels; 'unsupported' does not mean false. Instability and the amount of detail added both grow with texture."),
            "tile_level": {t: v["tile"] for t, v in tile.items()}, "scene_level": {t: v["scene"] for t, v in tile.items()}, "cell_level": cells}


def _cost_section(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    body = list(rows[1:]) if len(rows) > 1 else list(rows)               # the first tile of a run carries CUDA / worker warm-up; reported separately
    single = [r["tta"]["single_pass_seconds"] for r in body if r["tta"]["single_pass_seconds"] is not None]
    total = [r["tta"]["total_seconds"] for r in body]
    med_s, med_t = (float(np.median(single)) if single else None), (float(np.median(total)) if total else None)
    return {"n_members": int(rows[0]["tta"]["n_members"]), "transforms": rows[0]["tta"]["transforms"], "n_tiles_timed": len(body), "single_pass_seconds_median": med_s, "tta_seconds_median": med_t,
            "tta_over_single_pass": (med_t / med_s) if med_s and med_t else None, "first_tile_tta_seconds": float(rows[0]["tta"]["total_seconds"]),
            "definition": "single pass = median member latency of one full tiled pass over the scene; TTA = all members; the first tile of the run (warm-up) is excluded from the medians"}


# ---------------------------------------------------------------------------------------------------------------
# the whole analysis of one (dataset, system)
# ---------------------------------------------------------------------------------------------------------------

def analyse_system(rows: Sequence[Dict[str, Any]], arrays: Dict[str, Dict[str, np.ndarray]], config: ReliabilityConfig) -> Dict[str, Any]:
    """Every analysis for the eligible tiles of ONE dataset and ONE system (rows are that system's result rows; nothing is pooled across datasets)."""
    rows = list(rows)
    if not rows:
        return {"status": "no_eligible_evidence", "n_tiles": 0, "n_units": 0, "interpretation": INTERPRETATION}
    units = _units(rows)
    descriptive = len(units) < MIN_UNITS_CI
    out: Dict[str, Any] = {"status": "ok", "n_tiles": len(rows), "n_units": len(units), "units": units, "evidence_level": rows[0]["evidence_level"], "descriptive_only": descriptive,
                           "interpretation": INTERPRETATION + (" With fewer than 5 scene units every result here is descriptive only." if descriptive else "")}
    keys = ("spearman_abs_error", "pearson_abs_error", "spearman_sam", "spearman_texture_abs_error", "spearman_added_detail_abs_error", "partial_spearman_abs_error_given_baselines")
    pixel = {k: _aggregate(rows, f"pixel.{k}.value", config) for k in keys}
    pixel["per_band"] = {b: _aggregate(rows, f"pixel.per_band.{b}.value", config) for b in rows[0]["stability"]["band_mean_std"]}
    pixel["by_reference_displacement"] = {s: _aggregate(rows, f"pixel.by_reference_displacement.{s}.value", config) for s in rows[0]["pixel"]["by_reference_displacement"]}
    pixel["definition"] = "Spearman/Pearson correlation between per-pixel stability and per-pixel band-mean absolute error, WITHIN each tile, then summarised over scene units"
    out["pixel_level"] = pixel

    cells: Dict[str, Any] = {}
    for size in config.analysis.cell_sizes_hr_px:
        block = {k: _aggregate(rows, f"cells.{size}.{k}.value", config) for k in keys}
        names = [f"c{size}_stability", f"c{size}_abs_error", f"c{size}_sam_deg", f"c{size}_texture", f"c{size}_added_detail"]
        by_unit = _arrays_by_unit(rows, arrays, names)

        def stat(ps: List[Dict[str, np.ndarray]], names=names) -> Dict[str, Optional[float]]:
            cat = {n: np.concatenate([p[n] for p in ps]) for n in names}
            return {"abs_error": correlation(cat[names[0]], cat[names[1]])["value"], "sam": correlation(cat[names[0]], cat[names[2]])["value"],
                    "texture": correlation(cat[names[3]], cat[names[1]])["value"], "added_detail": correlation(cat[names[4]], cat[names[1]])["value"],
                    "partial": partial_spearman(cat[names[0]], cat[names[1]], [cat[names[3]], cat[names[4]]])["value"]}

        if by_unit:
            ci = _boot(by_unit, stat, config, pooled=True)
            block["pooled_spearman_abs_error"], block["pooled_spearman_sam"] = ci["abs_error"], ci["sam"]
            block["pooled_baselines"] = {"texture_vs_abs_error": ci["texture"], "added_detail_vs_abs_error": ci["added_detail"], "partial_stability_vs_abs_error_given_baselines": ci["partial"]}
        for name in ("unsupported", "omission", "supported"):
            block[f"spearman_{name}"] = _aggregate(rows, f"cells.{size}.spearman_{name}.value", config)
        block["definition"] = (f"{size} x {size} HR-pixel cells ({size * 2.5:g} m); within-tile Spearman over cells summarised over units; the pooled Spearman uses a seeded subsample of each tile's cells "
                               "and an interval that resamples whole units")
        cells[str(size)] = block
    out["cell_level"] = cells

    tile_level: Dict[str, Any] = {"definition": "tile mean stability against the tile error target, across tiles; the interval resamples scene units"}
    scene_level: Dict[str, Any] = {"definition": "the same after averaging the tiles of each scene unit; n is the number of scene units"}
    ctrl = ["baselines.texture_mean", "baselines.added_detail_mean"]
    for tag, x in (("stability_vs", "stability.mean"), ("texture_vs", "baselines.texture_mean"), ("added_detail_vs", "baselines.added_detail_mean")):
        tile_level[tag], scene_level[tag] = {}, {}
        for t in TARGETS:
            r = _tile_association(rows, x, f"targets.product.{t}", config)
            tile_level[tag][t], scene_level[tag][t] = r["tile"], r["scene"]
    tile_level["partial_stability_given_baselines"], scene_level["partial_stability_given_baselines"] = {}, {}
    for t in TARGETS:
        r = _tile_association(rows, "stability.mean", f"targets.product.{t}", config, controls=ctrl)
        tile_level["partial_stability_given_baselines"][t] = {k: r["tile"].get(k) for k in ("n_tiles", "partial_spearman", "partial_ci")}
        scene_level["partial_stability_given_baselines"][t] = {k: r["scene"].get(k) for k in ("n_units", "partial_spearman", "partial_ci")}
    out["tile_level"], out["scene_level"] = tile_level, scene_level
    out["risk_coverage"] = _risk_coverage_section(rows, arrays, config)
    out["high_error_detection"] = _high_error_section(rows, arrays, config)
    out["calibration"] = _calibration_section(rows, arrays, config)
    out["detail_relationship"] = _detail_section(rows, config)
    out["tta_cost"] = _cost_section(rows)
    return out


# ---------------------------------------------------------------------------------------------------------------
# comparing systems on the same evidence (descriptive; not a ranking)
# ---------------------------------------------------------------------------------------------------------------

COMPARE_PATHS = ("cells.4.spearman_abs_error.value", "pixel.spearman_abs_error.value", "stability.mean", "targets.product.rmse", "targets.product.sam_degrees", "tta.total_seconds",
                 "tta.single_pass_seconds")
COMPARE_NOTE = ("Descriptive comparison of the same protocol on the tiles that are eligible for BOTH systems, paired over scene units (A - B). It is not a ranking: a different spread magnitude or "
                "a different association is a property of each model's stability signal, not a verdict on which model is better.")


def compare_systems(rows_by_system: Dict[str, Sequence[Dict[str, Any]]], config: ReliabilityConfig) -> Dict[str, Any]:
    names = list(rows_by_system)
    ids = {n: {r["sample_id"] for r in rows_by_system[n]} for n in names}
    shared = set.intersection(*ids.values()) if names else set()
    only_in = {n: len(ids[n] - shared) for n in names}
    kept = {n: [r for r in rows_by_system[n] if r["sample_id"] in shared] for n in names}
    n_units = len({r["scene_unit"] for r in kept[names[0]]}) if names else 0
    out: Dict[str, Any] = {"n_shared_tiles": len(shared), "n_shared_units": n_units, "only_in": only_in, "pairs": [], "note": COMPARE_NOTE}
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            metrics: Dict[str, Any] = {}
            for path in COMPARE_PATHS:
                ia, va, _ = group_means([_get(r, path) for r in kept[a]], [r["scene_unit"] for r in kept[a]])
                ib, vb, _ = group_means([_get(r, path) for r in kept[b]], [r["scene_unit"] for r in kept[b]])
                common = sorted(set(ia) & set(ib))
                da, db = dict(zip(ia, va)), dict(zip(ib, vb))
                metrics[path] = paired_comparison([da[u] for u in common], [db[u] for u in common], n_boot=config.bootstrap.n_boot, alpha=config.bootstrap.alpha, seed=config.bootstrap.seed)
            out["pairs"].append({"a": a, "b": b, "metrics": metrics})
    return out


# ---------------------------------------------------------------------------------------------------------------
# reused by later phases (frame.downstream) under public names
# ---------------------------------------------------------------------------------------------------------------

aggregate_over_units = _aggregate           # one per-tile scalar summarised over scene units (mean of unit means, interval over units, share of tiles above 0)
across_unit_association = _tile_association  # association of tile-level x with tile-level y across tiles and across scene units (needs MIN_UNITS_ACROSS_UNITS for an interval)
unit_bootstrap = _boot                       # the unit-clustered bootstrap with the minimum-unit rules
