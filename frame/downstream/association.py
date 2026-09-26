"""Does the stability relate to downstream mistakes, once texture is controlled for? (Phase 7) The design of frame.reliability, with the downstream error as the target.

The downstream error of a region is |NDVI(system) - NDVI(reference)| (``ndvi_abs_error``) or its pixel decision disagreement rate (``disagreement``), at the primary threshold. Stability is the
region mean of the model's TTA spread (the deployed stability); texture is the region mean of the Sobel gradient of the bicubic input; added detail is how far the model moved from bicubic.

* ``association``     within EACH TILE the Spearman correlation of stability, texture and added detail with the error over the tile's regions, and the partial correlation of the stability controlling
                      for texture (and for texture and added detail); tiles of a scene unit are averaged, then the summary is over SCENE UNITS with an interval that resamples whole units (>= 5 units;
                      fewer is descriptive only). Across tiles / scene units (tile mean stability against tile mean error) an interval needs >= 10 units. The pooled-regions correlation
                      (a seeded subsample of each tile) also resamples whole units. Regions of one scene are never treated as independent replicates.
* ``risk_coverage``   within each scene unit, rank the unit's regions by instability, drop the predeclared most unstable fractions, and report the relative reduction of the remaining downstream
                      error; the texture ranking and the oracle (ranked by the true error) are reported the same way; the summary is over scene units. Not calibrated selective prediction.

Nothing here is causal, and nothing turns the stability into a calibrated uncertainty.
"""

from __future__ import annotations

import hashlib
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np

from frame.downstream.config import DownstreamConfig
from frame.downstream.metrics import region_metrics
from frame.evaluate.stats import bootstrap_ci, describe, paired_comparison
from frame.reliability.analysis import POOLED_MAX_BOOT, aggregate_over_units, across_unit_association, unit_bootstrap
from frame.reliability.association import correlation, partial_spearman
from frame.reliability.association import risk_coverage as _risk_coverage

TARGETS = ("ndvi_abs_error", "disagreement")
DEFINITION = ("region error = |NDVI(system) - NDVI(reference)| of the region mean, or the pixel decision disagreement rate; stability = region mean of the model's TTA spread; texture = region mean "
              "Sobel gradient of the bicubic input; added detail = |prediction - bicubic|. Regions are fixed square cells on the aligned grid. Associations are within tiles, summarised over scene units.")
RC_NOTE = ("Regions are ranked by instability within each scene unit and the most unstable predeclared fractions are dropped; the value is the relative reduction of the mean downstream error that remains. "
           "It describes an ordering and is NOT calibrated selective prediction; the oracle (ranked by the true error) and the texture ranking are shown for comparison.")


def _seed(config: DownstreamConfig, *parts: str) -> int:
    return int(hashlib.sha256("|".join([str(config.bootstrap.seed), *parts]).encode("utf-8")).hexdigest()[:8], 16)


def _arrays(table: Mapping[str, Any], system: str, threshold: str, config: DownstreamConfig) -> Dict[str, np.ndarray]:
    m = region_metrics(table, system, threshold, majority_fraction=config.decision.region_majority_fraction)
    out: Dict[str, np.ndarray] = {t: np.asarray(m[t], dtype=np.float64) for t in TARGETS}
    out["stab"] = np.asarray(table[f"mean:stab:{system}"], dtype=np.float64)
    out["tex"] = np.asarray(table["mean:texture"], dtype=np.float64)
    out["added"] = np.asarray(table[f"mean:added:{system}"], dtype=np.float64)
    return out


def _groups(table: Mapping[str, Any]) -> Dict[str, Dict[str, np.ndarray]]:
    """tile id -> {"unit": scene unit, "idx": region indices}: the identity every region carries."""
    tiles = np.asarray(table["tile_id"], dtype=object)
    units = np.asarray(table["scene_unit"], dtype=object)
    out: Dict[str, Dict[str, Any]] = {}
    for tile in sorted({str(t) for t in tiles}):
        idx = np.flatnonzero(np.asarray([str(t) == tile for t in tiles]))
        out[tile] = {"unit": str(units[idx[0]]), "idx": idx}
    return out


def association(table: Mapping[str, Any], system: str, threshold: str, config: DownstreamConfig) -> Dict[str, Any]:
    a = _arrays(table, system, threshold, config)
    groups = _groups(table)
    rows: List[Dict[str, Any]] = []
    for tile, g in groups.items():
        idx = g["idx"]
        row: Dict[str, Any] = {"scene_unit": g["unit"], "tile": tile, "assoc": {}, "t": {"stab": float(a["stab"][idx].mean()), "texture": float(a["tex"][idx].mean()), "added": float(a["added"][idx].mean())}}
        for target in TARGETS:
            e = a[target][idx]
            row["t"][target] = float(e.mean())
            s, x, ad = a["stab"][idx], a["tex"][idx], a["added"][idx]
            row["assoc"][target] = {"stability": correlation(s, e), "texture": correlation(x, e), "added_detail": correlation(ad, e), "partial_given_texture": partial_spearman(s, e, [x]),
                                    "partial_given_texture_and_added_detail": partial_spearman(s, e, [x, ad])}
        rows.append(row)
    out: Dict[str, Any] = {"system": system, "threshold": threshold, "n_regions": int(len(a["stab"])), "n_tiles": len(rows), "n_units": len({r["scene_unit"] for r in rows}), "definition": DEFINITION,
                           "targets": {}}
    for target in TARGETS:
        blk: Dict[str, Any] = {q: aggregate_over_units(rows, f"assoc.{target}.{q}.value", config)
                               for q in ("stability", "texture", "added_detail", "partial_given_texture", "partial_given_texture_and_added_detail")}
        for r in rows:
            r["tt"] = {"stab": r["t"]["stab"], "err": r["t"][target], "texture": r["t"]["texture"]}
        blk["across_tiles"] = across_unit_association(rows, "tt.stab", "tt.err", config, controls=("tt.texture",))
        # pooled regions: a seeded subsample of each tile, whole scene units resampled
        payload: Dict[str, Any] = {}
        for tile, g in groups.items():
            idx = g["idx"]
            cap = config.analysis.pooled_regions_per_tile
            if len(idx) > cap:
                idx = np.sort(np.random.default_rng(_seed(config, tile, "pool")).choice(idx, size=cap, replace=False))
            payload.setdefault(g["unit"], []).append((a["stab"][idx], a["tex"][idx], a[target][idx]))

        def stat(ps: List[Any]) -> Dict[str, Optional[float]]:
            s = np.concatenate([t[0] for p in ps for t in p])
            x = np.concatenate([t[1] for p in ps for t in p])
            e = np.concatenate([t[2] for p in ps for t in p])
            return {"stability": correlation(s, e)["value"], "texture": correlation(x, e)["value"], "partial_given_texture": partial_spearman(s, e, [x])["value"]}

        pooled = unit_bootstrap(payload, stat, config, pooled=True)
        blk["pooled_stability"], blk["pooled_texture"], blk["pooled_partial_given_texture"] = pooled["stability"], pooled["texture"], pooled["partial_given_texture"]
        out["targets"][target] = blk
    return out


def risk_coverage(table: Mapping[str, Any], system: str, threshold: str, config: DownstreamConfig) -> Dict[str, Any]:
    a = _arrays(table, system, threshold, config)
    units = np.asarray(table["scene_unit"], dtype=object)
    grid = config.analysis.retention_grid
    b = config.bootstrap
    out: Dict[str, Any] = {"system": system, "threshold": threshold, "retention_grid": list(grid), "note": RC_NOTE, "targets": {}}
    unit_ids = sorted({str(u) for u in units})
    for target in TARGETS:
        per_unit: Dict[str, List[Dict[str, Any]]] = {}
        for u in unit_ids:
            idx = np.flatnonzero(np.asarray([str(x) == u for x in units]))
            err = a[target][idx]
            rc_s = _risk_coverage(a["stab"][idx], err, grid, seed=_seed(config, u, "rc-stab"))
            rc_t = _risk_coverage(a["tex"][idx], err, grid, seed=_seed(config, u, "rc-tex"))
            per_unit[u] = []
            for i, c in enumerate(grid):
                full = rc_s["curve"][0]["risk"]
                per_unit[u].append({"retention": c, "risk": {"stability": rc_s["curve"][i]["risk"], "texture": rc_t["curve"][i]["risk"], "oracle": rc_s["oracle_curve"][i]["risk"]}, "full": full})
        by: List[Dict[str, Any]] = []
        for i, c in enumerate(grid):
            entry: Dict[str, Any] = {"retention": c, "dropped_fraction": round(1.0 - c, 10)}
            reductions: Dict[str, List[Optional[float]]] = {}
            for rank in ("stability", "texture", "oracle"):
                red = [(1.0 - per_unit[u][i]["risk"][rank] / per_unit[u][i]["full"]) if per_unit[u][i]["full"] > 0 else None for u in unit_ids]
                risk = [per_unit[u][i]["risk"][rank] for u in unit_ids]
                entry[rank] = {"reduction": {"unit": describe(red), "ci": bootstrap_ci(red, n_boot=b.n_boot, alpha=b.alpha, seed=b.seed)}, "risk": {"unit": describe(risk)}}
                reductions[rank] = red
            # the texture order is the trivial predictor the stability has to beat: the paired per-unit difference answers that, where two overlapping intervals of separate means cannot
            entry["stability_minus_texture"] = paired_comparison(reductions["stability"], reductions["texture"], n_boot=b.n_boot, alpha=b.alpha, seed=b.seed)
            by.append(entry)
        out["targets"][target] = {"by_retention": by, "n_units": len(unit_ids), "n_regions": int(len(units))}
    return out
