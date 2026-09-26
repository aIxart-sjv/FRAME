"""Association statistics between a stability signal and reconstruction error (Phase 6). numpy + scipy only.

Discipline inherited from the Phase 5 statistics layer (frame.evaluate.stats):

* Pixels and cells of one tile, and tiles of one scene, are NOT independent. Wherever an interval is reported it comes from a bootstrap that resamples whole scene
  UNITS (:func:`cluster_bootstrap`), never pixels. Below ``MIN_UNITS_CI`` units no interval is computed and the result is ``descriptive_only``.
* Non-finite values are dropped AND counted; an undefined statistic (constant input, too few points) is ``not_computable`` with a reason and NO value, never a zero.
* Every random draw takes an explicit seed, so a rerun is identical. Ties in a ranking are broken by a seeded random permutation, never by input order.
* Nothing here says "calibrated", "confidence" or "significant". The functions measure association; what it means is decided (and worded) by the report.
"""

from __future__ import annotations

import math
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy import stats as scipy_stats

from frame.evaluate.stats import MIN_UNITS_CI, group_means

#: unit-level mean of a per-tile value (tiles of one scene count once); re-exported so callers use one vocabulary
unit_means = group_means

MAX_FAILED_REPLICATE_FRACTION = 0.10


# ---------------------------------------------------------------------------------------------------------------
# correlation
# ---------------------------------------------------------------------------------------------------------------

def _pairs(x: Sequence[float], y: Sequence[float]) -> Tuple[np.ndarray, np.ndarray, int]:
    a, b = np.asarray(x, dtype=np.float64).ravel(), np.asarray(y, dtype=np.float64).ravel()
    if a.shape != b.shape:
        raise ValueError(f"x and y must have the same length, got {a.size} and {b.size}")
    keep = np.isfinite(a) & np.isfinite(b)
    return a[keep], b[keep], int(a.size - keep.sum())


def _not_computable(kind: str, n: int, dropped: int, reason: str) -> Dict[str, Any]:
    return {"kind": kind, "value": None, "n": n, "n_dropped": dropped, "status": "not_computable", "reason": reason}


def _pearson(a: np.ndarray, b: np.ndarray) -> Optional[float]:
    da, db = a - a.mean(), b - b.mean()
    denom = math.sqrt(float(np.dot(da, da)) * float(np.dot(db, db)))
    if denom <= 0.0 or not math.isfinite(denom):
        return None
    return float(np.clip(np.dot(da, db) / denom, -1.0, 1.0))


def correlation(x: Sequence[float], y: Sequence[float], kind: str = "spearman") -> Dict[str, Any]:
    """Spearman (rank; ties get average ranks) or Pearson correlation of the finite pairs. Undefined results say why and carry no value."""
    if kind not in ("spearman", "pearson"):
        raise ValueError(f"kind must be 'spearman' or 'pearson', got {kind!r}")
    a, b, dropped = _pairs(x, y)
    if a.size < 3:
        return _not_computable(kind, int(a.size), dropped, "too_few finite pairs (need at least 3)")
    if float(np.ptp(a)) == 0.0:
        return _not_computable(kind, int(a.size), dropped, "constant x: a constant has no rank order or variance")
    if float(np.ptp(b)) == 0.0:
        return _not_computable(kind, int(a.size), dropped, "constant y: a constant has no rank order or variance")
    if kind == "spearman":
        a, b = scipy_stats.rankdata(a), scipy_stats.rankdata(b)
    value = _pearson(a, b)
    if value is None:
        return _not_computable(kind, int(a.size), dropped, "zero variance after ranking")
    return {"kind": kind, "value": value, "n": int(a.size), "n_dropped": dropped, "status": "ok", "reason": None}


def partial_spearman(x: Sequence[float], y: Sequence[float], controls: Sequence[Sequence[float]]) -> Dict[str, Any]:
    """Rank correlation of x and y after removing what a set of control variables (linearly, on the ranks) explains of each.

    This asks whether the stability adds anything to a TRIVIAL predictor (image texture, the amount of detail the model added): a stability that only re-encodes texture
    correlates with error exactly as texture does, and the partial value shows it (near zero).
    """
    cols = [np.asarray(c, dtype=np.float64).ravel() for c in controls]
    a, b = np.asarray(x, dtype=np.float64).ravel(), np.asarray(y, dtype=np.float64).ravel()
    if any(c.shape != a.shape for c in cols) or a.shape != b.shape:
        raise ValueError("x, y and every control must have the same length")
    keep = np.isfinite(a) & np.isfinite(b)
    for c in cols:
        keep &= np.isfinite(c)
    n, dropped = int(keep.sum()), int(a.size - keep.sum())
    if n < 5:
        return _not_computable("partial_spearman", n, dropped, "too_few finite points (need at least 5)")
    ranks = lambda v: scipy_stats.rankdata(v[keep])                               # noqa: E731
    ra, rb = ranks(a), ranks(b)
    if float(np.ptp(ra)) == 0.0 or float(np.ptp(rb)) == 0.0:
        return _not_computable("partial_spearman", n, dropped, "constant x or y")
    design = np.column_stack([np.ones(n)] + [ranks(c) for c in cols])
    residuals = []
    for r in (ra, rb):
        coef, *_ = np.linalg.lstsq(design, r, rcond=None)
        residuals.append(r - design @ coef)
    value = _pearson(residuals[0], residuals[1])
    if value is None:
        return _not_computable("partial_spearman", n, dropped, "no variance left after removing the controls")
    return {"kind": "partial_spearman", "value": value, "n": n, "n_dropped": dropped, "status": "ok", "reason": None, "n_controls": len(cols)}


# ---------------------------------------------------------------------------------------------------------------
# bootstrap over scene units
# ---------------------------------------------------------------------------------------------------------------

def cluster_bootstrap_multi(units: Dict[str, Any], statistic: Callable[[List[Any]], Dict[str, Optional[float]]], *, n_boot: int, alpha: float, seed: int,
                            min_units: int = MIN_UNITS_CI) -> Dict[str, Dict[str, Any]]:
    """Percentile bootstrap over scene UNITS of several statistics computed from the SAME resamples.

    ``units`` maps a unit id to whatever payload the statistic needs (a list of per-tile values, a tuple of arrays, ...). Every replicate draws ``n_units`` units WITH
    replacement and passes their payloads (a unit drawn twice appears twice) to ``statistic``, which returns a dict of floats or None (undefined). Below ``min_units`` units no
    interval is computed (``descriptive_only``); a replicate where a statistic is undefined is counted for that statistic, and if more than 10% are, its interval is withdrawn
    (``unstable``) rather than computed from a biased remainder.
    """
    ids = sorted(units)
    payloads = [units[u] for u in ids]
    n = len(ids)
    base = {"n_units": n, "n_boot": int(n_boot), "level": 1.0 - alpha, "method": "percentile bootstrap over scene units", "seed": int(seed)}
    observed = statistic(payloads) if n else {}
    ok = lambda v: v is not None and math.isfinite(float(v))              # noqa: E731
    out: Dict[str, Dict[str, Any]] = {}
    resampled = n >= min_units and any(ok(v) for v in observed.values())
    draws: Dict[str, List[float]] = {k: [] for k in observed}
    failed: Dict[str, int] = {k: 0 for k in observed}
    if resampled:
        rng = np.random.default_rng(seed)
        for _ in range(int(n_boot)):
            idx = rng.integers(0, n, n)
            values = statistic([payloads[i] for i in idx])
            for k in observed:
                v = values.get(k)
                if ok(v):
                    draws[k].append(float(v))
                else:
                    failed[k] += 1
    for k, est in observed.items():
        if not ok(est):
            out[k] = {**base, "estimate": None, "ci_low": None, "ci_high": None, "n_failed_replicates": 0, "status": "not_computable", "reason": "the statistic is undefined on the observed data"}
        elif n < min_units:
            out[k] = {**base, "estimate": float(est), "ci_low": None, "ci_high": None, "n_failed_replicates": 0, "status": "descriptive_only",
                      "reason": f"{n} scene unit(s): fewer than the {min_units} needed for an interval"}
        elif failed[k] > MAX_FAILED_REPLICATE_FRACTION * n_boot or not draws[k]:
            out[k] = {**base, "estimate": float(est), "ci_low": None, "ci_high": None, "n_failed_replicates": failed[k], "status": "unstable",
                      "reason": f"the statistic was undefined in {failed[k]} of {n_boot} replicates (limit {int(MAX_FAILED_REPLICATE_FRACTION * 100)}%)"}
        else:
            lo, hi = np.percentile(np.asarray(draws[k]), [100 * alpha / 2, 100 * (1 - alpha / 2)])
            out[k] = {**base, "estimate": float(est), "ci_low": float(lo), "ci_high": float(hi), "n_failed_replicates": failed[k], "status": "ok", "reason": None}
    return out


def cluster_bootstrap(units: Dict[str, Any], statistic: Callable[[List[Any]], Optional[float]], *, n_boot: int, alpha: float, seed: int, min_units: int = MIN_UNITS_CI) -> Dict[str, Any]:
    """Percentile bootstrap of one statistic over scene units (see :func:`cluster_bootstrap_multi`)."""
    if not units:
        return cluster_bootstrap_multi(units, lambda p: {"value": None}, n_boot=n_boot, alpha=alpha, seed=seed, min_units=min_units).get("value") or {
            "n_units": 0, "estimate": None, "ci_low": None, "ci_high": None, "status": "not_computable", "reason": "no units", "n_boot": int(n_boot), "level": 1.0 - alpha,
            "method": "percentile bootstrap over scene units", "seed": int(seed), "n_failed_replicates": 0}
    return cluster_bootstrap_multi(units, lambda p: {"value": statistic(p)}, n_boot=n_boot, alpha=alpha, seed=seed, min_units=min_units)["value"]


# ---------------------------------------------------------------------------------------------------------------
# risk-coverage
# ---------------------------------------------------------------------------------------------------------------

def _order(values: np.ndarray, seed: int) -> np.ndarray:
    """Indices sorting ``values`` ascending; ties are broken by a seeded random permutation, never by input order."""
    perm = np.random.default_rng(seed).permutation(values.size)
    return np.lexsort((perm, values))


def _mean_risk(errors: np.ndarray) -> float:
    return float(np.mean(errors))


def risk_coverage(score: Sequence[float], error: Sequence[float], coverage_grid: Sequence[float], *, seed: int, risk_fn: Optional[Callable[[np.ndarray], float]] = None) -> Dict[str, Any]:
    """Rank items by ``score`` (higher = less stable), keep the most stable ``coverage`` fraction, and measure the error of what remains.

    * ``curve``            risk of the retained items at each coverage of ``coverage_grid`` (1.0 = nothing removed).
    * ``oracle_curve``     the same when the items are ranked by their TRUE error: the best any ranking could do (a bound, not an achievable result).
    * ``random_risk``      the risk of the full set: what removing items at random gives on average.
    * ``risk_reduction_at`` 1 - risk(c) / risk(1.0) for each coverage below 1 (positive = the removed items were the riskier ones).
    * ``aurc`` and ``selective_efficiency``  area under the risk-coverage curve over the reported coverage range (trapezoid on the grid), and
      (aurc_random - aurc) / (aurc_random - aurc_oracle): 1 = as good as the oracle, 0 = no better than random removal, negative = worse than random.

    ``risk_fn`` maps the retained errors to a risk (default: their mean; pass e.g. an RMSE-of-squared-errors for a tile-level RMSE risk). The result is
    a description of an ordering; it is not called calibrated selective prediction (nothing here is a calibrated probability).
    """
    risk_fn = risk_fn or _mean_risk
    s, e, dropped = _pairs(score, error)
    n = int(s.size)
    if n == 0:
        return {"status": "not_computable", "n": 0, "n_dropped": dropped, "reason": "no finite (score, error) pairs"}
    by_score, by_error = _order(s, seed), _order(e, seed + 1)

    def curve_for(order: np.ndarray) -> List[Dict[str, Any]]:
        out = []
        for c in coverage_grid:
            k = max(1, min(n, int(round(c * n))))
            out.append({"coverage": float(c), "n_retained": k, "risk": float(risk_fn(e[order[:k]]))})
        return out

    curve, oracle = curve_for(by_score), curve_for(by_error)
    full = float(risk_fn(e))
    grid = np.asarray(coverage_grid, dtype=np.float64)
    area = lambda risks: float(np.trapezoid(np.asarray(risks)[np.argsort(grid)], np.sort(grid))) if grid.size > 1 else float(risks[0])      # noqa: E731
    aurc, aurc_oracle = area([c["risk"] for c in curve]), area([c["risk"] for c in oracle])
    aurc_random = area([full] * grid.size)
    denom = aurc_random - aurc_oracle
    efficiency = float((aurc_random - aurc) / denom) if abs(denom) > 1e-15 else None
    reduction = {}
    for c in curve:
        if c["coverage"] < 1.0:
            reduction[str(c["coverage"])] = (1.0 - c["risk"] / full) if full > 0 else None
    return {"status": "ok", "n": n, "n_dropped": dropped, "coverage_grid": [float(c) for c in coverage_grid], "curve": curve, "oracle_curve": oracle, "random_risk": full,
            "aurc": aurc, "aurc_oracle": aurc_oracle, "aurc_random": aurc_random, "selective_efficiency": efficiency, "risk_reduction_at": reduction,
            "definition": "items ranked by instability (most unstable removed first); risk = error of the retained items; oracle = ranked by the true error"}


# ---------------------------------------------------------------------------------------------------------------
# development / test split of scene units
# ---------------------------------------------------------------------------------------------------------------

def dev_test_split(units: Sequence[str], *, seed: int, dev_fraction: float, min_units: int) -> Dict[str, Any]:
    """Seeded split of scene units into a development set (where a threshold may be chosen) and a test set (where it is only applied).

    The split is by UNIT, never by tile or pixel, and does not depend on the input order. Fewer than ``min_units`` units cannot be split meaningfully.
    """
    ids = sorted({str(u) for u in units})
    if len(ids) < min_units:
        return {"status": "not_splittable", "dev": [], "test": [], "n_units": len(ids), "reason": f"{len(ids)} scene unit(s): fewer than the {min_units} needed for a development/test split"}
    perm = np.random.default_rng(seed).permutation(len(ids))
    n_dev = max(1, min(len(ids) - 1, int(round(dev_fraction * len(ids)))))
    dev = sorted(ids[i] for i in perm[:n_dev])
    test = sorted(ids[i] for i in perm[n_dev:])
    return {"status": "ok", "dev": dev, "test": test, "n_units": len(ids), "seed": int(seed), "dev_fraction": float(dev_fraction), "reason": None}
