"""Is the stability a calibrated error predictor, and could it be made one on held-out evidence? (Phase 6). numpy + scipy only.

Three separate questions, kept separate:

1. **Is the raw spread a calibrated interval?**  ``interval_coverage`` asks how often the reference lies within mean +/- k*spread, against the nominal level a Gaussian
   of that spread would give, and ``scale_ratio`` how many times smaller the spread is than the error. A spread that is orders of magnitude below the error covers almost
   nothing: ensemble members share the model's bias, so their disagreement is a floor on the error, not a measure of it.
2. **Could it be recalibrated?**  ``recalibration_assessment`` learns a monotone (isotonic) map from stability to expected error on DEVELOPMENT evidence and judges it on
   TEST evidence it has never seen: reliability table, calibration line (observed = intercept + slope * predicted; ideal 1 and 0), expected calibration error, and skill over
   predicting the constant development mean.
3. **Is the relationship stable across scenes?**  Decided by the caller with the unit-clustered bootstrap (frame.reliability.association).

Nothing here adds conformal or probabilistic guarantees: a recalibrated stability, if it passes, is calibrated for that dataset, error metric and scale only.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional, Sequence

import numpy as np

MIN_POINTS_FOR_ASSESSMENT = 30


def _finite_pairs(a: Sequence[float], b: Sequence[float]):
    x, y = np.asarray(a, dtype=np.float64).ravel(), np.asarray(b, dtype=np.float64).ravel()
    if x.shape != y.shape:
        raise ValueError(f"inputs must have the same length, got {x.size} and {y.size}")
    keep = np.isfinite(x) & np.isfinite(y)
    return x[keep], y[keep], int(x.size - keep.sum())


# ---------------------------------------------------------------------------------------------------------------
# isotonic regression (pool-adjacent-violators)
# ---------------------------------------------------------------------------------------------------------------

def isotonic_fit(x: Sequence[float], y: Sequence[float], weights: Optional[Sequence[float]] = None) -> Dict[str, Any]:
    """Non-decreasing least-squares fit of y on x by pooling adjacent violators. Returns knots for :func:`isotonic_predict`."""
    xs, ys = np.asarray(x, dtype=np.float64).ravel(), np.asarray(y, dtype=np.float64).ravel()
    ws = np.ones_like(xs) if weights is None else np.asarray(weights, dtype=np.float64).ravel()
    keep = np.isfinite(xs) & np.isfinite(ys) & np.isfinite(ws) & (ws > 0)
    if not keep.any():
        raise ValueError("isotonic_fit needs at least one finite point with a positive weight")
    xs, ys, ws = xs[keep], ys[keep], ws[keep]
    order = np.argsort(xs, kind="stable")
    xs, ys, ws = xs[order], ys[order], ws[order]
    # each block: [sum w*y, sum w, x_min, x_max]
    blocks = []
    for xi, yi, wi in zip(xs, ys, ws):
        blocks.append([wi * yi, wi, xi, xi])
        while len(blocks) > 1 and blocks[-2][0] / blocks[-2][1] > blocks[-1][0] / blocks[-1][1]:
            top = blocks.pop()
            blocks[-1] = [blocks[-1][0] + top[0], blocks[-1][1] + top[1], blocks[-1][2], top[3]]
    knots_x, knots_y = [], []
    for wy, w, lo, hi in blocks:
        knots_x += [lo, hi]
        knots_y += [wy / w, wy / w]
    return {"x": knots_x, "y": knots_y, "n_blocks": len(blocks), "n": int(xs.size)}


def isotonic_predict(model: Dict[str, Any], x: Sequence[float]) -> np.ndarray:
    """A pooled block predicts its own value over the range it covers; between blocks the fit is interpolated; outside the fitted range it is held constant."""
    return np.interp(np.asarray(x, dtype=np.float64), np.asarray(model["x"], dtype=np.float64), np.asarray(model["y"], dtype=np.float64))


# ---------------------------------------------------------------------------------------------------------------
# reliability
# ---------------------------------------------------------------------------------------------------------------

def reliability_table(pred: Sequence[float], obs: Sequence[float], *, n_bins: int) -> Dict[str, Any]:
    """Equal-count bins of the predicted value: predicted mean, observed mean and their signed gap (observed - predicted) per bin, and the expected calibration error."""
    p, o, dropped = _finite_pairs(pred, obs)
    n = int(p.size)
    if n == 0:
        return {"status": "not_computable", "n": 0, "n_dropped": dropped, "bins": [], "ece": None, "reason": "no finite pairs"}
    order = np.argsort(p, kind="stable")
    bins, ece = [], 0.0
    for idx in np.array_split(order, min(int(n_bins), n)):
        mp, mo = float(p[idx].mean()), float(o[idx].mean())
        bins.append({"n": int(idx.size), "mean_predicted": mp, "mean_observed": mo, "gap": mo - mp})
        ece += idx.size / n * abs(mo - mp)
    return {"status": "ok", "n": n, "n_dropped": dropped, "bins": bins, "ece": float(ece), "definition": "equal-count bins of the predicted value; gap = observed - predicted; ece = sum_k n_k/N |gap_k|"}


def calibration_line(pred: Sequence[float], obs: Sequence[float]) -> Dict[str, Any]:
    """Least-squares line observed = intercept + slope * predicted. A calibrated predictor has slope 1 and intercept 0."""
    p, o, dropped = _finite_pairs(pred, obs)
    if p.size < 3 or float(np.ptp(p)) == 0.0:
        return {"status": "not_computable", "n": int(p.size), "n_dropped": dropped, "slope": None, "intercept": None, "r2": None, "reason": "fewer than 3 points or a constant prediction"}
    slope = float(np.cov(p, o, bias=True)[0, 1] / np.var(p))
    intercept = float(o.mean() - slope * p.mean())
    denom = float(np.var(o))
    r2 = float(np.corrcoef(p, o)[0, 1] ** 2) if denom > 0 else None
    return {"status": "ok", "n": int(p.size), "n_dropped": dropped, "slope": slope, "intercept": intercept, "r2": r2, "reason": None}


def interval_coverage(mean: Sequence[float], std: Sequence[float], ref: Sequence[float], *, k: float) -> Dict[str, Any]:
    """Fraction of finite elements whose reference lies within ``mean +/- k * std``, next to the level a Gaussian of that spread would give (erf(k / sqrt 2))."""
    m, s = np.asarray(mean, dtype=np.float64).ravel(), np.asarray(std, dtype=np.float64).ravel()
    r = np.asarray(ref, dtype=np.float64).ravel()
    if not (m.shape == s.shape == r.shape):
        raise ValueError("mean, std and ref must have the same length")
    keep = np.isfinite(m) & np.isfinite(s) & np.isfinite(r)
    n = int(keep.sum())
    nominal = math.erf(k / math.sqrt(2.0))
    if n == 0:
        return {"k": float(k), "n": 0, "n_dropped": int(m.size), "coverage": None, "nominal": nominal, "status": "not_computable"}
    covered = np.abs(r[keep] - m[keep]) <= k * s[keep]
    return {"k": float(k), "n": n, "n_dropped": int(m.size - n), "coverage": float(covered.mean()), "nominal": nominal, "status": "ok"}


def scale_ratio(std: Sequence[float], abs_error: Sequence[float]) -> Dict[str, Any]:
    """How many times larger the actual error is than the ensemble spread (medians and means)."""
    s, e, dropped = _finite_pairs(std, abs_error)
    if s.size == 0 or float(np.median(s)) <= 0 or float(s.mean()) <= 0:
        return {"status": "not_computable", "n": int(s.size), "n_dropped": dropped, "median_error_over_median_spread": None, "mean_error_over_mean_spread": None}
    return {"status": "ok", "n": int(s.size), "n_dropped": dropped, "median_spread": float(np.median(s)), "median_error": float(np.median(e)),
            "median_error_over_median_spread": float(np.median(e) / np.median(s)), "mean_error_over_mean_spread": float(e.mean() / s.mean())}


def recalibration_assessment(x_dev: Sequence[float], y_dev: Sequence[float], x_test: Sequence[float], y_test: Sequence[float], *, n_bins: int) -> Dict[str, Any]:
    """Learn stability -> expected error (isotonic) on development data; judge it on test data it has never seen."""
    xd, yd, _ = _finite_pairs(x_dev, y_dev)
    xt, yt, _ = _finite_pairs(x_test, y_test)
    if xd.size < MIN_POINTS_FOR_ASSESSMENT or xt.size < MIN_POINTS_FOR_ASSESSMENT:
        return {"status": "not_computable", "n_dev": int(xd.size), "n_test": int(xt.size), "reason": f"fewer than {MIN_POINTS_FOR_ASSESSMENT} finite points on the development or the test side"}
    model = isotonic_fit(xd, yd)
    pred = isotonic_predict(model, xt)
    mse_model = float(np.mean((pred - yt) ** 2))
    mse_constant = float(np.mean((yd.mean() - yt) ** 2))
    return {"status": "ok", "n_dev": int(xd.size), "n_test": int(xt.size), "n_blocks": model["n_blocks"], "line": calibration_line(pred, yt),
            "reliability": reliability_table(pred, yt, n_bins=n_bins), "skill_vs_constant": (1.0 - mse_model / mse_constant) if mse_constant > 0 else None,
            "definition": "isotonic map stability -> expected error fitted on development cells, judged on held-out test cells; skill = 1 - MSE_map / MSE_constant(development mean)"}
