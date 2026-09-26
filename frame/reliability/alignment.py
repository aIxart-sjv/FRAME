"""Reference-registration gate (Phase 6): is this reference registered well enough for a PIXEL-LEVEL comparison with the prediction?

Phase 5 measured that real references are not registered to the Sentinel-2 grid (median displacement of the bicubic baseline against the SEN2NEON references 1.0 HR pixel,
maximum 6.5), and that a displacement of even one HR pixel costs every pixel-level score. A stability-vs-error correlation on a misregistered reference is worse than noisy:
registration error grows with local contrast, and so does model instability, so the two would correlate through texture alone. Only registered evidence may enter.

The rule (settings in :class:`frame.reliability.config.AlignmentSpec`, declared before any result):

1. The displacement of the BICUBIC BASELINE against the reference is estimated (``estimate_displacement``: the Phase 5 estimator ``frame.evaluate.shift.estimate_alignment``,
   validated against a brute-force search, seeds a sub-pixel refinement of the normalised correlation peak over the valid pixels). The baseline, not the model, is used so every model is judged on identical evidence
   and no model chooses its own alignment.
2. Within ``tolerance_hr_px`` the pair is accepted as it is. Otherwise a single whole-pixel translation, at most ``max_correction_hr_px``, may be applied by CROPPING both
   grids to their overlap (``frame.evaluate.shift.displace_pair``): no resampling, no interpolation, no warp. The correction and the kept window are recorded.
3. After the correction the displacement is estimated again (the residual) and in each of the four quadrants (the spread): a global translation is the only misregistration
   model accepted, so quadrants that disagree mean a non-uniform misregistration that a translation cannot repair.
4. ``eligible``: residual within tolerance and quadrant spread within its tolerance. ``not_eligible`` (``reference_alignment_invalid``): the correction is out of range or
   disabled, or the residual / spread exceeds ``uncertain_factor`` tolerances. ``uncertain`` (``reference_alignment_uncertain``): anything in between, or an estimate that
   could not be computed. Only ``eligible`` evidence enters the analysis; the other two are excluded and reported with these codes.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from frame.evaluate.metrics import MetricConfig
from frame.evaluate.shift import displace_pair, estimate_alignment
from frame.reliability.config import AlignmentSpec

ALIGNED_NONE_REQUIRED = "none_required"
ALIGNED_TRANSLATION = "integer_translation_crop"
ESTIMATOR = ("frame.evaluate.shift.estimate_alignment seeds a sub-pixel refinement of the normalised correlation peak of the band-mean images over the valid overlap "
             "(frame.reliability.alignment.estimate_displacement; parabolic, ~0.1 px, valid pixels only)")
MIN_QUADRANT_VALID_PIXELS = 1024


MIN_NCC_PIXELS = 1024
_REFINE_MAX_MOVES = 4


def _ncc(a: np.ndarray, b: np.ndarray, m: np.ndarray, dy: int, dx: int) -> Optional[float]:
    """Pearson correlation over the valid overlap when the SR-side image is displaced back by the candidate displacement ``(dy, dx)``."""
    try:
        a_c, b_c, m_c = displace_pair(a[None], b[None], m, -dy, -dx)
    except ValueError:
        return None
    if int(m_c.sum()) < MIN_NCC_PIXELS:
        return None
    x, y = a_c[0][m_c], b_c[0][m_c]
    x, y = x - x.mean(), y - y.mean()
    denom = math.sqrt(float(np.dot(x, x)) * float(np.dot(y, y)))
    return float(np.dot(x, y) / denom) if denom > 0 else None


def _parabola(c_minus: float, c_zero: float, c_plus: float) -> float:
    """Offset of the peak of the parabola through three equally spaced samples, in [-1, 1]; 0 when the samples are not concave."""
    denom = c_minus - 2.0 * c_zero + c_plus
    if denom >= 0.0:
        return 0.0
    return float(np.clip(0.5 * (c_minus - c_plus) / denom, -1.0, 1.0))


def estimate_displacement(baseline: np.ndarray, hr: np.ndarray, mask: np.ndarray, cfg: MetricConfig) -> Dict[str, Any]:
    """Displacement (dy, dx) of the baseline against the reference in HR pixels, at sub-pixel resolution, from the valid pixels only.

    The Phase 5 estimator (``frame.evaluate.shift.estimate_alignment``) seeds the search. It has 0.1 px resolution on a tile without nodata but, with nodata, falls back to a
    masked correlation with WHOLE-pixel resolution: a true half-pixel shift then reads as +/-1 px and a one-pixel correction moves it to -1 px, so a 0.5 px tolerance cannot be
    judged. Here the seed is refined on the normalised correlation of the band-mean images over the valid overlap at the integer neighbours of the peak, and the peak of a parabola
    through them gives the sub-pixel position (a few dozen lines, no new registration system; validated below against known fractional shifts with and without nodata).
    """
    a, b, m = np.asarray(baseline, dtype=np.float64).mean(axis=0), np.asarray(hr, dtype=np.float64).mean(axis=0), np.asarray(mask, dtype=bool)
    seed = estimate_alignment(baseline, hr, m, cfg)
    if seed["status"] != "ok" or seed["raw_dy"] is None:
        return {"status": "not_computable", "reason": f"the seed estimate is not computable: {seed.get('reason')}", "dy": None, "dx": None}
    cy, cx = int(round(seed["raw_dy"])), int(round(seed["raw_dx"]))
    surface: Dict[Tuple[int, int], Optional[float]] = {}

    def value(y: int, x: int) -> Optional[float]:
        if (y, x) not in surface:
            surface[(y, x)] = _ncc(a, b, m, y, x)
        return surface[(y, x)]

    for _ in range(_REFINE_MAX_MOVES):
        cells = {(cy + i, cx + j): value(cy + i, cx + j) for i in (-1, 0, 1) for j in (-1, 0, 1)}
        finite = {k: v for k, v in cells.items() if v is not None}
        if (cy, cx) not in finite:
            return {"status": "not_computable", "reason": "the correlation could not be computed at the seed (too few valid pixels in the overlap, or a constant image)", "dy": None, "dx": None}
        best = max(finite, key=finite.get)
        if best == (cy, cx):
            break
        cy, cx = best
    else:
        return {"status": "not_computable", "reason": "the correlation peak did not settle", "dy": None, "dx": None}
    v = lambda y, x: value(y, x)                    # noqa: E731
    if None in (v(cy - 1, cx), v(cy + 1, cx), v(cy, cx - 1), v(cy, cx + 1)):
        return {"status": "ok", "dy": float(cy), "dx": float(cx), "method": "integer_peak (neighbours unavailable)", "coarse": [int(cy), int(cx)]}
    dy = cy + _parabola(v(cy - 1, cx), v(cy, cx), v(cy + 1, cx))
    dx = cx + _parabola(v(cy, cx - 1), v(cy, cx), v(cy, cx + 1))
    return {"status": "ok", "dy": float(dy), "dx": float(dx), "method": "correlation_peak_parabolic_refinement", "coarse": [int(cy), int(cx)], "peak_correlation": float(v(cy, cx))}


def apply_correction(stack: np.ndarray, hr: np.ndarray, mask: np.ndarray, correction: List[int]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply a recorded whole-pixel correction: the SR-side grids in ``stack`` are displaced by ``correction`` against the reference and everything is cropped to the overlap."""
    return displace_pair(stack, hr, mask, int(correction[0]), int(correction[1]))


def _crop_window(shape: Tuple[int, int], dy: int, dx: int) -> Tuple[List[int], List[int]]:
    height, width = shape
    return [max(dy, 0), height + min(dy, 0)], [max(dx, 0), width + min(dx, 0)]


def _quadrants(base: np.ndarray, ref: np.ndarray, mask: np.ndarray, cfg: MetricConfig) -> List[Optional[Tuple[float, float]]]:
    height, width = mask.shape
    hy, hx = height // 2, width // 2
    out: List[Optional[Tuple[float, float]]] = []
    for rows, cols in ((slice(0, hy), slice(0, hx)), (slice(0, hy), slice(hx, width)), (slice(hy, height), slice(0, hx)), (slice(hy, height), slice(hx, width))):
        m = mask[rows, cols]
        if int(m.sum()) < MIN_QUADRANT_VALID_PIXELS:
            out.append(None)
            continue
        est = estimate_displacement(base[:, rows, cols], ref[:, rows, cols], m, cfg)
        out.append((est["dy"], est["dx"]) if est["status"] == "ok" else None)
    return out


def _spread(estimates: List[Optional[Tuple[float, float]]]) -> Optional[float]:
    points = [e for e in estimates if e is not None]
    if len(points) < 2:
        return None
    return float(max(math.hypot(a[0] - b[0], a[1] - b[1]) for i, a in enumerate(points) for b in points[i + 1:]))


def classify_alignment(baseline: np.ndarray, hr: np.ndarray, mask: np.ndarray, spec: AlignmentSpec, cfg: MetricConfig) -> Dict[str, Any]:
    """Classify the registration of ``hr`` against the bicubic ``baseline`` (see the module docstring). Returns a JSON-ready provenance record."""
    base, ref, valid = np.asarray(baseline, dtype=np.float64), np.asarray(hr, dtype=np.float64), np.asarray(mask, dtype=bool)
    tol = float(spec.tolerance_hr_px)
    record: Dict[str, Any] = {
        "method": spec.method, "estimator": ESTIMATOR, "tolerance_hr_px": tol, "quadrant_tolerance_hr_px": float(spec.quadrant_tolerance_hr_px),
        "max_correction_hr_px": int(spec.max_correction_hr_px), "uncertain_factor": float(spec.uncertain_factor), "apply_translation_correction": bool(spec.apply_translation_correction),
        "status": None, "reason": None, "detail": None, "correction_method": None, "raw_dy": None, "raw_dx": None, "raw_magnitude": None, "correction": [0, 0], "applied": False,
        "residual_dy": None, "residual_dx": None, "residual_magnitude": None, "quadrant_estimates": [], "quadrant_spread": None, "overlap_fraction": None,
        "crop_rows": [0, int(valid.shape[0])], "crop_cols": [0, int(valid.shape[1])],
    }

    def finish(status: str, detail: str) -> Dict[str, Any]:
        record["status"], record["detail"] = status, detail
        record["reason"] = {"eligible": None, "uncertain": "reference_alignment_uncertain", "not_eligible": "reference_alignment_invalid"}[status]
        return record

    est = estimate_displacement(base, ref, valid, cfg)
    if est["status"] != "ok":
        return finish("uncertain", f"alignment estimate not_computable: {est.get('reason')}")
    raw_dy, raw_dx = float(est["dy"]), float(est["dx"])
    raw_mag = math.hypot(raw_dy, raw_dx)
    record.update(raw_dy=raw_dy, raw_dx=raw_dx, raw_magnitude=raw_mag)

    correction = (0, 0)
    if raw_mag > tol:
        cy, cx = -int(round(raw_dy)), -int(round(raw_dx))
        if not spec.apply_translation_correction:
            return finish("not_eligible", f"displacement {raw_mag:.2f} HR px exceeds the tolerance {tol} and translation correction is disabled")
        if max(abs(cy), abs(cx)) > spec.max_correction_hr_px:
            return finish("not_eligible", f"correction ({cy}, {cx}) HR px exceeds max_correction_hr_px={spec.max_correction_hr_px}: the reference is too far off to correct by a global translation")
        correction = (int(cy), int(cx))
    record["correction"] = [correction[0], correction[1]]
    record["applied"] = correction != (0, 0)
    record["correction_method"] = ALIGNED_TRANSLATION if record["applied"] else ALIGNED_NONE_REQUIRED

    base_c, ref_c, mask_c = displace_pair(base, ref, valid, correction[0], correction[1])
    rows, cols = _crop_window(valid.shape, correction[0], correction[1])
    record.update(crop_rows=rows, crop_cols=cols, overlap_fraction=float(mask_c.size) / float(valid.size))

    residual = est if not record["applied"] else estimate_displacement(base_c, ref_c, mask_c, cfg)
    if residual["status"] != "ok":
        return finish("uncertain", f"the residual displacement after the correction could not be computed: {residual.get('reason')}")
    r_dy, r_dx = float(residual["dy"]), float(residual["dx"])
    r_mag = math.hypot(r_dy, r_dx)
    quadrants = _quadrants(base_c, ref_c, mask_c, cfg)
    spread = _spread(quadrants)
    record.update(residual_dy=r_dy, residual_dx=r_dx, residual_magnitude=r_mag, quadrant_estimates=[list(q) if q is not None else None for q in quadrants], quadrant_spread=spread)

    q_tol, factor = float(spec.quadrant_tolerance_hr_px), float(spec.uncertain_factor)
    if r_mag > factor * tol:
        return finish("not_eligible", f"residual displacement {r_mag:.2f} HR px after the correction exceeds {factor:g} x the tolerance {tol}")
    if spread is not None and spread > factor * q_tol:
        return finish("not_eligible", f"the four quadrants disagree by {spread:.2f} HR px (> {factor:g} x {q_tol}): not one global translation")
    if spread is None:
        return finish("uncertain", "alignment not verifiable: fewer than two quadrants could be registered")
    if r_mag > tol or spread > q_tol:
        return finish("uncertain", f"residual {r_mag:.2f} HR px (tolerance {tol}), quadrant spread {spread:.2f} HR px (tolerance {q_tol}): between the accepted and the rejected range")
    return finish("eligible", f"residual {r_mag:.2f} HR px within {tol}, quadrant spread {spread:.2f} HR px within {q_tol}")
