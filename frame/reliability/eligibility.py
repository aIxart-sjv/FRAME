"""The eligibility gate (Phase 6): what may enter the stability-vs-error analysis, and the explicit reason for everything that may not.

Evidence is judged in this order and the FIRST failure is the recorded reason; nothing is ever silently dropped and an excluded tile never receives an error number:

1. ``reference_missing``            there is no reference.
2. ``reference_geometry_invalid``   the reference grid does not match the prediction grid.
3. ``reference_not_finite``         non-finite reference pixels beyond the tolerated fraction.
4. ``insufficient_valid_pixels``    the strict valid fraction (Phase 5 evaluation mask) is below the threshold; an all-nodata tile is this, not "zero error".
5. ``reference_alignment_invalid`` / ``reference_alignment_uncertain``   the registration gate (frame.reliability.alignment).
6. ``insufficient_valid_pixels``    again, after the alignment crop.

After inference (:func:`check_prediction`): ``prediction_not_finite`` (non-finite mean or spread on a valid pixel), ``too_few_ensemble_members`` (a one-member spread is zero by
construction), and, raised by the runner, ``model_failure`` / ``tta_member_failure``.

The evidence level is ``pixel_level_eligible``, ``uncertain`` or ``not_eligible``; only the first enters any correlation.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from frame.evaluate.metrics import MetricConfig
from frame.reliability.alignment import classify_alignment
from frame.reliability.config import AlignmentSpec, EligibilitySpec

#: the version of the gate (its order of checks, the registration rule and its estimator); a later analysis records it beside the settings it used
GATE_VERSION = "frame-reliability-gate/1"
EXCLUDED = "excluded_from_uncertainty_error_analysis"
REASONS = (
    "reference_missing", "reference_geometry_invalid", "reference_not_finite", "insufficient_valid_pixels", "reference_alignment_invalid", "reference_alignment_uncertain",
    "prediction_not_finite", "too_few_ensemble_members", "model_failure", "tta_member_failure", "sample_unreadable",
)
LEVEL_ELIGIBLE, LEVEL_UNCERTAIN, LEVEL_NOT = "pixel_level_eligible", "uncertain", "not_eligible"


def _result(status: str, reason: Optional[str], detail: Optional[str], **extra: Any) -> Dict[str, Any]:
    level = {"eligible": LEVEL_ELIGIBLE, "uncertain": LEVEL_UNCERTAIN, "not_eligible": LEVEL_NOT}[status]
    return {"gate_version": GATE_VERSION, "status": status, "reason": reason, "detail": detail, "evidence_level": level, "valid_fraction": extra.pop("valid_fraction", None),
            "nonfinite_fraction": extra.pop("nonfinite_fraction", None), "alignment": extra.pop("alignment", None), "valid_fraction_after_alignment": extra.pop("valid_fraction_after_alignment", None)}


def evaluate_reference(sample: Any, baseline: np.ndarray, eligibility: EligibilitySpec, alignment: AlignmentSpec, cfg: MetricConfig) -> Dict[str, Any]:
    """Judge one tile's reference (``sample`` is an EvalSample-like object or None) against the bicubic ``baseline`` on the SR grid."""
    if sample is None or getattr(sample, "hr", None) is None:
        return _result("not_eligible", "reference_missing", "no HR reference for this sample")
    hr = sample.hr.numpy() if hasattr(sample.hr, "numpy") else np.asarray(sample.hr)
    mask = sample.hr_mask.numpy() if hasattr(sample.hr_mask, "numpy") else np.asarray(sample.hr_mask)
    mask = mask.astype(bool)
    if hr.shape != np.asarray(baseline).shape or mask.shape != hr.shape[-2:]:
        return _result("not_eligible", "reference_geometry_invalid", f"reference {tuple(hr.shape)} does not match the prediction grid {tuple(np.asarray(baseline).shape)}")
    q = sample.quality
    nonfinite, valid_fraction = float(q.get("hr_nonfinite_fraction", 0.0)), float(mask.mean())
    common = {"valid_fraction": valid_fraction, "nonfinite_fraction": nonfinite}
    if nonfinite > eligibility.max_nonfinite_fraction:
        return _result("not_eligible", "reference_not_finite", f"{nonfinite:.4f} of the reference pixels are not finite (limit {eligibility.max_nonfinite_fraction})", **common)
    if valid_fraction < eligibility.min_valid_fraction:
        return _result("not_eligible", "insufficient_valid_pixels", f"valid fraction {valid_fraction:.4f} < {eligibility.min_valid_fraction}", **common)
    record = classify_alignment(baseline, hr, mask, alignment, cfg)
    if record["status"] != "eligible":
        return _result(record["status"], record["reason"], record["detail"], alignment=record, **common)
    after = _valid_fraction_after_crop(mask, record)
    if after < eligibility.min_valid_fraction:
        return _result("not_eligible", "insufficient_valid_pixels", f"valid fraction {after:.4f} after the alignment crop < {eligibility.min_valid_fraction}", alignment=record,
                       valid_fraction_after_alignment=after, **common)
    return _result("eligible", None, record["detail"], alignment=record, valid_fraction_after_alignment=after, **common)


def _valid_fraction_after_crop(mask: np.ndarray, record: Dict[str, Any]) -> float:
    r0, r1 = record["crop_rows"]
    c0, c1 = record["crop_cols"]
    return float(mask[r0:r1, c0:c1].mean())


def check_prediction(mean: np.ndarray, std: np.ndarray, mask: np.ndarray, *, n_members: int, min_members: int) -> Optional[Dict[str, str]]:
    """None when the prediction may be analysed, else ``{"reason", "detail"}``. Non-finite values under the mask (nodata) are never scored and do not disqualify a tile."""
    if n_members < min_members:
        return {"reason": "too_few_ensemble_members", "detail": f"{n_members} ensemble member(s) < {min_members}: a one-member spread is zero by construction, not stability"}
    valid = np.asarray(mask, dtype=bool)
    for name, arr in (("mean prediction", mean), ("ensemble spread", std)):
        a = np.asarray(arr)
        bad = ~np.isfinite(a) & valid[None] if a.ndim == 3 else ~np.isfinite(a) & valid
        if bad.any():
            return {"reason": "prediction_not_finite", "detail": f"{int(bad.sum())} non-finite value(s) in the {name} on valid pixels"}
    return None


def exclusion_row(*, dataset: str, system: Optional[str], sample_id: str, scene_unit: str, category: Optional[str], gate: Optional[Dict[str, Any]] = None,
                  reason: Optional[str] = None, detail: Optional[str] = None) -> Dict[str, Any]:
    """The machine-readable row of an excluded tile: status ``excluded_from_uncertainty_error_analysis``, a reason from :data:`REASONS`, and NO metric fields."""
    code = gate["reason"] if gate is not None else reason
    if code not in REASONS:
        raise ValueError(f"reason must be one of {REASONS}, got {code!r}")
    row: Dict[str, Any] = {"type": "tile_result", "status": EXCLUDED, "dataset": dataset, "system": system, "sample_id": sample_id, "scene_unit": scene_unit, "category": category,
                           "reason": code, "detail": gate["detail"] if gate is not None else detail}
    if gate is not None:
        row.update(evidence_level=gate["evidence_level"], alignment=gate["alignment"], valid_fraction=gate["valid_fraction"], nonfinite_fraction=gate["nonfinite_fraction"],
                   valid_fraction_after_alignment=gate["valid_fraction_after_alignment"])
    else:
        row["evidence_level"] = None
    return row
