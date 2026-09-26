"""frame.reliability.alignment / eligibility -- which evidence may enter the stability-vs-error analysis, and why the rest may not (Phase 6)."""

from __future__ import annotations

import numpy as np
import pytest
from scipy import ndimage

from frame.evaluate import metrics as M
from frame.reliability.alignment import ALIGNED_NONE_REQUIRED, ALIGNED_TRANSLATION, apply_correction, classify_alignment
from frame.reliability.config import AlignmentSpec, EligibilitySpec
from frame.reliability.eligibility import (
    EXCLUDED,
    REASONS,
    check_prediction,
    evaluate_reference,
    exclusion_row,
)

CFG = M.MetricConfig()
SIZE = 256


def periodic_field(seed, size=SIZE, bands=4):
    """A band-limited, exactly periodic texture (a reflectance-like field), so that circular shifts are exact translations."""
    rng = np.random.default_rng(seed)
    fy, fx = np.fft.fftfreq(size)[:, None], np.fft.fftfreq(size)[None, :]
    radius = np.sqrt(fy**2 + fx**2)
    amp = 1.0 / (radius + 0.02) ** 1.3
    amp[0, 0] = 0
    out = []
    for _ in range(bands):
        spectrum = amp * np.exp(2j * np.pi * rng.random((size, size)))
        f = np.real(np.fft.ifft2(spectrum))
        out.append(f / f.std())
    return (0.25 + 0.05 * np.stack(out)).astype("float64")


def blurred(hr, shift=(0, 0), sigma=2.0):
    """What a bicubic baseline is against the sharper reference: smoother, and (optionally) displaced."""
    x = ndimage.gaussian_filter(hr, (0, sigma, sigma), mode="wrap")
    if shift != (0, 0):
        x = ndimage.shift(x, (0, *shift), order=3, mode="grid-wrap")
    return x


def full(size=SIZE):
    return np.ones((size, size), bool)


def classify(baseline, hr, mask=None, **spec):
    return classify_alignment(baseline, hr, full() if mask is None else mask, AlignmentSpec(**spec), CFG)


# ============================================================================== registration classes


def test_a_well_registered_reference_is_eligible_and_nothing_is_corrected():
    hr = periodic_field(0)
    r = classify(blurred(hr), hr)
    assert r["status"] == "eligible" and r["correction_method"] == ALIGNED_NONE_REQUIRED and r["correction"] == [0, 0] and r["applied"] is False
    assert r["residual_magnitude"] <= 0.5 and r["raw_magnitude"] <= 0.5 and r["reason"] is None


def test_a_sub_tolerance_displacement_is_accepted_as_is():
    hr = periodic_field(1)
    r = classify(blurred(hr, shift=(0.3, -0.2)), hr)
    assert r["status"] == "eligible" and r["correction_method"] == ALIGNED_NONE_REQUIRED and r["applied"] is False


def test_a_whole_pixel_translation_is_corrected_and_the_correction_is_recorded():
    hr = periodic_field(2)
    r = classify(blurred(hr, shift=(2, -1)), hr)
    assert r["status"] == "eligible" and r["correction_method"] == ALIGNED_TRANSLATION and r["applied"] is True
    assert r["raw_dy"] == pytest.approx(2.0, abs=0.35) and r["raw_dx"] == pytest.approx(-1.0, abs=0.35)
    assert r["correction"] == [-2, 1]                                                       # what has to be done to the SR grid: displace it back
    assert r["residual_magnitude"] <= 0.5 and r["overlap_fraction"] == pytest.approx((SIZE - 2) * (SIZE - 1) / SIZE**2)
    assert r["crop_rows"] == [0, SIZE - 2] or r["crop_rows"] == [2, SIZE]                   # the window that was kept is recorded


def test_a_displacement_beyond_the_correctable_range_is_not_eligible_with_the_machine_readable_reason():
    hr = periodic_field(3)
    r = classify(blurred(hr, shift=(12, 9)), hr)
    assert r["status"] == "not_eligible" and r["reason"] == "reference_alignment_invalid" and r["applied"] is False
    assert "max_correction" in r["detail"]


def test_without_permission_to_correct_a_displaced_reference_is_not_eligible():
    hr = periodic_field(4)
    r = classify(blurred(hr, shift=(2, 0)), hr, apply_translation_correction=False)
    assert r["status"] == "not_eligible" and r["reason"] == "reference_alignment_invalid"
    assert classify(blurred(hr, shift=(0.2, 0.0)), hr, apply_translation_correction=False)["status"] == "eligible"      # nothing to correct: still fine


def test_a_misregistration_that_is_not_one_global_translation_is_never_eligible():
    hr = periodic_field(5)
    base = blurred(hr)
    left = blurred(hr, shift=(0, 3))[:, :, : SIZE // 2]
    right = blurred(hr, shift=(0, -3))[:, :, SIZE // 2:]
    r = classify(np.concatenate([left, right], axis=2), hr)
    assert r["status"] in ("not_eligible", "uncertain") and r["status"] != "eligible"
    assert r["reason"] in ("reference_alignment_invalid", "reference_alignment_uncertain") and base.shape == hr.shape
    assert r["quadrant_spread"] is not None and r["quadrant_spread"] > 1.0


def test_quadrants_that_disagree_reject_the_tile_even_when_the_whole_tile_residual_is_tolerated():
    """The spread test is independent of the residual test: here the residual tolerance is generous, only the quadrant tolerance can reject."""
    hr = periodic_field(5)
    half = SIZE // 2
    pieces = [blurred(hr, shift=(0, 3)), blurred(hr, shift=(0, -3))]
    top = np.concatenate([pieces[0][:, :half, :half], pieces[1][:, :half, half:]], axis=2)
    bottom = np.concatenate([pieces[1][:, half:, :half], pieces[0][:, half:, half:]], axis=2)
    r = classify(np.concatenate([top, bottom], axis=1), hr, tolerance_hr_px=6.0, quadrant_tolerance_hr_px=0.5, uncertain_factor=2.0, max_correction_hr_px=8)
    assert r["status"] == "not_eligible" and "quadrants disagree" in r["detail"] and r["quadrant_spread"] > 1.0


def test_a_borderline_disagreement_is_uncertain_not_eligible_and_not_rejected():
    hr = periodic_field(6)
    left = blurred(hr, shift=(0, 0.9))[:, :, : SIZE // 2]
    right = blurred(hr)[:, :, SIZE // 2:]
    r = classify(np.concatenate([left, right], axis=2), hr, quadrant_tolerance_hr_px=0.4, tolerance_hr_px=0.4, uncertain_factor=4.0)
    assert r["status"] == "uncertain" and r["reason"] == "reference_alignment_uncertain" and 0.4 < r["quadrant_spread"] <= 1.6


def test_an_image_that_cannot_be_registered_is_uncertain_never_silently_eligible():
    flat = np.full((4, SIZE, SIZE), 0.2)
    r = classify(flat, flat)
    assert r["status"] == "uncertain" and r["reason"] == "reference_alignment_uncertain" and "not_computable" in r["detail"]


def test_the_classification_is_deterministic():
    hr = periodic_field(7)
    a, b = classify(blurred(hr, shift=(2, 1)), hr), classify(blurred(hr, shift=(2, 1)), hr)
    assert a == b


def test_registration_uses_only_valid_pixels():
    hr = periodic_field(8)
    mask = full()
    mask[:, : SIZE // 3] = False
    hr_junk = hr.copy()
    hr_junk[:, :, : SIZE // 3] = 9.0                                                         # nodata content that must not steer the estimate
    r = classify(blurred(hr, shift=(2, 0)), hr_junk, mask)
    assert r["status"] == "eligible" and r["correction"] == [-2, 0]


def test_the_correction_crops_every_grid_identically():
    hr = periodic_field(9)
    stack = np.concatenate([blurred(hr, shift=(2, -1)), hr], axis=0)                       # two grids that live on the SR side
    mask = full()
    cropped, hr_c, mask_c = apply_correction(stack, hr, mask, [-2, 1])
    assert cropped.shape == (8, SIZE - 2, SIZE - 1) and hr_c.shape == (4, SIZE - 2, SIZE - 1) and mask_c.shape == (SIZE - 2, SIZE - 1)
    est = classify_alignment(cropped[:4], hr_c, mask_c, AlignmentSpec(), CFG)
    assert est["correction_method"] == ALIGNED_NONE_REQUIRED and est["raw_magnitude"] <= 0.5          # after the crop the pair is registered, and no further correction is asked for
    same, hr_same, _ = apply_correction(stack, hr, mask, [0, 0])
    assert np.array_equal(same, stack) and np.array_equal(hr_same, hr)


# ============================================================================== the reference gate


class Sample:
    """Just what the gate reads from an EvalSample."""

    def __init__(self, hr, mask, quality=None):
        import torch

        self.hr, self.hr_mask = torch.from_numpy(hr.astype("float32")), torch.from_numpy(mask)
        total = float(mask.size)
        self.quality = {"hr_valid_fraction": float(mask.mean()), "hr_nonfinite_fraction": 0.0, "hr_dataset_nodata_fraction": float(1 - mask.mean()), "hr_valid_pixels": int(mask.sum()),
                        "hr_total_pixels": total, **(quality or {})}


def gate(sample, baseline, **over):
    elig = EligibilitySpec(**{k: v for k, v in over.items() if k in EligibilitySpec.__dataclass_fields__})
    align = AlignmentSpec(**{k: v for k, v in over.items() if k in AlignmentSpec.__dataclass_fields__})
    return evaluate_reference(sample, baseline, elig, align, CFG)


def test_a_good_reference_passes_the_gate_with_full_provenance():
    hr = periodic_field(10)
    g = gate(Sample(hr, full()), blurred(hr))
    assert g["status"] == "eligible" and g["reason"] is None and g["alignment"]["status"] == "eligible"
    assert g["valid_fraction_after_alignment"] == pytest.approx(1.0) and g["evidence_level"] == "pixel_level_eligible"


def test_a_missing_reference_is_excluded_with_its_reason():
    g = evaluate_reference(None, blurred(periodic_field(11)), EligibilitySpec(), AlignmentSpec(), CFG)
    assert g["status"] == "not_eligible" and g["reason"] == "reference_missing" and g["alignment"] is None


def test_too_few_valid_pixels_exclude_the_tile_before_registration_is_even_attempted():
    hr = periodic_field(12)
    mask = full()
    mask[: int(0.7 * SIZE)] = False                                                         # 30% valid, threshold 50%
    g = gate(Sample(hr, mask), blurred(hr))
    assert g["status"] == "not_eligible" and g["reason"] == "insufficient_valid_pixels" and g["alignment"] is None and g["valid_fraction"] == pytest.approx(0.3, abs=0.01)


def test_an_all_nodata_tile_is_excluded_not_scored_as_zero_error():
    hr = periodic_field(13)
    g = gate(Sample(hr, np.zeros((SIZE, SIZE), bool)), blurred(hr))
    assert g["status"] == "not_eligible" and g["reason"] == "insufficient_valid_pixels" and g["valid_fraction"] == 0.0


def test_a_reference_with_non_finite_pixels_is_excluded():
    hr = periodic_field(14)
    g = gate(Sample(hr, full(), quality={"hr_nonfinite_fraction": 0.002}), blurred(hr))
    assert g["status"] == "not_eligible" and g["reason"] == "reference_not_finite"
    assert gate(Sample(hr, full(), quality={"hr_nonfinite_fraction": 0.002}), blurred(hr), max_nonfinite_fraction=0.01)["status"] == "eligible"


def test_a_shifted_reference_is_excluded_by_the_gate_with_the_alignment_reason_and_record():
    hr = periodic_field(15)
    g = gate(Sample(hr, full()), blurred(hr, shift=(14, 0)))
    assert g["status"] == "not_eligible" and g["reason"] == "reference_alignment_invalid" and g["evidence_level"] == "not_eligible"
    assert g["alignment"]["reason"] == "reference_alignment_invalid" and g["alignment"]["raw_dy"] > 4


def test_an_uncertain_alignment_is_its_own_evidence_level_and_is_still_excluded():
    flat = np.full((4, SIZE, SIZE), 0.2)
    g = gate(Sample(flat, full()), flat)
    assert g["status"] == "uncertain" and g["reason"] == "reference_alignment_uncertain" and g["evidence_level"] == "uncertain"


def test_the_valid_fraction_is_checked_again_after_the_alignment_crop():
    hr = periodic_field(16)
    mask = full()
    mask[: SIZE - 51] = False                                                              # only the last 51 rows (19.9%) are valid
    base = blurred(hr, shift=(3, 0))                                                       # correction (-3, 0) crops the LAST 3 rows: valid rows 205..252 of 253 remain (19.0%)
    passes = gate(Sample(hr, mask), base, min_valid_fraction=0.18)
    assert passes["status"] == "eligible" and passes["valid_fraction_after_alignment"] == pytest.approx(48 / 253, abs=1e-3)
    fails = gate(Sample(hr, mask), base, min_valid_fraction=0.195)                       # passes before the crop (0.199), fails after it (0.190)
    assert fails["status"] == "not_eligible" and fails["reason"] == "insufficient_valid_pixels" and fails["alignment"]["status"] == "eligible"
    assert "after the alignment crop" in fails["detail"]


def test_reasons_are_a_closed_machine_readable_vocabulary():
    assert set(REASONS) >= {"reference_missing", "reference_geometry_invalid", "reference_not_finite", "insufficient_valid_pixels", "reference_alignment_invalid",
                            "reference_alignment_uncertain", "prediction_not_finite", "model_failure", "tta_member_failure", "too_few_ensemble_members"}


# ============================================================================== prediction-side checks


def test_a_finite_prediction_passes():
    ok = check_prediction(np.zeros((4, 8, 8)), np.ones((4, 8, 8)) * 1e-3, full(8), n_members=6, min_members=2)
    assert ok is None


def test_a_non_finite_prediction_or_spread_on_valid_pixels_is_excluded_but_junk_under_the_mask_is_not():
    mean, std = np.zeros((4, 8, 8)), np.full((4, 8, 8), 1e-3)
    mean[0, 2, 2] = np.nan
    assert check_prediction(mean, std, full(8), n_members=6, min_members=2)["reason"] == "prediction_not_finite"
    mask = full(8)
    mask[2, 2] = False
    assert check_prediction(mean, std, mask, n_members=6, min_members=2) is None            # the NaN sits on a nodata pixel that is never scored
    std2 = np.full((4, 8, 8), 1e-3)
    std2[1, 5, 5] = np.inf
    assert check_prediction(np.zeros((4, 8, 8)), std2, full(8), n_members=6, min_members=2)["reason"] == "prediction_not_finite"


def test_an_ensemble_with_too_few_members_is_excluded_because_its_zero_spread_is_not_stability():
    r = check_prediction(np.zeros((4, 8, 8)), np.zeros((4, 8, 8)), full(8), n_members=1, min_members=2)
    assert r["reason"] == "too_few_ensemble_members"


# ============================================================================== the machine-readable exclusion row


def test_the_exclusion_row_is_explicit_and_never_a_zero():
    hr = periodic_field(17)
    g = gate(Sample(hr, full()), blurred(hr, shift=(14, 0)))
    row = exclusion_row(dataset="neon", system=None, sample_id="s1", scene_unit="u1", category="Forest", gate=g)
    assert row["type"] == "tile_result" and row["status"] == EXCLUDED == "excluded_from_uncertainty_error_analysis"
    assert row["reason"] == "reference_alignment_invalid" and row["evidence_level"] == "not_eligible" and row["alignment"]["status"] == "not_eligible"
    assert "metrics" not in row and "error" not in row                                     # no invented numbers on an excluded tile
    assert exclusion_row(dataset="d", system="lite", sample_id="s", scene_unit="u", category=None, reason="model_failure", detail="boom")["system"] == "lite"
    with pytest.raises(ValueError, match="reason"):
        exclusion_row(dataset="d", system=None, sample_id="s", scene_unit="u", category=None, reason="because")


# ============================================================================== sub-pixel resolution also where there is nodata


from frame.reliability.alignment import estimate_displacement          # noqa: E402


def _edge_mask(fraction=0.35):
    mask = full()
    mask[:, : int(fraction * SIZE)] = False                            # a nodata strip like the edge of an airborne flight line
    mask[: SIZE // 5, :] = mask[: SIZE // 5, :] & (np.arange(SIZE) > SIZE // 2)[None, :]
    return mask


@pytest.mark.parametrize("shift", [(0.0, 0.0), (0.3, -0.2), (0.5, 0.0), (-0.6, 0.4), (1.4, -0.7)])
def test_fractional_displacements_are_recovered_to_a_tenth_of_a_pixel_with_or_without_nodata(shift):
    hr = periodic_field(21)
    base = blurred(hr, shift=shift)
    for mask in (full(), _edge_mask()):
        est = estimate_displacement(base, hr, mask, CFG)
        assert est["status"] == "ok" and est["dy"] == pytest.approx(shift[0], abs=0.15) and est["dx"] == pytest.approx(shift[1], abs=0.15), (shift, est)


def test_the_refinement_uses_only_valid_pixels():
    hr = periodic_field(22)
    mask = _edge_mask()
    junk = hr.copy()
    junk[:, :, : int(0.35 * SIZE)] = 9.0
    est = estimate_displacement(blurred(hr, shift=(0.5, -0.3)), junk, mask, CFG)
    assert est["dy"] == pytest.approx(0.5, abs=0.15) and est["dx"] == pytest.approx(-0.3, abs=0.15)


def test_an_image_that_cannot_be_registered_has_no_displacement():
    flat = np.full((4, SIZE, SIZE), 0.2)
    assert estimate_displacement(flat, flat, full(), CFG)["status"] == "not_computable"
    assert estimate_displacement(periodic_field(23), periodic_field(23), np.zeros((SIZE, SIZE), bool), CFG)["status"] == "not_computable"


def test_a_half_pixel_displacement_on_a_tile_with_nodata_is_no_longer_read_as_a_whole_pixel():
    """The failure this refinement exists for: with nodata the Phase 5 estimator is integer-valued, so a true 0.5 px shift read as +/-1 px, and a correction of one pixel moved it to -1 px."""
    hr = periodic_field(24)
    base = blurred(hr, shift=(0.5, 0.0))
    r = classify(base, hr, _edge_mask())
    assert r["status"] == "eligible" and r["correction_method"] == ALIGNED_NONE_REQUIRED and r["raw_magnitude"] <= 0.6 and r["residual_magnitude"] <= 0.6


# ============================================================================== the gate is versioned so that later analyses can record which rule admitted their evidence


def test_the_gate_carries_a_version_that_downstream_analyses_record():
    from frame.reliability.eligibility import GATE_VERSION

    assert GATE_VERSION.startswith("frame-reliability-gate/")
    hr = periodic_field(30)
    assert gate(Sample(hr, full()), blurred(hr))["gate_version"] == GATE_VERSION
    assert gate(Sample(hr, np.zeros((SIZE, SIZE), bool)), blurred(hr))["gate_version"] == GATE_VERSION           # exclusions carry it too
