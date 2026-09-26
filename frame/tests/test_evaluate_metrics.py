"""frame.evaluate.metrics -- known values, nodata masking, and the separation of reference accuracy from self-consistency."""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch
from scipy import ndimage

from frame.evaluate import metrics as M

BANDS = ("B04", "B03", "B02", "B08")
CFG = M.MetricConfig()


def smooth_field(seed=0, size=64, bands=4):
    rng = np.random.default_rng(seed)
    x = rng.random((bands, size // 4, size // 4)).astype("float32")
    up = np.stack([ndimage.zoom(b, 4, order=3, mode="reflect") for b in x])
    return (0.05 + 0.4 * np.clip(up, 0, 1)).astype("float32")


def full_mask(size=64):
    return np.ones((size, size), dtype=bool)


# ============================================================================== reference accuracy


def test_a_constant_offset_has_hand_computable_errors():
    hr = np.full((4, 64, 64), 0.5, dtype="float32")
    sr = hr + 0.1
    r = M.reference_accuracy(sr, hr, full_mask(), BANDS, 4)
    assert r["rmse"] == pytest.approx(0.1, abs=1e-6) and r["mae"] == pytest.approx(0.1, abs=1e-6)
    assert r["psnr_db"] == pytest.approx(20.0, abs=1e-4)                                    # 10 log10(1 / 0.1^2), data range 1
    assert r["ergas"] == pytest.approx(100 / 4 * 0.2, abs=1e-4)                             # 100/scale * sqrt(mean((rmse / mean_hr)^2)) = 25 * 0.2
    assert r["n_valid_pixels"] == 64 * 64 and r["valid_fraction"] == 1.0


def test_mae_and_rmse_differ_when_errors_are_uneven():
    hr = np.full((4, 64, 64), 0.5, dtype="float32")
    sr = hr.copy()
    sr[:, :, :32] += 0.2                                                                    # half the pixels are off by 0.2
    r = M.reference_accuracy(sr, hr, full_mask(), BANDS, 4)
    assert r["mae"] == pytest.approx(0.1, abs=1e-6) and r["rmse"] == pytest.approx(math.sqrt(0.5 * 0.04), abs=1e-6) and r["rmse"] > r["mae"]


def test_identical_images_are_perfect_and_psnr_is_undefined_not_a_huge_number():
    hr = smooth_field()
    r = M.reference_accuracy(hr.copy(), hr, full_mask(), BANDS, 4)
    # SAM comes from the reused opensr-test distance, a float32 per-pixel arccos: identical spectra score ~0.006 degrees (measured), not exactly 0.
    # That numerical floor is negligible next to real values (1-3 degrees) and is documented in docs/EVALUATION.md rather than hidden.
    assert r["rmse"] == 0.0 and r["mae"] == 0.0 and r["sam_degrees"] < 0.02 and r["ssim"] == pytest.approx(1.0, abs=1e-6)
    assert r["psnr_db"] is None                                                             # infinite PSNR is reported as missing, never propagated


def test_nodata_pixels_are_never_scored_as_valid():
    hr = smooth_field()
    sr = hr + 0.02
    mask = full_mask()
    mask[:32, :] = False
    base = M.reference_accuracy(sr, hr, mask, BANDS, 4)
    sr_bad, hr_bad = sr.copy(), hr.copy()
    sr_bad[:, :32, :] = 1e3                                                                 # junk under the mask
    hr_bad[:, :32, :] = 0.0
    dirty = M.reference_accuracy(sr_bad, hr_bad, mask, BANDS, 4)
    for key in ("rmse", "mae", "psnr_db", "sam_degrees", "ergas", "ssim"):
        assert dirty[key] == pytest.approx(base[key], rel=1e-6, abs=1e-9), key
    assert base["n_valid_pixels"] == 32 * 64 and base["valid_fraction"] == 0.5


def test_a_fully_masked_tile_is_not_computable_rather_than_zero():
    r = M.reference_accuracy(smooth_field(), smooth_field(1), np.zeros((64, 64), dtype=bool), BANDS, 4)
    assert r["n_valid_pixels"] == 0 and r["rmse"] is None and r["mae"] is None and r["ssim"] is None and r["psnr_db"] is None


def test_ssim_falls_with_noise_and_only_counts_pixels_away_from_nodata():
    hr = smooth_field()
    rng = np.random.default_rng(0)
    noisy = hr + 0.05 * rng.standard_normal(hr.shape).astype("float32")
    assert M.reference_accuracy(noisy, hr, full_mask(), BANDS, 4)["ssim"] < 0.9
    mask = full_mask()
    mask[:, 30:34] = False                                                                  # a nodata stripe: SSIM windows must not straddle it
    assert M.reference_accuracy(hr.copy(), hr, mask, BANDS, 4)["ssim"] == pytest.approx(1.0, abs=1e-6)


def test_the_overall_ssim_agrees_with_the_existing_reference_implementation():
    from frame.validation import compute_reference_metrics

    hr = smooth_field()
    sr = hr + 0.03 * np.random.default_rng(1).standard_normal(hr.shape).astype("float32")
    ours = M.reference_accuracy(sr, hr, full_mask(), BANDS, 4)
    theirs = compute_reference_metrics(torch.from_numpy(sr), torch.from_numpy(hr), full_mask(), band_names=list(BANDS), scale_factor=4)
    assert ours["psnr_db"] == pytest.approx(theirs.psnr_db) and ours["rmse"] == pytest.approx(theirs.rmse) and ours["sam_degrees"] == pytest.approx(theirs.sam_degrees)
    assert ours["ergas"] == pytest.approx(theirs.ergas) and ours["ssim"] == pytest.approx(theirs.ssim, abs=0.01)


def test_torch_and_numpy_inputs_give_the_same_answer():
    hr = smooth_field()
    sr = hr + 0.01
    a = M.reference_accuracy(sr, hr, full_mask(), BANDS, 4)
    b = M.reference_accuracy(torch.from_numpy(sr), torch.from_numpy(hr), torch.from_numpy(full_mask()), BANDS, 4)
    assert a == b


def test_shape_mismatches_between_sr_and_reference_are_rejected():
    from frame.evaluate.errors import ReferenceMismatchError

    with pytest.raises(ReferenceMismatchError, match="shape"):
        M.reference_accuracy(np.zeros((4, 32, 32), "float32"), np.zeros((4, 64, 64), "float32"), full_mask(), BANDS, 4)
    with pytest.raises(ReferenceMismatchError, match="mask"):
        M.reference_accuracy(np.zeros((4, 64, 64), "float32"), np.zeros((4, 64, 64), "float32"), np.ones((32, 32), bool), BANDS, 4)


# ============================================================================== per band


def test_per_band_errors_recover_the_per_band_offsets():
    hr = smooth_field()
    offsets = np.array([0.01, 0.02, 0.03, 0.04], dtype="float32")
    sr = hr + offsets[:, None, None]
    per = M.per_band_metrics(sr, hr, full_mask(), BANDS)
    assert list(per) == list(BANDS)
    for band, off in zip(BANDS, offsets):
        assert per[band]["bias"] == pytest.approx(off, abs=1e-6) and per[band]["rmse"] == pytest.approx(off, abs=1e-6) and per[band]["mae"] == pytest.approx(off, abs=1e-6)
        assert per[band]["pbias_percent"] == pytest.approx(100 * off / hr[BANDS.index(band)].mean(), rel=1e-4) and per[band]["pearson_r"] == pytest.approx(1.0, abs=1e-6)


def test_per_band_metrics_respect_the_mask():
    hr = smooth_field()
    sr = hr + 0.05
    mask = full_mask()
    mask[:, :32] = False
    sr[:, :, :32] = 9.0
    assert M.per_band_metrics(sr, hr, mask, BANDS)["B08"]["rmse"] == pytest.approx(0.05, abs=1e-6)


# ============================================================================== indices and ratios


def test_identical_images_have_zero_index_error():
    hr = smooth_field()
    idx = M.index_metrics(hr.copy(), hr, full_mask(), BANDS, CFG)
    for name in ("NDVI", "NDWI"):
        assert idx[name]["mae"] == 0.0 and idx[name]["rmse"] == 0.0 and idx[name]["bias"] == 0.0 and idx[name]["pearson_r"] == pytest.approx(1.0)


def test_an_nir_offset_shifts_ndvi_by_the_analytic_amount():
    hr = np.zeros((4, 8, 8), dtype="float32")
    hr[0], hr[3] = 0.1, 0.3                                                                  # red 0.1, NIR 0.3 -> NDVI 0.5
    sr = hr.copy()
    sr[3] = 0.4                                                                              # NDVI 0.6
    idx = M.index_metrics(sr, hr, np.ones((8, 8), bool), BANDS, CFG)
    assert idx["NDVI"]["bias"] == pytest.approx(0.1, abs=1e-6) and idx["NDVI"]["mae"] == pytest.approx(0.1, abs=1e-6)
    hr[1] = 0.2                                                                              # green 0.2 -> NDWI = (0.2 - 0.3) / 0.5 = -0.2
    sr[1] = 0.2                                                                              # SR NDWI = (0.2 - 0.4) / 0.6
    idx = M.index_metrics(sr, hr, np.ones((8, 8), bool), BANDS, CFG)
    assert idx["NDWI"]["bias"] == pytest.approx((0.2 - 0.4) / 0.6 - (-0.2), abs=1e-6)


def test_pixels_with_a_near_zero_reflectance_sum_are_excluded_and_counted():
    hr = np.full((4, 8, 8), 0.3, dtype="float32")
    hr[0, :4], hr[3, :4] = 0.004, 0.004                                                      # NIR + red = 0.008 < 0.02: index is noise-dominated
    idx = M.index_metrics(hr * 1.01, hr, np.ones((8, 8), bool), BANDS, CFG)["NDVI"]
    assert idx["n_valid"] == 32 and idx["excluded_fraction"] == pytest.approx(0.5)


def test_indices_that_need_swir_bands_are_recorded_as_skipped_not_invented():
    idx = M.index_metrics(smooth_field(), smooth_field(), full_mask(), BANDS, CFG)
    assert set(idx["skipped"]) == {"MNDWI", "NDBI", "BSI"} and all("B11" in why for why in idx["skipped"].values())


def test_index_distribution_difference_is_reported():
    rng = np.random.default_rng(0)
    hr = smooth_field()
    sr = hr.copy()
    sr[3] *= 1.3
    assert M.index_metrics(sr, hr, full_mask(), BANDS, CFG)["NDVI"]["wasserstein"] > 0.0
    assert M.index_metrics(hr.copy(), hr, full_mask(), BANDS, CFG)["NDVI"]["wasserstein"] == pytest.approx(0.0, abs=1e-9)


def test_band_ratio_error_is_the_log_ratio_difference():
    hr = smooth_field()
    sr = hr.copy()
    sr[3] *= 1.1                                                                             # NIR too high by 10 %: log(B08/B04) off by ln(1.1)
    r = M.band_ratio_metrics(sr, hr, full_mask(), BANDS, CFG)["B08/B04"]
    assert r["bias_log_ratio"] == pytest.approx(math.log(1.1), abs=1e-5) and r["mae_log_ratio"] == pytest.approx(math.log(1.1), abs=1e-5)
    assert M.band_ratio_metrics(hr.copy(), hr, full_mask(), BANDS, CFG)["B03/B04"]["mae_log_ratio"] == 0.0


# ============================================================================== spatial / detail


def textured(seed=0, size=128):
    rng = np.random.default_rng(seed)
    return (0.2 + 0.1 * ndimage.gaussian_filter(rng.standard_normal((4, size, size)), (0, 1.0, 1.0)) * 4).astype("float32")


def test_identical_images_have_perfect_detail_and_edge_agreement():
    hr = textured()
    s = M.spatial_detail_metrics(hr.copy(), hr, np.ones((128, 128), bool), BANDS, CFG)
    assert s["hf_correlation"] == pytest.approx(1.0, abs=1e-6) and s["hf_energy_ratio"] == pytest.approx(1.0, abs=1e-6)
    assert s["hf_relative_error"] == pytest.approx(0.0, abs=1e-6) and s["gradient_correlation"] == pytest.approx(1.0, abs=1e-6)


def test_a_blurred_reconstruction_has_lost_detail_and_is_not_credited_with_synthesis():
    hr = textured()
    blurred = ndimage.gaussian_filter(hr, (0, 2.5, 2.5))
    s = M.spatial_detail_metrics(blurred, hr, np.ones((128, 128), bool), BANDS, CFG)
    assert s["hf_energy_ratio"] < 0.6 and 0.0 < s["hf_correlation"] < 1.0 and s["hf_relative_error"] > 0.4


def test_predicting_no_detail_at_all_scores_a_relative_error_of_exactly_one():
    """relative error = rms(SR_hf - HR_hf) / rms(HR_hf): 1.0 is the score of adding nothing, below 1 is useful detail, above 1 is worse than nothing."""
    hr = textured()
    smooth_sr = ndimage.gaussian_filter(hr, (0, 12, 12))                                     # essentially no high-frequency content
    s = M.spatial_detail_metrics(smooth_sr, hr, np.ones((128, 128), bool), BANDS, CFG)
    assert s["hf_energy_ratio"] < 0.05 and s["hf_relative_error"] == pytest.approx(1.0, abs=0.05)


def test_invented_detail_that_the_reference_does_not_have_scores_worse_than_nothing():
    hr = ndimage.gaussian_filter(textured(), (0, 3, 3))                                      # a soft reference
    rng = np.random.default_rng(3)
    invented = hr + 0.05 * ndimage.gaussian_filter(rng.standard_normal(hr.shape), (0, 0.7, 0.7)).astype("float32") * 4
    s = M.spatial_detail_metrics(invented, hr, np.ones((128, 128), bool), BANDS, CFG)
    assert s["hf_energy_ratio"] > 1.5 and s["hf_relative_error"] > 1.0


def test_a_global_displacement_is_measured_with_its_sign_convention():
    """Reported (dy, dx) is how far the SR content is displaced relative to the reference."""
    hr = textured(size=128)
    sr = np.roll(hr, (3, -2), axis=(1, 2))
    s = M.spatial_detail_metrics(sr, hr, np.ones((128, 128), bool), BANDS, CFG)["phase_shift_px"]
    assert s["dy"] == pytest.approx(3.0, abs=0.25) and s["dx"] == pytest.approx(-2.0, abs=0.25) and s["magnitude"] == pytest.approx(math.hypot(3, 2), abs=0.3)
    aligned = M.spatial_detail_metrics(hr.copy(), hr, np.ones((128, 128), bool), BANDS, CFG)["phase_shift_px"]
    assert abs(aligned["dy"]) < 0.15 and abs(aligned["dx"]) < 0.15


def test_the_displacement_is_still_found_when_part_of_the_reference_is_nodata():
    hr = textured(size=128)
    sr = np.roll(hr, (2, 1), axis=(1, 2))
    mask = np.ones((128, 128), bool)
    mask[:, :48] = False
    s = M.spatial_detail_metrics(sr, hr, mask, BANDS, CFG)["phase_shift_px"]
    assert s["dy"] == pytest.approx(2.0, abs=0.3) and s["dx"] == pytest.approx(1.0, abs=0.3)


def _bicubic_of_displaced_reference(hr, dy, dx):
    """What a bicubic baseline is against a co-located sharper reference: block-average the displaced reference to a 4x coarser grid and interpolate back up."""
    import torch.nn.functional as F

    shifted = np.roll(hr, (dy, dx), axis=(1, 2))
    lr = F.avg_pool2d(torch.from_numpy(shifted)[None], 4)
    return F.interpolate(lr, scale_factor=4, mode="bicubic", antialias=True)[0].clamp(min=0).numpy()


def test_the_displacement_is_found_between_a_blurry_sr_and_a_sharper_reference():
    """Found on real SEN2NEON tiles: with the default phase whitening the unmasked estimate ignored real 2-6 px misregistrations (it reported ~0) because the SR has no
    energy where the sharper reference does, so whitening turns those frequencies into noise. The estimate must not depend on the SR matching the reference's sharpness."""
    rng = np.random.default_rng(11)
    hr = (0.25 + 0.08 * ndimage.gaussian_filter(rng.standard_normal((4, 256, 256)), (0, 0.9, 0.9)) * 3).astype("float32")
    for dy, dx in ((3, 2), (-2, 3), (0, -3)):
        sr = _bicubic_of_displaced_reference(hr, dy, dx)
        s = M.spatial_detail_metrics(sr, hr, np.ones((256, 256), bool), BANDS, CFG)["phase_shift_px"]
        assert s["status"] == "ok"
        assert s["dy"] == pytest.approx(dy, abs=0.6) and s["dx"] == pytest.approx(dx, abs=0.6), (dy, dx, s)


def test_a_blurry_sr_that_is_not_displaced_is_reported_as_not_displaced():
    rng = np.random.default_rng(12)
    hr = (0.25 + 0.08 * ndimage.gaussian_filter(rng.standard_normal((4, 256, 256)), (0, 0.9, 0.9)) * 3).astype("float32")
    s = M.spatial_detail_metrics(_bicubic_of_displaced_reference(hr, 0, 0), hr, np.ones((256, 256), bool), BANDS, CFG)["phase_shift_px"]
    assert abs(s["dy"]) < 0.6 and abs(s["dx"]) < 0.6


def test_a_constant_image_has_no_measurable_displacement_instead_of_a_made_up_one():
    """Phase correlation of a constant image is 0/0; it returned a plausible-looking (1, 1). It must say it cannot be computed."""
    flat = np.full((4, 64, 64), 0.2, dtype="float32")
    s = M.spatial_detail_metrics(flat, flat, np.ones((64, 64), bool), BANDS, CFG)["phase_shift_px"]
    assert s["status"] == "not_computable" and s["dy"] is None and "constant" in s["reason"]
    hr = textured(size=64)
    s2 = M.spatial_detail_metrics(flat, hr, np.ones((64, 64), bool), BANDS, CFG)["phase_shift_px"]        # one constant image is as degenerate as two
    assert s2["status"] == "not_computable"


def test_detail_metrics_ignore_junk_under_the_nodata_mask():
    hr = textured()
    sr = hr.copy()
    mask = np.ones((128, 128), bool)
    mask[:40] = False
    sr[:, :40] = 5.0
    s = M.spatial_detail_metrics(sr, hr, mask, BANDS, CFG)
    assert s["hf_correlation"] == pytest.approx(1.0, abs=1e-3) and s["hf_relative_error"] == pytest.approx(0.0, abs=1e-3)


def test_seam_error_compares_error_near_tile_seams_with_the_interior():
    hr = np.full((4, 128, 128), 0.3, dtype="float32")
    sr = hr.copy()
    sr[:, :, 60:68] += 0.05                                                                  # an artefact right on the seam at column 64
    s = M.seam_error(sr, hr, np.ones((128, 128), bool), seam_rows=[], seam_cols=[64], half_width=4)
    assert s["seam_mae"] == pytest.approx(0.05, abs=1e-6) and s["interior_mae"] == pytest.approx(0.0, abs=1e-9) and s["n_seam_pixels"] == 8 * 128
    none = M.seam_error(hr.copy(), hr, np.ones((128, 128), bool), seam_rows=[], seam_cols=[], half_width=4)
    assert none["status"] == "no_seams" and none["seam_mae"] is None


# ============================================================================== improvement / supported synthesis / omission / unsupported detail


def detail_case():
    """A smooth base; the reference adds a +-0.03 pattern to the LEFT half only."""
    base = smooth_field(size=64)
    pattern = np.zeros_like(base)
    pattern[:, :, :32] = 0.03 * np.where(((np.arange(64)[:, None] // 4 + np.arange(32)[None, :] // 4) % 2) == 0, 1.0, -1.0)
    return base, pattern


def analyse(base, sr, hr, mask=None, tau=0.005):
    return M.hallucination_analysis(base, sr, hr, np.ones((64, 64), bool) if mask is None else mask, CFG)[str(tau)]


def test_perfect_supported_synthesis_is_recognised():
    base, pattern = detail_case()
    r = analyse(base, base + pattern, base + pattern)
    assert r["supported_synthesis"] == pytest.approx(0.5) and r["unsupported_detail"] == 0.0 and r["omission"] == 0.0 and r["neutral"] == pytest.approx(0.5)
    assert r["mean_error_reduction"] > 0.0 and r["improved_fraction"] == pytest.approx(0.5) and r["mse_skill_vs_baseline"] == pytest.approx(1.0)


def test_adding_no_detail_is_omission():
    base, pattern = detail_case()
    r = analyse(base, base.copy(), base + pattern)
    assert r["omission"] == pytest.approx(0.5) and r["supported_synthesis"] == 0.0 and r["unsupported_detail"] == 0.0 and r["mean_error_reduction"] == pytest.approx(0.0, abs=1e-9) and r["mse_skill_vs_baseline"] == pytest.approx(0.0, abs=1e-9)


def test_detail_of_the_wrong_sign_is_unsupported_and_makes_things_worse():
    base, pattern = detail_case()
    r = analyse(base, base - pattern, base + pattern)
    assert r["unsupported_detail"] == pytest.approx(0.5) and r["supported_synthesis"] == 0.0 and r["mean_error_reduction"] < 0.0
    assert r["unsupported_wrong_direction"] == pytest.approx(0.5) and r["unsupported_overshoot"] == 0.0


def test_detail_where_the_reference_has_none_is_unsupported_and_the_missed_detail_is_omission():
    base, pattern = detail_case()
    misplaced = np.zeros_like(pattern)
    misplaced[:, :, 32:] = 0.03 * np.where(((np.arange(64)[:, None] // 4 + np.arange(32)[None, :] // 4) % 2) == 0, 1.0, -1.0)
    r = analyse(base, base + misplaced, base + pattern)
    assert r["unsupported_detail"] == pytest.approx(0.5) and r["omission"] == pytest.approx(0.5) and r["supported_synthesis"] == 0.0


def test_overshooting_a_real_detail_is_unsupported_overshoot_not_wrong_direction():
    base, pattern = detail_case()
    r = analyse(base, base + 3 * pattern, base + pattern)                                     # right direction, three times too strong
    assert r["unsupported_overshoot"] == pytest.approx(0.5) and r["unsupported_wrong_direction"] == 0.0 and r["supported_synthesis"] == 0.0


def test_the_categories_partition_the_valid_elements():
    base, pattern = detail_case()
    rng = np.random.default_rng(0)
    sr = base + 0.5 * pattern + 0.004 * rng.standard_normal(base.shape).astype("float32")
    r = analyse(base, sr, base + pattern)
    assert r["supported_synthesis"] + r["unsupported_detail"] + r["omission"] + r["neutral"] == pytest.approx(1.0, abs=1e-9)
    assert r["unsupported_detail"] == pytest.approx(r["unsupported_wrong_direction"] + r["unsupported_overshoot"] + r["unsupported_where_reference_has_none"], abs=1e-9)


def test_masked_pixels_are_excluded_from_the_analysis():
    base, pattern = detail_case()
    mask = np.ones((64, 64), bool)
    mask[:, :32] = False                                                                       # hide exactly the region with reference detail
    r = analyse(base, base.copy(), base + pattern, mask)
    assert r["omission"] == 0.0 and r["neutral"] == pytest.approx(1.0) and r["n_valid_elements"] == 4 * 64 * 32


def test_the_analysis_is_reported_at_several_thresholds_for_sensitivity():
    base, pattern = detail_case()
    out = M.hallucination_analysis(base, base + pattern, base + pattern, np.ones((64, 64), bool), CFG)
    assert list(out) == [str(t) for t in CFG.hallucination_taus] and out[str(CFG.hallucination_taus[0])]["tau"] == CFG.hallucination_taus[0]


# ============================================================================== self-consistency is NOT reference accuracy


def test_an_sr_that_reduces_exactly_to_the_lr_is_perfectly_self_consistent():
    lr = smooth_field(size=16)
    sr = np.repeat(np.repeat(lr, 4, axis=1), 4, axis=2)
    sc = M.self_consistency(lr, sr, np.ones((16, 16), bool), BANDS, 4)
    assert sc["overall"]["mae"] == pytest.approx(0.0, abs=1e-7) and sc["ndvi"]["mae"] == pytest.approx(0.0, abs=1e-6)


def test_self_consistency_measures_a_radiometric_shift_against_the_lr_only():
    lr = smooth_field(size=16)
    sr = np.repeat(np.repeat(lr, 4, axis=1), 4, axis=2) + 0.02
    sc = M.self_consistency(lr, sr, np.ones((16, 16), bool), BANDS, 4)
    assert sc["overall"]["mae"] == pytest.approx(0.02, abs=1e-6) and sc["per_band"]["B04"]["mae"] == pytest.approx(0.02, abs=1e-6)


def test_self_consistency_is_labelled_and_never_carries_reference_accuracy_keys():
    sc = M.self_consistency(smooth_field(size=16), np.repeat(np.repeat(smooth_field(size=16), 4, 1), 4, 2), np.ones((16, 16), bool), BANDS, 4)
    assert "NOT accuracy against an HR reference" in sc["definition"]
    assert not ({"psnr_db", "ssim", "sam_degrees", "ergas"} & set(sc["overall"]))


def test_a_good_self_consistency_can_coexist_with_a_bad_reference_accuracy():
    """The reason they are separate: an SR can average back to the LR perfectly and still be far from the truth."""
    hr = textured(size=64)
    lr = hr.reshape(4, 16, 4, 16, 4).mean(axis=(2, 4))
    sr_bicubicish = np.repeat(np.repeat(lr, 4, axis=1), 4, axis=2)                            # blocky: consistent with LR, no detail
    sc = M.self_consistency(lr, sr_bicubicish, np.ones((16, 16), bool), BANDS, 4)
    acc = M.reference_accuracy(sr_bicubicish, hr, np.ones((64, 64), bool), BANDS, 4)
    assert sc["overall"]["mae"] == pytest.approx(0.0, abs=1e-6) and acc["mae"] > 0.01


# ============================================================================== data quality


def test_data_quality_reports_valid_and_nodata_fractions_of_both_grids():
    hr_mask = np.ones((64, 64), bool)
    hr_mask[:16] = False
    dq = M.data_quality(hr_mask, np.ones((16, 16), bool))
    assert dq["hr_valid_fraction"] == 0.75 and dq["hr_nodata_fraction"] == 0.25 and dq["lr_valid_fraction"] == 1.0 and dq["hr_valid_pixels"] == 48 * 64


# ============================================================================== opensr-test native metrics are their own group


def test_the_opensr_native_metrics_refuse_data_with_nodata_instead_of_scoring_zeros():
    r = M.opensr_native(np.zeros((4, 8, 8), "float32"), np.zeros((4, 32, 32), "float32"), np.zeros((4, 32, 32), "float32"), hr_mask=np.eye(32, dtype=bool))
    assert r["status"] == "not_computed" and "nodata" in r["reason"]


def test_the_opensr_native_metrics_are_computed_for_fully_valid_data_and_named_as_theirs():
    hr = smooth_field(size=128)
    lr = hr.reshape(4, 32, 4, 32, 4).mean(axis=(2, 4))
    sr = hr + 0.01 * np.random.default_rng(0).standard_normal(hr.shape).astype("float32")
    r = M.opensr_native(lr.astype("float32"), sr, hr, hr_mask=np.ones((128, 128), bool))
    assert r["status"] == "computed" and {"hallucination", "omission", "improvement", "synthesis", "reflectance", "spectral", "spatial"} <= set(r["values"])
    assert "opensr-test" in r["definition"] and r["opensr_test_version"]


# ============================================================================== element labels (shared with the reliability analysis, Phase 6)


def test_the_element_labels_partition_the_elements_and_reproduce_the_reported_fractions():
    rng = np.random.default_rng(21)
    base = (0.3 + 0.05 * rng.standard_normal((4, 32, 32))).astype("float64")
    hr = base + 0.02 * rng.standard_normal(base.shape)
    sr = base + 0.6 * (hr - base) + 0.01 * rng.standard_normal(base.shape)
    mask = np.ones((32, 32), bool)
    mask[:6] = False
    tau = 0.005
    labels = M.hallucination_labels(base, sr, hr, tau)
    assert set(labels) >= {"supported_synthesis", "unsupported_detail", "omission", "neutral"}
    valid = np.broadcast_to(mask[None], base.shape)
    total = sum(labels[k][valid].astype(int) for k in ("supported_synthesis", "unsupported_detail", "omission", "neutral"))
    assert (total == 1).all()                                                          # every valid element is in exactly one category
    reported = M.hallucination_analysis(base, sr, hr, mask, M.MetricConfig(hallucination_taus=(tau,)))[str(tau)]
    for key in ("supported_synthesis", "unsupported_detail", "omission", "neutral", "unsupported_wrong_direction", "unsupported_overshoot", "unsupported_where_reference_has_none"):
        assert labels[key][valid].mean() == pytest.approx(reported[key], abs=1e-12), key
