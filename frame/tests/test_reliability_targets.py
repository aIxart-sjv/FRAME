"""frame.reliability.targets -- the error targets and the trivial baseline predictors, on known tensors."""

from __future__ import annotations

import numpy as np
import pytest

from frame.evaluate import metrics as M
from frame.reliability import targets as T

BANDS = ("B04", "B03", "B02", "B08")


def field(seed=0, size=64):
    rng = np.random.default_rng(seed)
    return (0.25 + 0.05 * rng.standard_normal((4, size, size))).clip(0.02, 0.9)


def full(size=64):
    return np.ones((size, size), bool)


# ============================================================================== pixel error maps


def test_the_absolute_error_is_the_band_mean_of_the_absolute_reflectance_error():
    hr = field()
    pred = hr.copy()
    pred[0] += 0.1
    pred[1] -= 0.3
    maps = T.pixel_error_maps(pred, hr, full())
    assert np.allclose(maps["abs_error"], 0.1)                                              # (0.1 + 0.3 + 0 + 0) / 4
    assert np.allclose(maps["sq_error"], (0.01 + 0.09) / 4)
    assert maps["valid"].all()


def test_the_spectral_angle_is_zero_for_a_brightness_change_and_ninety_degrees_for_orthogonal_spectra():
    hr = np.zeros((4, 4, 4))
    hr[0], hr[1] = 0.2, 0.4
    pred = 2.0 * hr                                                                        # same spectral shape, twice as bright
    assert np.allclose(T.pixel_error_maps(pred, hr, full(4))["sam_deg"], 0.0, atol=1e-6)
    a, b = np.zeros((4, 4, 4)), np.zeros((4, 4, 4))
    a[0], b[1] = 0.3, 0.3
    assert np.allclose(T.pixel_error_maps(a, b, full(4))["sam_deg"], 90.0)


def test_an_undefined_spectral_angle_is_nan_and_never_zero():
    hr = field(size=8)
    pred = hr.copy()
    pred[:, 3, 3] = 0.0                                                                     # an all-zero predicted spectrum has no direction
    maps = T.pixel_error_maps(pred, hr, full(8))
    assert np.isnan(maps["sam_deg"][3, 3]) and np.isfinite(maps["sam_deg"][0, 0]) and maps["abs_error"][3, 3] > 0


def test_masked_pixels_are_invalid_and_non_finite_values_never_enter_the_valid_set():
    hr = field(size=16)
    pred = hr.copy()
    mask = full(16)
    mask[:4] = False
    pred[:, 8, 8] = np.nan
    maps = T.pixel_error_maps(pred, hr, mask)
    assert not maps["valid"][:4].any() and not maps["valid"][8, 8] and maps["valid"][12, 12]
    assert np.isfinite(maps["abs_error"][maps["valid"]]).all()


def test_shape_mismatches_are_errors():
    with pytest.raises(ValueError, match="shape"):
        T.pixel_error_maps(field(size=16), field(size=32), full(16))


# ============================================================================== trivial predictors of error (what stability has to beat)


def test_the_texture_baseline_is_the_sobel_gradient_of_the_band_mean_input():
    img = np.zeros((4, 32, 32))
    img[:, :, 16:] = 0.5                                                                    # a vertical edge
    g = T.texture_baseline(img)
    assert g.shape == (32, 32) and g[:, 15:17].min() > 0 and g[:, :12].max() == pytest.approx(0.0) and g[:, 20:].max() == pytest.approx(0.0)
    assert np.allclose(T.texture_baseline(np.full((4, 8, 8), 0.3)), 0.0)


def test_the_added_detail_baseline_is_how_far_the_model_moved_away_from_bicubic():
    bic = field(size=16)
    pred = bic + 0.02
    assert np.allclose(T.added_detail_baseline(pred, bic), 0.02)


# ============================================================================== cells


def test_block_reduce_averages_valid_pixels_and_drops_cells_that_are_too_empty():
    v = np.arange(64.0).reshape(8, 8)
    valid = np.ones((8, 8), bool)
    cells, ok = T.block_reduce(v, valid, size=4, min_valid_fraction=0.75)
    assert cells.shape == (2, 2) and ok.all() and cells[0, 0] == pytest.approx(v[:4, :4].mean()) and cells[1, 1] == pytest.approx(v[4:, 4:].mean())
    valid[:4, :4] = False
    valid[0, 0] = True                                                                       # 1 of 16 valid in the top-left cell
    valid[4:, :4] = False
    valid[4:7, :3] = True                                                                    # 9 of 16 valid: below 0.75
    valid[4:8, 3] = True                                                                     # 13 of 16 valid: above
    cells, ok = T.block_reduce(v, valid, size=4, min_valid_fraction=0.75)
    assert ok.tolist() == [[False, True], [True, True]] and np.isnan(cells[0, 0])           # an unusable cell is NaN, never 0
    assert cells[1, 0] == pytest.approx(v[4:, :4][valid[4:, :4]].mean())


def test_block_reduce_crops_the_ragged_edge_and_refuses_impossible_sizes():
    cells, ok = T.block_reduce(np.arange(100.0).reshape(10, 10), np.ones((10, 10), bool), size=4, min_valid_fraction=1.0)
    assert cells.shape == (2, 2) and ok.all()
    with pytest.raises(ValueError, match="size"):
        T.block_reduce(np.zeros((3, 3)), np.ones((3, 3), bool), size=4, min_valid_fraction=1.0)


def test_cell_arrays_pair_stability_and_error_over_the_same_valid_pixels():
    hr = field(size=32)
    pred = hr + 0.05
    stab = np.zeros((32, 32))
    stab[:16] = 0.001
    stab[16:] = 0.004
    mask = full(32)
    mask[:, 28:] = False
    maps = T.pixel_error_maps(pred, hr, mask)
    cells = T.cell_arrays({"stability": stab, "abs_error": maps["abs_error"], "sq_error": maps["sq_error"], "sam_deg": maps["sam_deg"]}, maps["valid"], size=4, min_valid_fraction=1.0)
    assert set(cells) >= {"stability", "abs_error", "sq_error", "sam_deg", "row", "col"}
    assert len(cells["stability"]) == 8 * 7                                                  # the seven fully valid columns of cells; the partially-masked column of cells is dropped
    assert np.allclose(cells["abs_error"], 0.05) and set(np.round(cells["stability"], 6)) == {0.001, 0.004}
    assert len({len(v) for v in cells.values()}) == 1


# ============================================================================== tile-level targets equal the Phase 5 definitions


def test_tile_targets_equal_the_phase_5_metrics():
    rng = np.random.default_rng(3)
    hr = field(1, 64)
    pred = (hr + 0.02 * rng.standard_normal(hr.shape)).clip(0.0, 1.0)
    mask = full()
    mask[:10] = False
    ours = T.tile_targets(pred, hr, mask, BANDS, scale=4)
    ref = M.reference_accuracy(pred, hr, mask, BANDS, 4)
    assert ours["rmse"] == pytest.approx(ref["rmse"], rel=1e-5) and ours["mae"] == pytest.approx(ref["mae"], rel=1e-9)
    assert ours["ergas"] == pytest.approx(ref["ergas"], rel=1e-5) and ours["sam_degrees"] == pytest.approx(ref["sam_degrees"], abs=0.02)   # the float32 floor of the reused SAM
    assert ours["n_valid_pixels"] == int(mask.sum())


def test_tile_targets_of_an_empty_valid_set_are_missing_not_zero():
    t = T.tile_targets(field(size=8), field(size=8), np.zeros((8, 8), bool), BANDS, scale=4)
    assert t["rmse"] is None and t["mae"] is None and t["sam_degrees"] is None and t["ergas"] is None and t["n_valid_pixels"] == 0


def test_detail_label_maps_reproduce_the_phase_5_fractions_per_tile():
    rng = np.random.default_rng(4)
    bic = field(2, 32)
    hr = bic + 0.03 * rng.standard_normal(bic.shape)
    pred = bic + 0.5 * (hr - bic) + 0.01 * rng.standard_normal(bic.shape)
    mask = full(32)
    mask[:5] = False
    maps = T.detail_label_maps(pred, bic, hr, tau=0.005)
    ref = M.hallucination_analysis(bic, pred, hr, mask, M.MetricConfig(hallucination_taus=(0.005,)))["0.005"]
    for key in ("supported_synthesis", "unsupported_detail", "omission"):
        assert maps[key][mask].mean() == pytest.approx(ref[key], abs=1e-12), key
    assert maps["supported_synthesis"].shape == (32, 32)                                      # per-pixel fraction of the four bands


def test_stability_map_is_the_band_mean_of_the_ensemble_std():
    std = np.stack([np.full((4, 4), v) for v in (0.001, 0.002, 0.003, 0.004)])
    assert np.allclose(T.stability_map(std), 0.0025)
