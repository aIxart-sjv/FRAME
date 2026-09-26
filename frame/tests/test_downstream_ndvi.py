"""frame.downstream.ndvi -- the NDVI of an RGBN stack: formula, band selection by name, denominators, nodata, finiteness."""

from __future__ import annotations

import numpy as np
import pytest

from frame.downstream.config import NdviSpec
from frame.downstream.errors import DownstreamError
from frame.downstream.ndvi import common_valid_mask, ndvi_map, reference_valid_mask

BANDS = ("B04", "B03", "B02", "B08")
SPEC = NdviSpec()


def stack(red, nir, shape=(4, 4)):
    s = np.full((4, *shape), 0.1)
    s[0], s[3] = red, nir
    return s


# ============================================================================== the formula


def test_ndvi_is_nir_minus_red_over_nir_plus_red():
    ndvi, ok = ndvi_map(stack(0.1, 0.5), BANDS, SPEC)
    assert np.allclose(ndvi, 0.4 / 0.6) and ok.all()
    assert np.allclose(ndvi_map(stack(0.3, 0.3), BANDS, SPEC)[0], 0.0)
    assert np.allclose(ndvi_map(stack(0.2, 0.0), BANDS, SPEC)[0], -1.0)
    assert np.allclose(ndvi_map(stack(0.0, 0.2), BANDS, SPEC)[0], 1.0)


def test_the_result_is_within_minus_one_and_one_for_reflectances_and_is_float64():
    rng = np.random.default_rng(0)
    s = rng.random((4, 32, 32)) * 0.6
    ndvi, ok = ndvi_map(s, BANDS, SPEC)
    assert ndvi.dtype == np.float64 and ok.all() and ndvi.min() >= -1 and ndvi.max() <= 1 and np.isfinite(ndvi).all()


# ============================================================================== band order


def test_bands_are_selected_by_name_not_by_position():
    """B04 is the red band and B08 the NIR band wherever they sit in the stack; an assumed position would flip the sign."""
    base = stack(0.1, 0.5)
    shuffled = base[[3, 1, 0, 2]]                                                        # order B08, B03, B04, B02
    a, _ = ndvi_map(base, BANDS, SPEC)
    b, _ = ndvi_map(shuffled, ("B08", "B03", "B04", "B02"), SPEC)
    assert np.allclose(a, b) and np.allclose(a, 0.4 / 0.6)
    swapped_names, _ = ndvi_map(base, ("B08", "B03", "B02", "B04"), SPEC)              # the same numbers under swapped names: red and NIR trade places
    assert np.allclose(swapped_names, -0.4 / 0.6)


def test_a_missing_or_duplicated_band_is_an_error():
    with pytest.raises(DownstreamError, match="B08"):
        ndvi_map(np.zeros((3, 2, 2)), ("B04", "B03", "B02"), SPEC)
    with pytest.raises(DownstreamError, match="duplicate"):
        ndvi_map(np.zeros((4, 2, 2)), ("B04", "B04", "B02", "B08"), SPEC)
    with pytest.raises(DownstreamError, match="channels"):
        ndvi_map(np.zeros((3, 2, 2)), BANDS, SPEC)


# ============================================================================== denominators, nodata, finiteness


def test_a_zero_or_tiny_denominator_is_not_computable_and_never_a_number():
    s = stack(0.2, 0.4, shape=(2, 3))
    s[0, 0, 0], s[3, 0, 0] = 0.0, 0.0                                                      # 0 / 0
    s[0, 0, 1], s[3, 0, 1] = 5e-7, 4e-7                                                    # below the epsilon floor
    s[0, 0, 2], s[3, 0, 2] = 5e-6, 4e-6                                                    # above it: computable (and extreme)
    ndvi, ok = ndvi_map(s, BANDS, SPEC)
    assert not ok[0, 0] and not ok[0, 1] and ok[0, 2] and np.isnan(ndvi[0, 0]) and np.isnan(ndvi[0, 1]) and np.isfinite(ndvi[0, 2])
    assert ok[1].all()


def test_nodata_and_non_finite_pixels_are_not_scored():
    s = stack(0.1, 0.5)
    s[0, 1, 1] = np.nan
    s[3, 2, 2] = np.inf
    valid = np.ones((4, 4), bool)
    valid[0, 0] = False
    ndvi, ok = ndvi_map(s, BANDS, SPEC, valid=valid)
    assert not ok[0, 0] and not ok[1, 1] and not ok[2, 2] and ok[3, 3] and int(ok.sum()) == 13
    assert np.isnan(ndvi[~ok]).all() and np.isfinite(ndvi[ok]).all()


def test_the_denominator_floor_is_the_configured_one():
    s = stack(2e-6, 2e-6)
    assert ndvi_map(s, BANDS, NdviSpec(denominator_epsilon=1e-6))[1].all()
    assert not ndvi_map(s, BANDS, NdviSpec(denominator_epsilon=1e-5))[1].any()


# ============================================================================== which pixels are scored (decided from the reference alone)


def test_a_pixel_enters_only_if_the_reference_reflectance_sum_exceeds_the_minimum():
    ref = stack(0.0, 0.0)
    ref[0, 0, 0], ref[3, 0, 0] = 0.009, 0.010                                              # sum 0.019: too dark
    ref[0, 0, 1], ref[3, 0, 1] = 0.010, 0.011                                              # sum 0.021
    ref[0, 1:], ref[3, 1:] = 0.1, 0.3
    valid = reference_valid_mask(ref, BANDS, np.ones((4, 4), bool), SPEC)
    assert not valid[0, 0] and valid[0, 1] and valid[1:].all()


def test_the_reference_mask_respects_the_strict_mask_and_non_finite_values():
    ref = stack(0.1, 0.3)
    mask = np.ones((4, 4), bool)
    mask[3, :] = False
    ref[0, 0, 0] = np.nan
    valid = reference_valid_mask(ref, BANDS, mask, SPEC)
    assert not valid[3].any() and not valid[0, 0] and int(valid.sum()) == 11


def test_every_system_is_scored_on_the_same_pixels():
    ref = stack(0.1, 0.4)
    a, b = stack(0.1, 0.4), stack(0.1, 0.4)
    a[0, 0, 0], a[3, 0, 0] = 0.0, 0.0                                                      # system a has no NDVI here
    valid = reference_valid_mask(ref, BANDS, np.ones((4, 4), bool), SPEC)
    common = common_valid_mask(valid, [ndvi_map(a, BANDS, SPEC)[1], ndvi_map(b, BANDS, SPEC)[1]])
    assert not common[0, 0] and int(common.sum()) == 15
    assert common_valid_mask(valid, []).sum() == valid.sum()
