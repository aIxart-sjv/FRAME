"""frame.data.geo -- geospatial validation of LR/HR pairs (CRS, resolution ratio, footprint, dimensions)."""

from __future__ import annotations

import pytest

from frame.data.contract import RasterSpec
from frame.data.errors import (
    CRSMismatchError,
    DimensionMismatchError,
    FootprintMismatchError,
    ResolutionRatioError,
)
from frame.data.geo import centroid_lonlat, expected_hr_transform, footprint, lr_transform_from_hr, same_crs, validate_pair_geometry

LR_T = (10.0, 0.0, 539120.0, 0.0, -10.0, 4146000.0)
HR_T = (2.5, 0.0, 539120.0, 0.0, -2.5, 4146000.0)


def lr(**kw):
    base = dict(band_names=("B04",), width=256, height=256, pixel_size_m=10.0, crs="EPSG:32617", transform=LR_T)
    base.update(kw)
    return RasterSpec(**base)


def hr(**kw):
    base = dict(band_names=("B04",), width=1024, height=1024, pixel_size_m=2.5, crs="EPSG:32617", transform=HR_T)
    base.update(kw)
    return RasterSpec(**base)


def test_a_correctly_aligned_x4_pair_is_valid_and_reported_as_fully_checked():
    assert validate_pair_geometry(lr(), hr(), 4) is True


def test_other_scale_factors_work():
    assert validate_pair_geometry(lr(width=128, height=128), hr(width=256, height=256, transform=(5.0, 0, 539120.0, 0, -5.0, 4146000.0)), 2)
    assert validate_pair_geometry(lr(width=100, height=100), hr(width=1000, height=1000, transform=(1.0, 0, 539120.0, 0, -1.0, 4146000.0)), 10)


def test_footprints_are_identical_for_a_valid_pair():
    assert footprint(lr()) == footprint(hr()) == (539120.0, 4146000.0 - 2560.0, 539120.0 + 2560.0, 4146000.0)


@pytest.mark.parametrize("shift", [(2.5, 0.0), (0.0, -2.5), (2.5, 2.5), (10.0, 0.0), (-25.0, 5.0)])
def test_an_intentionally_shifted_hr_transform_is_rejected(shift):
    """One HR pixel (2.5 m) or more of origin shift means LR and HR do not cover the same ground."""
    t = (2.5, 0.0, 539120.0 + shift[0], 0.0, -2.5, 4146000.0 + shift[1])
    with pytest.raises(FootprintMismatchError, match="same footprint"):
        validate_pair_geometry(lr(), hr(transform=t), 4)


def test_float_noise_in_the_origin_is_tolerated_but_a_hundredth_of_a_pixel_is_the_limit():
    ok = (2.5, 0.0, 539120.0 + 1e-7, 0.0, -2.5, 4146000.0 - 1e-7)
    assert validate_pair_geometry(lr(), hr(transform=ok), 4)
    bad = (2.5, 0.0, 539120.0 + 0.05, 0.0, -2.5, 4146000.0)  # 2 % of an HR pixel
    with pytest.raises(FootprintMismatchError):
        validate_pair_geometry(lr(), hr(transform=bad), 4)


def test_a_wrong_resolution_ratio_is_rejected():
    with pytest.raises(ResolutionRatioError, match="pixel size"):
        validate_pair_geometry(lr(), hr(transform=(2.0, 0, 539120.0, 0, -2.0, 4146000.0)), 4)
    with pytest.raises(ResolutionRatioError):
        validate_pair_geometry(lr(), hr(transform=(2.5, 0, 539120.0, 0, -2.0, 4146000.0)), 4)  # anisotropic


def test_hr_dimensions_must_be_scale_times_lr():
    with pytest.raises(DimensionMismatchError):
        validate_pair_geometry(lr(), hr(width=1000), 4)
    with pytest.raises(DimensionMismatchError):
        validate_pair_geometry(lr(), hr(height=1020), 4)
    with pytest.raises(DimensionMismatchError):
        validate_pair_geometry(lr(), hr(), 2)  # the sizes fit x4, not x2


def test_a_different_crs_is_rejected():
    with pytest.raises(CRSMismatchError):
        validate_pair_geometry(lr(), hr(crs="EPSG:32618"), 4)


def test_equivalent_crs_spellings_are_the_same_crs():
    assert same_crs("EPSG:32617", "EPSG:32617")
    assert same_crs("EPSG:32617", "epsg:32617")
    assert not same_crs("EPSG:32617", "EPSG:32618")
    assert not same_crs("EPSG:32617", "not a crs")
    assert validate_pair_geometry(lr(), hr(crs="epsg:32617"), 4)


def test_rotated_or_sheared_grids_are_refused_not_silently_accepted():
    rot = (2.5, 0.1, 539120.0, 0.1, -2.5, 4146000.0)
    with pytest.raises(ResolutionRatioError, match="north-up"):
        validate_pair_geometry(lr(), hr(transform=rot), 4)


def test_a_pair_without_georeferencing_is_size_checked_only_and_says_so():
    assert validate_pair_geometry(lr(crs=None, transform=None), hr(crs=None, transform=None), 4) is False
    with pytest.raises(DimensionMismatchError):  # still size-checked
        validate_pair_geometry(lr(crs=None, transform=None), hr(crs=None, transform=None, width=1000), 4)


def test_the_hr_transform_expected_from_an_lr_transform_reuses_the_phase2_relation():
    assert expected_hr_transform(LR_T, 4) == HR_T
    assert lr_transform_from_hr(HR_T, 4) == pytest.approx(LR_T)
    assert lr_transform_from_hr(expected_hr_transform(LR_T, 10), 10) == pytest.approx(LR_T)


def test_centroid_of_a_real_sen2neon_tile_matches_the_datasets_own_lon_lat():
    """Real tile 2018_MLBS_3__0_2 (EPSG:32617): the dataset publishes its WGS84 centroid; our CRS/affine maths must agree."""
    lon, lat = centroid_lonlat(lr())
    assert lon == pytest.approx(-80.54325020048908, abs=1e-6)
    assert lat == pytest.approx(37.448448298909405, abs=1e-6)


def test_centroid_is_none_without_georeferencing():
    assert centroid_lonlat(lr(crs=None, transform=None)) is None
