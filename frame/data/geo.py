"""Geospatial validation of an LR/HR pair (Phase 3).

For a x``scale`` pair the HR grid must be an exact refinement of the LR grid:

* same CRS;
* LR pixel size = ``scale`` x HR pixel size, on both axes;
* identical upper-left origin, hence the same footprint;
* HR raster size = ``scale`` x LR raster size.

The transform relationship is NOT redefined here. It is the one Phase 2 already uses in
the other direction: `frame.geospatial.derive_output_transform` divides a LR transform's
linear part by the scale and keeps the origin. This module derives the expected HR
transform with that same function and compares.

A non-georeferenced pair (a synthetic fixture, or a dataset that ships no CRS) is checked
for the size relationship only; that limitation is reported, not hidden.
"""

from __future__ import annotations

from typing import Optional, Tuple

from frame.data.contract import RasterSpec
from frame.data.errors import (
    CRSMismatchError,
    DimensionMismatchError,
    FootprintMismatchError,
    ResolutionRatioError,
)
from frame.geospatial import derive_output_transform
from frame.geospatial.metadata import bounds_from_transform

#: An origin may differ by this fraction of an HR pixel and still count as the same footprint
#: (float noise), while any real shift -- even one HR pixel -- is rejected.
DEFAULT_ORIGIN_TOLERANCE_HR_PX = 0.01
DEFAULT_RATIO_RTOL = 1e-6


def same_crs(a: str, b: str) -> bool:
    """CRS equality that understands equivalent spellings (``EPSG:32630`` vs a WKT of it)."""
    if a == b:
        return True
    from rasterio.crs import CRS

    try:
        return CRS.from_user_input(a) == CRS.from_user_input(b)
    except Exception:  # an unparsable CRS string is never "the same" as anything else
        return False


def expected_hr_transform(lr_transform: Tuple[float, ...], scale: int) -> Tuple[float, ...]:
    """The HR transform a x``scale`` refinement of ``lr_transform`` must have."""
    return derive_output_transform(lr_transform, scale)


def lr_transform_from_hr(hr_transform: Tuple[float, ...], scale: int) -> Tuple[float, ...]:
    """The LR transform of the grid that ``hr_transform`` refines by ``scale`` (origin unchanged)."""
    return derive_output_transform(hr_transform, 1.0 / scale)


def footprint(spec: RasterSpec) -> Tuple[float, float, float, float]:
    """(minx, miny, maxx, maxy) of a georeferenced raster."""
    if spec.transform is None:
        raise ValueError("footprint needs a transform")
    return bounds_from_transform(spec.transform, spec.width, spec.height)


def validate_pair_geometry(
    lr: RasterSpec,
    hr: RasterSpec,
    scale: int,
    *,
    origin_tolerance_hr_px: float = DEFAULT_ORIGIN_TOLERANCE_HR_PX,
    ratio_rtol: float = DEFAULT_RATIO_RTOL,
) -> bool:
    """Raise a `GeoPairError` subclass unless ``hr`` is an exact x``scale`` refinement of ``lr``.

    Returns True if the pair was georeferenced and fully checked, False if only the raster
    sizes could be checked (no CRS/transform to compare).
    """
    if (hr.height, hr.width) != (lr.height * scale, lr.width * scale):
        raise DimensionMismatchError(
            f"HR is {hr.width}x{hr.height} px but LR {lr.width}x{lr.height} x scale {scale} = "
            f"{lr.width * scale}x{lr.height * scale}."
        )
    if not (lr.georeferenced and hr.georeferenced):
        return False

    if not same_crs(lr.crs, hr.crs):  # type: ignore[arg-type]
        raise CRSMismatchError(f"LR CRS {lr.crs!r} differs from HR CRS {hr.crs!r}.")

    lr_t, hr_t = lr.transform, hr.transform
    assert lr_t is not None and hr_t is not None
    if abs(lr_t[1]) > 0 or abs(lr_t[3]) > 0 or abs(hr_t[1]) > 0 or abs(hr_t[3]) > 0:
        raise ResolutionRatioError("Rotated or sheared grids are not supported: the pair's transforms must be north-up.")

    expected = expected_hr_transform(lr_t, scale)
    for axis, index in (("x", 0), ("y", 4)):
        want, got = expected[index], hr_t[index]
        if abs(want - got) > ratio_rtol * abs(want):
            raise ResolutionRatioError(
                f"HR pixel size along {axis} is {abs(got)} but LR {abs(lr_t[index])} / scale {scale} = {abs(want)}."
            )

    tolerance = origin_tolerance_hr_px * abs(hr_t[0])
    dx, dy = abs(hr_t[2] - lr_t[2]), abs(hr_t[5] - lr_t[5])
    if dx > tolerance or dy > tolerance:
        raise FootprintMismatchError(
            f"HR origin ({hr_t[2]}, {hr_t[5]}) is shifted from LR origin ({lr_t[2]}, {lr_t[5]}) by "
            f"({dx:.6g}, {dy:.6g}) map units (tolerance {tolerance:.6g}): the pair does not cover the same footprint."
        )
    return True


def centroid_lonlat(spec: RasterSpec) -> Optional[Tuple[float, float]]:
    """WGS84 (lon, lat) of the raster's centre, or None if it is not georeferenced."""
    if not spec.georeferenced:
        return None
    from rasterio.warp import transform as warp_transform

    minx, miny, maxx, maxy = footprint(spec)
    lons, lats = warp_transform(spec.crs, "EPSG:4326", [(minx + maxx) / 2.0], [(miny + maxy) / 2.0])
    return float(lons[0]), float(lats[0])
