"""Tests for frame.product.tiles -- the minimal real COG -> PNG XYZ tile renderer
(Requirements 142 Requirement 11, Step I). Not titiler/rio-tiler (neither is a dependency here) --
see the module docstring for why. Fixture-only, no network.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from frame.geospatial.geotiff import write_cog
from frame.preprocessing.metadata import RasterMetadata
from frame.product.tiles import (
    TILE_SIZE,
    TileOutOfBoundsError,
    render_rgb_tile,
    render_single_band_tile,
    tile_bounds_3857,
)

CHENNAI_LON, CHENNAI_LAT = 81.0, 13.021


def _lonlat_to_tile(lon: float, lat: float, z: int):
    lat_rad = math.radians(lat)
    n = 2**z
    x = int((lon + 180.0) / 360.0 * n)
    y = int((1.0 - math.log(math.tan(lat_rad) + 1 / math.cos(lat_rad)) / math.pi) / 2.0 * n)
    return x, y


@pytest.fixture()
def real_product_cog(tmp_path):
    """A real COG at Chennai-area UTM coordinates (EPSG:32644), same CRS/area as the Step G/H
    live acquisition tests -- so a real lon/lat -> XYZ tile computation actually overlaps it."""
    meta = RasterMetadata(
        crs="EPSG:32644", transform=(2.5, 0.0, 500000.0, 0.0, -2.5, 1440752.0), bounds=None,
        resolution_m=2.5, width=512, height=512, band_names=("B04", "B03", "B02", "B08"),
        acquisition_timestamp=None, nodata_value=None, cloud_mask_coverage=None, sr_variant="x",
    )
    rng = np.random.default_rng(0)
    array = (rng.random((4, 512, 512)).astype("float32") * 0.4) + 0.1
    path = tmp_path / "product.tif"
    write_cog(path, array, meta)
    return path


def test_tile_bounds_3857_shrinks_by_a_factor_of_4_per_zoom_level():
    b0 = tile_bounds_3857(10, 500, 500)
    b1 = tile_bounds_3857(11, 1000, 1000)
    width_0 = b0[2] - b0[0]
    width_1 = b1[2] - b1[0]
    assert width_0 / width_1 == pytest.approx(2.0, rel=1e-6)


def test_tile_bounds_rejects_an_out_of_range_index():
    with pytest.raises(TileOutOfBoundsError):
        tile_bounds_3857(5, -1, 0)
    with pytest.raises(TileOutOfBoundsError):
        tile_bounds_3857(5, 32, 0)  # 2**5 == 32, so x must be < 32


def test_render_rgb_tile_over_the_real_product_extent_is_a_real_png(real_product_cog):
    z = 15
    x, y = _lonlat_to_tile(CHENNAI_LON, CHENNAI_LAT, z)
    png = render_rgb_tile(real_product_cog, z, x, y)
    assert png[:8] == b"\x89PNG\r\n\x1a\n"
    assert len(png) > 1000  # a near-blank/all-nodata tile would be a few hundred bytes only


def test_render_tile_far_from_the_product_is_blank_not_an_error(real_product_cog):
    # Antarctica: nowhere near the Chennai-area product -- must render an (all black) tile, not raise.
    z = 15
    x, y = _lonlat_to_tile(0.0, -80.0, z)
    png = render_rgb_tile(real_product_cog, z, x, y)
    assert png[:8] == b"\x89PNG\r\n\x1a\n"


def test_render_single_band_tile_for_a_stability_style_raster(real_product_cog):
    z = 15
    x, y = _lonlat_to_tile(CHENNAI_LON, CHENNAI_LAT, z)
    png = render_single_band_tile(real_product_cog, z, x, y, band_index=1)
    assert png[:8] == b"\x89PNG\r\n\x1a\n"


def test_tile_png_decodes_to_the_expected_size(real_product_cog):
    import rasterio
    from rasterio.io import MemoryFile

    z = 15
    x, y = _lonlat_to_tile(CHENNAI_LON, CHENNAI_LAT, z)
    png = render_rgb_tile(real_product_cog, z, x, y)
    with MemoryFile(png) as memfile, memfile.open() as dataset:
        assert dataset.width == TILE_SIZE
        assert dataset.height == TILE_SIZE
        assert dataset.count == 3
        assert dataset.dtypes[0] == "uint8"
