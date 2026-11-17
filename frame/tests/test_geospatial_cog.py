"""Tests for frame.geospatial.cog / write_cog (Requirements 142 Requirement 11, Step I).

Uses small synthetic rasters and a pytest tmp_path -- no network, no model weights. Mirrors
test_geospatial_geotiff.py's fixtures so a COG and a plain GeoTIFF of the same data can be
compared directly.
"""

from __future__ import annotations

import numpy as np
import pytest

from frame.geospatial.cog import validate_cog
from frame.geospatial.errors import MissingCRSError, UnreadableRasterError
from frame.geospatial.geotiff import read_geotiff, write_cog
from frame.preprocessing.metadata import RasterMetadata


def _metadata(**overrides):
    defaults = dict(
        crs="EPSG:32630",
        transform=(2.5, 0.0, 720285.0, 0.0, -2.5, 4375125.0),
        bounds=(720285.0, 4374805.0, 720605.0, 4375125.0),
        resolution_m=2.5,
        width=128,
        height=128,
        band_names=("B04", "B03", "B02", "B08"),
        acquisition_timestamp="2023-01-15T10:54:11.024000",
        nodata_value=0.0,
        cloud_mask_coverage=1.0,
        sr_variant="SEN2SRLite/NonReference_RGBN_x4",
    )
    defaults.update(overrides)
    return RasterMetadata(**defaults)


def _synthetic_array(bands=4, size=128):
    rng = np.random.default_rng(0)
    return rng.random((bands, size, size)).astype("float32")


def test_write_cog_round_trips_pixel_data(tmp_path):
    array = _synthetic_array()
    path = tmp_path / "product.tif"
    write_cog(path, array, _metadata())

    read_back, metadata = read_geotiff(path)
    assert np.allclose(read_back, array, atol=1e-6)
    assert metadata.crs == "EPSG:32630"
    assert metadata.band_names == ("B04", "B03", "B02", "B08")


def test_write_cog_produces_a_structurally_valid_cog(tmp_path):
    path = tmp_path / "product.tif"
    write_cog(path, _synthetic_array(), _metadata())

    result = validate_cog(path)
    assert result.is_valid_cog is True
    assert result.reasons == ()
    assert result.driver_reports_cog_layout is True
    assert result.is_tiled is True
    assert result.compression == "DEFLATE"
    assert result.has_crs is True
    assert result.has_transform is True
    assert result.band_count == 4
    assert result.width == 128 and result.height == 128


def test_a_small_raster_legitimately_has_no_overviews_and_that_is_not_a_defect(tmp_path):
    path = tmp_path / "small.tif"
    write_cog(path, _synthetic_array(bands=4, size=32), _metadata(width=32, height=32))
    result = validate_cog(path)
    assert result.overview_factors == ()
    assert result.is_valid_cog is True  # missing overviews on a tiny raster is not a validity failure


def test_a_512px_output_is_valid_even_though_rasterio_reports_is_tiled_false(tmp_path):
    """Real, measured GDAL/libtiff edge case: a 512x512 raster (blockxsize==blockysize==512, the
    common FRAME output size for a single 128x128 tile at scale 4) fits in exactly one block, so
    GDAL writes it as one TIFF *strip* rather than a *tile* -- rasterio's is_tiled reports False,
    but GDAL's own IMAGE_STRUCTURE LAYOUT tag still says 'COG'. Validity must follow the LAYOUT
    tag (the authoritative signal), not is_tiled, or every single-tile FRAME job would report an
    invalid COG."""
    path = tmp_path / "single_block.tif"
    write_cog(path, _synthetic_array(bands=4, size=512), _metadata(width=512, height=512))
    result = validate_cog(path)
    assert result.driver_reports_cog_layout is True
    assert result.is_tiled is False  # the measured quirk itself
    assert result.is_valid_cog is True  # must not be gated on is_tiled


def test_a_large_enough_raster_gets_real_overviews(tmp_path):
    path = tmp_path / "large.tif"
    write_cog(path, _synthetic_array(bands=4, size=1024), _metadata(width=1024, height=1024))
    result = validate_cog(path)
    assert len(result.overview_factors) > 0


def test_write_cog_requires_crs_like_write_geotiff(tmp_path):
    with pytest.raises(MissingCRSError):
        write_cog(tmp_path / "no_crs.tif", _synthetic_array(), _metadata(crs=None))


def test_extra_tags_are_written_and_readable_back(tmp_path):
    path = tmp_path / "product.tif"
    write_cog(path, _synthetic_array(), _metadata(), extra_tags={"FRAME_MODEL_ID": "lite", "FRAME_CHECKPOINT_SHA256": "ab" * 32})

    import rasterio

    with rasterio.open(path) as dataset:
        tags = dataset.tags()
    assert tags["FRAME_MODEL_ID"] == "lite"
    assert tags["FRAME_CHECKPOINT_SHA256"] == "ab" * 32


def test_validate_cog_on_a_plain_non_cog_geotiff_reports_invalid(tmp_path):
    from frame.geospatial.geotiff import write_geotiff

    path = tmp_path / "plain.tif"
    write_geotiff(path, _synthetic_array(), _metadata())

    result = validate_cog(path)
    assert result.is_valid_cog is False
    assert any("LAYOUT" in reason for reason in result.reasons)


def test_validate_cog_on_a_missing_file_raises_a_named_error(tmp_path):
    with pytest.raises(UnreadableRasterError):
        validate_cog(tmp_path / "does_not_exist.tif")


def test_validate_cog_never_fabricates_validity_for_a_georeference_free_file(tmp_path):
    import rasterio

    path = tmp_path / "no_geo.tif"
    with rasterio.open(path, "w", driver="GTiff", height=8, width=8, count=1, dtype="float32") as dst:
        dst.write(np.zeros((1, 8, 8), dtype="float32"))

    result = validate_cog(path)
    assert result.is_valid_cog is False
    assert result.has_crs is False
