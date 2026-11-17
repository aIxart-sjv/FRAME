"""Tests for frame.product.stac -- a real STAC Item for one completed SR product
(Requirements 142 Requirement 11, Step I). Fixture-only, no network, no model.
"""

from __future__ import annotations

import pytest

from frame.preprocessing.metadata import RasterMetadata
from frame.product import MissingGeospatialContextError, build_stac_item, validate_stac_item
from frame.product.stac import RECONSTRUCTED_GRID_STATEMENT


def _output_metadata(**overrides):
    defaults = dict(
        crs="EPSG:32644", transform=(2.5, 0.0, 500000.0, 0.0, -2.5, 1440752.0),
        bounds=(500000.0, 1440432.0, 500320.0, 1440752.0), resolution_m=2.5,
        width=128, height=128, band_names=("B04", "B03", "B02", "B08"),
        acquisition_timestamp="2024-03-30T04:56:59Z", nodata_value=None, cloud_mask_coverage=0.02,
        sr_variant="SEN2SRLite/NonReference_RGBN_x4",
    )
    defaults.update(overrides)
    return RasterMetadata(**defaults)


def _item(**overrides):
    kwargs = dict(
        item_id="frame-sr-job123", output_metadata=_output_metadata(), model_id="lite",
        model_name="SEN2SRLite/NonReference_RGBN_x4", architecture="CNNSR",
        checkpoint_sha256="479aa796d5068d0b1206118ccbca27bd3223df0214db1a9b31a1e18349ed1c7e",
        input_resolution_m=10.0, output_resolution_m=2.5, scale_factor=4,
        bands=("B04", "B03", "B02", "B08"), radiometric_convention="surface reflectance (fraction, 0-1 nominal)",
        processing_timestamp="2024-03-30T05:10:00Z",
        assets={"sr_cog": {"href": "/products/job123/cog", "type": "image/tiff; application=geotiff; profile=cloud-optimized", "roles": ["data"]}},
        scene_id="S2B_MSIL2A_20240330T045659_N0510_R119_T44PMV_20240330T072508", platform="sentinel-2b",
        constellation="sentinel-2", acquisition_datetime="2024-03-30T04:56:59.024000Z", source_provider="copernicus-cdse-process-api",
        scientific_caveats=["TTA stability is model-stability evidence, not calibrated uncertainty."],
    )
    kwargs.update(overrides)
    return build_stac_item(**kwargs)


def test_item_has_the_required_stac_shape():
    item = _item()
    assert item["type"] == "Feature"
    assert item["stac_version"] == "1.0.0"
    assert item["id"] == "frame-sr-job123"
    assert len(item["bbox"]) == 4
    assert item["geometry"]["type"] == "Polygon"
    assert validate_stac_item(item) == []


def test_bbox_is_reprojected_to_epsg4326_and_is_near_chennai():
    item = _item()
    lon_min, lat_min, lon_max, lat_max = item["bbox"]
    assert 79.0 < lon_min < lon_max < 81.1
    assert 12.0 < lat_min < lat_max < 14.0


def test_frame_namespaced_properties_carry_the_real_model_facts():
    item = _item()
    props = item["properties"]
    assert props["frame:model_id"] == "lite"
    assert props["frame:scale_factor"] == 4
    assert props["frame:bands"] == ["B04", "B03", "B02", "B08"]
    assert props["frame:checkpoint_sha256"].startswith("479aa796")
    assert props["proj:epsg"] == 32644


def test_reconstructed_grid_statement_is_present_and_says_not_native():
    item = _item()
    assert item["properties"]["frame:product_statement"] == RECONSTRUCTED_GRID_STATEMENT
    assert "not" in RECONSTRUCTED_GRID_STATEMENT.lower()
    assert "2.5 m" in RECONSTRUCTED_GRID_STATEMENT


def test_missing_acquisition_provenance_is_none_not_fabricated():
    item = _item(scene_id=None, platform=None, constellation=None, acquisition_datetime=None, source_provider=None)
    assert item["properties"]["frame:source_scene_id"] is None
    assert item["properties"]["frame:source_provider"] is None
    assert "platform" not in item["properties"]
    assert "constellation" not in item["properties"]
    # falls back to the processing timestamp, never a fabricated acquisition time
    assert item["properties"]["datetime"] == "2024-03-30T05:10:00Z"


def test_no_bounds_raises_rather_than_fabricating_geometry():
    with pytest.raises(MissingGeospatialContextError):
        _item(output_metadata=_output_metadata(bounds=None))


def test_validate_stac_item_catches_a_missing_required_field():
    item = _item()
    del item["bbox"]
    problems = validate_stac_item(item)
    assert any("bbox" in p for p in problems)


def test_validate_stac_item_catches_an_asset_missing_href():
    item = _item()
    item["assets"]["broken"] = {"type": "image/png"}
    problems = validate_stac_item(item)
    assert any("broken" in p and "href" in p for p in problems)


def test_validate_stac_item_rejects_empty_assets():
    item = _item(assets={})
    problems = validate_stac_item(item)
    assert any("assets" in p for p in problems)
