"""GIS compatibility smoke test (Requirements 142 Requirement 11, Step I Section 9).

Proves the generated SR product is usable as a real geospatial raster and a real STAC asset --
not just a NumPy array with some metadata attached. No `gdalinfo` CLI or `osgeo.gdal` binding is
installed in this environment (see docs/OPERATIONAL_DELIVERY.md); every check here goes through
rasterio directly instead, which bundles its own GDAL (see frame.geospatial.cog's module
docstring for why that is an honest substitute, not a weaker one, for what this test checks.
"""

from __future__ import annotations

import numpy as np
import rasterio

from frame.api.services import pipeline
from frame.api.services.product import build_cog_for_job
from frame.geospatial import RGBN_SCALE_FACTOR, validate_cog, write_geotiff
from frame.api.services.storage import JobStore
from frame.preprocessing import RGBN_BANDS
from frame.preprocessing.metadata import RasterMetadata
from frame.product import build_stac_item, validate_stac_item

H = 128


def _fake_model(x_batched):
    import torch.nn.functional as F

    return F.interpolate(x_batched, scale_factor=RGBN_SCALE_FACTOR, mode="bicubic", antialias=True)


def _real_job(tmp_path):
    rng = np.random.default_rng(0)
    array = (rng.random((4, H, H)).astype("float32") * 0.4) + 0.1
    metadata = RasterMetadata(
        crs="EPSG:32644", transform=(10.0, 0.0, 500000.0, 0.0, -10.0, 1440752.0),
        bounds=(500000.0, 1440752.0 - H * 10.0, 500000.0 + H * 10.0, 1440752.0), resolution_m=10.0,
        width=H, height=H, band_names=RGBN_BANDS, acquisition_timestamp="2024-03-30T04:56:59Z",
        nodata_value=0.0, cloud_mask_coverage=1.0, sr_variant=None,
    )
    upload_path = tmp_path / "scene.tif"
    write_geotiff(upload_path, array, metadata)
    store = JobStore()
    upload = pipeline.process_upload(store, upload_path, filename="scene.tif", input_scale="reflectance")
    job = pipeline.run_sr_job(store, upload.upload_id, model=_fake_model, device="cpu", seed=0, workspace_dir=tmp_path / "jobs")
    return job


def test_rasterio_can_reopen_the_product_and_every_geospatial_fact_is_real(tmp_path):
    job = _real_job(tmp_path)
    cog_path = build_cog_for_job(job)

    with rasterio.open(cog_path) as dataset:
        assert dataset.crs is not None
        assert dataset.transform is not None and not dataset.transform.is_identity
        bounds = dataset.bounds
        assert bounds.right > bounds.left and bounds.top > bounds.bottom  # bounds are sane, not swapped/degenerate
        pixel_size_x = abs(dataset.transform.a)
        pixel_size_y = abs(dataset.transform.e)
        assert pixel_size_x == pixel_size_y == 2.5  # the reconstructed grid, in the output's own CRS units (metres)
        assert dataset.count == 4
        assert list(dataset.descriptions) == list(RGBN_BANDS)  # band descriptions are correct, in the right order
        assert dataset.dtypes[0] == "float32"
        data = dataset.read()
        assert np.isfinite(data).all()


def test_cog_validation_passes_for_a_real_product(tmp_path):
    job = _real_job(tmp_path)
    cog_path = build_cog_for_job(job)
    result = validate_cog(cog_path)
    assert result.is_valid_cog
    assert result.reasons == ()


def test_stac_item_references_a_real_existing_asset_file(tmp_path):
    job = _real_job(tmp_path)
    cog_path = build_cog_for_job(job)

    from frame.models.selection import get_spec

    spec = get_spec(job.result["model_id"])
    item = build_stac_item(
        item_id=f"frame-sr-{job.job_id}", output_metadata=job.output_metadata, model_id=job.result["model_id"],
        model_name=job.result["model_name"], architecture=spec.architecture, checkpoint_sha256=None,
        input_resolution_m=job.result["resolution"]["native_resolution_m"], output_resolution_m=job.result["resolution"]["sr_resolution_m"],
        scale_factor=job.result["resolution"]["scale_factor"], bands=job.result["bands"],
        radiometric_convention="surface reflectance (fraction)", processing_timestamp=job.result["created_at"],
        assets={"sr_cog": {"href": str(cog_path), "type": "image/tiff; application=geotiff; profile=cloud-optimized", "roles": ["data"]}},
    )
    assert validate_stac_item(item) == []
    # the asset reference is not just a plausible-looking string -- the file it names is real and on disk
    from pathlib import Path

    assert Path(item["assets"]["sr_cog"]["href"]).exists()


def test_output_dimensions_are_exactly_the_declared_scale_factor(tmp_path):
    job = _real_job(tmp_path)
    with rasterio.open(build_cog_for_job(job)) as dataset:
        assert dataset.width == H * RGBN_SCALE_FACTOR
        assert dataset.height == H * RGBN_SCALE_FACTOR


def test_no_gdalinfo_validate_available_is_disclosed_not_silently_skipped():
    """Documents, as an executable fact rather than only in prose, that the official
    `gdalinfo --validate` / `rio-cogeo` COG validator is not available in this environment
    (docs/OPERATIONAL_DELIVERY.md) -- so a reader of the test suite sees the same limitation the
    docs state, not just the fallback rasterio-based checks passing silently."""
    import shutil

    assert shutil.which("gdalinfo") is None
    try:
        import rio_cogeo  # noqa: F401

        available = True
    except ImportError:
        available = False
    assert available is False
