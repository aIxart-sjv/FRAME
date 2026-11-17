"""Tests for frame.api.services.product -- the Requirement 11 GIS-ready product view
(Step I): COG writing, COG validation and STAC item composition around a real, already-completed
SR job. Uses the same synthetic-GeoTIFF + fake-model fixture as test_api_pipeline.py -- no
network, no real SEN2SRLite weights.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
import pytest

from frame.api.services import pipeline
from frame.api.services.product import build_cog_for_job, build_product_view
from frame.api.services.storage import JobStore
from frame.geospatial import RGBN_SCALE_FACTOR, write_geotiff
from frame.preprocessing import RGBN_BANDS
from frame.preprocessing.metadata import RasterMetadata
from frame.product.stac import validate_stac_item

H = 128


def _equivariant_model(x_batched: torch.Tensor) -> torch.Tensor:
    return F.interpolate(x_batched, scale_factor=RGBN_SCALE_FACTOR, mode="bicubic", antialias=True)


def _write_synthetic_geotiff(path, *, h=H):
    rng = np.random.default_rng(0)
    array = (rng.random((4, h, h)).astype("float32") * 0.4) + 0.1
    metadata = RasterMetadata(
        crs="EPSG:32644", transform=(10.0, 0.0, 500000.0, 0.0, -10.0, 1440752.0),
        bounds=(500000.0, 1440752.0 - h * 10.0, 500000.0 + h * 10.0, 1440752.0),
        resolution_m=10.0, width=h, height=h, band_names=RGBN_BANDS,
        acquisition_timestamp="2024-03-30T04:56:59Z", nodata_value=0.0, cloud_mask_coverage=1.0, sr_variant=None,
    )
    write_geotiff(path, array, metadata)


def _completed_job(tmp_path):
    upload_path = tmp_path / "scene.tif"
    _write_synthetic_geotiff(upload_path)
    store = JobStore()
    upload = pipeline.process_upload(store, upload_path, filename="scene.tif", input_scale="reflectance")
    job = pipeline.run_sr_job(store, upload.upload_id, model=_equivariant_model, device="cpu", seed=0, workspace_dir=tmp_path / "jobs")
    return store, job


def test_build_cog_for_job_writes_a_real_valid_cog(tmp_path):
    store, job = _completed_job(tmp_path)
    cog_path = build_cog_for_job(job)
    assert cog_path.exists()

    from frame.geospatial import validate_cog

    result = validate_cog(cog_path)
    assert result.is_valid_cog


def test_build_cog_for_job_is_cached_not_rewritten(tmp_path):
    store, job = _completed_job(tmp_path)
    first = build_cog_for_job(job)
    mtime_1 = first.stat().st_mtime_ns
    second = build_cog_for_job(job)
    assert second == first
    assert second.stat().st_mtime_ns == mtime_1


def test_cog_pixel_values_match_the_saved_sr_prediction(tmp_path):
    store, job = _completed_job(tmp_path)
    cog_path = build_cog_for_job(job)

    from frame.geospatial import read_geotiff

    array, _ = read_geotiff(cog_path)
    expected = torch.load(job.sr_mean_tensor_path, weights_only=True).numpy()
    assert np.allclose(array, expected, atol=1e-5)


def test_build_product_view_for_a_completed_job(tmp_path):
    store, job = _completed_job(tmp_path)
    view = build_product_view(store, job.job_id, base_url="http://testserver")

    assert view.status == "completed"
    assert view.cog_path is not None and view.cog_path.exists()
    assert view.cog_validation.is_valid_cog
    assert view.stac_item is not None
    assert validate_stac_item(view.stac_item) == []
    assert view.stac_item["properties"]["frame:model_id"] == "lite"
    # a manually-uploaded GeoTIFF has no acquisition provenance -- must be None, not fabricated
    assert view.stac_item["properties"]["frame:source_scene_id"] is None
    assert "stability_geotiff" in view.evidence_assets
    assert view.analyses == []  # no /analysis/ndvi call was made for this job


def test_product_view_stac_item_has_a_working_cog_asset_link(tmp_path):
    store, job = _completed_job(tmp_path)
    view = build_product_view(store, job.job_id, base_url="http://testserver")
    assert view.stac_item["assets"]["sr_cog"]["href"] == f"http://testserver/products/{job.job_id}/cog"


def test_product_view_includes_a_real_ndvi_analysis_when_one_was_run(tmp_path):
    store, job = _completed_job(tmp_path)
    pipeline.run_ndvi_analysis_job(store, job.job_id)

    view = build_product_view(store, job.job_id, base_url="")
    assert len(view.analyses) == 1
    assert any(k.endswith("_sr_ndvi") for k in view.evidence_assets)
    assert any(key.startswith("ndvi_") for key in view.stac_item["assets"])
