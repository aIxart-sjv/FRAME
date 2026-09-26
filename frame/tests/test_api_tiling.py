"""Arbitrary-size scenes through the HTTP API (Phase 2), with fake models: no GPU, no weights.

Keeps the real dependency chain (upload -> `get_model_callable` -> pipeline -> tile engine ->
GeoTIFFs) and patches only `model_service.get_model`, as test_api_model_selection.py does.
"""

from __future__ import annotations

import shutil

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from fastapi.testclient import TestClient

from frame.api import config, routes
from frame.api.app import create_app
from frame.api.services import model as model_service
from frame.api.services import pipeline
from frame.api.services.storage import JobStore
from frame.geospatial import read_geotiff, write_geotiff
from frame.models.errors import ModelWorkerError
from frame.preprocessing import RGBN_BANDS
from frame.preprocessing.metadata import RasterMetadata
from frame.tiling import InvalidTilingConfigError, TilingConfig, plan_tiles


def perfect(x):
    return F.interpolate(x, scale_factor=4, mode="nearest")


class SelfDescribing:
    def __init__(self):
        self.calls = 0

    def __call__(self, x):
        self.calls += 1
        return perfect(x)

    def describe(self):
        return {"isolated_worker": True, "worker_pid": 4242}


def scene_bytes(tmp_path, h, w) -> bytes:
    array = (np.random.default_rng(0).random((4, h, w)).astype("float32") * 0.4) + 0.1
    metadata = RasterMetadata(
        crs="EPSG:32630", transform=(10.0, 0.0, 720285.0, 0.0, -10.0, 4375125.0),
        bounds=(720285.0, 4375125.0 - h * 10.0, 720285.0 + w * 10.0, 4375125.0), resolution_m=10.0, width=w, height=h,
        band_names=tuple(RGBN_BANDS), acquisition_timestamp="2023-01-15T10:54:11.024000", nodata_value=0.0,
        cloud_mask_coverage=1.0, sr_variant=None,
    )
    path = tmp_path / f"scene_{h}x{w}.tif"
    write_geotiff(path, array, metadata)
    return path.read_bytes()


@pytest.fixture()
def env(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "WORKSPACE_DIR", tmp_path / "workspace")
    registry = {"lite": perfect, "mamba": SelfDescribing()}
    requested = []

    def fake_get_model(device=None, *, model_name="lite", **_):
        requested.append(model_name)
        entry = registry[model_name]
        if isinstance(entry, Exception):
            raise entry
        return entry

    monkeypatch.setattr(model_service, "get_model", fake_get_model)
    app = create_app()
    store = JobStore()
    app.dependency_overrides[routes.get_store] = lambda: store
    app.dependency_overrides[routes.get_device] = lambda: "cpu"
    yield TestClient(app), registry, requested, tmp_path
    # A job writes the SR raster, the uncertainty raster and two full-size tensors (~16x the
    # input pixels each): delete them so repeated runs cannot pile up in /tmp.
    shutil.rmtree(tmp_path, ignore_errors=True)


def upload(client, tmp_path, h, w):
    return client.post(
        "/upload", files={"file": ("scene.tif", scene_bytes(tmp_path, h, w), "image/tiff")}, data={"input_scale": "reflectance"}
    )


# ------------------------------------------------------- arbitrary sizes are accepted


@pytest.mark.parametrize("h,w", [(128, 256), (256, 128), (300, 500), (100, 90), (511, 777)])
def test_upload_accepts_non_square_and_non_multiple_of_128_scenes(env, h, w):  # upload only: no SR run, so no big outputs
    client, _, _, tmp_path = env
    response = upload(client, tmp_path, h, w)
    assert response.status_code == 200, response.text
    body = response.json()
    assert (body["height"], body["width"]) == (h, w) and body["valid"] is True


@pytest.mark.parametrize("model", ["lite", "mamba"])
def test_sr_run_on_a_rectangular_scene_works_for_both_models(env, model):
    client, _, requested, tmp_path = env
    upload_id = upload(client, tmp_path, 200, 300).json()["upload_id"]
    response = client.post("/sr/run", json={"upload_id": upload_id, "model": model})
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["input_shape"] == [4, 200, 300] and body["output_shape"] == [4, 800, 1200]
    assert body["resolution"]["sr_resolution_m"] == 2.5 and body["model_id"] == model
    assert requested == [model]  # the model was selected once, before the tiler; the tiler never chooses

    sr, meta = read_geotiff(body["artifacts"]["sr_geotiff"], require_crs=True)
    assert sr.shape == (4, 800, 1200) and meta.resolution_m == pytest.approx(2.5)
    assert torch.isfinite(torch.from_numpy(sr)).all()


def test_the_reconstruction_configuration_and_counts_are_recorded_in_the_result(env):
    client, _, _, tmp_path = env
    upload_id = upload(client, tmp_path, 300, 500).json()["upload_id"]
    tiling = client.post("/sr/run", json={"upload_id": upload_id}).json()["metadata"]["tiling"]
    assert (tiling["tile_size"], tiling["overlap"], tiling["stride"], tiling["scale"]) == (128, 32, 96, 4)
    assert (tiling["padding_mode"], tiling["blend_mode"]) == ("reflect", "linear")
    assert tiling["tile_grid"] == [3, 5] and tiling["tile_count"] == 15
    assert tiling["scene_passes"] == 6 and tiling["tile_inferences"] == 90  # 6 ensemble members x 15 tiles (rot90 -> 5 x 3, also 15)
    assert tiling["seam_diagnostic"]["status"] == "COMPUTABLE" and "not an accuracy measure" in tiling["seam_diagnostic"]["note"]


def test_the_mamba_model_is_called_once_per_tile_per_pass_on_one_object(env):
    client, registry, _, tmp_path = env
    upload_id = upload(client, tmp_path, 200, 300).json()["upload_id"]
    body = client.post("/sr/run", json={"upload_id": upload_id, "model": "mamba"}).json()
    assert registry["mamba"].calls == 36  # 6 ensemble passes x 6 tiles (200x300 and its transpose both tile 2x3): one object served all
    assert body["metadata"]["model_runtime"]["worker_pid"] == 4242


def test_the_configured_overlap_is_used(env, monkeypatch):
    client, _, _, tmp_path = env
    monkeypatch.setattr(config, "TILE_OVERLAP", 0)
    upload_id = upload(client, tmp_path, 200, 300).json()["upload_id"]
    tiling = client.post("/sr/run", json={"upload_id": upload_id}).json()["metadata"]["tiling"]
    assert tiling["overlap"] == 0 and tiling["tile_count"] == 6  # stride 128: 2 x 3 tiles
    assert tiling["seam_diagnostic"]["status"] == "NOT_COMPUTABLE"  # nothing overlaps, and it says so


def test_an_invalid_configured_overlap_fails_at_startup_not_on_the_first_request(monkeypatch):
    monkeypatch.setattr(config, "TILE_OVERLAP", 200)
    with pytest.raises(InvalidTilingConfigError):
        create_app()


def test_ndvi_analysis_and_downloads_work_on_a_tiled_job(env):
    client, _, _, tmp_path = env
    upload_id = upload(client, tmp_path, 200, 300).json()["upload_id"]
    job = client.post("/sr/run", json={"upload_id": upload_id}).json()
    analysis = client.post("/analysis/ndvi", json={"job_id": job["job_id"]})
    assert analysis.status_code == 200, analysis.text
    assert client.get(f"/sr/download/{job['job_id']}").status_code == 200
    assert client.get(f"/uncertainty/download/{job['job_id']}").status_code == 200


def test_a_128_by_128_scene_behaves_exactly_as_before(env):
    client, _, _, tmp_path = env
    upload_id = upload(client, tmp_path, 128, 128).json()["upload_id"]
    body = client.post("/sr/run", json={"upload_id": upload_id}).json()
    assert body["output_shape"] == [4, 512, 512]
    assert body["metadata"]["tiling"]["tile_count"] == 1 and body["metadata"]["tiling"]["tile_inferences"] == 6


def test_nodata_nan_in_an_uploaded_scene_is_zero_filled_by_preprocessing_so_the_output_is_finite(env):
    """Real Sentinel-2 windows can contain NaN outside the swath. frame.preprocessing.to_reflectance
    has always replaced NaN/Inf with 0 (as the upstream README does), so the tile engine never sees
    it: the scene is processed and the raster is finite. (The engine itself refuses non-finite
    input for direct callers -- see test_tiling_engine.py.)"""
    client, registry, _, work = env
    seen_finite = []
    registry["lite"] = lambda x: seen_finite.append(bool(torch.isfinite(x).all())) or perfect(x)
    array = np.full((4, 200, 300), 0.3, dtype="float32")
    array[:, 50, 60:70] = np.nan
    metadata = RasterMetadata(
        crs="EPSG:32630", transform=(10.0, 0.0, 720285.0, 0.0, -10.0, 4375125.0),
        bounds=(720285.0, 4375125.0 - 2000.0, 720285.0 + 3000.0, 4375125.0), resolution_m=10.0, width=300, height=200,
        band_names=tuple(RGBN_BANDS), acquisition_timestamp="2023-01-15T10:54:11.024000", nodata_value=None,
        cloud_mask_coverage=1.0, sr_variant=None,
    )
    path = work / "nan_scene.tif"
    write_geotiff(path, array, metadata)
    response = client.post("/upload", files={"file": ("nan.tif", path.read_bytes(), "image/tiff")}, data={"input_scale": "reflectance"})
    assert response.status_code == 200
    run = client.post("/sr/run", json={"upload_id": response.json()["upload_id"]})
    assert run.status_code == 200, run.text
    assert seen_finite and all(seen_finite)  # every tile the model saw was finite
    sr, _ = read_geotiff(run.json()["artifacts"]["sr_geotiff"], require_crs=True)
    assert np.isfinite(sr).all()


# ------------------------------------------------------------------ size guard


def test_oversized_scene_is_rejected_with_413_before_its_pixels_are_read(env, monkeypatch):
    client, _, _, tmp_path = env
    monkeypatch.setattr(config, "MAX_INPUT_PIXELS", 100_000)

    def must_not_read(*a, **k):
        raise AssertionError("the pixels were read before the size check")

    monkeypatch.setattr(pipeline, "read_geotiff", must_not_read)
    response = upload(client, tmp_path, 300, 500)
    assert response.status_code == 413
    body = response.json()
    assert body["code"] == "scene_too_large"
    assert "500 x 300" in body["detail"] and "150,000" in body["detail"] and "100,000" in body["detail"]


def test_a_rejected_upload_is_not_left_on_disk(env, monkeypatch):
    client, _, _, tmp_path = env
    monkeypatch.setattr(config, "MAX_INPUT_PIXELS", 100_000)
    upload(client, tmp_path, 300, 500)
    assert list((config.WORKSPACE_DIR / "uploads").glob("*")) == []


def test_the_size_limit_can_be_disabled_and_the_boundary_is_inclusive(env, monkeypatch):
    client, _, _, tmp_path = env
    monkeypatch.setattr(config, "MAX_INPUT_PIXELS", 300 * 500)
    assert upload(client, tmp_path, 300, 500).status_code == 200  # exactly at the limit
    monkeypatch.setattr(config, "MAX_INPUT_PIXELS", 300 * 500 - 1)
    assert upload(client, tmp_path, 300, 500).status_code == 413
    monkeypatch.setattr(config, "MAX_INPUT_PIXELS", 0)
    assert upload(client, tmp_path, 300, 500).status_code == 200


# ------------------------------------------------------------ failures mid-scene


def test_a_worker_failure_partway_through_a_scene_is_a_clean_error_with_no_job_or_artifacts(env):
    client, registry, _, tmp_path = env
    calls = {"n": 0}

    def dies_on_tile_five(x):
        calls["n"] += 1
        if calls["n"] == 5:
            raise ModelWorkerError("The SEN2SR-Mamba worker stopped unexpectedly.", technical_detail="mamba_ssm segfault")
        return perfect(x)

    registry["mamba"] = dies_on_tile_five
    upload_id = upload(client, tmp_path, 300, 500).json()["upload_id"]
    response = client.post("/sr/run", json={"upload_id": upload_id, "model": "mamba"})
    assert response.status_code == 503
    body = response.json()
    assert body["code"] == "model_runtime_error" and "segfault" not in response.text and "mamba_ssm" not in response.text
    assert "job_id" not in body
    assert list((config.WORKSPACE_DIR / "jobs").glob("*")) == []  # nothing partial was written
    assert calls["n"] == 5  # stopped at the failing tile
