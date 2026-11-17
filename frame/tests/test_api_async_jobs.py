"""API tests for the minimal background-job path (Step I Section 6):

    POST /jobs/sr
    GET  /jobs/{async_job_id}

`POST /sr/run` is unchanged and untouched by this feature -- these tests verify the NEW additive
path only. Same fixture pattern as test_api_products.py.
"""

from __future__ import annotations

import time

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from fastapi.testclient import TestClient

from frame.api import config, routes
from frame.api.app import create_app
from frame.api.services import model as model_service
from frame.api.services.async_jobs import AsyncJobRegistry
from frame.api.services.storage import JobStore
from frame.geospatial import RGBN_SCALE_FACTOR, write_geotiff
from frame.models.errors import ModelUnavailableError
from frame.preprocessing import RGBN_BANDS
from frame.preprocessing.metadata import RasterMetadata

H = 128


def _upsample(x_batched: torch.Tensor) -> torch.Tensor:
    return F.interpolate(x_batched, scale_factor=RGBN_SCALE_FACTOR, mode="bicubic", antialias=True)


def _scene_bytes(tmp_path) -> bytes:
    rng = np.random.default_rng(0)
    array = (rng.random((4, H, H)).astype("float32") * 0.4) + 0.1
    metadata = RasterMetadata(
        crs="EPSG:32644", transform=(10.0, 0.0, 500000.0, 0.0, -10.0, 1440752.0),
        bounds=(500000.0, 1440752.0 - H * 10.0, 500000.0 + H * 10.0, 1440752.0), resolution_m=10.0,
        width=H, height=H, band_names=tuple(RGBN_BANDS), acquisition_timestamp="2024-03-30T04:56:59Z",
        nodata_value=0.0, cloud_mask_coverage=1.0, sr_variant=None,
    )
    path = tmp_path / "scene.tif"
    write_geotiff(path, array, metadata)
    return path.read_bytes()


@pytest.fixture()
def env(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "WORKSPACE_DIR", tmp_path / "workspace")
    monkeypatch.setattr(model_service, "get_model", lambda device=None, *, model_name="lite", **_: _upsample)

    app = create_app()
    store = JobStore()
    registry = AsyncJobRegistry()
    app.dependency_overrides[routes.get_store] = lambda: store
    app.dependency_overrides[routes.get_device] = lambda: "cpu"
    app.dependency_overrides[routes.get_async_registry] = lambda: registry
    return TestClient(app), tmp_path


def _upload(client, tmp_path) -> str:
    return client.post("/upload", files={"file": ("scene.tif", _scene_bytes(tmp_path), "image/tiff")}, data={"input_scale": "reflectance"}).json()["upload_id"]


def _poll_until_terminal(client, async_job_id, timeout_s=10.0):
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        body = client.get(f"/jobs/{async_job_id}").json()
        if body["status"] in ("completed", "failed"):
            return body
        time.sleep(0.02)
    raise AssertionError("async job did not reach a terminal state in time")


def test_post_jobs_sr_returns_immediately_as_queued(env):
    client, tmp_path = env
    upload_id = _upload(client, tmp_path)

    response = client.post("/jobs/sr", json={"upload_id": upload_id})
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["status"] in ("queued", "running", "completed")  # a fast local thread may already be done
    assert body["async_job_id"]


def test_async_job_reaches_completed_and_links_to_a_real_sr_job(env):
    client, tmp_path = env
    upload_id = _upload(client, tmp_path)

    async_job_id = client.post("/jobs/sr", json={"upload_id": upload_id}).json()["async_job_id"]
    final = _poll_until_terminal(client, async_job_id)

    assert final["status"] == "completed"
    assert final["sr_job_id"]

    # the linked job is a real, complete SR job -- verify through the existing endpoint
    result = client.get(f"/sr/result/{final['sr_job_id']}")
    assert result.status_code == 200
    assert result.json()["model_id"] == "lite"


def test_sr_run_is_unaffected_and_still_fully_synchronous(env):
    client, tmp_path = env
    upload_id = _upload(client, tmp_path)

    response = client.post("/sr/run", json={"upload_id": upload_id})
    assert response.status_code == 200
    assert response.json()["status"] == "completed"  # no queued/running state ever appears here


def test_unknown_async_job_id_is_404(env):
    client, _ = env
    response = client.get("/jobs/not-a-real-async-job")
    assert response.status_code == 404


def test_an_unavailable_model_fails_the_request_immediately_not_inside_the_background_thread(env, monkeypatch):
    client, tmp_path = env
    upload_id = _upload(client, tmp_path)

    def _raise(*a, **k):
        raise ModelUnavailableError("mamba is not available on this host")

    monkeypatch.setattr(model_service, "get_model", _raise)
    response = client.post("/jobs/sr", json={"upload_id": upload_id, "model": "mamba"})
    assert response.status_code == 503  # failed synchronously, before any async_job_id was even created
