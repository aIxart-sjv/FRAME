"""API tests for the Requirement 11 product-registry routes (Step I):

    GET /products/{job_id}
    GET /products/{job_id}/stac
    GET /products/{job_id}/cog
    GET /products/{job_id}/stability
    GET /products/{job_id}/analysis

Same fixture pattern as test_api_model_selection.py: the real HTTP dependency chain, with only
`model_service.get_model` patched to a deterministic bicubic upsample -- no GPU, no weights.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from fastapi.testclient import TestClient

from frame.api import config, routes
from frame.api.app import create_app
from frame.api.services import model as model_service
from frame.api.services.storage import JobStore
from frame.geospatial import RGBN_SCALE_FACTOR, write_geotiff
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
    app.dependency_overrides[routes.get_store] = lambda: store
    app.dependency_overrides[routes.get_device] = lambda: "cpu"
    return TestClient(app), tmp_path


def _completed_job_id(client, tmp_path) -> str:
    upload_id = client.post("/upload", files={"file": ("scene.tif", _scene_bytes(tmp_path), "image/tiff")}, data={"input_scale": "reflectance"}).json()["upload_id"]
    response = client.post("/sr/run", json={"upload_id": upload_id})
    assert response.status_code == 200, response.text
    return response.json()["job_id"]


def test_get_product_for_a_completed_job(env):
    client, tmp_path = env
    job_id = _completed_job_id(client, tmp_path)

    response = client.get(f"/products/{job_id}")
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["job_id"] == job_id
    assert body["status"] == "completed"
    assert body["cog_available"] is True
    assert body["cog_validation"]["is_valid_cog"] is True
    assert "stability_geotiff" in body["evidence_assets"]
    assert body["stac_url"] == f"http://testserver/products/{job_id}/stac"


def test_get_product_for_an_unknown_job_is_404(env):
    client, _ = env
    response = client.get("/products/not-a-real-job")
    assert response.status_code == 404


def test_get_product_stac_returns_a_real_stac_item(env):
    client, tmp_path = env
    job_id = _completed_job_id(client, tmp_path)

    response = client.get(f"/products/{job_id}/stac")
    assert response.status_code == 200
    item = response.json()
    assert item["type"] == "Feature"
    assert item["id"] == f"frame-sr-{job_id}"
    assert item["properties"]["frame:model_id"] == "lite"
    assert item["assets"]["sr_cog"]["href"] == f"http://testserver/products/{job_id}/cog"


def test_download_product_cog_is_a_real_valid_cog(env):
    client, tmp_path = env
    job_id = _completed_job_id(client, tmp_path)

    response = client.get(f"/products/{job_id}/cog")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("image/tiff")

    from frame.geospatial import validate_cog

    written = tmp_path / "downloaded.tif"
    written.write_bytes(response.content)
    result = validate_cog(written)
    assert result.is_valid_cog


def test_get_product_stability_serves_the_same_file_as_uncertainty_download(env):
    client, tmp_path = env
    job_id = _completed_job_id(client, tmp_path)

    a = client.get(f"/products/{job_id}/stability")
    b = client.get(f"/uncertainty/download/{job_id}")
    assert a.status_code == b.status_code == 200
    assert a.content == b.content


def test_get_product_analysis_is_empty_before_any_ndvi_run(env):
    client, tmp_path = env
    job_id = _completed_job_id(client, tmp_path)

    response = client.get(f"/products/{job_id}/analysis")
    assert response.status_code == 200
    assert response.json()["analyses"] == []


def test_get_product_tile_rgb_returns_a_real_png(env):
    client, tmp_path = env
    job_id = _completed_job_id(client, tmp_path)

    response = client.get(f"/products/{job_id}/tiles/1/0/0.png")
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/png"
    assert response.content[:8] == b"\x89PNG\r\n\x1a\n"


def test_get_product_tile_stability_returns_a_real_png(env):
    client, tmp_path = env
    job_id = _completed_job_id(client, tmp_path)

    response = client.get(f"/products/{job_id}/tiles/1/0/0.png", params={"layer": "stability"})
    assert response.status_code == 200
    assert response.content[:8] == b"\x89PNG\r\n\x1a\n"


def test_get_product_tile_rejects_an_unknown_layer(env):
    client, tmp_path = env
    job_id = _completed_job_id(client, tmp_path)

    response = client.get(f"/products/{job_id}/tiles/1/0/0.png", params={"layer": "ndbi"})
    assert response.status_code == 400


def test_get_product_analysis_lists_a_real_run(env):
    client, tmp_path = env
    job_id = _completed_job_id(client, tmp_path)
    client.post("/analysis/ndvi", json={"job_id": job_id})

    response = client.get(f"/products/{job_id}/analysis")
    body = response.json()
    assert len(body["analyses"]) == 1
    assert "analysis_id" in body["analyses"][0]
    assert "result" in body["analyses"][0]
