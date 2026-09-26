"""INTEGRATION test: a multi-tile rectangular scene through the FRAME API, Mamba then Lite (Phase 2).

Real HTTP API -> upload -> model selection -> tile engine -> ONE Mamba worker (real weights, real
GPU) -> reconstruction -> GeoTIFFs; then the same scene with the Lite model. No overrides, no fakes.

    sen2sr_venv/bin/python -m pytest frame/tests/test_api_tiling_mamba_integration.py -m integration -v

The scene is the real Baseline 0 reflectance tile, mirror-extended to 160 x 224 with the engine's own
reflect indexing (offline, deterministic). 160 x 224 tiles as 2 x 2 at overlap 32 (its transpose too),
with the two tiles of the last row padded: 4 tile calls per pass, 24 for the 6-member ensemble.
Skipped -- never failed, never downloading -- when the GPU, environment or inputs are absent.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest
import torch
from fastapi.testclient import TestClient

from frame.api import config as api_config
from frame.geospatial import read_geotiff, write_geotiff
from frame.models import config as models_cfg
from frame.models.mamba_client import check_mamba_availability
from frame.preprocessing.metadata import RasterMetadata
from frame.tiling.padding import reflect_indices

BASELINE_TENSOR = Path(__file__).resolve().parents[2] / "experiments" / "baseline" / "outputs" / "input_tensor.pt"
BANDS = ["B04", "B03", "B02", "B08"]
HEIGHT, WIDTH = 160, 224
BOUNDS = (720285.0, 4375125.0 - HEIGHT * 10.0, 720285.0 + WIDTH * 10.0, 4375125.0)

_availability = check_mamba_availability("cuda")
_skip = None
if not _availability.available:
    _skip = f"SEN2SR-Mamba not runnable here: {_availability.message} ({_availability.reason_code})"
elif not BASELINE_TENSOR.is_file():
    _skip = f"missing {BASELINE_TENSOR}"

pytestmark = [pytest.mark.integration, pytest.mark.skipif(_skip is not None, reason=str(_skip))]


def _lite_cached() -> bool:
    return (api_config.MODEL_WEIGHTS_CACHE_DIR / "mlm.json").exists()


@pytest.fixture()
def client(tmp_path, monkeypatch):
    from frame.api.app import create_app
    from frame.api.services import model as model_service

    monkeypatch.setattr(api_config, "WORKSPACE_DIR", tmp_path / "workspace")
    monkeypatch.setattr(api_config, "DEVICE", "cuda")
    model_service.clear_cache()
    yield TestClient(create_app())
    model_service.clear_cache()  # stops the Mamba worker
    shutil.rmtree(tmp_path, ignore_errors=True)  # a job's rasters and tensors are large; do not leave them in /tmp


@pytest.fixture()
def upload_id(client, tmp_path) -> str:
    base = torch.load(BASELINE_TENSOR, weights_only=True)
    array = base[:, reflect_indices(128, HEIGHT)][:, :, reflect_indices(128, WIDTH)].numpy()
    metadata = RasterMetadata(
        crs="EPSG:32630", transform=(10.0, 0.0, 720285.0, 0.0, -10.0, 4375125.0), bounds=BOUNDS, resolution_m=10.0,
        width=WIDTH, height=HEIGHT, band_names=tuple(BANDS), acquisition_timestamp="2023-01-15T10:54:11.024000",
        nodata_value=None, cloud_mask_coverage=1.0, sr_variant=None,
    )
    path = tmp_path / "scene.tif"
    write_geotiff(path, array, metadata)
    response = client.post(
        "/upload", files={"file": ("scene.tif", path.read_bytes(), "image/tiff")}, data={"input_scale": "reflectance"}
    )
    assert response.status_code == 200, response.text
    assert (response.json()["height"], response.json()["width"]) == (HEIGHT, WIDTH)
    return response.json()["upload_id"]


def test_multi_tile_rectangular_scene_through_the_api_with_the_real_mamba_model(client, upload_id):
    response = client.post("/sr/run", json={"upload_id": upload_id, "model": "mamba"})
    assert response.status_code == 200, response.text
    body = response.json()

    assert body["model_id"] == "mamba" and body["model_name"] == models_cfg.MAMBA_MODEL_NAME
    assert body["input_shape"] == [4, HEIGHT, WIDTH] and body["output_shape"] == [4, HEIGHT * 4, WIDTH * 4]
    assert body["resolution"]["sr_resolution_m"] == 2.5 and body["metadata"]["device"] == "cuda"

    tiling = body["metadata"]["tiling"]
    assert (tiling["tile_size"], tiling["overlap"], tiling["padding_mode"], tiling["blend_mode"]) == (128, 32, "reflect", "linear")
    assert tiling["tile_grid"] == [2, 2] and tiling["tile_count"] == 4 and tiling["padded_tiles"] == 2  # the two tiles of the last row
    assert tiling["scene_passes"] == 6 and tiling["tile_inferences"] == 24
    assert tiling["seam_diagnostic"]["status"] == "COMPUTABLE"

    runtime = body["metadata"]["model_runtime"]
    assert runtime["isolated_worker"] is True and runtime["missing_keys"] == [] and runtime["parameter_count"] == 13_759_444

    array, meta = read_geotiff(body["artifacts"]["sr_geotiff"], require_crs=True)
    assert array.shape == (4, HEIGHT * 4, WIDTH * 4) and meta.crs == "EPSG:32630"
    assert meta.resolution_m == pytest.approx(2.5) and meta.bounds == pytest.approx(BOUNDS, abs=1e-9)
    assert meta.sr_variant == models_cfg.MAMBA_MODEL_NAME
    assert torch.isfinite(torch.from_numpy(array)).all()
    assert body["self_consistency"]["downsample_rmse"] < 0.01

    assert client.post("/analysis/ndvi", json={"job_id": body["job_id"]}).status_code == 200


@pytest.mark.skipif(not _lite_cached(), reason="Lite weights not cached; not downloading")
def test_lite_still_processes_the_same_multi_tile_scene(client, upload_id):
    lite = client.post("/sr/run", json={"upload_id": upload_id, "model": "lite"})
    assert lite.status_code == 200, lite.text
    body = lite.json()
    assert body["model_id"] == "lite" and body["output_shape"] == [4, HEIGHT * 4, WIDTH * 4]
    assert body["metadata"]["tiling"]["tile_count"] == 4 and "model_runtime" not in body["metadata"]

    array, meta = read_geotiff(body["artifacts"]["sr_geotiff"], require_crs=True)
    assert array.shape == (4, HEIGHT * 4, WIDTH * 4) and meta.bounds == pytest.approx(BOUNDS, abs=1e-9)
    assert meta.sr_variant == models_cfg.LITE_MODEL_NAME
    assert torch.isfinite(torch.from_numpy(array)).all()
