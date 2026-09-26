"""INTEGRATION test: the FRAME API with the REAL SEN2SR-Mamba model (Phase 1).

`POST /sr/run {"model": "mamba"}` through the real dependency chain (no
overrides, no fakes): API -> model selection -> isolated Mamba worker -> real
weights on the GPU -> TTA uncertainty ensemble -> GeoTIFFs -> self-consistency.
The same upload is also run with `"model": "lite"` (when the Lite weights are
cached) to show the baseline still works side by side.

Excluded from a normal run; execute with:

    sen2sr_venv/bin/python -m pytest frame/tests/test_api_mamba_integration.py -m integration -v

Skipped -- never failed, never downloading -- when the GPU, Mamba environment
or model files are absent.
"""

from __future__ import annotations

import pytest
import torch
from fastapi.testclient import TestClient

from frame.api import config as api_config
from frame.geospatial import read_geotiff, write_geotiff
from frame.models import config as models_cfg
from frame.models.mamba_client import check_mamba_availability
from frame.preprocessing.metadata import RasterMetadata

EXAMPLE_DATA = models_cfg.MAMBA_WEIGHTS_DIR / models_cfg.MAMBA_EXAMPLE_DATA_FILENAME
BANDS = ["B04", "B03", "B02", "B08"]
RGBN_FROM_TEN_BAND = [2, 1, 0, 6]

_availability = check_mamba_availability("cuda")
_skip = None
if not _availability.available:
    _skip = f"SEN2SR-Mamba not runnable here: {_availability.message} ({_availability.reason_code})"
elif not EXAMPLE_DATA.is_file():
    _skip = f"missing {EXAMPLE_DATA}"

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
    model_service.clear_cache()  # stops the worker


@pytest.fixture()
def upload_id(client, tmp_path) -> str:
    import safetensors.torch as st

    ten_band = st.load_file(str(EXAMPLE_DATA))["lr"]
    array = ten_band[0, RGBN_FROM_TEN_BAND].numpy()  # (4, 128, 128) reflectance, B04,B03,B02,B08
    metadata = RasterMetadata(
        crs="EPSG:32630",
        transform=(10.0, 0.0, 720285.0, 0.0, -10.0, 4375125.0),
        bounds=(720285.0, 4373845.0, 721565.0, 4375125.0),
        resolution_m=10.0,
        width=128,
        height=128,
        band_names=tuple(BANDS),
        acquisition_timestamp="2023-01-15T10:54:11.024000",
        nodata_value=None,
        cloud_mask_coverage=1.0,
        sr_variant=None,
    )
    path = tmp_path / "example_tile.tif"
    write_geotiff(path, array, metadata)
    response = client.post(
        "/upload",
        files={"file": ("example_tile.tif", path.read_bytes(), "image/tiff")},
        data={"input_scale": "reflectance"},
    )
    assert response.status_code == 200, response.text
    return response.json()["upload_id"]


def test_health_reports_mamba_available_on_this_machine(client):
    body = client.get("/health").json()
    mamba = next(m for m in body["available_models"] if m["id"] == "mamba")
    assert mamba["available"] is True and mamba["label"] == "SEN2SR-Mamba"


def test_sr_run_with_the_real_mamba_model(client, upload_id):
    response = client.post("/sr/run", json={"upload_id": upload_id, "model": "mamba"})
    assert response.status_code == 200, response.text
    body = response.json()

    assert body["status"] == "completed"
    assert body["model_id"] == "mamba" and body["model_name"] == models_cfg.MAMBA_MODEL_NAME
    assert body["input_shape"] == [4, 128, 128] and body["output_shape"] == [4, 512, 512]
    assert body["resolution"]["sr_resolution_m"] == 2.5 and body["bands"] == BANDS

    # provenance from the real worker: which weights, which runtime
    runtime = body["metadata"]["model_runtime"]
    assert runtime["isolated_worker"] is True
    assert runtime["parameter_count"] == 13_759_444
    assert runtime["missing_keys"] == [] and runtime["unexpected_keys"] == []
    assert len(runtime["weights_sha256"]) == 64
    assert runtime["executable_architecture"] == "MambaSR"

    # the TTA ensemble ran on the real model: 6 members, finite uncertainty
    uncertainty = body["uncertainty"]
    assert uncertainty["n"] == 6 and uncertainty["scalar_summary"] > 0
    assert body["metadata"]["device"] == "cuda"

    # exported product: 512x512 on the 2.5 m grid, correctly georeferenced, tagged with the model that ran
    array, metadata = read_geotiff(body["artifacts"]["sr_geotiff"], require_crs=True)
    assert array.shape == (4, 512, 512) and abs(metadata.resolution_m - 2.5) < 1e-6
    assert metadata.sr_variant == models_cfg.MAMBA_MODEL_NAME
    assert metadata.bounds == pytest.approx((720285.0, 4373845.0, 721565.0, 4375125.0))
    assert torch.isfinite(torch.from_numpy(array)).all()

    assert body["self_consistency"]["downsample_rmse"] < 0.01

    # downstream analysis works on a Mamba job exactly as on a Lite job
    assert client.post("/analysis/ndvi", json={"job_id": body["job_id"]}).status_code == 200
    assert client.get(f"/sr/download/{body['job_id']}").status_code == 200


def test_the_worker_is_started_once_and_reused_across_jobs(client, upload_id):
    first = client.post("/sr/run", json={"upload_id": upload_id, "model": "mamba"}).json()
    second = client.post("/sr/run", json={"upload_id": upload_id, "model": "mamba"}).json()
    assert first["metadata"]["model_runtime"]["worker_pid"] == second["metadata"]["model_runtime"]["worker_pid"]
    assert first["job_id"] != second["job_id"]


@pytest.mark.skipif(not _lite_cached(), reason="Lite weights not cached; not downloading")
def test_lite_still_works_through_the_same_api_alongside_mamba(client, upload_id):
    lite = client.post("/sr/run", json={"upload_id": upload_id, "model": "lite"})
    assert lite.status_code == 200, lite.text
    mamba = client.post("/sr/run", json={"upload_id": upload_id, "model": "mamba"})
    assert mamba.status_code == 200, mamba.text

    lite_body, mamba_body = lite.json(), mamba.json()
    assert lite_body["model_id"] == "lite" and lite_body["model_name"] == models_cfg.LITE_MODEL_NAME
    assert "model_runtime" not in lite_body["metadata"]  # Lite is in-process; no worker
    assert lite_body["output_shape"] == mamba_body["output_shape"] == [4, 512, 512]

    _, lite_meta = read_geotiff(lite_body["artifacts"]["sr_geotiff"], require_crs=True)
    _, mamba_meta = read_geotiff(mamba_body["artifacts"]["sr_geotiff"], require_crs=True)
    assert lite_meta.sr_variant == models_cfg.LITE_MODEL_NAME and mamba_meta.sr_variant == models_cfg.MAMBA_MODEL_NAME
    assert lite_meta.bounds == mamba_meta.bounds  # same georeferencing regardless of model
