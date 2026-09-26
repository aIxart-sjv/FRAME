"""Model selection through the API (Phase 1): `POST /sr/run {"model": "lite" | "mamba"}`.

Unlike test_api.py -- which replaces the whole model dependency with a fake --
these tests keep the real dependency chain (`get_model_callable` reading the
requested model from the body -> `model_service.get_model`) and patch only
`model_service.get_model`, so they verify that the selection actually reaches
the model layer. No GPU, no weights, no worker process.
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
from frame.geospatial import RGBN_SCALE_FACTOR, read_geotiff, write_geotiff
from frame.models import config as models_cfg
from frame.models.errors import (
    ModelContractError,
    ModelInferenceError,
    ModelUnavailableError,
    ModelWorkerError,
)
from frame.preprocessing import RGBN_BANDS
from frame.preprocessing.metadata import RasterMetadata

H = 128
FORBIDDEN_IN_USER_TEXT = ("triton", "mamba_ssm", "causal-conv1d", "causal_conv1d", "selective_scan")


def _upsample(x_batched: torch.Tensor) -> torch.Tensor:
    return F.interpolate(x_batched, scale_factor=RGBN_SCALE_FACTOR, mode="bicubic", antialias=True)


class DescribingModel:
    """A fake model that, like the Mamba worker client, can describe itself."""

    def __call__(self, x):
        return _upsample(x)

    def describe(self):
        return {"isolated_worker": True, "parameter_count": 13759444, "weights_sha256": "ab" * 32}


def _scene_bytes(tmp_path) -> bytes:
    rng = np.random.default_rng(0)
    array = (rng.random((4, H, H)).astype("float32") * 0.4) + 0.1
    metadata = RasterMetadata(
        crs="EPSG:32630",
        transform=(10.0, 0.0, 720285.0, 0.0, -10.0, 4375125.0),
        bounds=(720285.0, 4375125.0 - H * 10.0, 720285.0 + H * 10.0, 4375125.0),
        resolution_m=10.0,
        width=H,
        height=H,
        band_names=tuple(RGBN_BANDS),
        acquisition_timestamp="2023-01-15T10:54:11.024000",
        nodata_value=0.0,
        cloud_mask_coverage=1.0,
        sr_variant=None,
    )
    path = tmp_path / "scene.tif"
    write_geotiff(path, array, metadata)
    return path.read_bytes()


@pytest.fixture()
def env(tmp_path, monkeypatch):
    """(client, requested-models log, model registry) with `get_model` patched."""
    monkeypatch.setattr(config, "WORKSPACE_DIR", tmp_path / "workspace")
    requested = []
    registry = {"lite": _upsample, "mamba": DescribingModel()}

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
    return TestClient(app), requested, registry, tmp_path


def _upload(client, tmp_path) -> str:
    response = client.post(
        "/upload",
        files={"file": ("scene.tif", _scene_bytes(tmp_path), "image/tiff")},
        data={"input_scale": "reflectance"},
    )
    assert response.status_code == 200, response.text
    return response.json()["upload_id"]


# ---------------------------------------------------------------- selection


def test_default_model_is_lite_when_the_field_is_omitted(env):
    client, requested, _, tmp_path = env
    body = client.post("/sr/run", json={"upload_id": _upload(client, tmp_path)}).json()
    assert requested == ["lite"]
    assert body["model_id"] == "lite" and body["model_name"] == models_cfg.LITE_MODEL_NAME


def test_lite_can_be_selected_explicitly(env):
    client, requested, _, tmp_path = env
    body = client.post("/sr/run", json={"upload_id": _upload(client, tmp_path), "model": "lite"}).json()
    assert requested == ["lite"] and body["model_id"] == "lite"


def test_mamba_can_be_selected_and_reaches_the_model_layer(env):
    client, requested, _, tmp_path = env
    response = client.post("/sr/run", json={"upload_id": _upload(client, tmp_path), "model": "mamba"})
    assert response.status_code == 200, response.text
    body = response.json()
    assert requested == ["mamba"]  # the selection made it through the dependency into get_model
    assert body["model_id"] == "mamba" and body["model_name"] == models_cfg.MAMBA_MODEL_NAME
    assert body["output_shape"] == [4, 512, 512] and body["resolution"]["sr_resolution_m"] == 2.5


def test_selection_is_case_insensitive(env):
    client, requested, _, tmp_path = env
    body = client.post("/sr/run", json={"upload_id": _upload(client, tmp_path), "model": " Mamba "}).json()
    assert requested == ["mamba"] and body["model_id"] == "mamba"


def test_model_is_requested_once_per_job(env):
    client, requested, _, tmp_path = env
    client.post("/sr/run", json={"upload_id": _upload(client, tmp_path), "model": "mamba"})
    assert requested == ["mamba"]


@pytest.mark.parametrize("bad", ["swin", "", "sen2sr"])
def test_invalid_model_selection_fails_with_a_clear_422_naming_the_supported_ids(env, bad):
    client, requested, _, tmp_path = env
    response = client.post("/sr/run", json={"upload_id": _upload(client, tmp_path), "model": bad})
    assert response.status_code == 422
    assert "supported models are: lite, mamba" in response.json()["detail"]
    assert requested == []  # never reached the model layer


def test_non_string_model_is_rejected_with_a_422(env):
    client, requested, _, tmp_path = env
    response = client.post("/sr/run", json={"upload_id": _upload(client, tmp_path), "model": 5})
    assert response.status_code == 422 and requested == []


def test_validation_errors_are_reported_once_and_do_not_leak_source_paths(env):
    """POST /sr/run validates its body for both the route and model selection; the
    422 must still list each problem once, and never embed a server file path."""
    client, *_ = env
    detail = client.post("/sr/run", json={"seed": 1}).json()["detail"]
    assert detail == "1 validation error: body.upload_id: Field required"
    unknown = client.post("/sr/run", json={"upload_id": "u", "model": "swin"}).json()["detail"]
    assert unknown.count("Unknown model") == 1
    assert ".py" not in detail and ".py" not in unknown


def test_other_endpoints_keep_reporting_every_distinct_problem(env):
    client, *_ = env
    detail = client.post("/aoi/preview", json={}).json()["detail"]
    assert detail.startswith("4 validation errors:")


# ------------------------------------------------------------------ provenance


def test_mamba_output_geotiff_is_tagged_with_the_mamba_model_not_lite(env):
    client, _, _, tmp_path = env
    body = client.post("/sr/run", json={"upload_id": _upload(client, tmp_path), "model": "mamba"}).json()
    _, metadata = read_geotiff(body["artifacts"]["sr_geotiff"], require_crs=True)
    assert metadata.sr_variant == models_cfg.MAMBA_MODEL_NAME
    _, uncertainty_metadata = read_geotiff(body["artifacts"]["uncertainty_geotiff"], require_crs=True)
    assert uncertainty_metadata.sr_variant == models_cfg.MAMBA_MODEL_NAME


def test_lite_output_geotiff_is_still_tagged_with_lite(env):
    client, _, _, tmp_path = env
    body = client.post("/sr/run", json={"upload_id": _upload(client, tmp_path), "model": "lite"}).json()
    _, metadata = read_geotiff(body["artifacts"]["sr_geotiff"], require_crs=True)
    assert metadata.sr_variant == models_cfg.LITE_MODEL_NAME


def test_a_self_describing_model_adds_its_runtime_to_the_result_metadata(env):
    client, _, _, tmp_path = env
    body = client.post("/sr/run", json={"upload_id": _upload(client, tmp_path), "model": "mamba"}).json()
    runtime = body["metadata"]["model_runtime"]
    assert runtime["isolated_worker"] is True and runtime["parameter_count"] == 13759444


def test_a_plain_callable_adds_no_runtime_block(env):
    client, _, _, tmp_path = env
    body = client.post("/sr/run", json={"upload_id": _upload(client, tmp_path), "model": "lite"}).json()
    assert "model_runtime" not in body["metadata"]


def test_ndvi_analysis_still_works_after_a_mamba_job(env):
    client, _, _, tmp_path = env
    job = client.post("/sr/run", json={"upload_id": _upload(client, tmp_path), "model": "mamba"}).json()
    response = client.post("/analysis/ndvi", json={"job_id": job["job_id"]})
    assert response.status_code == 200, response.text


# ----------------------------------------------------------- error handling


def test_unavailable_model_is_a_503_with_a_user_safe_message(env):
    client, _, registry, tmp_path = env
    registry["mamba"] = ModelUnavailableError(
        "SEN2SR-Mamba requires a CUDA-capable GPU.", technical_detail="mamba_ssm ImportError: libcudart"
    )
    response = client.post("/sr/run", json={"upload_id": _upload(client, tmp_path), "model": "mamba"})
    assert response.status_code == 503
    body = response.json()
    assert body["code"] == "model_unavailable"
    assert body["detail"] == "SEN2SR-Mamba requires a CUDA-capable GPU."
    assert "libcudart" not in response.text  # technical detail is never sent to the client


@pytest.mark.parametrize(
    "error",
    [
        ModelWorkerError("The SEN2SR-Mamba worker stopped unexpectedly.", technical_detail="mamba_ssm segfault at 0x0"),
        ModelInferenceError("SEN2SR-Mamba ran out of GPU memory.", technical_detail="triton CUDA OOM"),
    ],
)
def test_runtime_failures_get_a_generic_message_and_leak_no_internals(env, error):
    client, _, registry, tmp_path = env
    registry["mamba"] = error
    response = client.post("/sr/run", json={"upload_id": _upload(client, tmp_path), "model": "mamba"})
    assert response.status_code in (500, 503)
    body = response.json()
    assert body["code"] == "model_runtime_error"
    assert not any(word in response.text.lower() for word in FORBIDDEN_IN_USER_TEXT)


def test_contract_violation_is_a_422_with_the_actionable_message(env):
    client, _, registry, tmp_path = env

    def reject(x):
        raise ModelContractError("Input must be surface reflectance as a fraction (divided by 10000).")

    registry["mamba"] = reject
    response = client.post("/sr/run", json={"upload_id": _upload(client, tmp_path), "model": "mamba"})
    assert response.status_code == 422
    assert response.json()["code"] == "model_input_invalid"
    assert "divided by 10000" in response.json()["detail"]


def test_lite_still_works_when_mamba_is_unavailable(env):
    client, _, registry, tmp_path = env
    registry["mamba"] = ModelUnavailableError("SEN2SR-Mamba requires a CUDA-capable GPU.")
    response = client.post("/sr/run", json={"upload_id": _upload(client, tmp_path), "model": "lite"})
    assert response.status_code == 200


# --------------------------------------------------------------------- health


def test_health_lists_both_models_and_the_default(env, monkeypatch):
    client, *_ = env
    body = client.get("/health").json()
    assert body["default_model"] == "lite"
    assert body["model_name"] == models_cfg.LITE_MODEL_NAME  # backward compatible field
    assert [m["id"] for m in body["available_models"]] == ["lite", "mamba"]
    labels = {m["id"]: m["label"] for m in body["available_models"]}
    assert labels == {"lite": "SEN2SR-Lite", "mamba": "SEN2SR-Mamba"}
    assert next(m for m in body["available_models"] if m["id"] == "lite")["available"] is True


def test_health_reports_mamba_unavailable_with_a_user_safe_reason(env, monkeypatch):
    client, *_ = env
    monkeypatch.setattr(config, "DEVICE", "cpu")
    body = client.get("/health").json()
    mamba = next(m for m in body["available_models"] if m["id"] == "mamba")
    assert mamba["available"] is False
    assert "CUDA" in mamba["reason"]
    assert not any(word in mamba["reason"].lower() for word in FORBIDDEN_IN_USER_TEXT)
