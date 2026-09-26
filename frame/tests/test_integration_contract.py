"""Phase 8 integration contract: the production path (HTTP upload -> preprocessing -> model selection -> tile engine -> TTA -> GeoTIFF) end to end,
with fake models (no GPU, no weights), and the failure cases a demo can hit.

It re-uses the `env` fixture of test_api_tiling.py: the real dependency chain, with only `model_service.get_model` replaced. Nothing here is scientific evidence.
The real-model equivalents are the `integration`-marked tests (test_api_integration.py, test_api_mamba_integration.py, test_api_tiling_mamba_integration.py).
"""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from frame.geospatial import read_geotiff, write_geotiff
from frame.models.errors import ModelUnavailableError, ModelWorkerError
from frame.models.selection import MODEL_SPECS
from frame.preprocessing import RGBN_BANDS
from frame.preprocessing.metadata import RasterMetadata
from frame.tests.test_api_tiling import env, perfect, scene_bytes, upload  # noqa: F401  (`env` is a fixture)

ORIGIN_X, ORIGIN_Y = 720285.0, 4375125.0


def raster_bytes(tmp_path, array, *, nodata=0.0, name="custom", bands=RGBN_BANDS):
    """A GeoTIFF from an explicit array (EPSG:32630, 10 m, origin ORIGIN_X / ORIGIN_Y)."""
    _, h, w = array.shape
    metadata = RasterMetadata(
        crs="EPSG:32630", transform=(10.0, 0.0, ORIGIN_X, 0.0, -10.0, ORIGIN_Y), bounds=(ORIGIN_X, ORIGIN_Y - h * 10.0, ORIGIN_X + w * 10.0, ORIGIN_Y), resolution_m=10.0,
        width=w, height=h, band_names=tuple(bands), acquisition_timestamp="2023-01-15T10:54:11.024000", nodata_value=nodata, cloud_mask_coverage=1.0, sr_variant=None,
    )
    path = tmp_path / f"{name}.tif"
    write_geotiff(path, array, metadata)
    return path.read_bytes()


def post_scene(client, data, *, scale="reflectance"):
    return client.post("/upload", files={"file": ("scene.tif", data, "image/tiff")}, data={"input_scale": scale})


def base(h=100, w=130, seed=1):
    return np.random.default_rng(seed).random((4, h, w)).astype("float32") * 0.4 + 0.1


# ============================================================================== the output geometry of a rectangular scene, for both models


@pytest.mark.parametrize("model", ["lite", "mamba"])
@pytest.mark.parametrize("h,w", [(200, 300), (97, 131)])
def test_the_output_geotiff_keeps_crs_origin_and_footprint_and_divides_the_pixel_size_by_four(env, model, h, w):
    client, _, _, tmp_path = env
    body = client.post("/sr/run", json={"upload_id": upload(client, tmp_path, h, w).json()["upload_id"], "model": model}).json()
    sr, out = read_geotiff(body["artifacts"]["sr_geotiff"], require_crs=True)
    unc, unc_geo = read_geotiff(body["artifacts"]["uncertainty_geotiff"], require_crs=True)

    assert sr.shape == (4, 4 * h, 4 * w) and body["output_shape"] == [4, 4 * h, 4 * w]                  # exact 4x, no rounding of an odd size
    assert out.crs == "EPSG:32630"
    assert out.transform == pytest.approx((2.5, 0.0, ORIGIN_X, 0.0, -2.5, ORIGIN_Y), abs=1e-9)          # same origin, 10 m -> 2.5 m
    assert out.bounds == pytest.approx((ORIGIN_X, ORIGIN_Y - h * 10.0, ORIGIN_X + w * 10.0, ORIGIN_Y), abs=1e-6)  # the footprint of the input
    assert (out.width, out.height) == (4 * w, 4 * h) and out.band_names == tuple(RGBN_BANDS)
    assert unc.shape[1:] == sr.shape[1:] and unc_geo.transform == out.transform and unc_geo.crs == out.crs        # the stability raster is on the very same grid

    reported = body["metadata"]["output_geospatial"]
    assert reported["crs"] == out.crs and reported["width"] == out.width and reported["height"] == out.height
    assert tuple(reported["transform"]) == pytest.approx(out.transform) and tuple(reported["bounds"]) == pytest.approx(out.bounds)
    assert body["resolution"]["native_resolution_m"] == 10.0 and body["resolution"]["sr_resolution_m"] == 2.5 and body["resolution"]["scale_factor"] == 4


@pytest.mark.parametrize("model", ["lite", "mamba"])
def test_the_model_that_was_asked_for_is_the_model_recorded_everywhere(env, model):
    client, _, requested, tmp_path = env
    body = client.post("/sr/run", json={"upload_id": upload(client, tmp_path, 130, 170).json()["upload_id"], "model": model}).json()
    _, out = read_geotiff(body["artifacts"]["sr_geotiff"], require_crs=True)
    assert body["model_id"] == model and body["model_name"] == MODEL_SPECS[model].model_name and out.sr_variant == MODEL_SPECS[model].model_name
    _, unc = read_geotiff(body["artifacts"]["uncertainty_geotiff"], require_crs=True)
    assert unc.sr_variant == MODEL_SPECS[model].model_name
    assert requested == [model]


def test_a_model_consistent_across_overlaps_leaves_no_seam_at_the_integration_layer(env):
    """The tile engine blends overlaps; with a model whose tiles agree exactly (a nearest x4 replication) the seam diagnostic must be ~0, and the mosaic must equal the model run on the whole scene."""
    client, _, _, tmp_path = env
    h, w = 260, 300
    body = client.post("/sr/run", json={"upload_id": upload(client, tmp_path, h, w).json()["upload_id"]}).json()
    seam = body["metadata"]["tiling"]["seam_diagnostic"]
    assert seam["status"] == "COMPUTABLE" and seam["max_abs_difference"] < 1e-5
    sr, _ = read_geotiff(body["artifacts"]["sr_geotiff"], require_crs=True)
    whole = perfect(torch.from_numpy(np.random.default_rng(0).random((4, h, w)).astype("float32") * 0.4 + 0.1)[None])[0].numpy()
    assert sr.shape == whole.shape and np.abs(sr - whole).max() < 1e-5                                       # tiling changed nothing about the pixel values or their places


# ============================================================================== no silent model substitution


def test_an_unavailable_mamba_is_a_503_and_lite_is_never_run_in_its_place(env):
    client, registry, requested, tmp_path = env
    lite_calls = []
    registry["lite"] = lambda x: lite_calls.append(1) or perfect(x)
    registry["mamba"] = ModelUnavailableError("SEN2SR-Mamba requires a CUDA-capable GPU.")
    upload_id = upload(client, tmp_path, 130, 170).json()["upload_id"]
    response = client.post("/sr/run", json={"upload_id": upload_id, "model": "mamba"})
    assert response.status_code == 503 and response.json()["code"] == "model_unavailable"
    assert requested == ["mamba"] and lite_calls == []                                                       # Lite was not even requested from the model service
    assert list((tmp_path / "workspace" / "jobs").glob("*")) == []                                           # and no job exists


def test_a_mamba_worker_that_dies_mid_scene_is_a_clean_error_and_lite_is_never_run_in_its_place(env):
    client, registry, _, tmp_path = env
    lite_calls = []
    registry["lite"] = lambda x: lite_calls.append(1) or perfect(x)
    calls = []

    def dies(x):
        calls.append(1)
        if len(calls) == 3:
            raise ModelWorkerError("The SEN2SR-Mamba worker stopped unexpectedly.", technical_detail="segfault")
        return perfect(x)

    registry["mamba"] = dies
    response = client.post("/sr/run", json={"upload_id": upload(client, tmp_path, 200, 300).json()["upload_id"], "model": "mamba"})
    assert response.status_code in (500, 503) and response.json()["code"] == "model_runtime_error" and "segfault" not in response.text
    assert lite_calls == []
    assert list((tmp_path / "workspace" / "jobs").glob("*")) == []                                           # no half-written job directory


def test_the_default_model_is_lite_and_only_lite(env):
    client, registry, requested, tmp_path = env
    registry["mamba"] = ModelUnavailableError("must not be touched")
    body = client.post("/sr/run", json={"upload_id": upload(client, tmp_path, 130, 170).json()["upload_id"]}).json()
    assert body["model_id"] == "lite" and requested == ["lite"]


# ============================================================================== input validation at the door


def test_a_file_that_is_not_a_geotiff_is_a_400_with_a_json_body_and_no_server_path(env):
    client, _, _, tmp_path = env
    response = client.post("/upload", files={"file": ("scene.tif", b"this is not a tiff", "image/tiff")}, data={"input_scale": "reflectance"})
    assert response.status_code == 400 and response.json()["code"] == "frame_error"
    assert "scene.tif" in response.json()["detail"] and str(tmp_path) not in response.text
    assert list((tmp_path / "workspace" / "uploads").glob("*")) == []                                       # the rejected file is not kept


def test_a_scene_with_no_valid_pixel_is_rejected_at_upload_not_processed_into_a_zero_image(env):
    client, _, _, tmp_path = env
    for name, array in (("nodata", np.zeros((4, 100, 130), "float32")), ("nan", np.full((4, 100, 130), np.nan, "float32"))):
        response = post_scene(client, raster_bytes(tmp_path, array, name=name))
        assert response.status_code == 400 and "no valid pixel" in response.json()["detail"], name
    assert list((tmp_path / "workspace" / "uploads").glob("*")) == []


def test_reflectance_declared_as_raw_digital_numbers_is_rejected_instead_of_silently_producing_a_black_image(env):
    client, _, _, tmp_path = env
    response = post_scene(client, raster_bytes(tmp_path, base()), scale="raw_digital_number")
    assert response.status_code == 400 and "reflectance" in response.json()["detail"]


def test_digital_numbers_declared_as_reflectance_are_rejected_for_every_model_not_only_mamba(env):
    client, _, _, tmp_path = env
    response = post_scene(client, raster_bytes(tmp_path, base() * 10000), scale="reflectance")
    assert response.status_code == 400 and "raw_digital_number" in response.json()["detail"]


def test_digital_numbers_declared_correctly_are_processed(env):
    client, _, _, tmp_path = env
    upload_response = post_scene(client, raster_bytes(tmp_path, base() * 10000), scale="raw_digital_number")
    assert upload_response.status_code == 200
    body = client.post("/sr/run", json={"upload_id": upload_response.json()["upload_id"]}).json()
    sr, _ = read_geotiff(body["artifacts"]["sr_geotiff"], require_crs=True)
    assert 0.09 < float(sr.mean()) < 0.51                                                                    # 0.1-0.5 reflectance came back out, not 1e-5


def test_the_scale_is_checked_again_at_run_time_for_a_record_that_bypassed_the_upload_check(env):
    client, _, _, tmp_path = env
    from frame.api import routes

    upload_id = post_scene(client, raster_bytes(tmp_path, base()), scale="reflectance").json()["upload_id"]
    record = client.app.dependency_overrides[routes.get_store]().get_upload(upload_id)
    object.__setattr__(record, "input_scale", "raw_digital_number")                                          # a stale or hand-edited record, not something the upload route would accept
    response = client.post("/sr/run", json={"upload_id": upload_id})
    assert response.status_code == 400 and "reflectance" in response.json()["detail"]


def test_non_finite_pixels_are_excluded_from_the_reported_coverage_and_the_output_stays_finite(env):
    client, _, _, tmp_path = env
    array = base()
    array[:, 10:20, 10:20] = np.nan
    array[0, 50, 50] = np.inf
    upload_id = post_scene(client, raster_bytes(tmp_path, array)).json()["upload_id"]
    body = client.post("/sr/run", json={"upload_id": upload_id}).json()
    assert body["metadata"]["preprocessing_mask_coverage"] == pytest.approx(1 - 101 / (100 * 130))
    sr, _ = read_geotiff(body["artifacts"]["sr_geotiff"], require_crs=True)
    assert np.isfinite(sr).all()


def test_a_nodata_heavy_scene_is_processed_and_its_low_coverage_is_reported(env):
    client, _, _, tmp_path = env
    array = base()
    array[:, :, :100] = 0.0                                                                                   # 77 % nodata
    body = client.post("/sr/run", json={"upload_id": post_scene(client, raster_bytes(tmp_path, array)).json()["upload_id"]}).json()
    assert body["metadata"]["preprocessing_mask_coverage"] == pytest.approx(30 / 130)


@pytest.mark.parametrize("bands", [RGBN_BANDS + ("B11",), RGBN_BANDS[:3]])
def test_an_unsupported_band_configuration_is_a_400_naming_the_problem(env, bands):
    client, _, _, tmp_path = env
    array = np.random.default_rng(0).random((len(bands), 100, 130)).astype("float32") * 0.4 + 0.1
    response = post_scene(client, raster_bytes(tmp_path, array, bands=bands))
    assert response.status_code == 400 and "Band set does not match" in response.json()["detail"]


# ============================================================================== a misbehaving model never produces a wrong result or a leaky error


def _run_with(env, model_callable):
    client, registry, _, tmp_path = env
    registry["lite"] = model_callable
    upload_id = upload(client, tmp_path, 130, 170).json()["upload_id"]
    return client, client.post("/sr/run", json={"upload_id": upload_id})


def test_a_model_that_returns_the_wrong_number_of_channels_is_a_json_error_not_a_result(env):
    client, response = _run_with(env, lambda x: perfect(x)[:, :3])
    assert response.status_code == 500 and response.json()["code"] == "model_runtime_error"


def test_a_model_that_returns_the_wrong_scale_is_a_json_error_not_a_result(env):
    client, response = _run_with(env, lambda x: F.interpolate(x, scale_factor=3, mode="nearest"))
    assert response.status_code == 500 and response.json()["code"] == "model_runtime_error"


def test_a_model_that_returns_nan_is_a_json_error_not_a_result(env):
    client, response = _run_with(env, lambda x: perfect(x) * float("nan"))
    assert response.status_code == 500 and response.json()["code"] == "model_runtime_error"


def test_an_unexpected_exception_is_a_json_500_that_leaks_nothing(env):
    def boom(x):
        raise RuntimeError("cuda oom at /home/someone/secret/path")

    client, _ = env[0], None
    app = client.app
    from fastapi.testclient import TestClient

    quiet = TestClient(app, raise_server_exceptions=False)                                                  # what a real server does: answer, do not re-raise into the test
    env[1]["lite"] = boom
    upload_id = upload(quiet, env[3], 130, 170).json()["upload_id"]
    response = quiet.post("/sr/run", json={"upload_id": upload_id})
    assert response.status_code == 500 and response.headers["content-type"].startswith("application/json")
    body = response.json()
    assert body["code"] == "internal_error" and set(body) == {"error", "code", "detail"}
    assert "secret" not in response.text and "cuda" not in response.text.lower() and "Traceback" not in response.text


# ============================================================================== the reference gate is ONE gate for the reliability and the downstream layers


def test_reliability_and_downstream_use_the_same_gate_function_and_version():
    import frame.downstream.runner as downstream
    import frame.reliability.eligibility as gate
    import frame.reliability.runner as reliability

    assert downstream.evaluate_reference is gate.evaluate_reference and reliability.evaluate_reference is gate.evaluate_reference
    assert gate.GATE_VERSION == "frame-reliability-gate/1"


def test_the_same_tiles_are_excluded_with_the_same_reasons_by_both_layers(tmp_path):
    """An ineligible reference must not enter EITHER analysis, and the two layers must agree on which tiles those are and why."""
    from frame.tests import test_downstream_runner as ds
    from frame.tests import test_reliability_runner as rel

    dataset = ds.excluded_dataset()
    _, ds_out = ds.run(tmp_path / "ds", datasets=[dataset], names=("a",))
    _, rel_out = rel.run(tmp_path / "rel", datasets=[ds.excluded_dataset()], names=("a",))

    downstream_rows = {t["sample_id"]: t for t in ds.tiles(ds_out)}
    reliability_rows = {r["sample_id"]: r for r in rel.rows_of(rel_out) if r.get("type") == "tile" or "sample_id" in r}
    excluded_ds = {k for k, v in downstream_rows.items() if v["status"] == "excluded_from_downstream_primary_analysis"}
    excluded_rel = {k for k, v in reliability_rows.items() if v.get("status") == "excluded_from_uncertainty_error_analysis"}
    assert excluded_ds == excluded_rel and len(excluded_ds) == 2
    for sample_id in excluded_ds:
        assert downstream_rows[sample_id]["reason"] == reliability_rows[sample_id]["reason"]
        assert downstream_rows[sample_id]["evidence_level"] == reliability_rows[sample_id]["evidence_level"]


# ============================================================================== what the product says about its own evidence


def test_the_ndvi_response_carries_the_demonstration_note_and_the_stability_is_labelled_a_diagnostic(env):
    from frame.api.schemas import NDVI_DEMONSTRATION_NOTE, UNCERTAINTY_LABEL

    client, _, _, tmp_path = env
    job = client.post("/sr/run", json={"upload_id": upload(client, tmp_path, 130, 170).json()["upload_id"]}).json()
    assert job["uncertainty"]["label"] == UNCERTAINTY_LABEL and "diagnostic" in UNCERTAINTY_LABEL
    assert "weakly associated" in job["uncertainty"]["disclaimer"] and any("weakly associated" in c for c in job["scientific_caveats"])
    ndvi = client.post("/analysis/ndvi", json={"job_id": job["job_id"]}).json()
    assert ndvi["scientific_caveats"][-1] == NDVI_DEMONSTRATION_NOTE and len(ndvi["scientific_caveats"]) == 5
    joined = " ".join(ndvi["scientific_caveats"])
    assert "TTA stability diagnostic" in joined and "Phase 5" not in joined and "Uncertainty here is" not in joined      # the old wording of frame.analysis is not shown to users
    fetched = client.get(f"/analysis/{ndvi['analysis_id']}").json()
    assert fetched["scientific_caveats"] == ndvi["scientific_caveats"]
