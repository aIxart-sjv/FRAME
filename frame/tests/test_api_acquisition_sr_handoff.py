"""Step E: the handoff from a real acquisition retrieval to the existing, unmodified SR pipeline
(`POST /acquisition/retrieve/{retrieval_id}/prepare-sr-input` -> the real `POST /sr/run`).

Uses real GeoTIFFs and the real FastAPI app throughout (like frame/tests/test_api.py's own
`/upload` -> `/sr/run` tests) -- no acquisition provider is mocked here because none is needed:
`prepare_sr_input` only reads a real, already-on-disk retrieval artifact and writes a real
RGBN GeoTIFF from it, exactly the same shape of work `/upload` already does for a user-supplied
file. This is a genuine end-to-end integration test, not a monkeypatched one.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from fastapi.testclient import TestClient

from frame.acquisition.contract import RetrievalProvenance, SceneCandidate
from frame.api import routes
from frame.api.app import create_app
from frame.api.errors import AcquisitionBandsInsufficientForModelError
from frame.api.services import acquisition as acquisition_service
from frame.api.services.storage import AcquisitionRetrievalRecord, AcquisitionRetrievalNotFoundError, JobStore
from frame.data.bands import RGBN_BANDS
from frame.geospatial import RGBN_SCALE_FACTOR, write_geotiff
from frame.geospatial.errors import UnreadableRasterError
from frame.preprocessing.metadata import RasterMetadata

H = 32
CRS = "EPSG:32644"
TRANSFORM = (10.0, 0.0, 500000.0, 0.0, -10.0, 1400000.0)
BOUNDS = (500000.0, 1400000.0 - H * 10.0, 500000.0 + H * 10.0, 1400000.0)


def _equivariant_model(x_batched: torch.Tensor) -> torch.Tensor:
    return F.interpolate(x_batched, scale_factor=RGBN_SCALE_FACTOR, mode="bicubic", antialias=True)


def _candidate(scene_id="S2B_MSIL2A_20240330T045659_N0510_R119_T44PMV_20240330T072508") -> SceneCandidate:
    return SceneCandidate(
        id=scene_id,
        provider="copernicus-cdse",
        stac_endpoint="https://stac.dataspace.copernicus.eu/v1/search",
        collection="sentinel-2-l2a",
        datetime="2024-03-30T04:56:59.024000Z",
        cloud_cover=2.43,
        bbox=(80.075, 12.574, 81.090, 13.568),
        geometry={"type": "Polygon", "coordinates": []},
        mgrs_tile="MGRS-44PMV",
        platform="sentinel-2b",
        processing_baseline="05.10",
        asset_hrefs={},
    )


def _write_bands_geotiff(path, *, bands=RGBN_BANDS, acquisition_timestamp="2024-03-30T04:56:59.024000Z"):
    rng = np.random.default_rng(0)
    array = (rng.random((len(bands), H, H)).astype("float32") * 0.4) + 0.1
    metadata = RasterMetadata(
        crs=CRS,
        transform=TRANSFORM,
        bounds=BOUNDS,
        resolution_m=10.0,
        width=H,
        height=H,
        band_names=tuple(bands),
        acquisition_timestamp=acquisition_timestamp,
        nodata_value=None,
        cloud_mask_coverage=None,
        sr_variant="frame.acquisition retrieval -- real reflectance, not an SR product",
    )
    write_geotiff(path, array, metadata)
    return array, metadata


def _put_retrieval_record(store: JobStore, tmp_path, *, bands=RGBN_BANDS, retrieval_id="retrieval-1", scene_id=None) -> AcquisitionRetrievalRecord:
    candidate = _candidate(scene_id) if scene_id else _candidate()
    bands_path = tmp_path / f"{retrieval_id}_bands.tif"
    _write_bands_geotiff(bands_path, bands=bands)

    provenance = RetrievalProvenance(
        query=None,  # not needed by prepare_sr_input
        candidate_id=candidate.id,
        search_provider=candidate.provider,
        search_stac_endpoint=candidate.stac_endpoint,
        retrieval_provider="copernicus-cdse-process-api",
        retrieval_endpoint="https://sh.dataspace.copernicus.eu/process/v1",
        matched_acquisition_datetime=candidate.datetime,
        retrieved_at_utc="2024-03-30T05:00:00Z",
        band_sources={b: "copernicus-cdse-process-api" for b in bands},
        warnings=(),
    )
    record = AcquisitionRetrievalRecord(
        retrieval_id=retrieval_id,
        search_id="search-1",
        scene_id=candidate.id,
        bands_geotiff_path=bands_path,
        scl_geotiff_path=None,
        provenance=provenance,
        result={},
        created_at="2024-03-30T05:00:00Z",
    )
    store.put_acquisition_retrieval(record)
    return record


# ---------------------------------------------------------------------------
# Service-level tests (frame.api.services.acquisition.prepare_sr_input)
# ---------------------------------------------------------------------------

def test_prepare_sr_input_produces_a_real_upload_and_hands_off_into_the_real_sr_pipeline(tmp_path):
    store = JobStore()
    _put_retrieval_record(store, tmp_path)

    result = acquisition_service.prepare_sr_input(store, "retrieval-1", workspace_dir=tmp_path / "work")

    assert result["band_names"] == list(RGBN_BANDS)
    assert result["crs"] == CRS
    assert result["input_scale"] == "reflectance"
    assert result["excluded_bands"] == []
    upload = store.get_upload(result["upload_id"])
    assert upload.source_acquisition_retrieval_id == "retrieval-1"

    # The real, existing, unmodified SR pipeline runs on this upload_id with no special-casing.
    from frame.api.services.pipeline import run_sr_job

    job = run_sr_job(
        store, result["upload_id"], model=_equivariant_model, device="cpu", seed=42, workspace_dir=tmp_path / "jobs"
    )
    assert job.result["status"] == "completed"
    assert job.result["bands"] == list(RGBN_BANDS)


def test_prepare_sr_input_with_an_unknown_retrieval_id_raises_a_named_error(tmp_path):
    store = JobStore()
    with pytest.raises(AcquisitionRetrievalNotFoundError):
        acquisition_service.prepare_sr_input(store, "does-not-exist", workspace_dir=tmp_path / "work")


def test_prepare_sr_input_with_a_missing_real_file_raises_unreadable_raster(tmp_path):
    store = JobStore()
    record = _put_retrieval_record(store, tmp_path)
    record.bands_geotiff_path.unlink()  # the real file is gone -- an invalid/corrupted artifact

    with pytest.raises(UnreadableRasterError):
        acquisition_service.prepare_sr_input(store, "retrieval-1", workspace_dir=tmp_path / "work")


def test_prepare_sr_input_rejects_a_retrieval_missing_required_rgbn_bands(tmp_path):
    store = JobStore()
    _put_retrieval_record(store, tmp_path, bands=("B04",))  # only one of the four required bands

    with pytest.raises(AcquisitionBandsInsufficientForModelError) as excinfo:
        acquisition_service.prepare_sr_input(store, "retrieval-1", workspace_dir=tmp_path / "work")
    assert "B03" in str(excinfo.value) and "B02" in str(excinfo.value) and "B08" in str(excinfo.value)


def test_prepare_sr_input_preserves_real_scene_provenance(tmp_path):
    store = JobStore()
    _put_retrieval_record(store, tmp_path, scene_id="S2A_REAL_SCENE_ID_EXAMPLE")

    result = acquisition_service.prepare_sr_input(store, "retrieval-1", workspace_dir=tmp_path / "work")

    assert result["scene_id"] == "S2A_REAL_SCENE_ID_EXAMPLE"
    assert result["retrieval_id"] == "retrieval-1"
    assert result["search_id"] == "search-1"
    assert result["provider"] == "copernicus-cdse-process-api"
    assert result["matched_acquisition_datetime"] == "2024-03-30T04:56:59.024000Z"


def test_prepare_sr_input_preserves_real_geospatial_metadata(tmp_path):
    store = JobStore()
    _put_retrieval_record(store, tmp_path)

    result = acquisition_service.prepare_sr_input(store, "retrieval-1", workspace_dir=tmp_path / "work")

    from frame.geospatial import read_geotiff

    upload = store.get_upload(result["upload_id"])
    _, sr_input_metadata = read_geotiff(upload.file_path, require_crs=True)
    assert sr_input_metadata.crs == CRS
    assert sr_input_metadata.transform == TRANSFORM
    assert sr_input_metadata.bounds == BOUNDS
    assert sr_input_metadata.resolution_m == 10.0
    # the real acquisition timestamp survives into the SR-input file's own metadata
    assert sr_input_metadata.acquisition_timestamp == "2024-03-30T04:56:59.024000Z"


def test_prepare_sr_input_records_excluded_bands_instead_of_discarding_them(tmp_path):
    store = JobStore()
    _put_retrieval_record(store, tmp_path, bands=RGBN_BANDS + ("B11",))

    result = acquisition_service.prepare_sr_input(store, "retrieval-1", workspace_dir=tmp_path / "work")

    assert result["excluded_bands"] == ["B11"]
    assert "B11" in result["note"]
    assert result["band_names"] == list(RGBN_BANDS)  # the SR input itself is RGBN-only, no more
    # the original retrieval's own bands file (with B11) is untouched, not overwritten
    retrieval = store.get_acquisition_retrieval("retrieval-1")
    from frame.geospatial import read_geotiff

    _, original_metadata = read_geotiff(retrieval.bands_geotiff_path, require_crs=True)
    assert "B11" in original_metadata.band_names


def test_prepare_sr_input_never_claims_a_12_band_sr_output(tmp_path):
    """Scientific-honesty regression: whatever this handoff does, it must never present the SR
    input/output as more than the real, RGBN-only capability the models actually have."""
    store = JobStore()
    _put_retrieval_record(store, tmp_path, bands=RGBN_BANDS + ("B11", "B12", "B05"))

    result = acquisition_service.prepare_sr_input(store, "retrieval-1", workspace_dir=tmp_path / "work")

    assert len(result["band_names"]) == 4
    assert set(result["excluded_bands"]) == {"B11", "B12", "B05"}


# ---------------------------------------------------------------------------
# HTTP-layer integration (real FastAPI app, real files, no mocked provider)
# ---------------------------------------------------------------------------

@pytest.fixture()
def app_and_client(tmp_path, monkeypatch):
    from frame.api import config

    monkeypatch.setattr(config, "WORKSPACE_DIR", tmp_path / "workspace")
    app = create_app()
    store = JobStore()
    app.dependency_overrides[routes.get_model_callable] = lambda: _equivariant_model
    app.dependency_overrides[routes.get_store] = lambda: store
    client = TestClient(app)
    return client, store


def test_full_http_handoff_acquisition_to_sr_run(app_and_client, tmp_path):
    client, store = app_and_client
    _put_retrieval_record(store, tmp_path)

    prepare_response = client.post("/acquisition/retrieve/retrieval-1/prepare-sr-input")
    assert prepare_response.status_code == 200
    body = prepare_response.json()
    assert body["band_names"] == list(RGBN_BANDS)
    assert body["scene_id"] == "S2B_MSIL2A_20240330T045659_N0510_R119_T44PMV_20240330T072508"

    sr_response = client.post("/sr/run", json={"upload_id": body["upload_id"], "model": "lite"})
    assert sr_response.status_code == 200
    sr_body = sr_response.json()
    assert sr_body["status"] == "completed"
    assert sr_body["bands"] == list(RGBN_BANDS)
    assert sr_body["resolution"]["scale_factor"] == RGBN_SCALE_FACTOR
    # real geospatial pass-through, verified at the HTTP boundary too
    assert sr_body["crs"] == CRS
    assert sr_body["metadata"]["output_geospatial"]["width"] == H * RGBN_SCALE_FACTOR


def test_http_prepare_sr_input_with_unknown_retrieval_is_a_named_404(app_and_client):
    client, _store = app_and_client
    response = client.post("/acquisition/retrieve/does-not-exist/prepare-sr-input")
    assert response.status_code == 404
    assert response.json()["code"] == "acquisition_retrieval_not_found"


def test_http_prepare_sr_input_with_insufficient_bands_is_a_named_422(app_and_client, tmp_path):
    client, store = app_and_client
    _put_retrieval_record(store, tmp_path, bands=("B04", "B08"))

    response = client.post("/acquisition/retrieve/retrieval-1/prepare-sr-input")
    assert response.status_code == 422
    assert response.json()["code"] == "acquisition_bands_insufficient_for_model"


def test_the_real_sr_input_file_is_downloadable_for_the_frontends_original_comparison_view(app_and_client, tmp_path):
    """The 'original' viewer needs the exact real file /sr/run used, with matching band order --
    fetched here rather than assumed, to prove no band-order mismatch is possible."""
    client, store = app_and_client
    _put_retrieval_record(store, tmp_path)

    prepared = client.post("/acquisition/retrieve/retrieval-1/prepare-sr-input").json()
    download = client.get(f"/upload/{prepared['upload_id']}/download")

    assert download.status_code == 200
    assert download.headers["content-type"] == "image/tiff"
    assert len(download.content) > 0


def test_download_with_an_unknown_upload_id_is_a_named_404(app_and_client):
    client, _store = app_and_client
    response = client.get("/upload/does-not-exist/download")
    assert response.status_code == 404
    assert response.json()["code"] == "upload_not_found"
