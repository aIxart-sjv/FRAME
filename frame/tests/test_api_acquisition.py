"""HTTP-layer tests for the acquisition endpoints (`POST /acquisition/search`,
`GET /acquisition/scenes/{search_id}/{scene_id}`, `POST /acquisition/retrieve`,
`GET /acquisition/download/{retrieval_id}/{bands,scl}`), via FastAPI's
TestClient. `frame.api.services.acquisition` binds `real_search_scenes` /
`real_retrieve_scene` by name at import time (`from ... import x as
real_x`), so these tests monkeypatch those names directly on the service
module -- patching `frame.acquisition.search.search_scenes` itself would not
reach the already-bound alias. No real network access or CDSE credentials
are used; see frame/tests/test_acquisition_live.py for the real, live path.
"""

from __future__ import annotations

import numpy as np
import pytest
from fastapi.testclient import TestClient

from frame.acquisition.contract import RetrievalProvenance, RetrievedScene, SceneCandidate
from frame.acquisition.errors import AuthenticationError, NoScenesFoundError, ProviderUnavailableError, SceneNotFoundError
from frame.acquisition.harmonise import RADIOMETRIC_METHOD_BY_PROVIDER
from frame.api import routes
from frame.api.app import create_app
from frame.api.services import acquisition as acquisition_service
from frame.api.services.storage import JobStore

VALID_SEARCH_BODY = {
    "bbox": [80.12, 12.85, 80.32, 13.15],
    "start_date": "2024-01-01",
    "end_date": "2024-04-01",
    "max_cloud_cover": 20.0,
}


def _real_shaped_candidate(scene_id="S2B_MSIL2A_20240330T045659_N0510_R119_T44PMV_20240330T072508"):
    return SceneCandidate(
        id=scene_id,
        provider="copernicus-cdse",
        stac_endpoint="https://stac.dataspace.copernicus.eu/v1/search",
        collection="sentinel-2-l2a",
        datetime="2024-03-30T04:56:59.024000Z",
        cloud_cover=2.43,
        bbox=(80.075, 12.574, 81.090, 13.568),
        geometry={"type": "Polygon", "coordinates": []},
        mgrs_tile="44PMV",
        platform="sentinel-2b",
        processing_baseline="05.10",
        asset_hrefs={"B04_10m": "s3://eodata/Sentinel-2/.../B04_10m.jp2", "B03_10m": "s3://eodata/..."},
    )


@pytest.fixture()
def app_and_client(tmp_path, monkeypatch):
    from frame.api import config

    monkeypatch.setattr(config, "WORKSPACE_DIR", tmp_path / "workspace")
    app = create_app()
    store = JobStore()
    app.dependency_overrides[routes.get_store] = lambda: store
    client = TestClient(app)
    return app, client, store


@pytest.fixture()
def client(app_and_client):
    return app_and_client[1]


@pytest.fixture()
def store(app_and_client):
    return app_and_client[2]


# ---------------------------------------------------------------------------
# POST /acquisition/search
# ---------------------------------------------------------------------------

def test_search_returns_real_shaped_scene_candidates(client, monkeypatch):
    monkeypatch.setattr(acquisition_service, "real_search_scenes", lambda query: [_real_shaped_candidate()])

    response = client.post("/acquisition/search", json=VALID_SEARCH_BODY)

    assert response.status_code == 200
    body = response.json()
    assert body["scene_count"] == 1
    assert body["scenes"][0]["id"] == "S2B_MSIL2A_20240330T045659_N0510_R119_T44PMV_20240330T072508"
    assert body["scenes"][0]["cloud_cover"] == 2.43
    assert body["scenes"][0]["asset_count"] == 2
    assert body["note"] is None
    assert "search_id" in body


def test_search_with_zero_real_results_is_200_not_an_error(client, monkeypatch):
    def fake_search(query):
        raise NoScenesFoundError("no real scenes matched")

    monkeypatch.setattr(acquisition_service, "real_search_scenes", fake_search)

    response = client.post("/acquisition/search", json=VALID_SEARCH_BODY)

    assert response.status_code == 200
    body = response.json()
    assert body["scene_count"] == 0
    assert body["scenes"] == []
    assert body["note"] is not None


def test_search_accepts_a_geojson_aoi_instead_of_bbox(client, monkeypatch):
    captured = {}

    def fake_search(query):
        captured["bbox"] = query.bbox
        return [_real_shaped_candidate()]

    monkeypatch.setattr(acquisition_service, "real_search_scenes", fake_search)

    body = {
        "aoi_geojson": {
            "type": "Polygon",
            "coordinates": [[[80.12, 12.85], [80.32, 12.85], [80.32, 13.15], [80.12, 13.15], [80.12, 12.85]]],
        },
        "start_date": "2024-01-01",
        "end_date": "2024-04-01",
    }
    response = client.post("/acquisition/search", json=body)

    assert response.status_code == 200
    assert captured["bbox"] == (80.12, 12.85, 80.32, 13.15)


def test_search_requires_bbox_or_aoi_geojson(client):
    body = {"start_date": "2024-01-01", "end_date": "2024-04-01"}
    response = client.post("/acquisition/search", json=body)
    assert response.status_code == 422


def test_search_rejects_an_invalid_bbox_as_a_named_error(client):
    body = dict(VALID_SEARCH_BODY, bbox=[80.32, 12.85, 80.12, 13.15])  # min_lon > max_lon
    response = client.post("/acquisition/search", json=body)
    assert response.status_code == 400
    assert response.json()["code"] == "invalid_acquisition_query"


def test_search_reports_a_provider_outage_as_503(client, monkeypatch):
    def fake_search(query):
        raise ProviderUnavailableError("CDSE STAC search failed: connection refused", provider="copernicus-cdse")

    monkeypatch.setattr(acquisition_service, "real_search_scenes", fake_search)

    response = client.post("/acquisition/search", json=VALID_SEARCH_BODY)

    assert response.status_code == 503
    assert response.json()["code"] == "acquisition_provider_unavailable"


# ---------------------------------------------------------------------------
# GET /acquisition/scenes/{search_id}/{scene_id}
# ---------------------------------------------------------------------------

def test_scene_detail_returns_the_real_candidate(client, monkeypatch):
    monkeypatch.setattr(acquisition_service, "real_search_scenes", lambda query: [_real_shaped_candidate()])
    search_response = client.post("/acquisition/search", json=VALID_SEARCH_BODY)
    search_id = search_response.json()["search_id"]
    scene_id = search_response.json()["scenes"][0]["id"]

    response = client.get(f"/acquisition/scenes/{search_id}/{scene_id}")

    assert response.status_code == 200
    assert response.json()["scene"]["id"] == scene_id
    assert "asset_hrefs" not in response.json()["scene"]  # the raw per-asset href dict is never echoed to the client


def test_scene_detail_with_an_unknown_search_id_is_404(client):
    response = client.get("/acquisition/scenes/does-not-exist/scene-1")
    assert response.status_code == 404
    assert response.json()["code"] == "acquisition_search_not_found"


def test_scene_detail_with_an_unknown_scene_id_is_a_named_404(client, monkeypatch):
    monkeypatch.setattr(acquisition_service, "real_search_scenes", lambda query: [_real_shaped_candidate()])
    search_id = client.post("/acquisition/search", json=VALID_SEARCH_BODY).json()["search_id"]

    response = client.get(f"/acquisition/scenes/{search_id}/not-a-real-scene-id")

    assert response.status_code == 404
    assert response.json()["code"] == "acquisition_scene_not_found"


# ---------------------------------------------------------------------------
# POST /acquisition/retrieve
# ---------------------------------------------------------------------------

def _fake_retrieved_scene(candidate, query, *, bands, resolution_m, include_scl, cdse_auth):
    height, width = 8, 8
    band_arrays = {b: np.full((height, width), 0.2, dtype="float32") for b in bands}
    return RetrievedScene(
        candidate=candidate,
        bands=band_arrays,
        band_native_resolution_m={b: 10.0 for b in bands},
        scl=np.full((height, width), 4, dtype="float32") if include_scl else None,
        crs="EPSG:32644",
        transform=(10.0, 0.0, 500000.0, 0.0, -10.0, 1400000.0),
        resolution_m=resolution_m,
        width=width,
        height=height,
        provenance=RetrievalProvenance(
            query=query,
            candidate_id=candidate.id,
            search_provider=candidate.provider,
            search_stac_endpoint=candidate.stac_endpoint,
            retrieval_provider="copernicus-cdse-process-api",
            retrieval_endpoint="https://sh.dataspace.copernicus.eu/process/v1",
            matched_acquisition_datetime=candidate.datetime,
            retrieved_at_utc="2024-03-30T05:00:00Z",
            band_sources={b: "copernicus-cdse-process-api" for b in bands},
            warnings=(),
        ),
    )


def _search_then_get_ids(client, monkeypatch):
    monkeypatch.setattr(acquisition_service, "real_search_scenes", lambda query: [_real_shaped_candidate()])
    search_response = client.post("/acquisition/search", json=VALID_SEARCH_BODY).json()
    return search_response["search_id"], search_response["scenes"][0]["id"]


def test_retrieve_returns_real_shaped_bands_and_writes_a_real_geotiff(client, monkeypatch, tmp_path):
    search_id, scene_id = _search_then_get_ids(client, monkeypatch)
    monkeypatch.setattr(acquisition_service, "real_retrieve_scene", _fake_retrieved_scene)

    response = client.post(
        "/acquisition/retrieve",
        json={"search_id": search_id, "scene_id": scene_id, "bands": ["B02", "B03", "B04", "B08"], "resolution_m": 10.0, "include_scl": True},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["bands"] == ["B02", "B03", "B04", "B08"]
    assert body["scl_included"] is True
    assert body["width"] == 8 and body["height"] == 8
    assert "bands_geotiff" in body["artifacts"]
    assert "scl_geotiff" in body["artifacts"]
    assert body["retrieval_provider"] == "copernicus-cdse-process-api"
    # Step F: real harmonisation metadata -- an RGBN-only retrieval is not a complete L2A set,
    # and the fixture's SCL (all "vegetation", class 4) is fully valid.
    assert body["is_complete_l2a"] is False
    assert body["validity_mask_coverage"] == 1.0
    assert body["band_radiometric_method"]["B04"] == RADIOMETRIC_METHOD_BY_PROVIDER["copernicus-cdse-process-api"]

    # download the real written artifact
    download = client.get(body["artifacts"]["bands_geotiff"])
    assert download.status_code == 200
    assert download.headers["content-type"] == "image/tiff"
    assert len(download.content) > 0


def test_retrieve_with_all_twelve_l2a_bands_reports_a_complete_product(client, monkeypatch):
    from frame.data.bands import L2A_BANDS

    search_id, scene_id = _search_then_get_ids(client, monkeypatch)
    monkeypatch.setattr(acquisition_service, "real_retrieve_scene", _fake_retrieved_scene)

    response = client.post(
        "/acquisition/retrieve",
        json={"search_id": search_id, "scene_id": scene_id, "bands": list(L2A_BANDS), "resolution_m": 10.0, "include_scl": False},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["is_complete_l2a"] is True
    assert body["validity_mask_coverage"] is None  # include_scl was false -- no mask to report
    assert set(body["band_radiometric_method"].keys()) == set(L2A_BANDS)


def test_retrieve_without_scl_has_no_scl_artifact_and_404s_on_download(client, monkeypatch):
    search_id, scene_id = _search_then_get_ids(client, monkeypatch)
    monkeypatch.setattr(acquisition_service, "real_retrieve_scene", _fake_retrieved_scene)

    response = client.post(
        "/acquisition/retrieve",
        json={"search_id": search_id, "scene_id": scene_id, "bands": ["B04"], "resolution_m": 10.0, "include_scl": False},
    )
    body = response.json()
    assert body["scl_included"] is False
    assert "scl_geotiff" not in body["artifacts"]

    retrieval_id = body["retrieval_id"]
    download = client.get(f"/acquisition/download/{retrieval_id}/scl")
    assert download.status_code == 404
    assert download.json()["code"] == "acquisition_artifact_not_available"


def test_retrieve_rejects_an_invalid_band_name_as_a_named_400(client, monkeypatch):
    search_id, scene_id = _search_then_get_ids(client, monkeypatch)
    monkeypatch.setattr(acquisition_service, "real_retrieve_scene", _fake_retrieved_scene)

    response = client.post(
        "/acquisition/retrieve",
        json={"search_id": search_id, "scene_id": scene_id, "bands": ["B04", "B10"], "resolution_m": 10.0},
    )
    assert response.status_code == 400
    assert response.json()["code"] == "invalid_acquisition_query"


def test_retrieve_with_an_unknown_scene_is_a_named_404(client, monkeypatch):
    search_id, _ = _search_then_get_ids(client, monkeypatch)
    response = client.post(
        "/acquisition/retrieve",
        json={"search_id": search_id, "scene_id": "not-a-real-scene", "bands": ["B04"], "resolution_m": 10.0},
    )
    assert response.status_code == 404
    assert response.json()["code"] == "acquisition_scene_not_found"


def test_retrieve_reports_authentication_failure_distinctly_from_provider_outage(client, monkeypatch):
    search_id, scene_id = _search_then_get_ids(client, monkeypatch)

    def failing_retrieve(*args, **kwargs):
        raise AuthenticationError("CDSE rejected the configured credentials", provider="copernicus-cdse-process-api")

    monkeypatch.setattr(acquisition_service, "real_retrieve_scene", failing_retrieve)

    response = client.post(
        "/acquisition/retrieve",
        json={"search_id": search_id, "scene_id": scene_id, "bands": ["B04"], "resolution_m": 10.0},
    )
    assert response.status_code == 401
    assert response.json()["code"] == "acquisition_authentication_failed"


def test_retrieve_reports_a_generic_retrieval_failure_as_provider_unavailable(client, monkeypatch):
    search_id, scene_id = _search_then_get_ids(client, monkeypatch)

    def failing_retrieve(*args, **kwargs):
        raise ProviderUnavailableError("both providers failed", provider="microsoft-planetary-computer")

    monkeypatch.setattr(acquisition_service, "real_retrieve_scene", failing_retrieve)

    response = client.post(
        "/acquisition/retrieve",
        json={"search_id": search_id, "scene_id": scene_id, "bands": ["B04"], "resolution_m": 10.0},
    )
    assert response.status_code == 503
    assert response.json()["code"] == "acquisition_provider_unavailable"


def test_download_with_an_unknown_retrieval_id_is_404(client):
    response = client.get("/acquisition/download/does-not-exist/bands")
    assert response.status_code == 404
    assert response.json()["code"] == "acquisition_retrieval_not_found"
