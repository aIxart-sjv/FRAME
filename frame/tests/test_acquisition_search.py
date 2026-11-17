"""Unit tests for frame.acquisition.search -- a fake `http_post` stands in for
the network, so these never require real CDSE access, matching the fixture
approach docs/ACQUISITION.md documents for CI."""

import pytest

from frame.acquisition.contract import AcquisitionQuery
from frame.acquisition.errors import NoScenesFoundError, ProviderUnavailableError
from frame.acquisition.search import search_scenes

VALID_QUERY = AcquisitionQuery(bbox=(80.12, 12.85, 80.32, 13.15), start_date="2024-01-01", end_date="2024-04-01")


class _FakeResponse:
    def __init__(self, status_code, payload=None, text=""):
        self.status_code = status_code
        self._payload = payload or {}
        self.text = text

    def json(self):
        return self._payload


def _real_feature(cloud_cover=2.43):
    return {
        "id": "S2B_MSIL2A_20240330T045659_N0510_R119_T44PMV_20240330T072508",
        "bbox": [80.075, 12.574, 81.090, 13.568],
        "geometry": {"type": "Polygon", "coordinates": []},
        "properties": {
            "datetime": "2024-03-30T04:56:59.024000Z",
            "eo:cloud_cover": cloud_cover,
            "platform": "sentinel-2b",
            "s2:mgrs_tile": "44PMV",
            "s2:processing_baseline": "05.10",
        },
        "assets": {"B04_10m": {"href": "s3://eodata/Sentinel-2/.../B04_10m.jp2"}},
    }


def test_a_real_feature_list_becomes_scene_candidates():
    def fake_post(url, json, timeout):
        assert url.endswith("/search")
        assert json["collections"] == ["sentinel-2-l2a"]
        return _FakeResponse(200, {"features": [_real_feature()]})

    candidates = search_scenes(VALID_QUERY, http_post=fake_post)
    assert len(candidates) == 1
    candidate = candidates[0]
    assert candidate.id == "S2B_MSIL2A_20240330T045659_N0510_R119_T44PMV_20240330T072508"
    assert candidate.provider == "copernicus-cdse"
    assert candidate.cloud_cover == 2.43
    assert candidate.mgrs_tile == "44PMV"
    assert candidate.asset_hrefs["B04_10m"].startswith("s3://")


def test_an_empty_feature_list_raises_no_scenes_found():
    def fake_post(url, json, timeout):
        return _FakeResponse(200, {"features": []})

    with pytest.raises(NoScenesFoundError):
        search_scenes(VALID_QUERY, http_post=fake_post)


def test_scenes_above_the_cloud_threshold_are_excluded():
    def fake_post(url, json, timeout):
        return _FakeResponse(200, {"features": [_real_feature(cloud_cover=95.0)]})

    query = AcquisitionQuery(bbox=(80.12, 12.85, 80.32, 13.15), start_date="2024-01-01", end_date="2024-04-01", max_cloud_cover=20.0)
    with pytest.raises(NoScenesFoundError):
        search_scenes(query, http_post=fake_post)


def test_a_feature_with_no_reported_cloud_cover_is_kept_not_excluded():
    def fake_post(url, json, timeout):
        return _FakeResponse(200, {"features": [_real_feature(cloud_cover=None)]})

    query = AcquisitionQuery(bbox=(80.12, 12.85, 80.32, 13.15), start_date="2024-01-01", end_date="2024-04-01", max_cloud_cover=5.0)
    candidates = search_scenes(query, http_post=fake_post)
    assert len(candidates) == 1
    assert candidates[0].cloud_cover is None


def test_a_non_2xx_response_raises_provider_unavailable():
    def fake_post(url, json, timeout):
        return _FakeResponse(500, text="internal server error")

    with pytest.raises(ProviderUnavailableError):
        search_scenes(VALID_QUERY, http_post=fake_post)


def test_a_network_exception_raises_provider_unavailable():
    import requests

    def fake_post(url, json, timeout):
        raise requests.ConnectionError("no route to host")

    with pytest.raises(ProviderUnavailableError):
        search_scenes(VALID_QUERY, http_post=fake_post)
