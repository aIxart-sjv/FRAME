"""Unit tests for frame.acquisition.providers.cdse -- OAuth2 token caching and
Process API request construction, all against a fake `http_post` (no real
network, no real credentials required)."""

import io

import numpy as np
import pytest
import rasterio
from rasterio.transform import Affine

from frame.acquisition.errors import AuthenticationError, ProviderUnavailableError
from frame.acquisition.providers.cdse import CDSEAuth, retrieve_bands


class _FakeResponse:
    def __init__(self, status_code, *, json_body=None, content=b"", text=""):
        self.status_code = status_code
        self._json_body = json_body or {}
        self.content = content
        self.text = text

    def json(self):
        return self._json_body


def _fake_tiff_bytes(bands=4, height=8, width=8):
    buf = io.BytesIO()
    with rasterio.open(
        buf, "w", driver="GTiff", height=height, width=width, count=bands, dtype="float32",
        crs="EPSG:32644", transform=Affine(10.0, 0.0, 500000.0, 0.0, -10.0, 1400000.0),
    ) as dst:
        dst.write(np.ones((bands, height, width), dtype="float32"))
    return buf.getvalue()


def test_auth_is_not_configured_without_credentials(monkeypatch):
    monkeypatch.delenv("FRAME_CDSE_CLIENT_ID", raising=False)
    monkeypatch.delenv("FRAME_CDSE_CLIENT_SECRET", raising=False)
    auth = CDSEAuth()
    assert not auth.configured()
    with pytest.raises(ProviderUnavailableError):
        auth.get_token()


def test_auth_fetches_and_caches_a_real_shaped_token():
    calls = []

    def fake_post(url, data, timeout):
        calls.append(url)
        return _FakeResponse(200, json_body={"access_token": "fake-token-value", "expires_in": 1800})

    auth = CDSEAuth(client_id="id", client_secret="secret", token_url="https://example.test/token", http_post=fake_post)
    token1 = auth.get_token()
    token2 = auth.get_token()
    assert token1 == token2 == "fake-token-value"
    assert len(calls) == 1  # second call served from cache, not a second HTTP request


def test_auth_raises_authentication_error_on_401():
    def fake_post(url, data, timeout):
        return _FakeResponse(401, text="invalid client")

    auth = CDSEAuth(client_id="id", client_secret="secret", token_url="https://example.test/token", http_post=fake_post)
    with pytest.raises(AuthenticationError):
        auth.get_token()


def test_auth_raises_provider_unavailable_on_other_http_error():
    def fake_post(url, data, timeout):
        return _FakeResponse(500, text="internal server error")

    auth = CDSEAuth(client_id="id", client_secret="secret", token_url="https://example.test/token", http_post=fake_post)
    with pytest.raises(ProviderUnavailableError):
        auth.get_token()


def _auth_with_fake_token():
    def fake_post(url, data, timeout):
        return _FakeResponse(200, json_body={"access_token": "fake-token", "expires_in": 1800})

    return CDSEAuth(client_id="id", client_secret="secret", token_url="https://example.test/token", http_post=fake_post)


def test_retrieve_bands_returns_a_real_shaped_array_and_sends_bearer_auth():
    captured = {}

    def fake_post(url, json, headers, timeout):
        captured["url"] = url
        captured["json"] = json
        captured["headers"] = headers
        return _FakeResponse(200, content=_fake_tiff_bytes(bands=4, height=8, width=8))

    array, crs, transform = retrieve_bands(
        auth=_auth_with_fake_token(),
        utm_crs="EPSG:32644",
        utm_bbox=(500000.0, 1400000.0, 500080.0, 1400080.0),
        time_range=("2024-03-29T00:00:00Z", "2024-03-31T00:00:00Z"),
        bands=["B02", "B03", "B04", "B08"],
        resolution_m=10.0,
        endpoint="https://example.test/process",
        http_post=fake_post,
    )
    assert array.shape == (4, 8, 8)
    assert captured["headers"]["Authorization"] == "Bearer fake-token"
    assert captured["json"]["input"]["data"][0]["processing"]["upsampling"] == "BILINEAR"
    assert '"B02"' in captured["json"]["evalscript"]


def test_retrieve_bands_uses_nearest_resampling_for_categorical_scl():
    captured = {}

    def fake_post(url, json, headers, timeout):
        captured["json"] = json
        return _FakeResponse(200, content=_fake_tiff_bytes(bands=1, height=4, width=4))

    retrieve_bands(
        auth=_auth_with_fake_token(),
        utm_crs="EPSG:32644",
        utm_bbox=(500000.0, 1400000.0, 500080.0, 1400080.0),
        time_range=("2024-03-29T00:00:00Z", "2024-03-31T00:00:00Z"),
        bands=["SCL"],
        resolution_m=20.0,
        categorical=True,
        endpoint="https://example.test/process",
        http_post=fake_post,
    )
    data_config = captured["json"]["input"]["data"][0]
    assert data_config["processing"]["upsampling"] == "NEAREST"
    assert data_config["processing"]["downsampling"] == "NEAREST"
    assert '"DN"' in captured["json"]["evalscript"] or "DN" in captured["json"]["evalscript"]


def test_retrieve_bands_raises_provider_unavailable_on_http_error():
    def fake_post(url, json, headers, timeout):
        return _FakeResponse(400, text="bad request")

    with pytest.raises(ProviderUnavailableError):
        retrieve_bands(
            auth=_auth_with_fake_token(),
            utm_crs="EPSG:32644",
            utm_bbox=(500000.0, 1400000.0, 500080.0, 1400080.0),
            time_range=("2024-03-29T00:00:00Z", "2024-03-31T00:00:00Z"),
            bands=["B02"],
            resolution_m=10.0,
            endpoint="https://example.test/process",
            http_post=fake_post,
        )


def test_retrieve_bands_without_credentials_raises_before_any_request():
    calls = []

    def fake_post(*args, **kwargs):
        calls.append(1)
        return _FakeResponse(200)

    auth = CDSEAuth(client_id=None, client_secret=None, http_post=fake_post)
    with pytest.raises(ProviderUnavailableError):
        retrieve_bands(
            auth=auth,
            utm_crs="EPSG:32644",
            utm_bbox=(500000.0, 1400000.0, 500080.0, 1400080.0),
            time_range=("2024-03-29T00:00:00Z", "2024-03-31T00:00:00Z"),
            bands=["B02"],
            resolution_m=10.0,
            http_post=fake_post,
        )
    assert not calls
