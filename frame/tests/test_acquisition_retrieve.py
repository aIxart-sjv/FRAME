"""Unit tests for frame.acquisition.retrieve -- real orchestration logic, tested
against fakes for both providers (frame.acquisition.providers.cdse.retrieve_bands
and frame.acquisition.providers.planetary_computer.retrieve_bands) so these never
require real network access or credentials."""

from __future__ import annotations

import numpy as np
import pytest

from frame.acquisition import geometry, retrieve
from frame.acquisition.contract import AcquisitionQuery, SceneCandidate
from frame.acquisition.errors import ProviderUnavailableError
from frame.acquisition.providers import cdse as cdse_provider
from frame.acquisition.providers import planetary_computer

FIXED_UTM_CRS = "EPSG:32644"
FIXED_UTM_BBOX = (500000.0, 1400000.0, 501000.0, 1401000.0)  # a real-shaped 1000 m x 1000 m box

QUERY = AcquisitionQuery(bbox=(80.20, 13.00, 80.21, 13.01), start_date="2024-01-01", end_date="2024-04-01")
CANDIDATE = SceneCandidate(
    id="S2B_MSIL2A_20240330T045659_N0510_R119_T44PMV_20240330T072508",
    provider="copernicus-cdse",
    stac_endpoint="https://stac.dataspace.copernicus.eu/v1/search",
    collection="sentinel-2-l2a",
    datetime="2024-03-30T04:56:59.024000Z",
    cloud_cover=2.43,
    bbox=(80.075, 12.574, 81.090, 13.568),
    geometry=None,
    mgrs_tile="44PMV",
    platform="sentinel-2b",
    processing_baseline="05.10",
    asset_hrefs={},
)


class _ConfiguredAuth:
    def configured(self):
        return True


class _UnconfiguredAuth:
    def configured(self):
        return False

    def get_token(self):
        raise AssertionError("get_token should never be called when unconfigured")


def _native_shape(resolution_m: float) -> tuple[int, int]:
    min_x, min_y, max_x, max_y = FIXED_UTM_BBOX
    return max(1, round((max_y - min_y) / resolution_m)), max(1, round((max_x - min_x) / resolution_m))


@pytest.fixture(autouse=True)
def _fixed_geometry(monkeypatch):
    monkeypatch.setattr(geometry, "bbox_to_utm", lambda bbox: (FIXED_UTM_CRS, FIXED_UTM_BBOX))


def test_retrieve_scene_via_cdse_harmonises_all_resolution_groups(monkeypatch):
    def fake_cdse_retrieve_bands(*, bands, resolution_m, categorical, **kwargs):
        height, width = _native_shape(resolution_m)
        fill = 99 if categorical else 0.2
        return np.full((len(bands), height, width), fill, dtype="float32"), FIXED_UTM_CRS, (10.0, 0, 0, 0, -10.0, 0)

    monkeypatch.setattr(cdse_provider, "retrieve_bands", fake_cdse_retrieve_bands)

    scene = retrieve.retrieve_scene(
        CANDIDATE, QUERY,
        bands=["B02", "B03", "B04", "B08", "B05", "B11", "B01", "B09"],
        resolution_m=10.0,
        include_scl=True,
        cdse_auth=_ConfiguredAuth(),
    )

    target_h, target_w = _native_shape(10.0)
    assert scene.width == target_w and scene.height == target_h
    for band in ["B02", "B03", "B04", "B08", "B05", "B11", "B01", "B09"]:
        assert scene.bands[band].shape == (target_h, target_w)
    # native resolution is recorded per band, not overwritten by the harmonised grid
    assert scene.band_native_resolution_m["B01"] == 60.0
    assert scene.band_native_resolution_m["B02"] == 10.0
    assert scene.scl is not None and scene.scl.shape == (target_h, target_w)
    assert all(source == cdse_provider.PROVIDER_NAME for source in scene.provenance.band_sources.values())
    # 20 m and 60 m groups were genuinely resampled -- this must be disclosed, not hidden
    assert any("60.0 m" in w for w in scene.provenance.warnings)
    assert any("20.0 m" in w for w in scene.provenance.warnings)


def test_a_target_coarser_than_every_native_band_is_fetched_directly_not_at_full_native_size(monkeypatch):
    """Regression: live testing found that a large AOI at a coarse target resolution (e.g. 100 m)
    still triggered a real Process API request at each band's full NATIVE resolution (10/20/60 m) --
    for an 80 km AOI that is a ~7900x7900-pixel real request regardless of the requested output grid.
    Fetching must never ask a real provider for more detail than the target grid can use."""
    captured_resolutions = []

    def fake_cdse_retrieve_bands(*, bands, resolution_m, categorical, **kwargs):
        captured_resolutions.append(resolution_m)
        height, width = _native_shape(resolution_m)
        return np.zeros((len(bands), height, width), dtype="float32"), FIXED_UTM_CRS, (10.0, 0, 0, 0, -10.0, 0)

    monkeypatch.setattr(cdse_provider, "retrieve_bands", fake_cdse_retrieve_bands)

    scene = retrieve.retrieve_scene(
        CANDIDATE, QUERY,
        bands=["B02", "B08", "B05", "B01"],  # 10 m, 10 m, 20 m, 60 m natives
        resolution_m=100.0,  # coarser than every native resolution above
        include_scl=True,
        cdse_auth=_ConfiguredAuth(),
    )

    # Every real fetch happened at the 100 m target, never at a band's finer native resolution.
    assert captured_resolutions == [100.0, 100.0, 100.0, 100.0]  # 3 native-resolution groups (10/20/60 m) + SCL
    target_h, target_w = _native_shape(100.0)
    assert scene.width == target_w and scene.height == target_h
    for band in ["B02", "B08", "B05", "B01"]:
        assert scene.bands[band].shape == (target_h, target_w)
    # native resolution provenance is still honest even though the fetch itself was coarser
    assert scene.band_native_resolution_m["B01"] == 60.0
    assert scene.band_native_resolution_m["B02"] == 10.0
    # no resampling happened (nothing to resample -- already fetched at the target grid), but the
    # coarser-than-native fetch is still disclosed
    assert not any("resampled from native" in w for w in scene.provenance.warnings)
    assert any("requested directly at the coarser" in w for w in scene.provenance.warnings)


def test_a_10m_only_request_needs_no_resampling_warning(monkeypatch):
    def fake_cdse_retrieve_bands(*, bands, resolution_m, categorical, **kwargs):
        height, width = _native_shape(resolution_m)
        return np.zeros((len(bands), height, width), dtype="float32"), FIXED_UTM_CRS, (10.0, 0, 0, 0, -10.0, 0)

    monkeypatch.setattr(cdse_provider, "retrieve_bands", fake_cdse_retrieve_bands)

    scene = retrieve.retrieve_scene(
        CANDIDATE, QUERY, bands=["B02", "B03", "B04", "B08"], resolution_m=10.0,
        include_scl=False, cdse_auth=_ConfiguredAuth(),
    )
    assert scene.scl is None
    assert scene.provenance.warnings == ()


def test_falls_back_to_planetary_computer_when_cdse_is_unconfigured(monkeypatch):
    calls = []

    class _FakeDataArray:
        def __init__(self, n_bands, height, width):
            self._array = np.full((3, n_bands, height, width), 0.3, dtype="float32")
            self.sizes = {"time": 3}
            self._time = np.array(
                ["2024-03-29T12:00:00", "2024-03-30T04:56:59", "2024-03-31T12:00:00"], dtype="datetime64[s]"
            )

        def __getitem__(self, key):
            if key == "time":
                return type("T", (), {"values": self._time})()
            raise KeyError(key)

        def isel(self, time):
            return type("Sel", (), {"values": self._array[time]})()

    def fake_pc_retrieve_bands(*, bands, resolution_m, **kwargs):
        height, width = _native_shape(resolution_m)
        calls.append((tuple(bands), resolution_m))
        return _FakeDataArray(len(bands), height, width)

    monkeypatch.setattr(planetary_computer, "retrieve_bands", fake_pc_retrieve_bands)

    scene = retrieve.retrieve_scene(
        CANDIDATE, QUERY, bands=["B02", "B03", "B04", "B08"], resolution_m=10.0,
        include_scl=False, cdse_auth=_UnconfiguredAuth(),
    )
    assert calls  # the fallback provider was genuinely invoked
    assert all(source == planetary_computer.PROVIDER_NAME for source in scene.provenance.band_sources.values())
    assert planetary_computer.PROVIDER_NAME in scene.provenance.retrieval_provider


def test_raises_provider_unavailable_when_both_providers_fail(monkeypatch):
    def failing_pc_retrieve_bands(**kwargs):
        raise ProviderUnavailableError("no scenes", provider=planetary_computer.PROVIDER_NAME)

    monkeypatch.setattr(planetary_computer, "retrieve_bands", failing_pc_retrieve_bands)

    with pytest.raises(ProviderUnavailableError):
        retrieve.retrieve_scene(
            CANDIDATE, QUERY, bands=["B02"], resolution_m=10.0,
            include_scl=False, cdse_auth=_UnconfiguredAuth(),
        )


def test_falls_back_per_group_when_cdse_fails_for_one_call(monkeypatch):
    def failing_cdse_retrieve_bands(**kwargs):
        raise ProviderUnavailableError("token expired", provider=cdse_provider.PROVIDER_NAME)

    class _FakeDataArray:
        def __init__(self, n_bands, height, width):
            self._array = np.zeros((1, n_bands, height, width), dtype="float32")
            self.sizes = {"time": 1}
            self._time = np.array(["2024-03-30T04:56:59"], dtype="datetime64[s]")

        def __getitem__(self, key):
            if key == "time":
                return type("T", (), {"values": self._time})()
            raise KeyError(key)

        def isel(self, time):
            return type("Sel", (), {"values": self._array[time]})()

    monkeypatch.setattr(cdse_provider, "retrieve_bands", failing_cdse_retrieve_bands)
    monkeypatch.setattr(
        planetary_computer, "retrieve_bands",
        lambda *, bands, resolution_m, **kwargs: _FakeDataArray(len(bands), *_native_shape(resolution_m)),
    )

    scene = retrieve.retrieve_scene(
        CANDIDATE, QUERY, bands=["B02", "B03", "B04", "B08"], resolution_m=10.0,
        include_scl=False, cdse_auth=_ConfiguredAuth(),
    )
    assert all(source == planetary_computer.PROVIDER_NAME for source in scene.provenance.band_sources.values())
