"""Real, live, network-touching acquisition test -- the actual end-to-end path
(AOI -> real CDSE STAC search -> real scene selection -> real multi-resolution
CDSE Process API retrieval -> harmonised 12-band-capable RetrievedScene), run
against the genuine Copernicus Data Space Ecosystem, not a fixture.

This is NOT part of the default CI-safe suite: it is skipped automatically
whenever FRAME_CDSE_CLIENT_ID/SECRET are not present in the environment (the
project's own .env is git-ignored and never assumed to exist in CI). Where
live credentials are unavailable, the fixture-backed tests in
test_acquisition_search.py / test_acquisition_cdse.py / test_acquisition_retrieve.py
are the ones that exercise this code's actual logic -- this file exists to
prove that logic also holds against the real API, without pretending the
fixture-backed tests are themselves live data.
"""

from __future__ import annotations

import os

import pytest

from frame.acquisition.contract import AcquisitionQuery
from frame.acquisition.errors import NoScenesFoundError
from frame.acquisition.providers.cdse import CDSEAuth
from frame.acquisition.retrieve import retrieve_scene
from frame.acquisition.search import search_scenes

_HAS_LIVE_CDSE_CREDENTIALS = bool(os.environ.get("FRAME_CDSE_CLIENT_ID")) and bool(
    os.environ.get("FRAME_CDSE_CLIENT_SECRET")
)

pytestmark = [
    pytest.mark.live_cdse,  # exempts these tests from conftest.py's usual CDSE-credential stripping
    pytest.mark.skipif(
        not _HAS_LIVE_CDSE_CREDENTIALS,
        reason="FRAME_CDSE_CLIENT_ID/SECRET not set -- live CDSE test skipped (see fixture-backed tests instead).",
    ),
]

# A small AOI over Chennai, India, chosen only because it was already used and
# verified (real scenes, low cloud cover) during this project's acquisition audit.
_LIVE_QUERY = AcquisitionQuery(
    bbox=(80.204, 13.006, 80.207, 13.009),  # ~330 m x 330 m -- deliberately tiny to keep this call cheap
    start_date="2024-01-01",
    end_date="2024-04-01",
    max_cloud_cover=20.0,
)


def test_live_search_returns_a_real_scene():
    candidates = search_scenes(_LIVE_QUERY)
    assert len(candidates) >= 1
    candidate = candidates[0]
    assert candidate.id
    assert candidate.provider == "copernicus-cdse"
    assert candidate.cloud_cover is None or candidate.cloud_cover <= 20.0


def test_live_retrieval_harmonises_all_three_resolution_groups_plus_scl():
    try:
        candidates = search_scenes(_LIVE_QUERY)
    except NoScenesFoundError:
        pytest.skip("No real scenes matched this AOI/date range at search time.")
    candidate = candidates[0]

    scene = retrieve_scene(
        candidate,
        _LIVE_QUERY,
        bands=["B02", "B03", "B04", "B08", "B11", "B01"],
        resolution_m=10.0,
        include_scl=True,
        cdse_auth=CDSEAuth(),
    )

    assert scene.width > 0 and scene.height > 0
    for band in ["B02", "B03", "B04", "B08", "B11", "B01"]:
        assert scene.bands[band].shape == (scene.height, scene.width)
        assert scene.bands[band].dtype.kind == "f"
    assert scene.band_native_resolution_m["B01"] == 60.0
    assert scene.band_native_resolution_m["B11"] == 20.0
    assert scene.band_native_resolution_m["B02"] == 10.0
    assert scene.scl is not None and scene.scl.shape == (scene.height, scene.width)
    # real reflectance is a small positive fraction, not a placeholder/fabricated value
    b04 = scene.bands["B04"]
    assert 0.0 <= float(b04.mean()) < 1.0
    assert scene.provenance.retrieval_provider  # non-empty: some real provider actually served this
