"""Shared pytest fixtures for frame/tests."""

import pytest

_CDSE_ENV_VARS = (
    "FRAME_CDSE_CLIENT_ID",
    "FRAME_CDSE_CLIENT_SECRET",
    "FRAME_CDSE_STAC_URL",
    "FRAME_CDSE_PROCESS_URL",
    "FRAME_CDSE_TOKEN_URL",
)


@pytest.fixture(autouse=True)
def _no_real_cdse_credentials_by_default(request, monkeypatch):
    """Every test not marked `live_cdse` runs with the real FRAME_CDSE_*
    environment variables removed, even though the developer's own .env may
    be present (frame.api.config loads it at import time, into the whole
    pytest process, not just one test module).

    Without this, a fixture-backed acquisition/API test that forgets its own
    `monkeypatch.delenv` could silently pass or fail differently depending on
    whether the real .env happens to be loadable in the environment running
    the suite -- exactly the kind of environment-dependent, non-deterministic
    test behaviour this project's own testing philosophy rejects. Worse, a
    test could silently and unintentionally fall back to the secondary
    Planetary Computer provider instead of genuinely exercising CDSE, which
    is exactly what happened here once (see git history) before this
    exemption was switched from a hardcoded module-name check to a real
    pytest marker (pyproject.toml's `live_cdse`) that every live-network test
    must apply for itself -- no test is silently exempted just by its name.
    """
    if request.node.get_closest_marker("live_cdse") is not None:
        return
    for var in _CDSE_ENV_VARS:
        monkeypatch.delenv(var, raising=False)
