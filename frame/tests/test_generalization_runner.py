"""Step H: frame.generalization -- the domain-shift protocol runner (Requirements 142
Requirement 12). Fixture-backed: no network. Mirrors test_acquisition_harmonise.py's
`_retrieved_scene`/`equivariant_model` pattern so this test exercises the exact same real
harmonise -> select_rgbn_adapter -> tiled-model -> consistency chain, only with real network
calls replaced by monkeypatched real-shaped fixtures.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch.nn.functional as F

from frame.acquisition.contract import AcquisitionQuery, RetrievalProvenance, RetrievedScene, SceneCandidate
from frame.acquisition.providers import cdse as cdse_provider
from frame.data.bands import L2A_BANDS
from frame.data.contract import HRStatus
from frame.generalization import runner as gen_runner
from frame.generalization.errors import NoIndianSceneFoundError

H = 16
CRS = "EPSG:32644"
TRANSFORM = (10.0, 0.0, 500000.0, 0.0, -10.0, 1400000.0)


def _candidate() -> SceneCandidate:
    return SceneCandidate(
        id="S2TEST_INDIA_20240330", provider="copernicus-cdse", stac_endpoint="https://stac.dataspace.copernicus.eu/v1/search",
        collection="sentinel-2-l2a", datetime="2024-03-30T04:56:59Z", cloud_cover=5.0,
        bbox=(80.0, 13.0, 80.1, 13.1), geometry={"type": "Polygon", "coordinates": []},
        mgrs_tile="MGRS-44PMV", platform="sentinel-2b", processing_baseline="05.10", asset_hrefs={},
    )


def _retrieved_scene() -> RetrievedScene:
    rng = np.random.default_rng(0)
    band_arrays = {b: (rng.random((H, H)).astype("float32") * 0.4 + 0.1) for b in L2A_BANDS}
    band_native_resolution_m = {
        b: (10.0 if b in ("B02", "B03", "B04", "B08") else 20.0 if b in ("B05", "B06", "B07", "B8A", "B11", "B12") else 60.0)
        for b in L2A_BANDS
    }
    provenance = RetrievalProvenance(
        query=None, candidate_id=_candidate().id, search_provider="copernicus-cdse",
        search_stac_endpoint="https://stac.dataspace.copernicus.eu/v1/search",
        retrieval_provider=cdse_provider.PROVIDER_NAME, retrieval_endpoint="https://sh.dataspace.copernicus.eu/process/v1",
        matched_acquisition_datetime=_candidate().datetime, retrieved_at_utc="2024-03-30T05:00:00Z",
        band_sources={b: cdse_provider.PROVIDER_NAME for b in L2A_BANDS}, warnings=(),
    )
    return RetrievedScene(
        candidate=_candidate(), bands=band_arrays, band_native_resolution_m=band_native_resolution_m, scl=None,
        crs=CRS, transform=TRANSFORM, resolution_m=10.0, width=H, height=H, provenance=provenance,
    )


def _equivariant_model(x_batched):
    return F.interpolate(x_batched, scale_factor=4, mode="bicubic", antialias=True)


def _query() -> AcquisitionQuery:
    return AcquisitionQuery(bbox=(80.0, 13.0, 80.1, 13.1), start_date="2024-01-01", end_date="2024-04-01")


def test_run_domain_shift_experiment_produces_a_real_record(monkeypatch):
    monkeypatch.setattr(gen_runner, "search_scenes", lambda query: [_candidate()])
    monkeypatch.setattr(gen_runner, "retrieve_scene", lambda *a, **k: _retrieved_scene())
    monkeypatch.setattr(gen_runner, "get_model", lambda device, *, model_name: _equivariant_model)
    monkeypatch.setattr(gen_runner, "resolve_device", lambda device: "cpu")

    record = gen_runner.run_domain_shift_experiment(_query(), model_name="lite", device="cpu", seed=0)

    assert record.scene_id == "S2TEST_INDIA_20240330"
    assert record.provider == cdse_provider.PROVIDER_NAME
    assert record.protocol.hr_reference_status is HRStatus.UNAVAILABLE
    assert record.protocol.training_domain_known is False
    assert record.protocol.model_id == "lite"
    assert len(record.excluded_bands) == 8  # 12 L2A bands - 4 RGBN
    assert record.self_consistency_overall["status"] == "COMPUTABLE"
    assert record.ndvi_self_consistency["status"] == "COMPUTABLE"
    assert record.ndwi_self_consistency["status"] == "COMPUTABLE"
    assert record.stability_scalar_summary >= 0.0
    assert record.output_shape[-1] == record.input_shape[-1] * 4
    assert record.device == "cpu"


def test_no_candidates_raises_a_named_error_not_a_fallback(monkeypatch):
    monkeypatch.setattr(gen_runner, "search_scenes", lambda query: [])
    with pytest.raises(NoIndianSceneFoundError):
        gen_runner.run_domain_shift_experiment(_query(), model_name="lite")


def test_unknown_model_is_rejected_before_any_network_call(monkeypatch):
    from frame.models.errors import UnknownModelError

    called = {"search": False}

    def _fail_if_called(query):
        called["search"] = True
        return [_candidate()]

    monkeypatch.setattr(gen_runner, "search_scenes", _fail_if_called)
    with pytest.raises(UnknownModelError):
        gen_runner.run_domain_shift_experiment(_query(), model_name="not-a-real-model")
    assert called["search"] is False
