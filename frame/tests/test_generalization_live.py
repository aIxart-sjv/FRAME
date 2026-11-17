"""Step H: live (real network, real Copernicus API) domain-shift experiment -- skipped without
credentials, same convention as test_acquisition_live.py / test_acquisition_harmonise.py.

This is the one test that proves `frame.generalization` runs against a REAL Indian Sentinel-2
scene end to end (real discovery, real retrieval, real harmonisation, real RGBN adapter, real
registered model, real consistency/stability diagnostics) rather than only against fixtures.
"""

from __future__ import annotations

import os

import pytest

_HAS_LIVE_CDSE_CREDENTIALS = bool(os.environ.get("FRAME_CDSE_CLIENT_ID")) and bool(os.environ.get("FRAME_CDSE_CLIENT_SECRET"))


@pytest.mark.live_cdse  # exempts this test from conftest.py's usual CDSE-credential stripping
@pytest.mark.skipif(not _HAS_LIVE_CDSE_CREDENTIALS, reason="FRAME_CDSE_CLIENT_ID/SECRET not set -- live domain-shift test skipped.")
def test_live_indian_domain_shift_experiment_runs_end_to_end():
    from frame.acquisition.contract import AcquisitionQuery
    from frame.data.contract import HRStatus
    from frame.generalization.runner import run_domain_shift_experiment

    query = AcquisitionQuery(
        bbox=(80.204, 13.006, 80.2168, 13.0188),  # the same real Chennai-area AOI already verified live (Step G)
        start_date="2024-01-01", end_date="2024-04-01", max_cloud_cover=20.0,
    )
    record = run_domain_shift_experiment(query, model_name="lite", device="cpu", seed=0)

    assert record.protocol.hr_reference_status is HRStatus.UNAVAILABLE
    assert record.protocol.training_domain_known is False
    assert len(record.excluded_bands) == 8
    assert record.output_shape[-1] == record.input_shape[-1] * 4
    assert record.self_consistency_overall["status"] == "COMPUTABLE"
    assert record.stability_scalar_summary >= 0.0
