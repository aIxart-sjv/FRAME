"""Step G: the real model-capability registry (`GET /models`, Requirements 142 Requirement 4).

`frame.models.selection.MODEL_SPECS` is the single source of truth `GET /health`'s
`available_models` and `GET /models` both read -- these tests check the richer view's real
content, and that it never claims more than the verified RGBN, 4x, 10 m -> 2.5 m contract.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from frame.api.app import create_app
from frame.api.services import model as model_service
from frame.data.bands import RGBN_BANDS
from frame.models import config as models_cfg
from frame.models.selection import CAPABILITY_ONLY_SPECS, MODEL_SPECS, TTA_STABILITY_DESCRIPTION

ALL_SPECS = {**MODEL_SPECS, **CAPABILITY_ONLY_SPECS}


def test_every_registered_model_has_the_verified_rgbn_4x_contract():
    for model_id, spec in ALL_SPECS.items():
        assert spec.input_bands == RGBN_BANDS, model_id
        assert spec.output_bands == RGBN_BANDS, model_id
        assert spec.input_resolution_m == 10.0, model_id
        assert spec.output_resolution_m == 2.5, model_id
        assert spec.scale_factor == 4, model_id


def test_no_registered_model_claims_a_12_band_or_finer_than_2_5m_contract():
    """Scientific-honesty regression: whatever gets added to this registry later, it must never
    silently claim more than a verified model/checkpoint actually does."""
    for spec in ALL_SPECS.values():
        assert len(spec.input_bands) == 4
        assert len(spec.output_bands) == 4
        assert spec.output_resolution_m is None or spec.output_resolution_m >= 2.5


def test_uncertainty_is_never_described_as_calibrated():
    for spec in ALL_SPECS.values():
        if spec.uncertainty:
            assert spec.uncertainty_method is not None
            assert "calibrat" not in spec.uncertainty_method.lower() or "uncalibrated" in spec.uncertainty_method.lower()


def test_capability_only_models_cannot_be_selected_for_sr_run():
    """LDSR-S2 is real and verified, but must never be reachable through the `/sr/run` selection
    boundary -- only through the honest, always-unavailable `GET /models` capability listing."""
    from frame.models.errors import UnknownModelError
    from frame.models.selection import normalize_model_name

    assert "ldsrs2" not in MODEL_SPECS
    assert "ldsrs2" in CAPABILITY_ONLY_SPECS
    try:
        normalize_model_name("ldsrs2")
        assert False, "ldsrs2 must not resolve as a selectable /sr/run model"
    except UnknownModelError:
        pass


def test_capability_only_models_are_always_reported_unavailable_with_a_reason():
    for spec in CAPABILITY_ONLY_SPECS.values():
        assert spec.always_unavailable_reason
        assert len(spec.always_unavailable_reason) > 20  # a real explanation, not a placeholder


def test_checkpoint_sha256_reads_the_real_cached_weight_file_when_present():
    """Step I: frame.api.services.product needs a real checkpoint hash for STAC metadata.
    Whatever this returns must be independently computable from the file on disk, not copied
    from docs/REGISTRY.md -- so this test recomputes it itself rather than hard-coding the value."""
    import hashlib

    lite_hash = model_service.checkpoint_sha256("lite")
    weight_path = model_service.config.MODEL_WEIGHTS_CACHE_DIR / "model.safetensor"
    if not weight_path.exists():
        pytest.skip("Lite weights not cached on this machine")
    expected = hashlib.sha256(weight_path.read_bytes()).hexdigest()
    assert lite_hash == expected


def test_checkpoint_sha256_returns_none_for_an_unknown_model():
    assert model_service.checkpoint_sha256("not-a-model") is None


def test_lite_and_mamba_share_the_identical_tta_stability_wording():
    # Both are deterministic RGBN CNNs sharing the exact same TTA ensemble (frame.uncertainty) --
    # they must report the identical honest wording, not two independently-drifted strings.
    assert MODEL_SPECS[models_cfg.MODEL_LITE].uncertainty_method == TTA_STABILITY_DESCRIPTION
    assert MODEL_SPECS[models_cfg.MODEL_MAMBA].uncertainty_method == TTA_STABILITY_DESCRIPTION


def test_list_model_capabilities_matches_the_specs(monkeypatch):
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)  # deterministic: no GPU assumed
    entries = model_service.list_model_capabilities(device="cpu")
    by_id = {e["id"]: e for e in entries}

    assert by_id["lite"]["available"] is True  # Lite has no GPU requirement
    assert by_id["mamba"]["available"] is False  # Mamba requires CUDA
    assert by_id["mamba"]["reason"] is not None
    assert by_id["lite"]["input_bands"] == list(RGBN_BANDS)
    assert by_id["lite"]["scale_factor"] == 4
    assert by_id["ldsrs2"]["available"] is False  # verified but not deployable here (Step G Section 7)
    assert by_id["ldsrs2"]["reason"]


def test_http_get_models_returns_the_real_registry(monkeypatch):
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    client = TestClient(create_app())

    response = client.get("/models")

    assert response.status_code == 200
    body = response.json()
    assert {entry["id"] for entry in body} == {"lite", "mamba", "ldsrs2"}
    lite = next(e for e in body if e["id"] == "lite")
    assert lite["input_bands"] == list(RGBN_BANDS)
    assert lite["output_bands"] == list(RGBN_BANDS)
    assert lite["scale_factor"] == 4
    assert lite["uncertainty"] is True
    assert "uncalibrated" in lite["uncertainty_method"].lower()

    ldsrs2 = next(e for e in body if e["id"] == "ldsrs2")
    assert ldsrs2["available"] is False
    assert ldsrs2["scale_factor"] == 4  # verified by a real forward pass, not asserted
    assert "calibrat" not in ldsrs2["uncertainty_method"].lower() or "uncalibrated" in ldsrs2["uncertainty_method"].lower()


def test_get_models_and_get_health_report_the_same_availability(monkeypatch):
    """The two endpoints must never disagree about whether a model can actually run."""
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    client = TestClient(create_app())

    models_body = {e["id"]: e["available"] for e in client.get("/models").json()}
    health_body = {e["id"]: e["available"] for e in client.get("/health").json()["available_models"]}

    assert models_body == health_body
