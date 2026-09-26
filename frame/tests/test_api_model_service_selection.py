"""frame.api.services.model -- selecting between the Lite and Mamba models (Phase 1).

Complements test_api_model_service.py (which covers the Lite caching that
predates model selection and must keep passing unchanged). All loaders are
injected fakes: no network, no weights, no GPU, no worker process.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from frame.api.services import model as model_service
from frame.models.errors import ModelUnavailableError, UnknownModelError


def setup_function() -> None:
    model_service.clear_cache()


def teardown_function() -> None:
    model_service.clear_cache()


class Recorder:
    """Fake loaders that record which path was taken."""

    def __init__(self):
        self.calls = []

    def ensure_weights(self, cache_dir: Path) -> Path:
        self.calls.append("ensure_weights")
        return cache_dir

    def load_compiled_model(self, weight_dir: Path, device: str):
        self.calls.append(f"lite:{device}")
        return f"lite-model-{device}"

    def load_mamba(self, device: str):
        self.calls.append(f"mamba:{device}")
        return FakeMambaClient(device)

    def kwargs(self):
        return dict(
            ensure_weights=self.ensure_weights, load_compiled_model=self.load_compiled_model, load_mamba=self.load_mamba
        )


class FakeMambaClient:
    def __init__(self, device):
        self.device = device
        self.closed = False

    def close(self):
        self.closed = True


def test_default_selection_is_lite_and_never_touches_the_mamba_loader():
    rec = Recorder()
    assert model_service.get_model("cpu", **rec.kwargs()) == "lite-model-cpu"
    assert rec.calls == ["ensure_weights", "lite:cpu"]


def test_explicit_lite_matches_the_default():
    rec = Recorder()
    assert model_service.get_model("cpu", model_name="lite", **rec.kwargs()) == "lite-model-cpu"


def test_mamba_uses_the_mamba_loader_and_does_not_load_the_lite_weights():
    rec = Recorder()
    client = model_service.get_model("cuda", model_name="mamba", **rec.kwargs())
    assert isinstance(client, FakeMambaClient) and client.device == "cuda"
    assert rec.calls == ["mamba:cuda"]  # no ensure_weights, no lite load


def test_selection_is_case_insensitive():
    rec = Recorder()
    assert isinstance(model_service.get_model("cuda", model_name=" MAMBA ", **rec.kwargs()), FakeMambaClient)


def test_models_are_cached_independently_per_model_and_device():
    rec = Recorder()
    lite = model_service.get_model("cuda", model_name="lite", **rec.kwargs())
    mamba = model_service.get_model("cuda", model_name="mamba", **rec.kwargs())
    assert lite != mamba
    assert model_service.get_model("cuda", model_name="lite", **rec.kwargs()) is lite
    assert model_service.get_model("cuda", model_name="mamba", **rec.kwargs()) is mamba
    assert rec.calls.count("mamba:cuda") == 1 and rec.calls.count("lite:cuda") == 1  # each loaded once


def test_invalid_model_selection_fails_clearly_before_loading_anything():
    rec = Recorder()
    with pytest.raises(UnknownModelError, match="supported models are: lite, mamba"):
        model_service.get_model("cpu", model_name="swin", **rec.kwargs())
    assert rec.calls == []


def test_a_failed_mamba_load_is_not_cached_so_the_next_request_retries():
    attempts = []

    def flaky(device):
        attempts.append(device)
        if len(attempts) == 1:
            raise ModelUnavailableError("SEN2SR-Mamba requires a CUDA-capable GPU.")
        return FakeMambaClient(device)

    rec = Recorder()
    with pytest.raises(ModelUnavailableError):
        model_service.get_model("cuda", model_name="mamba", ensure_weights=rec.ensure_weights,
                                load_compiled_model=rec.load_compiled_model, load_mamba=flaky)
    client = model_service.get_model("cuda", model_name="mamba", ensure_weights=rec.ensure_weights,
                                     load_compiled_model=rec.load_compiled_model, load_mamba=flaky)
    assert isinstance(client, FakeMambaClient) and len(attempts) == 2


def test_a_mamba_failure_does_not_affect_the_lite_path():
    def unavailable(device):
        raise ModelUnavailableError("nope")

    rec = Recorder()
    with pytest.raises(ModelUnavailableError):
        model_service.get_model("cuda", model_name="mamba", ensure_weights=rec.ensure_weights,
                                load_compiled_model=rec.load_compiled_model, load_mamba=unavailable)
    assert model_service.get_model("cuda", model_name="lite", **rec.kwargs()) == "lite-model-cuda"


def test_clear_cache_stops_the_mamba_worker_and_forces_a_reload():
    rec = Recorder()
    first = model_service.get_model("cuda", model_name="mamba", **rec.kwargs())
    model_service.clear_cache()
    assert first.closed is True
    second = model_service.get_model("cuda", model_name="mamba", **rec.kwargs())
    assert second is not first and rec.calls.count("mamba:cuda") == 2


def test_clear_cache_tolerates_models_without_a_close_method():
    rec = Recorder()
    model_service.get_model("cpu", **rec.kwargs())  # a plain string "model"
    model_service.clear_cache()  # must not raise


def test_default_mamba_loader_refuses_cleanly_when_there_is_no_gpu():
    with pytest.raises(ModelUnavailableError, match="CUDA"):
        model_service._default_load_mamba("cpu")


def test_list_models_reports_both_models_without_loading_anything():
    models = model_service.list_models("cpu")
    assert [m["id"] for m in models] == ["lite", "mamba"]
    lite, mamba = models
    assert lite["available"] is True and lite["reason"] is None
    assert mamba["available"] is False and "CUDA" in mamba["reason"]
    assert model_service._model_cache == {}  # listing never starts a worker
