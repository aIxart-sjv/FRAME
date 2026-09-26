"""frame.models.mamba_adapter -- the parts that do not need the real model.

The module is importable in the main environment (heavy imports are inside
`MambaRGBNModel.load`), so its contract handling can be unit-tested with a
fake module. Loading the real weights is covered by the GPU integration test.
"""

from __future__ import annotations

import json
import sys

import pytest
import torch
import torch.nn.functional as F

from frame.models import config
from frame.models.errors import ModelContractError, ModelInferenceError, ModelUnavailableError
from frame.models.mamba_adapter import MambaLoadReport, MambaRGBNModel, sha256_of


def _report(**overrides) -> MambaLoadReport:
    base = dict(
        model_name=config.MAMBA_MODEL_NAME,
        executable_architecture="MambaSR",
        artifact_metadata_label="Swin2SR",
        architecture=config.MAMBA_ARCHITECTURE.kwargs(),
        parameter_count=1,
        state_tensors=1,
        missing_keys=[],
        unexpected_keys=[],
        weights_file="sr_model.safetensor",
        weights_sha256="0" * 64,
        hard_constraint_file="sr_hard_constraint.safetensor",
        hard_constraint_sha256="1" * 64,
        device="cpu",
        load_seconds=0.0,
    )
    base.update(overrides)
    return MambaLoadReport(**base)


class RecordingModule:
    """Stands in for the wrapped MambaSR+HardConstraint module."""

    def __init__(self, output_fn=None):
        self.seen_shapes = []
        self.output_fn = output_fn or (lambda x: F.interpolate(x, scale_factor=4, mode="nearest"))

    def __call__(self, x):
        self.seen_shapes.append(tuple(x.shape))
        return self.output_fn(x)


def _model(module) -> MambaRGBNModel:
    return MambaRGBNModel(module, "cpu", _report())


def _x(*lead) -> torch.Tensor:
    return torch.rand(*lead, 4, 128, 128, dtype=torch.float32) * 0.5


# ------------------------------------------------------------ inference logic


def test_unbatched_input_gives_unbatched_output():
    y = _model(RecordingModule())(_x())
    assert y.shape == (4, 512, 512) and y.dtype == torch.float32


def test_batched_input_gives_batched_output():
    y = _model(RecordingModule())(_x(1))
    assert y.shape == (1, 4, 512, 512)


def test_tiles_of_a_batch_run_one_at_a_time_to_bound_gpu_memory():
    module = RecordingModule()
    y = _model(module)(_x(3))
    assert y.shape == (3, 4, 512, 512)
    assert module.seen_shapes == [(1, 4, 128, 128)] * 3


def test_output_is_deterministic_for_a_deterministic_module():
    model = _model(RecordingModule())
    x = _x(1)
    assert torch.equal(model(x), model(x))


def test_channel_order_is_not_altered_by_the_adapter():
    """No hidden reordering: what the module receives is exactly what was passed in."""
    seen = {}

    def capture(x):
        seen["x"] = x.clone()
        return F.interpolate(x, scale_factor=4, mode="nearest")

    x = _x(1)
    _model(RecordingModule(capture))(x)
    assert torch.equal(seen["x"], x)


@pytest.mark.parametrize(
    "bad",
    [torch.rand(1, 3, 128, 128), torch.rand(1, 4, 64, 64), torch.rand(1, 4, 128, 128).double(), torch.rand(1, 4, 128, 128) * 5000],
)
def test_contract_violations_are_rejected_before_the_module_runs(bad):
    module = RecordingModule()
    with pytest.raises(ModelContractError):
        _model(module)(bad)
    assert module.seen_shapes == []


def test_non_finite_module_output_is_reported_as_an_inference_error():
    def nan_output(x):
        y = F.interpolate(x, scale_factor=4, mode="nearest")
        y[0, 0, 0, 0] = float("nan")
        return y

    with pytest.raises(ModelInferenceError, match="invalid output"):
        _model(RecordingModule(nan_output))(_x(1))


def test_wrongly_shaped_module_output_is_reported_as_an_inference_error():
    with pytest.raises(ModelInferenceError, match="invalid output"):
        _model(RecordingModule(lambda x: F.interpolate(x, scale_factor=2, mode="nearest")))(_x(1))


# ------------------------------------------------------------------- loading


def test_loading_on_cpu_is_refused_rather_than_silently_falling_back():
    with pytest.raises(ModelUnavailableError, match="CUDA"):
        MambaRGBNModel.load(device="cpu")


def test_loading_with_missing_weights_is_reported_as_unavailable(tmp_path):
    with pytest.raises(ModelUnavailableError) as excinfo:
        MambaRGBNModel.load(tmp_path, device="cuda")  # no CUDA -> also "unavailable"; either way, same type
    assert "mamba_ssm" not in str(excinfo.value).lower()


def test_load_report_is_json_serialisable_and_carries_the_band_order():
    payload = json.loads(json.dumps(_report().as_dict()))
    assert payload["input_band_order"] == ["B04", "B03", "B02", "B08"]
    assert payload["executable_architecture"] == "MambaSR" and payload["artifact_metadata_label"] == "Swin2SR"


def test_sha256_of_matches_hashlib(tmp_path):
    import hashlib

    path = tmp_path / "blob.bin"
    path.write_bytes(b"frame" * 1000)
    assert sha256_of(path) == hashlib.sha256(b"frame" * 1000).hexdigest()


def test_adapter_module_is_importable_without_the_mamba_runtime():
    import subprocess

    code = "import sys, frame.models.mamba_adapter; print('mamba_ssm' in sys.modules or 'sen2sr' in sys.modules)"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "False"
