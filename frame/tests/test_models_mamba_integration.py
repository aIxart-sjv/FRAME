"""INTEGRATION test: the REAL SEN2SR-Mamba RGBN model on a REAL GPU (Phase 1).

Excluded from a normal `pytest frame/tests/` run (`addopts = "-m 'not integration'"`
in pyproject.toml). Run explicitly with:

    sen2sr_venv/bin/python -m pytest frame/tests/test_models_mamba_integration.py -m integration -v

It launches the actual isolated worker in the dedicated Mamba environment,
loads the actual `sr_model.safetensor` + `sr_hard_constraint.safetensor`, and
feeds it the tile shipped in `models/SEN2SR/example_data.safetensor` (RGBN
slice). It is skipped -- never failed -- when the GPU, the Mamba environment
or the model files are absent, and it never downloads anything.

The unit tests in test_models_*.py cover everything that does not need the
real model; this module covers what only the real model can show.
"""

from __future__ import annotations

import hashlib
import json

import pytest
import torch
import torch.nn.functional as F

from frame.consistency.downsample import downsample_to_lr_grid
from frame.models import config
from frame.models.errors import ModelContractError
from frame.models.mamba_client import MambaWorkerClient, check_mamba_availability


EXAMPLE_DATA = config.MAMBA_WEIGHTS_DIR / config.MAMBA_EXAMPLE_DATA_FILENAME

_availability = check_mamba_availability("cuda")
_skip = None
if not _availability.available:
    _skip = f"SEN2SR-Mamba not runnable here: {_availability.message} ({_availability.reason_code})"
elif not EXAMPLE_DATA.is_file():
    _skip = f"missing {EXAMPLE_DATA}"

pytestmark = [pytest.mark.integration, pytest.mark.skipif(_skip is not None, reason=str(_skip))]

#: 10-band order of example_data.safetensor (see models/SEN2SR/mlm.json) -> RGBN in FRAME's order.
RGBN_FROM_TEN_BAND = [2, 1, 0, 6]  # B04, B03, B02, B08 -- the indices sen2sr/referencex4.py uses


@pytest.fixture(scope="module")
def client():
    with MambaWorkerClient(device="cuda") as started:
        yield started


@pytest.fixture(scope="module")
def tile() -> torch.Tensor:
    import safetensors.torch as st

    ten_band = st.load_file(str(EXAMPLE_DATA))["lr"]  # (1, 10, 128, 128) float32 reflectance
    return ten_band[:, RGBN_FROM_TEN_BAND].contiguous()


# --------------------------------------------------------------- model / load


def test_weights_load_with_no_missing_or_unexpected_keys(client):
    info = client.describe()
    assert info["missing_keys"] == [] and info["unexpected_keys"] == []


def test_the_shipped_weights_match_the_recorded_architecture_facts(client):
    info = client.describe()
    assert info["parameter_count"] == config.MAMBA_EXPECTED_PARAMETER_COUNT == 13_759_444
    assert info["state_tensors"] == config.MAMBA_EXPECTED_STATE_TENSORS == 1228
    # the report crosses a JSON wire, where tuples become lists
    assert info["architecture"] == json.loads(json.dumps(config.MAMBA_ARCHITECTURE.kwargs()))


def test_provenance_records_the_real_weights_hash(client):
    info = client.describe()
    weights = config.MAMBA_WEIGHTS_DIR / config.MAMBA_SR_WEIGHTS_FILENAME
    assert info["weights_sha256"] == hashlib.sha256(weights.read_bytes()).hexdigest()
    assert len(info["hard_constraint_sha256"]) == 64


def test_executable_architecture_and_artifact_label_are_both_recorded(client):
    info = client.describe()
    assert info["executable_architecture"] == "MambaSR"
    assert info["artifact_metadata_label"] == "Swin2SR"  # the known mlm.json inconsistency, kept visible


def test_the_worker_runs_in_the_isolated_environment_not_this_one(client):
    info = client.describe()
    assert info["isolated_worker"] is True
    assert info["worker_torch"] == "2.6.0+cu118"
    assert info["worker_torch"] != torch.__version__  # two different torch builds, side by side
    assert info["worker_cuda"] == "11.8"
    assert info["worker_gpu"] == torch.cuda.get_device_name(0)  # the worker is on the same physical GPU


# ------------------------------------------------------------------ inference


def test_real_inference_4x128x128_to_4x512x512_on_cuda(client, tile):
    y = client(tile[0].contiguous())  # unbatched (4, 128, 128) in ...
    assert y.shape == (4, 512, 512)  # ... unbatched (4, 512, 512) out
    y = client(tile)  # batched (1, 4, 128, 128)
    assert y.shape == (1, 4, 512, 512) and y.dtype == torch.float32
    assert client.describe()["last_request"]["peak_memory_mib"] > 0  # measured on the GPU, not a CPU fallback


def test_output_is_finite_and_in_a_plausible_reflectance_range(client, tile):
    y = client(tile)
    assert torch.isfinite(y).all()
    assert float(y.min()) > -0.05  # the constraint's clamp keeps reflectance ~non-negative
    assert float(y.max()) < 2.0


def test_output_is_deterministic(client, tile):
    assert torch.equal(client(tile), client(tile))


def test_hard_constraint_keeps_the_output_consistent_with_its_lr_input(client, tile):
    """SR area-averaged back to the 10 m grid must match the LR input. Measured
    overall RMSE on this tile: 0.0031 (bicubic: 0.0036); 0.01 is a generous sanity
    bound that a missing/mis-applied hard constraint would exceed."""
    y = client(tile)
    downsampled = downsample_to_lr_grid(y[0], config.SCALE_FACTOR)
    rmse = float(((downsampled - tile[0]) ** 2).mean().sqrt())
    assert rmse < 0.01


def test_sr_adds_detail_beyond_a_bicubic_upsample(client, tile):
    y = client(tile)
    bicubic = F.interpolate(tile, scale_factor=config.SCALE_FACTOR, mode="bicubic", antialias=True)
    assert float((y - bicubic).abs().mean()) > 1e-3  # not just a resample


def test_channel_order_is_meaningful_to_the_model(client, tile):
    """Swapping red and blue changes the network's output, so the documented
    B04,B03,B02,B08 order is a real contract, not a cosmetic one."""
    swapped = tile[:, [2, 1, 0, 3]].contiguous()
    difference = client(tile) - client(swapped)[:, [2, 1, 0, 3]]
    assert float(difference.abs().mean()) > 1e-4


def test_peak_gpu_memory_fits_a_4gb_card_with_a_wide_margin(client, tile):
    client(tile)
    assert client.describe()["last_request"]["peak_memory_mib"] < 2048


def test_a_batch_runs_tile_by_tile_without_growing_peak_memory(client, tile):
    client(tile)
    single = client.describe()["last_request"]["peak_memory_mib"]
    y = client(torch.cat([tile, tile.flip(-1)], dim=0))
    assert y.shape == (2, 4, 512, 512)
    assert client.describe()["last_request"]["peak_memory_mib"] < single * 1.25


# ------------------------------------------------------------- contract, real


def test_raw_digital_numbers_are_rejected_not_silently_processed(client, tile):
    with pytest.raises(ModelContractError, match="divided by 10000"):
        client(tile * 10000.0)


def test_wrong_channel_count_is_rejected(client, tile):
    with pytest.raises(ModelContractError, match="channels"):
        client(torch.rand(1, 10, 128, 128))


def test_worker_survives_a_rejected_request(client, tile):
    with pytest.raises(ModelContractError):
        client(tile * 10000.0)
    assert client(tile).shape == (1, 4, 512, 512)
