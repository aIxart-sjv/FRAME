"""INTEGRATION test: the tile engine on a multi-tile scene with the REAL SEN2SR-Mamba model (Phase 2).

Excluded from a normal run (`addopts = "-m 'not integration'"`); execute with:

    sen2sr_venv/bin/python -m pytest frame/tests/test_tiling_mamba_integration.py -m integration -v

It launches the isolated worker in the Mamba environment with the real weights and runs
a rectangular scene that needs several tiles, overlap blending and padded edge tiles. The
scene is real Sentinel-2 reflectance -- the deterministic Baseline 0 tile
(experiments/baseline/outputs/input_tensor.pt) -- mirror-extended to 200 x 300 with the
engine's own reflect indexing. That keeps the test offline and reproducible; it exercises
the tiling machinery, and is not a claim about the SR quality of a natural 200 x 300 scene.

Skipped -- never failed, never downloading -- when the GPU, the Mamba environment or the
inputs are absent.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from frame.consistency.downsample import downsample_to_lr_grid
from frame.consistency.status import ComputationStatus
from frame.models import config as models_cfg
from frame.models.mamba_client import MambaWorkerClient, check_mamba_availability
from frame.tiling import TiledModel, TilingConfig, plan_tiles, run_tiled
from frame.tiling.padding import reflect_indices
from frame.uncertainty import DEFAULT_TRANSFORMS, run_stochastic_uncertainty

BASELINE_TENSOR = Path(__file__).resolve().parents[2] / "experiments" / "baseline" / "outputs" / "input_tensor.pt"

_availability = check_mamba_availability("cuda")
_skip = None
if not _availability.available:
    _skip = f"SEN2SR-Mamba not runnable here: {_availability.message} ({_availability.reason_code})"
elif not BASELINE_TENSOR.is_file():
    _skip = f"missing {BASELINE_TENSOR}"

pytestmark = [pytest.mark.integration, pytest.mark.skipif(_skip is not None, reason=str(_skip))]

HEIGHT, WIDTH = 200, 300  # 2 x 3 tiles at overlap 32; both last tiles are padded (104 and 108 valid px)


def mirrored_real_scene(height: int, width: int) -> torch.Tensor:
    base = torch.load(BASELINE_TENSOR, weights_only=True)  # (4, 128, 128) real reflectance, B04,B03,B02,B08
    return base[:, reflect_indices(128, height)][:, :, reflect_indices(128, width)].contiguous()


@pytest.fixture(scope="module")
def client():
    with MambaWorkerClient(device="cuda") as started:
        yield started


@pytest.fixture(scope="module")
def scene() -> torch.Tensor:
    return mirrored_real_scene(HEIGHT, WIDTH)


@pytest.fixture(scope="module")
def first_run(client, scene):
    pids = []
    result = run_tiled(client, scene, TilingConfig(), on_tile=lambda done, total, tile: pids.append(client._proc.pid))
    return result, pids


def test_the_scene_really_is_multi_tile_rectangular_and_padded(scene):
    plan = plan_tiles(HEIGHT, WIDTH)
    assert scene.shape == (4, HEIGHT, WIDTH) and HEIGHT != WIDTH
    assert plan.tile_count == 6 and (plan.n_rows, plan.n_cols) == (2, 3)
    assert sum(t.is_padded for t in plan.tiles) == 4  # all 3 tiles of the last row, plus the top tile of the last column


def test_real_mamba_reconstructs_the_scene_at_4x(first_run):
    result, _ = first_run
    assert result.sr.shape == (4, HEIGHT * 4, WIDTH * 4)
    assert result.sr.dtype == torch.float32 and torch.isfinite(result.sr).all()
    assert 0.0 < float(result.sr.min()) and float(result.sr.max()) < 2.0
    assert len(result.tile_seconds) == 6


def test_one_worker_process_served_every_tile(first_run, client):
    _, pids = first_run
    assert len(pids) == 6 and len(set(pids)) == 1  # never respawned between tiles
    assert client.is_running


def test_the_reconstruction_is_bit_identical_across_runs(client, scene, first_run):
    """The real model is deterministic (Phase 1: bit-identical repeat calls); the tiler adds
    only fixed-order float32 arithmetic, so the tolerance is exactly zero."""
    again = run_tiled(client, scene, TilingConfig())
    assert torch.equal(first_run[0].sr, again.sr)


def test_a_one_tile_scene_equals_calling_the_model_directly(client):
    tile = mirrored_real_scene(128, 128)
    assert torch.equal(run_tiled(client, tile).sr, client(tile[None])[0])


def test_the_reconstruction_stays_consistent_with_its_input_at_10m(first_run, scene):
    """Each tile's hard constraint pins its low frequencies to the input, so the blended
    raster, area-averaged back to 10 m, must still match the input (single-tile RMSE in
    Phase 1: ~0.003)."""
    down = downsample_to_lr_grid(first_run[0].sr, 4)
    rmse = float(((down - scene) ** 2).mean().sqrt())
    assert rmse < 0.01


def test_the_seam_diagnostic_is_computed_on_the_real_overlaps(first_run):
    seam = first_run[0].seam_diagnostic
    assert seam.status == ComputationStatus.COMPUTABLE.value
    assert seam.pair_count == 7  # 2 x 3 grid: 4 horizontal + 3 vertical neighbour pairs
    assert seam.overlap_pixel_count > 0 and 0.0 < seam.rmse < 0.05 and seam.max_abs_difference >= seam.mean_abs_difference


def test_overlap_zero_still_reconstructs_the_full_scene(client, scene):
    result = run_tiled(client, scene, TilingConfig(overlap=0))
    assert result.sr.shape == (4, HEIGHT * 4, WIDTH * 4) and torch.isfinite(result.sr).all()
    assert result.plan.tile_count == 6  # stride 128: rows 0,128; cols 0,128,256 -> 2 x 3 tiles
    assert result.seam_diagnostic.status == ComputationStatus.NOT_COMPUTABLE.value


def test_gpu_memory_stays_at_the_single_tile_figure_for_a_whole_scene(client, first_run):
    assert client.describe()["last_request"]["peak_memory_mib"] < 2048  # Phase 1 single tile: ~606 MiB


def test_the_uncertainty_ensemble_runs_through_the_tiler_and_the_same_worker(client, scene):
    """The ensemble (6 transforms, including rot90/rot270 which transpose the scene) drives the
    tiler on the real model: the API path, at the library level."""
    tiled = TiledModel(client)
    pid_before = client.describe()["worker_pid"]
    result = run_stochastic_uncertainty(
        tiled, mirrored_real_scene(160, 224), transforms=DEFAULT_TRANSFORMS, seed=42,
        band_names=("B04", "B03", "B02", "B08"), keep_per_member_predictions=False,
    )
    assert result.n == 6 and result.mean_prediction.shape == (4, 640, 896)
    assert torch.isfinite(result.std_prediction).all() and float(result.std_prediction.max()) > 0
    assert client.describe()["worker_pid"] == pid_before
    assert tiled.summary()["scene_passes"] == 6 and tiled.summary()["tile_inferences"] == 24  # 4 tiles per orientation
