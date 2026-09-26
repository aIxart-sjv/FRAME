"""One long-lived worker serves a whole tiled scene (Phase 2).

Drives the REAL `MambaWorkerClient` and the REAL tile engine against the small stub
worker from test_models_mamba_client.py (same wire protocol, nearest-neighbour x4
"model", no GPU): every tile of every ensemble pass must go through a single worker
process, and a worker failure partway through a scene must abort it cleanly.
"""

from __future__ import annotations

import subprocess
import sys
import time

import pytest
import torch
import torch.nn.functional as F

from frame.models import mamba_client
from frame.models.errors import ModelContractError, ModelWorkerError
from frame.models.mamba_client import MambaWorkerClient
from frame.tests.test_models_mamba_client import STUB_WORKER
from frame.tiling import TiledModel, plan_tiles, run_tiled
from frame.uncertainty import DEFAULT_TRANSFORMS, run_stochastic_uncertainty


def truth(scene: torch.Tensor) -> torch.Tensor:
    return F.interpolate(scene[None], scale_factor=4, mode="nearest")[0]


def scene_of(h: int, w: int) -> torch.Tensor:
    # reflectance-like values: the client enforces the model's input contract on every tile
    return torch.rand(4, h, w, generator=torch.Generator().manual_seed(1)) * 0.5


@pytest.fixture()
def spawns(monkeypatch):
    """Counts every worker process the client launches."""
    started = []
    real_popen = subprocess.Popen

    def counting_popen(*args, **kwargs):
        process = real_popen(*args, **kwargs)
        started.append(process.pid)
        return process

    monkeypatch.setattr(mamba_client.subprocess, "Popen", counting_popen)
    return started


@pytest.fixture()
def make_client(tmp_path):
    script = tmp_path / "stub_worker.py"
    script.write_text(STUB_WORKER)
    clients = []

    def _make(mode: str = "ok", request_timeout: float = 30.0) -> MambaWorkerClient:
        client = MambaWorkerClient(
            worker_command=[sys.executable, str(script), mode, str(tmp_path / "crash.marker")],
            device="cpu",
            startup_timeout=30,
            request_timeout=request_timeout,
        )
        clients.append(client)
        return client

    yield _make
    for client in clients:
        client.close()


def test_every_tile_of_a_scene_is_served_by_one_worker_process(make_client, spawns):
    client = make_client()
    scene = scene_of(300, 500)
    pids_seen = set()

    result = run_tiled(client, scene, on_tile=lambda done, total, tile: pids_seen.add(client._proc.pid))

    assert len(spawns) == 1  # the worker was started once, not once per tile
    assert pids_seen == {spawns[0]}  # and it was the same process for all 15 tiles
    assert result.plan.tile_count == 15
    assert torch.allclose(result.sr, truth(scene), atol=1e-6)


def test_the_worker_is_reused_across_all_ensemble_passes_of_a_scene(make_client, spawns):
    client = make_client()
    tiled = TiledModel(client)
    scene = scene_of(200, 300)

    result = run_stochastic_uncertainty(
        tiled, scene, transforms=DEFAULT_TRANSFORMS, seed=42, band_names=("B04", "B03", "B02", "B08"),
        keep_per_member_predictions=False,
    )

    assert len(spawns) == 1
    assert tiled.summary()["tile_inferences"] == 36  # 6 passes x 6 tiles, all through the same worker
    assert torch.allclose(result.mean_prediction, truth(scene), atol=1e-5)


def test_a_worker_reused_across_scenes_is_still_one_process(make_client, spawns):
    client = make_client()
    run_tiled(client, scene_of(150, 150))
    run_tiled(client, scene_of(200, 260))
    assert len(spawns) == 1


def test_a_worker_crash_partway_through_a_scene_aborts_it_and_the_next_scene_recovers(make_client, spawns):
    client = make_client("crash_once")
    scene = scene_of(300, 500)

    with pytest.raises(ModelWorkerError, match="stopped unexpectedly"):
        run_tiled(client, scene)  # the stub dies on its first request; nothing partial comes back
    assert len(spawns) == 1 and not client.is_running

    result = run_tiled(client, scene)  # the client restarts the worker for the next request
    assert len(spawns) == 2
    assert torch.allclose(result.sr, truth(scene), atol=1e-6)


def test_a_hung_worker_times_out_instead_of_blocking_the_scene_forever(make_client):
    client = make_client("hang", request_timeout=1.0)
    started = time.monotonic()
    with pytest.raises(ModelWorkerError, match="timed out"):
        run_tiled(client, scene_of(200, 200))
    assert time.monotonic() - started < 15
    assert not client.is_running


def test_the_models_input_contract_is_enforced_on_every_tile_before_any_worker_starts(make_client, spawns):
    client = make_client()
    raw_digital_numbers = scene_of(200, 200) * 20000.0  # a scene that was never divided by 10000
    with pytest.raises(ModelContractError, match="divided by 10000"):
        run_tiled(client, raw_digital_numbers)
    assert spawns == []  # rejected in this process, so no worker was even started


def test_a_scene_with_a_tile_count_of_one_makes_exactly_one_request(make_client, spawns):
    client = make_client()
    scene = scene_of(128, 128)
    assert plan_tiles(128, 128).tile_count == 1
    assert torch.equal(run_tiled(client, scene).sr, truth(scene))
    assert len(spawns) == 1
