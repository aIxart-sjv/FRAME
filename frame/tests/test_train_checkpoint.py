"""frame.train.checkpoint -- atomic, safely-loadable, compatibility-checked, resumable checkpoints."""

from __future__ import annotations

import random

import numpy as np
import pytest
import torch
import torch.nn as nn

from frame.train import checkpoint as ckpt
from frame.train.config import TrainConfig
from frame.train.errors import CheckpointError

DIGEST = "a" * 64


def make_config(**overrides) -> TrainConfig:
    d = {"name": "unit", "model": {"name": "tiny_cnn", "params": {"width": 8, "depth": 2}}, "data": {"manifest": "m.jsonl"}, "steps": 10, "output_dir": "out"}
    d.update(overrides)
    return TrainConfig.from_dict(d)


def make_state(seed=0):
    torch.manual_seed(seed)
    model = nn.Sequential(nn.Linear(4, 6), nn.ReLU(), nn.Linear(6, 2))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.5)
    scaler = torch.amp.GradScaler("cpu", enabled=False)
    return model, optimizer, scheduler, scaler


def take_steps(model, optimizer, scheduler, n=3):
    for _ in range(n):
        optimizer.zero_grad()
        model(torch.ones(3, 4)).pow(2).mean().backward()
        optimizer.step()
        scheduler.step()


def save(tmp_path, config=None, **kw):
    model, optimizer, scheduler, scaler = make_state()
    take_steps(model, optimizer, scheduler)
    path = ckpt.save_checkpoint(tmp_path / "step_000003.pt", model=model, optimizer=optimizer, scheduler=scheduler, scaler=scaler, step=3, micro_step=6,
                                config=config or make_config(), manifest_digest=DIGEST, git_revision="abc123", **kw)
    return path, (model, optimizer, scheduler, scaler)


# ============================================================================== save / load


def test_a_checkpoint_carries_everything_needed_to_continue_and_to_audit_the_run(tmp_path):
    path, _ = save(tmp_path)
    c = ckpt.load_checkpoint(path)
    for key in ("model", "optimizer", "scheduler", "scaler", "step", "micro_step", "config", "resume_signature", "seed", "manifest_digest",
                "git_revision", "torch_version", "format_version", "rng"):
        assert key in c, key
    assert (c["step"], c["micro_step"], c["seed"], c["manifest_digest"], c["git_revision"]) == (3, 6, 0, DIGEST, "abc123")
    assert c["config"] == make_config().to_dict() and c["resume_signature"] == make_config().resume_signature()
    assert c["torch_version"] == torch.__version__ and c["format_version"] == ckpt.FORMAT_VERSION


def test_state_survives_the_round_trip_exactly(tmp_path):
    path, (model, optimizer, scheduler, scaler) = save(tmp_path)
    fresh_model, fresh_opt, fresh_sched, fresh_scaler = make_state(seed=99)
    ckpt.apply_checkpoint(ckpt.load_checkpoint(path), fresh_model, fresh_opt, fresh_sched, fresh_scaler)
    assert all(torch.equal(a, b) for a, b in zip(model.state_dict().values(), fresh_model.state_dict().values()))
    assert fresh_sched.last_epoch == scheduler.last_epoch and fresh_opt.param_groups[0]["lr"] == optimizer.param_groups[0]["lr"]
    a, b = optimizer.state_dict()["state"], fresh_opt.state_dict()["state"]
    assert a.keys() == b.keys() and all(torch.equal(a[k]["exp_avg"], b[k]["exp_avg"]) for k in a)


def test_a_restored_run_continues_exactly_like_the_original(tmp_path):
    path, (model, optimizer, scheduler, scaler) = save(tmp_path)
    fresh = make_state(seed=99)
    ckpt.apply_checkpoint(ckpt.load_checkpoint(path), *fresh)
    take_steps(model, optimizer, scheduler, 4)
    take_steps(fresh[0], fresh[1], fresh[2], 4)
    assert all(torch.equal(a, b) for a, b in zip(model.state_dict().values(), fresh[0].state_dict().values()))


def test_checkpoints_hold_only_tensors_and_primitives_so_they_load_without_unpickling_arbitrary_objects(tmp_path):
    path, _ = save(tmp_path)
    loaded = torch.load(path, weights_only=True, map_location="cpu")                  # would raise on any pickled object
    assert loaded["step"] == 3


def test_saving_is_atomic_and_a_failure_leaves_the_previous_checkpoint_intact(tmp_path, monkeypatch):
    path, state = save(tmp_path)
    before = path.read_bytes()

    def boom(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(torch, "save", boom)
    with pytest.raises(OSError):
        ckpt.save_checkpoint(path, model=state[0], optimizer=state[1], scheduler=state[2], scaler=state[3], step=9, micro_step=9,
                             config=make_config(), manifest_digest=DIGEST, git_revision=None)
    assert path.read_bytes() == before and [p.name for p in tmp_path.iterdir()] == [path.name]      # and no temp file is left behind


def test_parent_directories_are_created(tmp_path):
    model, optimizer, scheduler, scaler = make_state()
    path = ckpt.save_checkpoint(tmp_path / "a" / "b" / "c.pt", model=model, optimizer=optimizer, scheduler=scheduler, scaler=scaler, step=0, micro_step=0,
                                config=make_config(), manifest_digest=DIGEST, git_revision=None)
    assert path.is_file()


# ============================================================================== compatibility


def test_a_compatible_checkpoint_is_accepted_even_if_the_step_budget_changed(tmp_path):
    path, _ = save(tmp_path)
    c = ckpt.load_checkpoint(path)
    ckpt.check_resumable(c, make_config(steps=500, output_dir="elsewhere"), DIGEST)


@pytest.mark.parametrize("override", [{"seed": 7}, {"batch_size": 2}, {"optim": {"lr": 5e-3}}, {"loss": {"reconstruction": "charbonnier"}}])
def test_a_changed_training_config_refuses_to_resume(tmp_path, override):
    path, _ = save(tmp_path)
    with pytest.raises(CheckpointError, match="config"):
        ckpt.check_resumable(ckpt.load_checkpoint(path), make_config(**override), DIGEST)


def test_a_different_manifest_refuses_to_resume(tmp_path):
    path, _ = save(tmp_path)
    with pytest.raises(CheckpointError, match="manifest"):
        ckpt.check_resumable(ckpt.load_checkpoint(path), make_config(), "b" * 64)


def test_an_unknown_format_version_is_refused(tmp_path):
    path, _ = save(tmp_path)
    c = ckpt.load_checkpoint(path)
    c["format_version"] = 999
    with pytest.raises(CheckpointError, match="format"):
        ckpt.check_resumable(c, make_config(), DIGEST)


def test_missing_and_corrupt_files_are_checkpoint_errors(tmp_path):
    with pytest.raises(CheckpointError, match="not found"):
        ckpt.load_checkpoint(tmp_path / "absent.pt")
    (tmp_path / "junk.pt").write_bytes(b"this is not a checkpoint")
    with pytest.raises(CheckpointError, match="could not be read"):
        ckpt.load_checkpoint(tmp_path / "junk.pt")
    torch.save({"step": 1}, tmp_path / "partial.pt")
    with pytest.raises(CheckpointError, match="missing"):
        ckpt.load_checkpoint(tmp_path / "partial.pt")


def test_loading_into_a_model_of_a_different_architecture_is_an_error(tmp_path):
    path, _ = save(tmp_path)
    other = nn.Sequential(nn.Linear(4, 7), nn.ReLU(), nn.Linear(7, 2))
    opt = torch.optim.AdamW(other.parameters())
    with pytest.raises(CheckpointError, match="architecture"):
        ckpt.apply_checkpoint(ckpt.load_checkpoint(path), other, opt, torch.optim.lr_scheduler.StepLR(opt, 2), torch.amp.GradScaler("cpu", enabled=False))


# ============================================================================== RNG


def test_rng_capture_and_restore_reproduces_the_random_streams():
    random.seed(1), np.random.seed(1), torch.manual_seed(1)
    state = ckpt.capture_rng()
    expected = (random.random(), float(np.random.rand()), float(torch.rand(1)))
    random.seed(5), np.random.seed(5), torch.manual_seed(5)                        # disturb everything
    ckpt.restore_rng(state)
    assert (random.random(), float(np.random.rand()), float(torch.rand(1))) == expected


def test_captured_rng_state_is_safe_to_serialise():
    state = ckpt.capture_rng()
    torch.save(state, "/dev/null")
    import json

    json.dumps({k: v for k, v in state.items() if k in ("python", "numpy")})       # primitives only


# ============================================================================== directory management


def test_the_latest_checkpoint_is_the_one_with_the_highest_step(tmp_path):
    assert ckpt.latest_checkpoint(tmp_path) is None and ckpt.latest_checkpoint(tmp_path / "missing") is None
    for step in (5, 20, 100):
        (tmp_path / ckpt.checkpoint_name(step)).write_bytes(b"x")
    (tmp_path / "notes.txt").write_text("x")
    assert ckpt.latest_checkpoint(tmp_path).name == ckpt.checkpoint_name(100)
    assert ckpt.checkpoint_name(7) == "step_000007.pt"


def test_pruning_keeps_only_the_last_k_and_touches_nothing_else(tmp_path):
    for step in (1, 2, 3, 4):
        (tmp_path / ckpt.checkpoint_name(step)).write_bytes(b"x")
    (tmp_path / "keep_me.txt").write_text("x")
    removed = ckpt.prune_checkpoints(tmp_path, keep_last=2)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["keep_me.txt", ckpt.checkpoint_name(3), ckpt.checkpoint_name(4)]
    assert sorted(r.name for r in removed) == [ckpt.checkpoint_name(1), ckpt.checkpoint_name(2)]
