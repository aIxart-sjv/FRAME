"""frame.evaluate.systems -- frozen systems behind one interface, with provenance; no GPU needed."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from frame.evaluate.config import SystemSpec
from frame.evaluate.errors import RoleSafetyError, SystemUnavailableError
from frame.evaluate.systems import (
    CallableSystem,
    build_system,
    check_no_training_overlap,
    seam_lines,
    sha256_file,
)
from frame.tiling.plan import TilingConfig, plan_tiles
from frame.train import checkpoint as ckpt
from frame.train.config import ModelConfig, TrainConfig
from frame.train.models import build_model

LITE_DIR = Path(os.environ.get("SEN2SR_BASELINE_WEIGHTS_DIR", str(Path.home() / ".cache" / "sen2sr_baseline" / "SEN2SRLite_RGBN")))
needs_lite = pytest.mark.skipif(not (LITE_DIR / "model.safetensor").is_file(), reason="Lite weights are not cached locally; not downloading")
TILING = TilingConfig(tile_size=128, overlap=32)


def spec(kind, **kw):
    return SystemSpec(name=kw.pop("name", kind), kind=kind, **kw)


# ============================================================================== bicubic


def test_bicubic_is_frames_existing_operator_applied_to_the_whole_scene():
    from frame.validation import bicubic_upsample

    lr = torch.rand(4, 20, 30)
    system = build_system(spec("bicubic"), TILING, device="cpu")
    result = system.infer(lr)
    assert torch.equal(result.sr, bicubic_upsample(lr, 4)) and result.sr.shape == (4, 80, 120) and result.plan is None
    p = system.provenance()
    assert p["kind"] == "bicubic" and p["hard_constraint"] is None and p["weights"] is None and "bicubic_upsample" in p["operator"] and p["tiling"] == "none (whole scene)"


def test_every_system_records_the_common_preprocessing_and_band_order():
    p = build_system(spec("bicubic"), TILING, device="cpu").provenance()
    assert p["input_bands"] == ["B04", "B03", "B02", "B08"] and "DN" in p["value_convention"] and "no BOA offset" in p["value_convention"]


# ============================================================================== tiled inference through the existing engine


class Nearest4(nn.Module):
    """A 'model' whose output does not depend on tiling: nearest-neighbour x4."""

    def forward(self, x):
        return F.interpolate(x, scale_factor=4, mode="nearest")


def test_tiled_inference_of_a_tiling_independent_model_equals_whole_scene_inference():
    lr = torch.rand(4, 200, 150)
    system = CallableSystem("nn", Nearest4(), TILING, hard_constraint=False, provenance={"kind": "test"}, device="cpu")
    r = system.infer(lr)
    assert torch.allclose(r.sr, F.interpolate(lr[None], scale_factor=4, mode="nearest")[0], atol=1e-6)
    assert r.plan.tile_count > 1 and r.tile_count == r.plan.tile_count and r.seam_diagnostic is not None


def test_a_single_tile_scene_has_no_seams():
    r = CallableSystem("nn", Nearest4(), TILING, hard_constraint=False, provenance={}, device="cpu").infer(torch.rand(4, 128, 128))
    assert r.plan.tile_count == 1 and seam_lines(r.plan, TILING) == ([], [])


def test_seam_lines_sit_in_the_middle_of_each_blend_overlap():
    plan = plan_tiles(256, 256, TILING)
    rows, cols = seam_lines(plan, TILING)
    assert rows == cols and len(rows) >= 1 and all(r % 1 == 0 for r in rows)
    stride_sr = TILING.stride * 4
    assert rows[0] == stride_sr + (TILING.overlap * 4) // 2


def test_the_tiling_configuration_is_recorded():
    p = CallableSystem("nn", Nearest4(), TILING, hard_constraint=False, provenance={"kind": "x"}, device="cpu").provenance()
    assert p["tiling"] == {"tile_size": 128, "overlap": 32, "stride": 96, "scale": 4, "padding_mode": "reflect", "blend_mode": "linear"}


# ============================================================================== Lite (cached weights, CPU)


@needs_lite
def test_lite_with_its_hard_constraint_is_the_published_model_and_records_its_weights():
    import mlstac

    system = build_system(spec("lite"), TILING, device="cpu")
    x = torch.rand(4, 128, 128) * 0.4
    published = mlstac.load(str(LITE_DIR)).compiled_model(device="cpu")
    with torch.no_grad():
        assert torch.equal(system.infer(x).sr, published(x[None])[0])
    p = system.provenance()
    assert p["hard_constraint"] is True and p["model_name"] == "SEN2SRLite/NonReference_RGBN_x4" and p["weights"]["model.safetensor"] == sha256_file(LITE_DIR / "model.safetensor")
    assert len(p["weights"]["hard_constraint.safetensor"]) == 64 and p["parameters"] == 572_336
    system.close()


@needs_lite
def test_lite_without_its_hard_constraint_is_a_different_named_system():
    import mlstac

    system = build_system(spec("lite", name="lite_nc", hard_constraint=False), TILING, device="cpu")
    x = torch.rand(4, 128, 128) * 0.4
    published = mlstac.load(str(LITE_DIR)).compiled_model(device="cpu")
    with torch.no_grad():
        assert torch.equal(system.infer(x).sr, published.sr_model(x[None]).clamp(min=0)[0])
        assert not torch.equal(system.infer(x).sr, published(x[None])[0])
    assert system.provenance()["hard_constraint"] is False and "clamped" in system.provenance()["note"]


def test_missing_lite_weights_are_an_unavailable_system_not_a_crash(tmp_path):
    with pytest.raises(SystemUnavailableError, match="not found"):
        build_system(spec("lite", params={"weights_dir": str(tmp_path)}), TILING, device="cpu")


# ============================================================================== Mamba (worker; faked here)


class FakeClient:
    def __init__(self, **kw):
        self.started = False
        self.closed = False

    def start(self):
        self.started = True
        return {}

    def __call__(self, x):
        return F.interpolate(x, scale_factor=4, mode="nearest")

    def describe(self):
        return {"model_name": "SEN2SR/MambaSR_RGBN_x4", "executable_architecture": "MambaSR", "parameter_count": 13_759_444, "weights_file": "sr_model.safetensor",
                "weights_sha256": "a" * 64, "hard_constraint_file": "sr_hard_constraint.safetensor", "hard_constraint_sha256": "b" * 64, "device": "cuda", "input_band_order": ["B04", "B03", "B02", "B08"]}

    def close(self):
        self.closed = True


def test_mamba_runs_through_its_worker_and_records_the_workers_own_load_report():
    system = build_system(spec("mamba"), TILING, device="cuda", mamba_client_factory=FakeClient)
    r = system.infer(torch.rand(4, 128, 128))
    assert r.sr.shape == (4, 512, 512)
    p = system.provenance()
    assert p["hard_constraint"] is True and p["weights"]["weights_sha256"] == "a" * 64 and p["parameters"] == 13_759_444 and p["model_name"] == "SEN2SR/MambaSR_RGBN_x4"
    system.close()
    assert system._client.closed


def test_mamba_without_a_gpu_or_worker_is_unavailable_with_a_reason(monkeypatch):
    from frame.models.mamba_client import Availability

    monkeypatch.setattr("frame.evaluate.systems.check_mamba_availability", lambda device: Availability(available=False, reason_code="no_cuda", message="no GPU"))
    with pytest.raises(SystemUnavailableError, match="no GPU"):
        build_system(spec("mamba"), TILING, device="cpu")


# ============================================================================== frozen trained checkpoints


def make_checkpoint(tmp_path, seed=0, manifest_digest="d" * 64, name="ck"):
    config = TrainConfig.from_dict({"name": "t", "model": {"name": "tiny_cnn", "params": {"width": 8, "depth": 2}}, "data": {"manifest": "m.jsonl"}, "steps": 5, "output_dir": "o", "seed": seed})
    model = build_model(config.model, seed=seed)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    path = ckpt.save_checkpoint(tmp_path / f"{name}.pt", model=model, optimizer=opt, scheduler=torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 1.0),
                                scaler=torch.amp.GradScaler("cpu", enabled=False), step=5, micro_step=5, config=config, manifest_digest=manifest_digest, git_revision="abc123")
    return path, config


def test_a_checkpoint_system_is_a_frozen_model_with_full_provenance(tmp_path):
    path, config = make_checkpoint(tmp_path, seed=4)
    system = build_system(spec("checkpoint", name="t4", checkpoint=str(path)), TILING, device="cpu")
    lr = torch.rand(4, 128, 128)                                                          # one unpadded tile: the model sees exactly the scene
    from frame.validation import bicubic_upsample

    assert torch.equal(system.infer(lr).sr, bicubic_upsample(lr, 4))                      # an untrained tiny_cnn IS bicubic
    p = system.provenance()
    assert p["hard_constraint"] is False and p["weights"]["checkpoint_sha256"] == sha256_file(path) and p["training"]["step"] == 5 and p["training"]["seed"] == 4
    assert p["training"]["manifest_digest"] == "d" * 64 and p["training"]["git_revision"] == "abc123" and p["training"]["config_digest"] == config.digest()
    assert p["model_name"] == "tiny_cnn" and p["parameters"] > 0 and all(not q.requires_grad for q in system._model.parameters()) and not system._model.training


def test_a_missing_or_corrupt_checkpoint_is_unavailable(tmp_path):
    with pytest.raises(SystemUnavailableError):
        build_system(spec("checkpoint", checkpoint=str(tmp_path / "absent.pt")), TILING, device="cpu")
    (tmp_path / "junk.pt").write_bytes(b"not a checkpoint")
    with pytest.raises(SystemUnavailableError):
        build_system(spec("checkpoint", checkpoint=str(tmp_path / "junk.pt")), TILING, device="cpu")


def write_summary(tmp_path, train=("R00_S00", "R01_S00"), val=("R02_S00",), datasets=("synthetic_smoke",)):
    p = tmp_path / "summary.json"
    p.write_text(json.dumps({"dataset": {"datasets": list(datasets), "train_scenes": list(train), "val_scenes": list(val), "manifest_digest": "d" * 64}}))
    return p


def test_the_training_summary_supplies_the_scenes_used_and_is_recorded(tmp_path):
    path, _ = make_checkpoint(tmp_path)
    summary = write_summary(tmp_path)
    system = build_system(spec("checkpoint", checkpoint=str(path), train_summary=str(summary)), TILING, device="cpu")
    t = system.provenance()["training"]
    assert t["train_scenes"] == ["R00_S00", "R01_S00"] and t["val_scenes"] == ["R02_S00"] and t["training_datasets"] == ["synthetic_smoke"] and t["summary_sha256"] == sha256_file(summary)


def test_evaluating_on_scenes_the_model_saw_is_refused(tmp_path):
    path, _ = make_checkpoint(tmp_path)
    system = build_system(spec("checkpoint", checkpoint=str(path), train_summary=str(write_summary(tmp_path))), TILING, device="cpu")
    with pytest.raises(RoleSafetyError) as excinfo:
        check_no_training_overlap(system, dataset_kind="synthetic_smoke", scene_ids=["R02_S00", "R09_S00"])           # R02_S00 was the validation scene
    assert "train_eval_overlap" in excinfo.value.codes and "R02_S00" in str(excinfo.value)
    check_no_training_overlap(system, dataset_kind="synthetic_smoke", scene_ids=["R09_S00"])                            # a disjoint scene is fine


def test_a_model_trained_on_a_benchmark_cannot_be_evaluated_on_it(tmp_path):
    path, _ = make_checkpoint(tmp_path)
    system = build_system(spec("checkpoint", checkpoint=str(path), train_summary=str(write_summary(tmp_path, datasets=("sen2neon", "synthetic_smoke")))), TILING, device="cpu")
    with pytest.raises(RoleSafetyError) as excinfo:
        check_no_training_overlap(system, dataset_kind="sen2neon", scene_ids=["2018_MLBS_3"])
    assert "trained_on_benchmark" in excinfo.value.codes


def test_a_checkpoint_without_its_training_summary_is_evaluated_but_flagged_unverifiable(tmp_path):
    path, _ = make_checkpoint(tmp_path)
    system = build_system(spec("checkpoint", checkpoint=str(path)), TILING, device="cpu")
    assert system.provenance()["training"]["overlap_check"] == "not performed: no train_summary was given"
    check_no_training_overlap(system, dataset_kind="sen2neon", scene_ids=["x"])                                          # nothing to check against; must not raise


def test_non_checkpoint_systems_have_no_training_scenes_to_overlap(tmp_path):
    system = build_system(spec("bicubic"), TILING, device="cpu")
    check_no_training_overlap(system, dataset_kind="sen2neon", scene_ids=["x"])


def test_sha256_file_is_the_streaming_digest(tmp_path):
    p = tmp_path / "f.bin"
    p.write_bytes(b"abc" * 100_000)
    assert sha256_file(p) == hashlib.sha256(b"abc" * 100_000).hexdigest()


def test_tiled_inference_of_a_small_scene_reflect_pads_to_a_full_tile_so_border_pixels_differ_from_whole_scene_bicubic(tmp_path):
    """A protocol property, not a bug: the tile engine pads a 40x40 scene to 128x128 by reflection, so interpolation at the scene border sees reflected
    pixels. The interior is identical to whole-scene bicubic; only a border of a few HR pixels differs. Bicubic itself is evaluated on the whole scene."""
    from frame.validation import bicubic_upsample

    path, _ = make_checkpoint(tmp_path)
    system = build_system(spec("checkpoint", checkpoint=str(path)), TILING, device="cpu")
    lr = torch.rand(4, 40, 40)
    tiled, whole = system.infer(lr).sr, bicubic_upsample(lr, 4)
    assert not torch.equal(tiled, whole) and torch.allclose(tiled[:, 16:-16, 16:-16], whole[:, 16:-16, 16:-16], atol=1e-5)


def test_a_network_system_exposes_its_single_tile_model_and_device_for_test_time_augmentation():
    """Phase 6 runs the same frozen model through frame.uncertainty's TTA ensemble, wrapped in the tile engine; it needs the tile model, not a private attribute."""
    model = Nearest4()
    system = CallableSystem("nn", model, TILING, hard_constraint=False, provenance={}, device="cpu")
    assert system.tile_model is model and system.device == "cpu" and system.tiling == TILING
