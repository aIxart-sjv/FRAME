"""frame.train.models -- the model registry behind the simple ``sr = model(lr)`` interface."""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from frame.train.config import ModelConfig
from frame.train.errors import ModelBuildError
from frame.train.models import build_model, count_parameters, parameter_megabytes

LITE_DIR = Path(os.environ.get("SEN2SR_BASELINE_WEIGHTS_DIR", str(Path.home() / ".cache" / "sen2sr_baseline" / "SEN2SRLite_RGBN")))
needs_lite = pytest.mark.skipif(not (LITE_DIR / "model.safetensor").is_file(), reason="Lite weights are not cached locally; not downloading")


def tiny(**params):
    return build_model(ModelConfig("tiny_cnn", {"width": 8, "depth": 2, **params}), seed=0)


# ============================================================================== tiny CNN


def test_the_tiny_cnn_follows_the_sr_equals_model_of_lr_contract():
    model = tiny()
    lr = torch.rand(2, 4, 16, 16)
    sr = model(lr)
    assert sr.shape == (2, 4, 64, 64) and sr.dtype == torch.float32


def test_before_any_training_the_tiny_cnn_is_exactly_frames_bicubic_baseline():
    """Zero-initialised residual head: step 0 IS the bicubic baseline, so any later gain is attributable to training.
    'Bicubic' means FRAME's existing convention (frame.validation.bicubic_upsample: antialiased kernel, negatives clamped);
    torch's default bicubic uses a different cubic coefficient and is NOT the same operator (measured: RMSE 0.0069 vs 0.0100)."""
    from frame.validation import bicubic_upsample

    model = tiny()
    lr = torch.rand(2, 4, 16, 16)
    out = model(lr)
    assert all(torch.equal(out[i], bicubic_upsample(lr[i], 4)) for i in range(2))
    assert not torch.equal(out, F.interpolate(lr, scale_factor=4, mode="bicubic", align_corners=False))


def test_the_tiny_cnn_scale_and_channels_are_configurable():
    model = build_model(ModelConfig("tiny_cnn", {"width": 8, "depth": 2, "scale": 2, "in_channels": 3}), seed=0)
    assert model(torch.rand(1, 3, 10, 10)).shape == (1, 3, 20, 20)


def test_construction_is_deterministic_under_the_seed():
    a, b, c = tiny(), tiny(), build_model(ModelConfig("tiny_cnn", {"width": 8, "depth": 2}), seed=1)
    assert all(torch.equal(x, y) for x, y in zip(a.state_dict().values(), b.state_dict().values()))
    assert not all(torch.equal(x, y) for x, y in zip(a.state_dict().values(), c.state_dict().values()))


def test_every_parameter_receives_a_gradient_after_the_first_update():
    """The zero-initialised head blocks gradients to earlier layers at step 0 only; after one step it must not."""
    model = tiny()
    lr, hr = torch.rand(2, 4, 16, 16), torch.rand(2, 4, 64, 64)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    for _ in range(2):
        optimizer.zero_grad()
        F.l1_loss(model(lr), hr).backward()
        if _ == 0:
            optimizer.step()
    assert all(p.grad is not None and float(p.grad.abs().sum()) > 0.0 for p in model.parameters())


def test_the_tiny_cnn_is_fully_convolutional_so_train_and_validation_patch_sizes_may_differ():
    model = tiny()
    assert model(torch.rand(1, 4, 8, 8)).shape[-1] == 32 and model(torch.rand(1, 4, 24, 40)).shape[-2:] == (96, 160)


def test_unknown_constructor_parameters_are_refused():
    with pytest.raises(ModelBuildError, match="depht"):
        build_model(ModelConfig("tiny_cnn", {"depht": 2}), seed=0)
    with pytest.raises(ModelBuildError, match="width"):
        build_model(ModelConfig("tiny_cnn", {"width": 0}), seed=0)


def test_tiny_cnn_has_no_pretrained_weights():
    with pytest.raises(ModelBuildError, match="pretrained"):
        build_model(ModelConfig("tiny_cnn", {}, pretrained=True), seed=0)


def test_parameter_accounting():
    model = tiny()
    n = count_parameters(model)
    assert n == sum(p.numel() for p in model.parameters()) and n > 0
    assert parameter_megabytes(model) == pytest.approx(n * 4 / 2**20)
    assert count_parameters(model, trainable_only=True) == n


# ============================================================================== SEN2SR-Lite architecture


def test_the_lite_architecture_builds_with_the_published_size_and_handles_small_patches():
    model = build_model(ModelConfig("sen2sr_lite", {}), seed=0)
    assert count_parameters(model) == 572_336
    assert model(torch.rand(1, 4, 32, 32)).shape == (1, 4, 128, 128)


def test_lite_is_built_in_train_mode_because_the_published_deploy_mode_trains_almost_nothing():
    """Measured upstream facts. CNNSR(..., train_mode=False) rebuilds each fused eval_conv from its branch weights under
    .detach() on every forward, so only the layers that are not re-parameterised (conv_cat, upsampler) get gradients:
    4 tensors / 16,216 parameters of 472,496. In train mode 68 tensors get gradients; the rest are unused by the published
    forward, which feeds every block the same input (blocks 1-4 never reach the output). sen2sr/ is left untouched."""
    from sen2sr.models.opensr_baseline.cnn import CNNSR

    x, target = torch.rand(1, 4, 16, 16), torch.rand(1, 4, 64, 64)

    def tensors_with_gradient(model):
        model.zero_grad()
        F.l1_loss(model(x), target).backward()
        return [n for n, p in model.named_parameters() if p.grad is not None and float(p.grad.abs().sum()) > 0]

    deploy = CNNSR(in_channels=4, out_channels=4, feature_channels=24, upscale=4, bias=True, train_mode=False, num_blocks=6)
    assert sorted(tensors_with_gradient(deploy)) == ["conv_cat.bias", "conv_cat.weight", "upsampler.0.bias", "upsampler.0.weight"]
    trainable = build_model(ModelConfig("sen2sr_lite", {}), seed=0)
    with_grad = tensors_with_gradient(trainable)
    assert len(with_grad) == 68 and any(".sk." in n for n in with_grad) and not any("eval_conv" in n for n in with_grad)


def test_the_lite_trainable_parameter_count_excludes_only_the_fused_eval_convs():
    model = build_model(ModelConfig("sen2sr_lite", {}), seed=0)
    assert count_parameters(model, trainable_only=True) == 572_336 - 99_840
    assert all(("eval_conv" in n) != p.requires_grad for n, p in model.named_parameters())


@needs_lite
def test_pretrained_lite_weights_reproduce_the_published_model_before_its_hard_constraint():
    import mlstac

    model = build_model(ModelConfig("sen2sr_lite", {}, pretrained=True), seed=0).eval()
    published = mlstac.load(str(LITE_DIR)).compiled_model(device="cpu")
    x = torch.rand(1, 4, 128, 128) * 0.4
    with torch.no_grad():
        # train-mode branches vs the published fused conv: the same function up to float rounding
        assert torch.allclose(model(x), published.sr_model(x), atol=1e-4)


@needs_lite
def test_pretrained_lite_actually_updates_under_an_optimizer_step():
    model = build_model(ModelConfig("sen2sr_lite", {}, pretrained=True), seed=0)
    before = {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}
    optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=1e-2)
    x, target = torch.rand(1, 4, 16, 16) * 0.3, torch.rand(1, 4, 64, 64) * 0.3
    F.l1_loss(model(x), target).backward()
    optimizer.step()
    changed = sum(not torch.equal(before[n], p) for n, p in model.named_parameters() if p.requires_grad)
    assert changed >= 60                                  # the 68 tensors the published forward actually uses (see above)


def test_missing_lite_weights_are_a_clear_error_and_never_trigger_a_download(tmp_path):
    with pytest.raises(ModelBuildError, match="not found"):
        build_model(ModelConfig("sen2sr_lite", {"weights_dir": str(tmp_path)}, pretrained=True), seed=0)


# ============================================================================== Mamba (isolated environment)


@pytest.mark.skipif(importlib.util.find_spec("mamba_ssm") is not None, reason="this environment has mamba_ssm")
def test_mamba_is_refused_with_a_pointer_to_the_isolated_environment_when_it_cannot_be_imported():
    with pytest.raises(ModelBuildError, match="mamba"):
        build_model(ModelConfig("sen2sr_mamba", {}), seed=0)


def test_an_unknown_model_name_is_refused():
    with pytest.raises(ModelBuildError, match="Unknown model"):
        build_model(SimpleNamespace(name="resnet", params={}, pretrained=False), seed=0)    # bypasses config validation on purpose


def test_the_tiny_cnn_escapes_its_bicubic_start_even_for_initialisations_that_kill_relus(tmp_path):
    """Regression for a measured failure, through the real Trainer + data path: with a plain ReLU body, seed 2 left ~94% of the last
    layer's channels dead and validation stayed at EXACTLY the bicubic output (33.106 dB) for the whole run (also seed 3; 2 of 6 seeds
    stalled at batch 1 and 3 of 6 at batch 4). A leaky activation keeps every channel trainable (35.9-36.2 dB)."""
    from frame.data.adapters.synthetic import build_synthetic_dataset
    from frame.data.contract import Split
    from frame.data.manifest import write_manifest
    from frame.data.splits import apply_splits, assign_splits
    from frame.train.config import TrainConfig
    from frame.train.data import prepare_training_data
    from frame.train.trainer import Trainer

    records = build_synthetic_dataset(tmp_path, n_regions=2, scenes_per_region=1, seed=1)
    records = apply_splits(records, assign_splits(records, {Split.TRAIN: 0.5, Split.VAL: 0.5, Split.TEST: 0.0}, seed=0, level="region"), level="region")
    write_manifest(tmp_path / "m.jsonl", records)
    config = TrainConfig.from_dict({
        "name": "stall", "model": {"name": "tiny_cnn", "params": {"width": 32, "depth": 4}}, "seed": 2, "steps": 600, "batch_size": 4, "device": "cpu",
        "output_dir": str(tmp_path / "run"), "loss": {"reconstruction": "charbonnier", "charbonnier_eps": 0.01}, "optim": {"lr": 0.001}, "log_every": 1000,
        "data": {"manifest": str(tmp_path / "m.jsonl"), "data_root": str(tmp_path), "lr_patch": 32, "val_lr_patch": 128, "patches_per_pair": 16, "augment": True},
    })
    data = prepare_training_data(config)
    trainer = Trainer(config, build_model(config.model, seed=config.seed), data.train, val_data=data.val, manifest_digest="x", scale=4)
    before = trainer.validate()["metrics"]["psnr_db"]                       # step 0 == the bicubic baseline
    trainer.run()
    assert trainer.validate()["metrics"]["psnr_db"] > before + 1.0


def test_the_tiny_cnn_body_uses_no_dying_activation():
    model = tiny()
    assert not any(isinstance(m, torch.nn.ReLU) for m in model.modules())
    assert any(isinstance(m, torch.nn.LeakyReLU) and m.negative_slope > 0 for m in model.modules())
