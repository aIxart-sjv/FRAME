"""frame.train.validate and frame.train.baselines -- deterministic evaluation and the reference upsamplers."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from frame.train.baselines import LITE_NATIVE_LR, BicubicUpsampler, build_baselines, load_lite_baseline
from frame.train.config import LossConfig
from frame.train.errors import ModelBuildError, TrainError
from frame.train.losses import CompositeLoss
from frame.train.validate import basic_metrics, evaluate, reference_metrics
from frame.tests.test_train_trainer import PairSet

LITE_DIR = Path(os.environ.get("SEN2SR_BASELINE_WEIGHTS_DIR", str(Path.home() / ".cache" / "sen2sr_baseline" / "SEN2SRLite_RGBN")))
needs_lite = pytest.mark.skipif(not (LITE_DIR / "model.safetensor").is_file(), reason="Lite weights are not cached locally; not downloading")
BANDS = ("B04", "B03", "B02", "B08")


# ============================================================================== bicubic


def test_the_bicubic_baseline_is_exactly_frames_existing_bicubic_convention():
    from frame.validation import bicubic_upsample

    lr = torch.rand(2, 4, 16, 16)
    out = BicubicUpsampler(4)(lr)
    assert out.shape == (2, 4, 64, 64) and all(torch.equal(out[i], bicubic_upsample(lr[i], 4)) for i in range(2))
    assert float(out.min()) >= 0.0                                        # the convention clamps negative ringing


def test_the_bicubic_baseline_has_no_parameters_and_is_not_trainable():
    assert list(BicubicUpsampler(4).parameters()) == []


# ============================================================================== Lite


@needs_lite
def test_the_lite_baseline_is_the_published_model_unchanged():
    import mlstac

    baseline = load_lite_baseline("cpu")
    published = mlstac.load(str(LITE_DIR)).compiled_model(device="cpu")
    x = torch.rand(1, 4, 128, 128) * 0.4
    with torch.no_grad():
        assert torch.equal(baseline(x), published(x)) and baseline(x).shape == (1, 4, 512, 512)


@needs_lite
def test_the_lite_baseline_only_accepts_its_native_tile_size():
    baseline = load_lite_baseline("cpu")
    with pytest.raises(TrainError, match="128"):
        baseline(torch.rand(1, 4, 64, 64))
    assert LITE_NATIVE_LR == 128


def test_missing_lite_weights_are_a_clear_error_and_never_trigger_a_download(tmp_path):
    with pytest.raises(ModelBuildError, match="not found"):
        load_lite_baseline("cpu", weights_dir=tmp_path)


def test_build_baselines_returns_the_requested_ones_by_name():
    assert list(build_baselines(("bicubic",), device="cpu", scale=4)) == ["bicubic"]
    with pytest.raises(ModelBuildError, match="Unknown baseline"):
        build_baselines(("magic",), device="cpu", scale=4)


# ============================================================================== metrics


def test_basic_metrics_match_hand_computation_and_respect_the_mask():
    hr = torch.full((4, 8, 8), 0.5)
    sr = hr + 0.1
    mask = np.ones((8, 8), dtype=bool)
    m = basic_metrics(sr, hr, mask, BANDS, 4)
    assert m["rmse"] == pytest.approx(0.1, abs=1e-6) and m["psnr_db"] == pytest.approx(20.0, abs=1e-4)
    bad = sr.clone()
    bad[:, :4] += 9.0
    half = np.ones((8, 8), dtype=bool)
    half[:4] = False
    assert basic_metrics(bad, hr, half, BANDS, 4)["rmse"] == pytest.approx(0.1, abs=1e-5)
    assert basic_metrics(sr, hr, np.zeros((8, 8), dtype=bool), BANDS, 4) == {"rmse": None, "psnr_db": None}


def test_reference_metrics_delegate_to_the_existing_validation_code():
    from frame.validation import compute_reference_metrics

    g = torch.Generator().manual_seed(0)
    hr = torch.rand(4, 32, 32, generator=g) * 0.5 + 0.1
    sr = hr + 0.02 * torch.randn(4, 32, 32, generator=g)
    mask = np.ones((32, 32), dtype=bool)
    ours = reference_metrics(sr, hr, mask, BANDS, 4)
    theirs = compute_reference_metrics(sr, hr, mask, band_names=list(BANDS), scale_factor=4)
    assert ours == {"psnr_db": theirs.psnr_db, "ssim": theirs.ssim, "rmse": theirs.rmse, "sam_degrees": theirs.sam_degrees, "ergas": theirs.ergas}
    assert set(ours) == {"psnr_db", "ssim", "rmse", "sam_degrees", "ergas"} and 0.0 < ours["ssim"] < 1.0


# ============================================================================== evaluate()


def loss():
    return CompositeLoss(LossConfig())


def evaluate_on(model, data, metrics_fn=None):
    return evaluate(model, data, loss(), device=torch.device("cpu"), scale=4, metrics_fn=metrics_fn, band_names=BANDS)


def test_evaluate_reports_loss_and_metrics_averaged_over_patches():
    data = PairSet(n=3, seed=5)
    result = evaluate_on(BicubicUpsampler(4), data)
    assert result["n_patches"] == 3 and result["skipped_patches"] == 0
    assert result["loss"]["total"] == pytest.approx(result["loss"]["reconstruction"]) and result["loss"]["total"] > 0
    expected = [float(F.l1_loss(BicubicUpsampler(4)(data[i]["lr"][None]), data[i]["hr"][None])) for i in range(3)]
    assert result["loss"]["total"] == pytest.approx(sum(expected) / 3, rel=1e-5)
    assert {"rmse", "psnr_db"} <= set(result["metrics"])


def test_evaluate_is_deterministic():
    data = PairSet(n=3, seed=5)
    assert evaluate_on(BicubicUpsampler(4), data) == evaluate_on(BicubicUpsampler(4), data)


def test_evaluate_uses_eval_mode_and_restores_training_mode():
    modes = []

    class Probe(torch.nn.Module):
        def forward(self, x):
            modes.append(self.training)
            return F.interpolate(x, scale_factor=4, mode="bilinear")

    model = Probe().train()
    evaluate_on(model, PairSet(n=2))
    assert modes == [False, False] and model.training


def test_evaluate_leaves_gradients_alone():
    model = torch.nn.Conv2d(4, 4, 1)
    class Wrap(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.c = model

        def forward(self, x):
            return F.interpolate(self.c(x), scale_factor=4, mode="bilinear")

    wrapped = Wrap()
    evaluate_on(wrapped, PairSet(n=2))
    assert all(p.grad is None for p in wrapped.parameters())


def test_fully_masked_patches_are_skipped_and_counted_not_scored():
    data = PairSet(n=2, masks=True)
    data.items[0]["hr_mask"] = torch.zeros_like(data.items[0]["hr_mask"])
    result = evaluate_on(BicubicUpsampler(4), data)
    assert (result["n_patches"], result["skipped_patches"]) == (1, 1)


def test_metric_values_that_are_infinite_or_missing_are_dropped_from_the_mean_not_propagated():
    identical = PairSet(n=2)
    for item in identical.items:
        item["hr"] = BicubicUpsampler(4)(item["lr"][None])[0]
    result = evaluate_on(BicubicUpsampler(4), identical)
    assert result["loss"]["total"] == pytest.approx(0.0, abs=1e-7)
    assert result["metrics"]["psnr_db"] is None and result["metrics"]["rmse"] == 0.0            # PSNR is infinite: dropped, never propagated


def test_a_custom_metrics_function_plugs_in_without_touching_the_loop():
    result = evaluate_on(BicubicUpsampler(4), PairSet(n=2), metrics_fn=lambda sr, hr, mask, bands, scale: {"answer": 42.0})
    assert result["metrics"] == {"answer": 42.0}


def test_reference_metrics_work_end_to_end_inside_evaluate():
    result = evaluate_on(BicubicUpsampler(4), PairSet(n=2, lr_size=16), metrics_fn=reference_metrics)
    assert set(result["metrics"]) == {"psnr_db", "ssim", "rmse", "sam_degrees", "ergas"} and all(v is not None for v in result["metrics"].values())
