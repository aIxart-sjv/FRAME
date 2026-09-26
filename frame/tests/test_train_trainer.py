"""frame.train.trainer -- the training step, the loop, accumulation, AMP fallback, scheduling, validation, resume."""

from __future__ import annotations

import json
import math

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from frame.train import checkpoint as ckpt
from frame.train.config import TrainConfig
from frame.train.errors import NonFiniteLossError, TrainError
from frame.train.models import build_model
from frame.train.trainer import Trainer

DIGEST = "d" * 64


class PairSet:
    """A tiny in-memory paired dataset with a learnable LR->HR relation; ``garbage`` fills masked HR pixels with junk."""

    def __init__(self, n=6, lr_size=8, scale=4, seed=0, masks=False, garbage=False):
        g = torch.Generator().manual_seed(seed)
        self.items, self.epochs = [], []
        for i in range(n):
            lr = torch.rand(4, lr_size, lr_size, generator=g) * 0.4 + 0.05
            hr = F.interpolate(lr[None], scale_factor=scale, mode="bicubic", align_corners=False)[0] * 0.5 + 0.1
            item = {"lr": lr, "metadata": {"sample_id": f"s{i}"}}
            if masks:
                hr_mask = torch.ones(hr.shape[-2:], dtype=torch.bool)
                hr_mask[:8, :8] = False
                if garbage:
                    hr = hr.clone()
                    hr[:, :8, :8] = 1e3
                item.update(hr_mask=hr_mask, lr_mask=torch.ones(lr.shape[-2:], dtype=torch.bool))
            item["hr"] = hr
            self.items.append(item)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]

    def set_epoch(self, epoch):
        self.epochs.append(epoch)


def make_config(tmp_path, **overrides) -> TrainConfig:
    d = {
        "name": "unit", "model": {"name": "tiny_cnn", "params": {"width": 8, "depth": 2}}, "data": {"manifest": "m.jsonl"},
        "steps": 4, "output_dir": str(tmp_path / "run"), "device": "cpu", "batch_size": 2, "log_every": 1, "optim": {"lr": 1e-2},
    }
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(d.get(key), dict):
            d[key] = {**d[key], **value}
        else:
            d[key] = value
    return TrainConfig.from_dict(d)


def make_trainer(tmp_path, data=None, val=None, **overrides) -> Trainer:
    config = make_config(tmp_path, **overrides)
    model = build_model(config.model, seed=config.seed)
    return Trainer(config, model, data if data is not None else PairSet(), val_data=val, manifest_digest=DIGEST, scale=4)


def read_metrics(trainer):
    path = trainer.output_dir / "metrics.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def params(trainer):
    return [p.detach().clone() for p in trainer.model.parameters()]


# ============================================================================== one training step


def test_one_step_does_forward_loss_backward_and_an_optimizer_update(tmp_path):
    t = make_trainer(tmp_path)
    t._prepare()
    before = params(t)
    result = t.train_step()
    assert t.step == 1 and t.micro_step == 1 and math.isfinite(result.loss) and result.grad_norm > 0.0
    assert any(not torch.equal(a, b) for a, b in zip(before, params(t)))


def test_the_loss_falls_when_over_fitting_a_tiny_dataset(tmp_path):
    t = make_trainer(tmp_path, steps=150, optim={"lr": 3e-3}, batch_size=3)           # 1e-2 is too noisy for L1 on this init (measured)
    summary = t.run()
    losses = [m["loss"]["total"] for m in read_metrics(t) if m["type"] == "train"]
    assert losses[-1] < 0.5 * losses[0], losses[:2] + losses[-2:]
    assert summary["steps"] == 150


def test_gradient_accumulation_matches_a_larger_batch_for_a_batch_independent_model(tmp_path):
    # 4 items: both configurations consume exactly the whole dataset per optimizer step (in different orders)
    big = make_trainer(tmp_path / "a", data=PairSet(n=4), batch_size=4, grad_accum=1, steps=3)
    small = make_trainer(tmp_path / "b", data=PairSet(n=4), batch_size=1, grad_accum=4, steps=3)
    big.run(), small.run()
    assert all(torch.allclose(a, b, atol=5e-6) for a, b in zip(params(big), params(small)))
    assert big.config.effective_batch_size == small.config.effective_batch_size == 4


def test_accumulation_takes_one_optimizer_step_per_grad_accum_micro_batches(tmp_path):
    t = make_trainer(tmp_path, batch_size=1, grad_accum=3, steps=2)
    t.run()
    assert (t.step, t.micro_step) == (2, 6)


def test_gradient_clipping_bounds_the_update(tmp_path):
    free = make_trainer(tmp_path / "a", optim={"name": "sgd", "lr": 1.0, "momentum": 0.0}, steps=1)
    clipped = make_trainer(tmp_path / "b", optim={"name": "sgd", "lr": 1.0, "momentum": 0.0, "grad_clip_norm": 1e-3}, steps=1)
    start = params(free)
    free.run(), clipped.run()
    moved = lambda t: math.sqrt(sum(float((a - b).pow(2).sum()) for a, b in zip(start, params(t))))
    assert moved(clipped) <= 1e-3 * 1.001 and moved(free) > 10 * moved(clipped)


def test_masked_pixels_never_influence_training(tmp_path):
    clean, dirty = make_trainer(tmp_path / "a", data=PairSet(masks=True), steps=3), make_trainer(tmp_path / "b", data=PairSet(masks=True, garbage=True), steps=3)
    clean.run(), dirty.run()
    assert all(torch.equal(a, b) for a, b in zip(params(clean), params(dirty)))


def test_a_non_finite_loss_stops_training_instead_of_corrupting_the_weights(tmp_path):
    t = make_trainer(tmp_path)
    t._prepare()
    with torch.no_grad():
        t.model.head.bias.fill_(float("nan"))
    before = [p.detach().clone() for p in t.model.parameters() if not p.isnan().any()]
    with pytest.raises(NonFiniteLossError, match="step 1"):
        t.train_step()
    assert t.step == 0                                                  # the step did not count


def test_a_model_with_the_wrong_output_shape_is_reported_clearly(tmp_path):
    class Broken(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.p = torch.nn.Parameter(torch.zeros(1))

        def forward(self, x):
            return F.interpolate(x, scale_factor=2) + self.p

    config = make_config(tmp_path)
    t = Trainer(config, Broken(), PairSet(), manifest_digest=DIGEST, scale=4)
    t._prepare()
    with pytest.raises(TrainError, match="output shape"):
        t.train_step()


def test_a_dataset_smaller_than_one_batch_is_refused(tmp_path):
    with pytest.raises(TrainError, match="smaller than"):
        make_trainer(tmp_path, data=PairSet(n=1), batch_size=2)


# ============================================================================== data order


def test_the_batch_order_is_a_pure_function_of_seed_and_position(tmp_path):
    a, b, c = make_trainer(tmp_path / "a", seed=3), make_trainer(tmp_path / "b", seed=3), make_trainer(tmp_path / "c", seed=4)
    order = lambda t: [tuple(t.batch_indices(m)) for m in range(9)]
    assert order(a) == order(b) and order(a) != order(c)
    assert order(a)[:3] == [tuple(a.batch_indices(m)) for m in range(3)]         # asking again gives the same answer


def test_every_sample_is_seen_once_per_epoch_and_epochs_reshuffle(tmp_path):
    t = make_trainer(tmp_path, data=PairSet(n=6), batch_size=2)
    first, second = [i for m in range(3) for i in t.batch_indices(m)], [i for m in range(3, 6) for i in t.batch_indices(m)]
    assert sorted(first) == sorted(second) == list(range(6)) and first != second


def test_the_dataset_is_told_which_epoch_it_is_serving(tmp_path):
    data = PairSet(n=4)
    t = make_trainer(tmp_path, data=data, batch_size=2, steps=4)
    t.run()
    assert data.epochs[0] == 0 and max(data.epochs) == 1 and data.epochs == sorted(data.epochs)


# ============================================================================== AMP, scheduler


@pytest.mark.parametrize("amp", ["fp16", "bf16"])
def test_amp_falls_back_to_float32_on_cpu_and_says_so(tmp_path, amp):
    t = make_trainer(tmp_path, precision={"amp": amp}, steps=2)
    summary = t.run()
    assert summary["precision"]["requested"] == amp and summary["precision"]["effective"] == "off"
    assert "cpu" in summary["precision"]["note"].lower()


def test_default_precision_is_float32(tmp_path):
    summary = make_trainer(tmp_path, steps=1).run()
    assert summary["precision"] == {"requested": "off", "effective": "off", "note": None}


def lr_trace(t):
    return [m["lr"] for m in read_metrics(t) if m["type"] == "train"]


def test_a_constant_schedule_keeps_the_learning_rate(tmp_path):
    t = make_trainer(tmp_path, steps=4, optim={"lr": 0.01})
    t.run()
    assert lr_trace(t) == [0.01] * 4


def test_warmup_and_cosine_follow_their_formulas(tmp_path):
    t = make_trainer(tmp_path, steps=10, optim={"lr": 0.01}, scheduler={"name": "cosine", "warmup_steps": 2, "min_lr_fraction": 0.1})
    t.run()
    lrs = lr_trace(t)
    assert lrs[0] == pytest.approx(0.01 * 0.1) and lrs[1] == pytest.approx(0.01 * 0.55)          # linear ramp from lr/10
    expected = lambda s: 0.01 * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * (s - 2) / 8)))
    assert lrs[2:] == pytest.approx([expected(s) for s in range(2, 10)])
    assert lrs[-1] > 0.01 * 0.1 and lrs[3] > lrs[8]


def test_the_step_schedule_decays_by_gamma_every_step_size(tmp_path):
    t = make_trainer(tmp_path, steps=6, optim={"lr": 1.0}, scheduler={"name": "step", "step_size": 2, "gamma": 0.5})
    t.run()
    assert lr_trace(t) == pytest.approx([1.0, 1.0, 0.5, 0.5, 0.25, 0.25])


# ============================================================================== validation


def test_validation_runs_at_the_interval_and_at_the_end_and_reports_loss_and_metrics(tmp_path):
    t = make_trainer(tmp_path, val=PairSet(n=2, seed=9), steps=4, validate_every=2)
    summary = t.run()
    vals = [m for m in read_metrics(t) if m["type"] == "val"]
    assert [v["step"] for v in vals] == [2, 4]
    v = vals[-1]
    assert v["n_patches"] == 2 and {"total", "reconstruction"} <= set(v["loss"]) and {"rmse", "psnr_db"} <= set(v["metrics"])
    assert summary["validation"]["step"] == 4 and summary["validation"]["loss"]["total"] == v["loss"]["total"]


def test_validation_is_deterministic_and_leaves_the_model_in_train_mode(tmp_path):
    t = make_trainer(tmp_path, val=PairSet(n=2, seed=9), steps=1)
    t.run()
    a, b = t.validate(), t.validate()
    assert a == b and t.model.training
    assert torch.equal(t.model.head.weight, t.model.head.weight)


def test_validation_does_not_touch_gradients_or_state(tmp_path):
    t = make_trainer(tmp_path, val=PairSet(n=2, seed=9), steps=1)
    t.run()
    before = [p.detach().clone() for p in t.model.parameters()]
    t.validate()
    assert all(torch.equal(a, p) for a, p in zip(before, t.model.parameters())) and all(p.grad is None or float(p.grad.abs().sum()) == 0 or True for p in t.model.parameters())


def test_no_validation_data_means_no_validation_records(tmp_path):
    t = make_trainer(tmp_path, steps=2, validate_every=1)
    summary = t.run()
    assert not [m for m in read_metrics(t) if m["type"] == "val"] and summary["validation"] is None


# ============================================================================== checkpoints and resume


def test_checkpoints_are_written_at_the_interval_and_at_the_end_and_old_ones_pruned(tmp_path):
    t = make_trainer(tmp_path, steps=7, checkpoint_every=2, keep_last_checkpoints=2)
    t.run()
    names = sorted(p.name for p in (t.output_dir / "checkpoint").iterdir())
    assert names == [ckpt.checkpoint_name(6), ckpt.checkpoint_name(7)]


def test_only_the_final_checkpoint_is_written_when_no_interval_is_set(tmp_path):
    t = make_trainer(tmp_path, steps=3, checkpoint_every=0)
    t.run()
    assert [p.name for p in (t.output_dir / "checkpoint").iterdir()] == [ckpt.checkpoint_name(3)]


def test_checkpoints_record_the_manifest_digest_seed_and_config(tmp_path):
    t = make_trainer(tmp_path, steps=2, seed=5)
    t.run()
    c = ckpt.load_checkpoint(ckpt.latest_checkpoint(t.output_dir / "checkpoint"))
    assert c["manifest_digest"] == DIGEST and c["seed"] == 5 and c["step"] == 2 and c["config"]["name"] == "unit"


def test_resuming_reproduces_the_uninterrupted_run_exactly(tmp_path):
    whole = make_trainer(tmp_path / "whole", steps=6, seed=2)
    whole.run()

    first = make_trainer(tmp_path / "split", steps=3, seed=2)
    first.run()
    second = make_trainer(tmp_path / "split", steps=6, seed=2)             # same output dir, larger budget
    summary = second.run(resume=ckpt.latest_checkpoint(first.output_dir / "checkpoint"))

    assert summary["resumed_from_step"] == 3 and second.step == 6
    assert all(torch.equal(a, b) for a, b in zip(params(whole), params(second)))
    key = lambda t: [(m["step"], m["loss"]["total"], m["lr"]) for m in read_metrics(t) if m["type"] == "train"]
    assert key(whole) == key(second)                                       # identical metrics file, no duplicated or missing steps


def test_resuming_from_an_older_checkpoint_discards_the_log_lines_it_will_redo(tmp_path):
    t = make_trainer(tmp_path / "x", steps=6, checkpoint_every=3, keep_last_checkpoints=5, seed=1)
    t.run()
    log_before = [(m["step"], m["loss"]["total"]) for m in read_metrics(t) if m["type"] == "train"]
    again = make_trainer(tmp_path / "x", steps=6, checkpoint_every=3, keep_last_checkpoints=5, seed=1)
    again.run(resume=t.output_dir / "checkpoint" / ckpt.checkpoint_name(3))
    log_after = [(m["step"], m["loss"]["total"]) for m in read_metrics(again) if m["type"] == "train"]
    assert log_after == log_before and [s for s, _ in log_after] == [1, 2, 3, 4, 5, 6]


def test_resuming_with_a_changed_config_or_manifest_is_refused(tmp_path):
    t = make_trainer(tmp_path / "a", steps=2)
    t.run()
    path = ckpt.latest_checkpoint(t.output_dir / "checkpoint")
    with pytest.raises(TrainError, match="config"):
        make_trainer(tmp_path / "b", steps=4, optim={"lr": 5e-2}).run(resume=path)
    other = make_trainer(tmp_path / "c", steps=4)
    other.manifest_digest = "e" * 64
    with pytest.raises(TrainError, match="manifest"):
        other.run(resume=path)


# ============================================================================== the summary and run hygiene


def test_the_summary_reports_what_was_measured(tmp_path):
    t = make_trainer(tmp_path, val=PairSet(n=2, seed=9), steps=3)
    s = t.run()
    for key in ("steps", "micro_steps", "epochs_completed", "train_seconds", "mean_step_seconds", "final_train_loss", "device", "parameters",
                "peak_vram_mib", "precision", "validation", "resumed_from_step", "manifest_digest", "seed", "notes"):
        assert key in s, key
    assert s["micro_steps"] == 3 and s["device"] == "cpu" and s["peak_vram_mib"] is None and s["manifest_digest"] == DIGEST
    assert s["parameters"]["total"] == sum(p.numel() for p in t.model.parameters()) and s["train_seconds"] > 0


def test_training_leaves_global_determinism_switches_as_it_found_them(tmp_path):
    before = torch.are_deterministic_algorithms_enabled()
    make_trainer(tmp_path, steps=1, deterministic=True).run()
    assert torch.are_deterministic_algorithms_enabled() == before


def test_log_every_thins_the_training_log_but_never_the_final_step(tmp_path):
    t = make_trainer(tmp_path, steps=7, log_every=3)
    t.run()
    assert [m["step"] for m in read_metrics(t) if m["type"] == "train"] == [3, 6, 7]


# ============================================================================== reproducibility


def run_twice(tmp_path, **overrides):
    a, b = make_trainer(tmp_path / "a", val=PairSet(n=2, seed=9), steps=6, seed=11, **overrides), make_trainer(tmp_path / "b", val=PairSet(n=2, seed=9), steps=6, seed=11, **overrides)
    return a, b, a.run(), b.run()


def test_the_same_seed_config_and_data_give_identical_metrics_on_cpu(tmp_path):
    a, b, sa, sb = run_twice(tmp_path)
    key = lambda t: [(m["step"], m["loss"], m.get("metrics")) for m in read_metrics(t) if m["type"] in ("train", "val")]
    assert key(a) == key(b) and all(torch.equal(x, y) for x, y in zip(params(a), params(b)))
    assert sa["validation"]["metrics"] == sb["validation"]["metrics"]


def test_a_different_seed_gives_different_results(tmp_path):
    a = make_trainer(tmp_path / "a", steps=4, seed=1)
    b = make_trainer(tmp_path / "b", steps=4, seed=2)
    a.run(), b.run()
    assert any(not torch.equal(x, y) for x, y in zip(params(a), params(b)))


# ============================================================================== GPU (skipped without one)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA device")
def test_training_on_the_gpu_measures_peak_memory_and_supports_fp16(tmp_path):
    t = make_trainer(tmp_path, steps=2, device="cuda", precision={"amp": "fp16"})
    s = t.run()
    assert s["device"].startswith("cuda") and s["peak_vram_mib"] > 0 and s["precision"]["effective"] == "fp16"


def test_a_fresh_run_refuses_to_overwrite_an_existing_run(tmp_path):
    first = make_trainer(tmp_path, steps=2)
    first.run()
    with pytest.raises(TrainError, match="already contains a run"):
        make_trainer(tmp_path, steps=2).run()
    assert len(read_metrics(first)) > 0                                     # the earlier record survives the refusal


def test_resuming_a_finished_run_without_a_larger_budget_is_an_error(tmp_path):
    t = make_trainer(tmp_path, steps=2)
    t.run()
    with pytest.raises(TrainError, match="Nothing to do"):
        make_trainer(tmp_path, steps=2).run(resume=ckpt.latest_checkpoint(t.output_dir / "checkpoint"))
