"""Step 2 of the Mamba fine-tuning smoke (MAMBA environment): a real, tiny fine-tune of the pretrained SEN2SR-Mamba through frame.train.

    sen2sr_mamba_venv/bin/python experiments/training/mamba_finetune_smoke/run_mamba_smoke.py [--steps 20 --resume-to 30]

The same `Trainer` that trains the smoke CNN drives ``MambaSR`` (13,759,444 parameters, pretrained weights, activation checkpointing, fp32,
batch 1, AdamW, L1) on the patches exported by ``export_patches.py``. It records: memory and speed, the training loss, validation loss/RMSE/PSNR
of the pretrained model BEFORE fine-tuning, of the fine-tuned model, and of bicubic on the same validation patches, and it checks that a
checkpoint of the real model saves and resumes. The checkpoint stays outside the repository (it is ~165 MiB).

This is a capability and feasibility check on synthetic data. It is NOT a fine-tuning result: a few dozen steps on 32 synthetic patches say
nothing about real imagery, and the model is the pretrained SEN2SR-Mamba WITHOUT its low-frequency hard constraint (the constraint's mask is
fixed at 512x512 and cannot be applied to 64x64-crop training).
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2]))

from frame.train import checkpoint as ckpt  # noqa: E402
from frame.train.baselines import build_baselines  # noqa: E402
from frame.train.config import TrainConfig  # noqa: E402
from frame.train.models import build_model  # noqa: E402
from frame.train.record import environment_info  # noqa: E402
from frame.train.trainer import Trainer  # noqa: E402
from frame.train.validate import evaluate  # noqa: E402

DESKTOP_RESERVE_MIB = 350


class TensorPairs:
    def __init__(self, lr: torch.Tensor, hr: torch.Tensor):
        self.lr, self.hr = lr, hr

    def __len__(self) -> int:
        return len(self.lr)

    def __getitem__(self, i: int):
        return {"lr": self.lr[i], "hr": self.hr[i]}


def make_config(steps: int, output_dir: Path, lr_patch: int) -> TrainConfig:
    return TrainConfig.from_dict({
        "name": "mamba_finetune_smoke", "model": {"name": "sen2sr_mamba", "params": {"activation_checkpointing": True}, "pretrained": True},
        "data": {"manifest": "exported-patches", "lr_patch": lr_patch, "val_lr_patch": 128}, "steps": steps, "batch_size": 1, "grad_accum": 2, "seed": 0,
        "optim": {"name": "adamw", "lr": 1e-5}, "loss": {"reconstruction": "l1"}, "precision": {"amp": "off"}, "checkpoint_every": 10, "validate_every": 10,
        "log_every": 1, "device": "cuda", "deterministic": False, "output_dir": str(output_dir),
    })


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--patches", default=str(Path.home() / ".cache" / "frame_data" / "exports" / "mamba_smoke_patches.pt"))
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--resume-to", type=int, default=30)
    parser.add_argument("--work-dir", default=str(Path.home() / ".cache" / "frame_tmp" / "mamba_finetune_smoke"))
    args = parser.parse_args()

    payload = torch.load(args.patches, weights_only=True)
    free, total = torch.cuda.mem_get_info()
    torch.cuda.set_per_process_memory_fraction((free / 2**20 - DESKTOP_RESERVE_MIB) * 2**20 / total)
    work = Path(args.work_dir)
    shutil.rmtree(work, ignore_errors=True)

    train = TensorPairs(payload["train"]["lr"], payload["train"]["hr"])
    val = TensorPairs(payload["val"]["lr"], payload["val"]["hr"])
    scale, digest = payload["scale"], payload["manifest_digest"]

    def build(steps: int) -> Trainer:
        config = make_config(steps, work, payload["lr_patch"])
        return Trainer(config, build_model(config.model, seed=config.seed), train, val_data=val, manifest_digest=digest, scale=scale,
                       band_names=payload["bands"], output_dir=work)

    first = build(args.steps)
    first.prepare()
    step0 = first.validate()
    bicubic = evaluate(build_baselines(("bicubic",), device=first.device, scale=scale)["bicubic"], val, first.loss_fn, device=first.device, scale=scale,
                       band_names=payload["bands"])
    torch.cuda.reset_peak_memory_stats()
    part1 = first.run()
    del first
    torch.cuda.empty_cache()

    second = build(args.resume_to)                                    # a NEW process-level object resuming the real model's checkpoint
    part2 = second.run(resume=ckpt.latest_checkpoint(work / "checkpoint"))
    losses = [json.loads(line) for line in (work / "metrics.jsonl").read_text().splitlines()]
    train_loss = [m["loss"]["total"] for m in losses if m["type"] == "train"]
    steady = [m["step_seconds"] for m in losses if m["type"] == "train"][1:]
    checkpoint_mib = round(ckpt.latest_checkpoint(work / "checkpoint").stat().st_size / 2**20, 1)

    summary = {
        "what": "Tiny fine-tune of the pretrained SEN2SR-Mamba (no hard constraint) through frame.train on exported SYNTHETIC patches; a feasibility check, not a result.",
        "model": {"name": "sen2sr_mamba", "parameters": part1["parameters"], "activation_checkpointing": True, "pretrained": True},
        "data": {"train_patches": len(train), "lr_patch": payload["lr_patch"], "val_patches": len(val), "val_lr_patch": 128, "train_scenes": payload["train_scenes"],
                 "val_scenes": payload["val_scenes"], "ignored_records": payload["ignored_records"], "manifest_digest": digest, "synthetic": True},
        "config": {"batch_size": 1, "grad_accum": 2, "optimizer": "adamw", "lr": 1e-5, "loss": "l1", "precision": "off (float32)", "steps_first_session": args.steps,
                   "steps_total": args.resume_to},
        "results": {"validation_before_finetuning": step0, "validation_after_finetuning": part2["validation"], "bicubic_on_same_patches": bicubic},
        "training": {"first_loss": train_loss[0], "last_loss": train_loss[-1], "steps_completed": part2["steps"], "resumed_from_step": part2["resumed_from_step"],
                     "mean_step_seconds_steady": round(sum(steady) / len(steady), 3), "peak_vram_mib_second_session": part2["peak_vram_mib"],
                     "peak_vram_mib_first_session": part1["peak_vram_mib"], "checkpoint_mib": checkpoint_mib},
        "environment": environment_info("cuda"),
    }
    (HERE / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    shutil.copy(work / "metrics.jsonl", HERE / "metrics.jsonl")
    m = lambda block: {k: round(v, 4) if v is not None else None for k, v in block["metrics"].items()}
    print(f"steps {part2['steps']} (resumed from {part2['resumed_from_step']}), peak VRAM {part1['peak_vram_mib']:.0f}/{part2['peak_vram_mib']:.0f} MiB, {summary['training']['mean_step_seconds_steady']} s/step")
    print("val before :", m(step0), "| loss", round(step0["loss"]["total"], 5))
    print("val after  :", m(part2["validation"]), "| loss", round(part2["validation"]["loss"]["total"], 5))
    print("bicubic    :", m(bicubic), "| loss", round(bicubic["loss"]["total"], 5))
    print(f"train loss {train_loss[0]:.4f} -> {train_loss[-1]:.4f}; checkpoint {checkpoint_mib} MiB (kept outside the repo in {work})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
