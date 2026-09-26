"""The training loop (Phase 4): the smallest thing that proves paired data -> model -> loss -> backward ->
optimizer -> checkpoint -> validation -> a reproducible record. Torch, numpy and the standard library only.

Design
------
* The model is any ``nn.Module`` with ``sr = model(lr)``; the loop never learns which architecture it drives.
* Data comes from any indexable dataset of ``{"lr", "hr", ["lr_mask", "hr_mask"]}`` items (the Phase 3
  `PairedPatchDataset` with ``return_masks=True``). The loop NEVER splits data: the caller hands it the
  train and validation datasets built from the manifest's own split assignments (frame.train.data).
* Batch order is a pure function of ``(seed, micro_step)``: epoch ``e`` uses a permutation drawn from
  ``default_rng([seed, e])``. Resuming therefore continues on exactly the batch an uninterrupted run would
  see next, and a resumed run reproduces the uninterrupted one (tested).
* One optimizer step consumes ``grad_accum`` micro-batches of ``batch_size``. Accumulation raises the effective
  batch size at constant activation memory; it does NOT reduce the activation memory of one micro-batch.
* AMP is opt-in and CUDA-only: on CPU (or bf16 without hardware support) the loop falls back to float32 and
  records why. Losses and validation always run in float32.
* A NaN/Inf loss stops training (`NonFiniteLossError`) before the weights are updated.
* ``deterministic`` seeds every RNG and requests deterministic kernels for the duration of `run` only; it
  claims reproducibility on the same machine and software, not bitwise equality across GPUs (measured in docs/TRAINING.md).
"""

from __future__ import annotations

import contextlib
import json
import math
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence

import numpy as np
import torch

from frame.train import checkpoint as ckpt
from frame.train.config import TrainConfig
from frame.train.errors import CheckpointError, NonFiniteLossError, TrainError
from frame.train.losses import CompositeLoss
from frame.train.models import count_parameters, parameter_megabytes
from frame.train.validate import MetricsFn, evaluate


@dataclass(frozen=True)
class StepResult:
    loss: float
    components: Dict[str, float]
    grad_norm: float
    lr: float
    seconds: float
    samples: int


@contextlib.contextmanager
def determinism(enabled: bool, seed: int) -> Iterator[None]:
    """Seed every RNG and (if ``enabled``) request deterministic kernels, restoring the global switches on exit."""
    saved = (torch.are_deterministic_algorithms_enabled(), torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark)
    random.seed(seed), np.random.seed(seed), torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if enabled:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")      # must precede the first CUDA context to take effect
        torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = True, False
        torch.use_deterministic_algorithms(True, warn_only=True)
    try:
        yield
    finally:
        torch.use_deterministic_algorithms(saved[0])
        torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = saved[1], saved[2]


def _collate(items: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    batch: Dict[str, Any] = {"lr": torch.stack([i["lr"] for i in items]), "hr": torch.stack([i["hr"] for i in items])}
    for key in ("lr_mask", "hr_mask"):
        if key in items[0]:
            batch[key] = torch.stack([i[key] for i in items])
    return batch


class Trainer:
    def __init__(
        self,
        config: TrainConfig,
        model: torch.nn.Module,
        train_data: Any,
        *,
        val_data: Any = None,
        metrics_fn: Optional[MetricsFn] = None,
        manifest_digest: str,
        scale: int = 4,
        band_names: Optional[Sequence[str]] = None,
        git_revision: Optional[str] = None,
        output_dir: Optional[Path] = None,
        notes: Optional[List[str]] = None,
    ):
        if len(train_data) < config.batch_size:
            raise TrainError(f"The training set has {len(train_data)} item(s), smaller than batch_size={config.batch_size}.")
        self.config, self.model, self.train_data, self.val_data = config, model, train_data, val_data
        self.metrics_fn, self.manifest_digest, self.scale = metrics_fn, manifest_digest, scale
        self.band_names, self.git_revision = band_names, git_revision
        self.output_dir = Path(output_dir if output_dir is not None else config.output_dir)
        self.checkpoint_dir = self.output_dir / "checkpoint"
        self.metrics_path = self.output_dir / "metrics.jsonl"
        self.notes: List[str] = list(notes or [])
        self.batches_per_epoch = len(train_data) // config.batch_size          # drop_last: every batch has the same shape
        self.step = 0                                                          # optimizer updates completed
        self.micro_step = 0                                                    # micro-batches consumed
        self.resumed_from_step: Optional[int] = None
        self._prepared = False
        self._perm_cache: Dict[int, np.ndarray] = {}
        self._epoch_told: Optional[int] = None
        self._train_seconds = 0.0
        self._shape_checked = False

    # ------------------------------------------------------------------ setup

    def _resolve_device(self) -> torch.device:
        requested = self.config.device
        if requested == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if requested.startswith("cuda") and not torch.cuda.is_available():
            raise TrainError(f"device={requested!r} was requested but CUDA is not available.")
        return torch.device(requested)

    def _resolve_precision(self) -> None:
        requested = self.config.precision.amp
        self.amp_dtype: Optional[torch.dtype] = None
        self.precision_note: Optional[str] = None
        if requested != "off":
            if self.device.type != "cuda":
                self.precision_note = f"{requested} requested but the device is cpu; using float32."
            elif requested == "bf16" and not torch.cuda.is_bf16_supported():
                self.precision_note = "bf16 requested but this GPU does not support it; using float32."
            else:
                self.amp_dtype = torch.float16 if requested == "fp16" else torch.bfloat16
        self.effective_amp = "off" if self.amp_dtype is None else requested

    def _make_optimizer(self) -> torch.optim.Optimizer:
        c = self.config.optim
        trainable = [p for p in self.model.parameters() if p.requires_grad]
        if not trainable:
            raise TrainError("The model has no trainable parameters.")
        if c.name == "sgd":
            return torch.optim.SGD(trainable, lr=c.lr, momentum=c.momentum, weight_decay=c.weight_decay)
        cls = torch.optim.AdamW if c.name == "adamw" else torch.optim.Adam
        return cls(trainable, lr=c.lr, betas=tuple(c.betas), weight_decay=c.weight_decay)

    def _make_scheduler(self) -> torch.optim.lr_scheduler.LRScheduler:
        s, steps = self.config.scheduler, self.config.steps

        def factor(step: int) -> float:
            if step < s.warmup_steps:
                return 0.1 + 0.9 * step / s.warmup_steps                        # linear ramp from lr / 10
            t = step - s.warmup_steps
            if s.name == "cosine":
                progress = min(1.0, t / max(1, steps - s.warmup_steps))
                return s.min_lr_fraction + (1.0 - s.min_lr_fraction) * 0.5 * (1.0 + math.cos(math.pi * progress))
            if s.name == "step":
                return s.gamma ** (t // s.step_size)
            return 1.0

        return torch.optim.lr_scheduler.LambdaLR(self.optimizer, factor)

    def _prepare(self) -> None:
        if self._prepared:
            return
        self.device = self._resolve_device()
        self._resolve_precision()
        if self.precision_note:
            self.notes.append(self.precision_note)
        self.model.to(self.device)
        self.optimizer = self._make_optimizer()
        self.scheduler = self._make_scheduler()
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.effective_amp == "fp16")
        self.loss_fn = CompositeLoss(self.config.loss)
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)
        self._prepared = True

    def prepare(self) -> None:
        """Resolve the device and build the optimizer/scheduler/scaler/loss (idempotent). Called by `run`; public so a caller can
        evaluate baselines on the training device before training starts."""
        self._prepare()

    def assert_fresh_output(self) -> None:
        """Refuse to start a new run inside a directory that already holds one (nothing is overwritten)."""
        if self.metrics_path.exists() or ckpt.latest_checkpoint(self.checkpoint_dir) is not None:
            raise TrainError(f"{self.output_dir} already contains a run; resume it (--resume) or choose a new output directory. Nothing was overwritten.")

    # ------------------------------------------------------------------ data order

    def _permutation(self, epoch: int) -> np.ndarray:
        if epoch not in self._perm_cache:
            self._perm_cache = {epoch: np.random.default_rng([self.config.seed, epoch]).permutation(len(self.train_data))}
        return self._perm_cache[epoch]

    def batch_indices(self, micro_step: int) -> List[int]:
        """The dataset indices of micro-batch ``micro_step``: a pure function of (seed, micro_step)."""
        epoch, offset = divmod(micro_step, self.batches_per_epoch)
        size = self.config.batch_size
        return [int(i) for i in self._permutation(epoch)[offset * size : (offset + 1) * size]]

    def _next_batch(self) -> Dict[str, torch.Tensor]:
        epoch = self.micro_step // self.batches_per_epoch
        if epoch != self._epoch_told:
            if hasattr(self.train_data, "set_epoch"):
                self.train_data.set_epoch(epoch)
            self._epoch_told = epoch
        batch = _collate([self.train_data[i] for i in self.batch_indices(self.micro_step)])
        self.micro_step += 1
        return batch

    # ------------------------------------------------------------------ one optimizer step

    def train_step(self) -> StepResult:
        self._prepare()
        self.model.train()
        cfg, device = self.config, self.device
        parameters = [p for p in self.model.parameters() if p.requires_grad]
        lr_now = self.optimizer.param_groups[0]["lr"]
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        started = time.perf_counter()

        self.optimizer.zero_grad(set_to_none=True)
        components: Dict[str, float] = {}
        for _ in range(cfg.grad_accum):
            batch = self._next_batch()
            lr, hr = batch["lr"].to(device), batch["hr"].to(device)
            hr_mask = batch["hr_mask"].to(device) if "hr_mask" in batch else None
            lr_mask = batch["lr_mask"].to(device) if "lr_mask" in batch else None
            with torch.autocast(device_type=device.type, dtype=self.amp_dtype, enabled=self.amp_dtype is not None):
                sr = self.model(lr)
            if sr.shape != hr.shape:
                raise TrainError(f"The model output shape {tuple(sr.shape)} != the HR target shape {tuple(hr.shape)}; check model scale vs the data's scale factor.")
            out = self.loss_fn(sr.float(), hr, lr=lr, hr_mask=hr_mask, lr_mask=lr_mask, scale=self.scale)
            if not math.isfinite(out.components["total"]):
                self.micro_step -= 1
                raise NonFiniteLossError(f"Non-finite loss ({out.components['total']}) at step {self.step + 1}; stopping before the weights change.")
            self.scaler.scale(out.total / cfg.grad_accum).backward()
            for key, value in out.components.items():
                components[key] = components.get(key, 0.0) + value / cfg.grad_accum

        self.scaler.unscale_(self.optimizer)
        grad_norm = float(torch.nn.utils.clip_grad_norm_(parameters, cfg.optim.grad_clip_norm if cfg.optim.grad_clip_norm else float("inf")))
        if not math.isfinite(grad_norm) and not self.scaler.is_enabled():
            raise NonFiniteLossError(f"Non-finite gradient norm at step {self.step + 1}.")
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.scheduler.step()
        self.step += 1

        if device.type == "cuda":
            torch.cuda.synchronize(device)
        seconds = time.perf_counter() - started
        self._train_seconds += seconds
        return StepResult(loss=components["total"], components=components, grad_norm=grad_norm, lr=lr_now, seconds=seconds,
                          samples=cfg.batch_size * cfg.grad_accum)

    # ------------------------------------------------------------------ validation

    def validate(self) -> Optional[Dict[str, Any]]:
        """Deterministic evaluation on the validation split (None when there is none). Pure: it logs nothing."""
        if self.val_data is None or len(self.val_data) == 0:
            return None
        self._prepare()
        result = evaluate(self.model, self.val_data, self.loss_fn, device=self.device, scale=self.scale, metrics_fn=self.metrics_fn,
                          band_names=self.band_names)
        return {"type": "val", "step": self.step, **result}

    # ------------------------------------------------------------------ recording

    def _log(self, record: Dict[str, Any]) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        with open(self.metrics_path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")

    def _truncate_log(self, keep_through_step: int) -> None:
        if not self.metrics_path.exists():
            return
        kept = [line for line in self.metrics_path.read_text(encoding="utf-8").splitlines() if line.strip() and json.loads(line).get("step", 0) <= keep_through_step]
        self.metrics_path.write_text("".join(line + "\n" for line in kept), encoding="utf-8")

    def _peak_vram_mib(self) -> Optional[float]:
        return torch.cuda.max_memory_allocated(self.device) / 2**20 if self.device.type == "cuda" else None

    def _save_checkpoint(self) -> Path:
        path = ckpt.save_checkpoint(
            self.checkpoint_dir / ckpt.checkpoint_name(self.step), model=self.model, optimizer=self.optimizer, scheduler=self.scheduler,
            scaler=self.scaler, step=self.step, micro_step=self.micro_step, config=self.config, manifest_digest=self.manifest_digest,
            git_revision=self.git_revision,
        )
        ckpt.prune_checkpoints(self.checkpoint_dir, self.config.keep_last_checkpoints)
        return path

    # ------------------------------------------------------------------ the loop

    def run(self, resume: Optional[Path] = None) -> Dict[str, Any]:
        cfg = self.config
        with determinism(cfg.deterministic, cfg.seed):
            self._prepare()
            if resume is not None:
                payload = ckpt.load_checkpoint(resume)
                ckpt.check_resumable(payload, cfg, self.manifest_digest)
                ckpt.apply_checkpoint(payload, self.model, self.optimizer, self.scheduler, self.scaler)
                self.step, self.micro_step = int(payload["step"]), int(payload["micro_step"])
                self.resumed_from_step = self.step
                self._truncate_log(self.step)
            else:
                self.assert_fresh_output()
            if self.step >= cfg.steps:
                raise TrainError(f"Nothing to do: the run is already at step {self.step} of {cfg.steps}.")

            last: Optional[StepResult] = None
            validation: Optional[Dict[str, Any]] = None
            while self.step < cfg.steps:
                last = self.train_step()
                if self.step % cfg.log_every == 0 or self.step == cfg.steps:
                    self._log({
                        "type": "train", "step": self.step, "micro_step": self.micro_step, "epoch": round(self.micro_step / self.batches_per_epoch, 4),
                        "loss": last.components, "lr": last.lr, "grad_norm": last.grad_norm, "step_seconds": round(last.seconds, 5),
                        "samples_per_second": round(last.samples / last.seconds, 3) if last.seconds > 0 else None,
                        "peak_vram_mib": self._peak_vram_mib(), "amp_scale": self.scaler.get_scale() if self.scaler.is_enabled() else None,
                    })
                if cfg.validate_every and self.step % cfg.validate_every == 0 and self.step != cfg.steps:
                    validation = self.validate()
                    if validation:
                        self._log(validation)
                if cfg.checkpoint_every and self.step % cfg.checkpoint_every == 0 and self.step != cfg.steps:
                    self._save_checkpoint()

            validation = self.validate()
            if validation:
                self._log(validation)
            self._save_checkpoint()

        steps_this_session = self.step - (self.resumed_from_step or 0)
        return {
            "steps": self.step, "micro_steps": self.micro_step, "epochs_completed": round(self.micro_step / self.batches_per_epoch, 4),
            "train_seconds": round(self._train_seconds, 4), "mean_step_seconds": round(self._train_seconds / max(1, steps_this_session), 5),
            "final_train_loss": last.components["total"] if last else None, "device": str(self.device),
            "parameters": {"total": count_parameters(self.model), "trainable": count_parameters(self.model, trainable_only=True),
                           "megabytes": round(parameter_megabytes(self.model), 4)},
            "peak_vram_mib": self._peak_vram_mib(),
            "precision": {"requested": cfg.precision.amp, "effective": self.effective_amp, "note": self.precision_note},
            "validation": validation, "resumed_from_step": self.resumed_from_step, "manifest_digest": self.manifest_digest, "seed": cfg.seed,
            "effective_batch_size": cfg.effective_batch_size, "notes": self.notes,
        }
