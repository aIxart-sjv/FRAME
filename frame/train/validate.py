"""Deterministic validation (Phase 4): evaluate any ``sr = model(lr)`` callable on a paired dataset.

* Patch by patch (batch of 1), ``model.eval()``, ``torch.no_grad()``, float32, no augmentation, in dataset
  order: the same model on the same data always gives the same numbers, and a bicubic baseline goes through
  exactly the same code as the trained model.
* Reports the (evaluation-side) loss with the SAME composite the model trained on, plus metrics computed on
  the valid pixels only. Metrics are per-patch means; they are pluggable (``metrics_fn``) so later phases can add
  SAM/ERGAS-style analyses without touching the loop.
* Torch only. The default ``basic_metrics`` (RMSE, PSNR) needs nothing else; `reference_metrics` adds SSIM, SAM
  and ERGAS by delegating to frame.validation.compute_reference_metrics (main environment only, imported lazily),
  so evaluation numbers match the rest of FRAME's validation and are never re-implemented here.
* These are TRAINING-LOOP diagnostics on the validation split of the training manifest. They are not a
  benchmark: nothing here touches SEN2NEON / OpenSR-Test, and a good number is not evidence about real imagery.
"""

from __future__ import annotations

import math
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np
import torch

MetricsFn = Callable[[torch.Tensor, torch.Tensor, np.ndarray, Sequence[str], int], Dict[str, Optional[float]]]


def basic_metrics(sr: torch.Tensor, hr: torch.Tensor, mask: np.ndarray, band_names: Sequence[str], scale: int) -> Dict[str, Optional[float]]:
    """Masked RMSE and PSNR (data range 1, the reflectance convention used throughout FRAME). (C, H, W) tensors."""
    valid = torch.from_numpy(np.asarray(mask, dtype=bool))
    if not bool(valid.any()):
        return {"rmse": None, "psnr_db": None}
    mse = float(((sr.double() - hr.double()) ** 2)[:, valid].mean())
    return {"rmse": math.sqrt(mse), "psnr_db": 10.0 * math.log10(1.0 / mse) if mse > 0 else float("inf")}


def reference_metrics(sr: torch.Tensor, hr: torch.Tensor, mask: np.ndarray, band_names: Sequence[str], scale: int) -> Dict[str, Optional[float]]:
    """PSNR, SSIM, RMSE, SAM (degrees) and ERGAS via frame.validation.compute_reference_metrics."""
    from frame.validation import compute_reference_metrics

    m = compute_reference_metrics(sr, hr, mask, band_names=list(band_names), scale_factor=scale)
    return {"psnr_db": m.psnr_db, "ssim": m.ssim, "rmse": m.rmse, "sam_degrees": m.sam_degrees, "ergas": m.ergas}


def _mean(values: List[Optional[float]]) -> Optional[float]:
    finite = [v for v in values if v is not None and math.isfinite(v)]
    return float(sum(finite) / len(finite)) if finite else None


def _mask(item: Dict[str, Any], key: str, like: torch.Tensor) -> torch.Tensor:
    return item[key] if key in item else torch.ones(like.shape[-2:], dtype=torch.bool)


@torch.no_grad()
def evaluate(
    model: Callable[[torch.Tensor], torch.Tensor],
    dataset: Any,
    loss_fn: Callable[..., Any],
    *,
    device: torch.device,
    scale: int,
    metrics_fn: Optional[MetricsFn] = None,
    band_names: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """Loss and metrics of ``model`` over every item of ``dataset`` (see the module docstring)."""
    metrics_fn = metrics_fn or basic_metrics
    was_training = getattr(model, "training", False)
    if hasattr(model, "eval"):
        model.eval()
    losses: Dict[str, List[float]] = {}
    metrics: Dict[str, List[Optional[float]]] = {}
    evaluated = skipped = 0
    try:
        for index in range(len(dataset)):
            item = dataset[index]
            lr = item["lr"][None].to(device)
            hr = item["hr"][None].to(device)
            hr_mask, lr_mask = _mask(item, "hr_mask", item["hr"]), _mask(item, "lr_mask", item["lr"])
            if not bool(hr_mask.any()):
                skipped += 1
                continue
            sr = model(lr).float()
            out = loss_fn(sr, hr, lr=lr, hr_mask=hr_mask[None].to(device), lr_mask=lr_mask[None].to(device), scale=scale)
            for key, value in out.components.items():
                losses.setdefault(key, []).append(value)
            names = list(band_names) if band_names is not None else [f"band{i}" for i in range(hr.shape[1])]
            for key, value in metrics_fn(sr[0].cpu(), hr[0].cpu(), hr_mask.numpy(), names, scale).items():
                metrics.setdefault(key, []).append(value)
            evaluated += 1
    finally:
        if was_training and hasattr(model, "train"):
            model.train()
    return {
        "n_patches": evaluated,
        "skipped_patches": skipped,
        "loss": {key: _mean(values) for key, values in losses.items()},
        "metrics": {key: _mean(values) for key, values in metrics.items()},
    }
