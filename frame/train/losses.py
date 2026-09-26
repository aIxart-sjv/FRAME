"""Training losses (Phase 4): one reconstruction loss plus two optional, independently weighted terms.

    L_total = L_reconstruction  +  lambda_spectral * L_spectral  +  lambda_consistency * L_consistency

These are TRAINING objectives. They are deliberately separate from the evaluation metrics
(frame.validation: PSNR, SSIM, RMSE, SAM, ERGAS) -- a training term that pushes a number down does not
prove that the number it resembles is good on real data.

Requirements grounding (docs/Requirements 142.txt, Requirement 4 sections 33 and 36, Requirement 5
sections 24-25): the baseline objective is a pixel reconstruction loss, optionally with a spectral term;
perceptual / adversarial terms are excluded because in Earth observation "perceptual hallucination can be
dangerous". The SEN2SR low-frequency HARD CONSTRAINT (a model-side mechanism) is not a loss and is not
implemented here.

Conventions
-----------
* Tensors are (B, C, H, W). An optional boolean ``mask`` is (B, H, W) and selects the pixels that count;
  masked pixels contribute nothing to the value or the gradient. A batch with no valid pixel gives a zero
  loss with zero (finite) gradients, never NaN.
* Every loss is computed in float32, whatever the model's autocast dtype.
* Each function is a pure function of its arguments and is tested alone (test_train_losses.py).

Terms
-----
l1_loss             mean absolute error over valid pixels and channels.
charbonnier_loss    mean of sqrt(d^2 + eps^2) - eps: a smooth L1 that is exactly 0 for identical inputs
                    (the plain Charbonnier form bottoms out at eps) and has a finite gradient at 0.
spectral_angle_loss mean over valid pixels of (1 - cos(theta)), theta = the angle between the predicted and
                    the true spectral vector. It is 0 for identical spectral DIRECTION (brightness-invariant),
                    1 for orthogonal spectra, and smooth at 0 -- unlike arccos, whose gradient is infinite
                    there. A monotone surrogate of the SAM angle, not SAM itself.
consistency_loss    L1 between the area-average downsampling of the SR output and the LR input (the
                    requirements' primary reduction operator). CAUTION: it presumes the LR really is an area
                    average of the HR. That holds for synthetic pairs, not for cross-sensor pairs, so its
                    weight defaults to 0 and it should only be used deliberately.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from frame.train.config import LossConfig
from frame.train.errors import LossInputError

_EPS = 1e-8


def _check_pair(pred: torch.Tensor, target: torch.Tensor, mask: Optional[torch.Tensor], what: str = "prediction") -> None:
    if pred.ndim != 4 or target.ndim != 4:
        raise LossInputError(f"Expected 4-D (B, C, H, W) tensors, got {what}={tuple(pred.shape)} and target={tuple(target.shape)}.")
    if pred.shape != target.shape:
        raise LossInputError(f"{what} shape {tuple(pred.shape)} != target shape {tuple(target.shape)}.")
    _check_mask(mask, pred)


def _check_mask(mask: Optional[torch.Tensor], like: torch.Tensor) -> None:
    if mask is not None and tuple(mask.shape) != (like.shape[0], like.shape[2], like.shape[3]):
        raise LossInputError(f"mask shape {tuple(mask.shape)} != (B, H, W) = {(like.shape[0], like.shape[2], like.shape[3])}.")


def _masked_mean(per_pixel: torch.Tensor, mask: Optional[torch.Tensor], channels: int) -> torch.Tensor:
    """Mean of ``per_pixel`` (B, C, H, W) or (B, H, W) over valid pixels; a zero, gradient-safe value if none."""
    if mask is None:
        return per_pixel.mean()
    weight = mask.to(per_pixel.dtype)
    valid = weight.sum()
    if per_pixel.ndim == 4:
        weight = weight[:, None]
        count = valid * channels
    else:
        count = valid
    if float(count) == 0.0:
        return (per_pixel * 0.0).sum()
    return (per_pixel * weight).sum() / count


def l1_loss(pred: torch.Tensor, target: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    _check_pair(pred, target, mask)
    return _masked_mean((pred.float() - target.float()).abs(), mask, pred.shape[1])


def charbonnier_loss(pred: torch.Tensor, target: torch.Tensor, mask: Optional[torch.Tensor] = None, eps: float = 1e-3) -> torch.Tensor:
    _check_pair(pred, target, mask)
    diff = pred.float() - target.float()
    return _masked_mean(torch.sqrt(diff * diff + eps * eps) - eps, mask, pred.shape[1])


def spectral_angle_loss(pred: torch.Tensor, target: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    _check_pair(pred, target, mask)
    if pred.shape[1] < 2:
        raise LossInputError("The spectral term needs at least 2 bands: one band has no spectral direction.")
    p, t = pred.float(), target.float()
    dot = (p * t).sum(dim=1)
    norm = torch.sqrt((p * p).sum(dim=1) * (t * t).sum(dim=1) + _EPS)          # +eps inside the sqrt keeps the gradient finite at 0
    return _masked_mean(1.0 - dot / norm, mask, 1)


def consistency_loss(sr: torch.Tensor, lr: torch.Tensor, scale: int, lr_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    if sr.ndim != 4 or lr.ndim != 4:
        raise LossInputError(f"Expected 4-D tensors, got sr={tuple(sr.shape)} and lr={tuple(lr.shape)}.")
    if sr.shape[1] != lr.shape[1] or sr.shape[2] != lr.shape[2] * scale or sr.shape[3] != lr.shape[3] * scale:
        raise LossInputError(f"sr {tuple(sr.shape)} is not lr {tuple(lr.shape)} x scale {scale}.")
    _check_mask(lr_mask, lr)
    down = F.avg_pool2d(sr.float(), kernel_size=scale, stride=scale)
    return _masked_mean((down - lr.float()).abs(), lr_mask, lr.shape[1])


@dataclass(frozen=True)
class LossOutput:
    total: torch.Tensor                  # differentiable; backpropagate this
    components: Dict[str, float]         # detached floats for logging: reconstruction, [spectral], [consistency], total


class CompositeLoss(nn.Module):
    """``reconstruction + lambda_spectral * spectral + lambda_consistency * consistency`` from a `LossConfig`."""

    def __init__(self, config: LossConfig):
        super().__init__()
        self.config = config

    def forward(
        self,
        sr: torch.Tensor,
        hr: torch.Tensor,
        *,
        lr: Optional[torch.Tensor] = None,
        hr_mask: Optional[torch.Tensor] = None,
        lr_mask: Optional[torch.Tensor] = None,
        scale: Optional[int] = None,
    ) -> LossOutput:
        cfg = self.config
        if cfg.reconstruction == "charbonnier":
            reconstruction = charbonnier_loss(sr, hr, hr_mask, eps=cfg.charbonnier_eps)
        else:
            reconstruction = l1_loss(sr, hr, hr_mask)
        total = reconstruction
        components = {"reconstruction": float(reconstruction.detach())}

        if cfg.spectral_weight > 0:
            spectral = spectral_angle_loss(sr, hr, hr_mask)
            total = total + cfg.spectral_weight * spectral
            components["spectral"] = float(spectral.detach())
        if cfg.consistency_weight > 0:
            if lr is None or scale is None:
                raise LossInputError("The consistency term needs the lr input and the scale factor (lr=..., scale=...).")
            consistency = consistency_loss(sr, lr, scale, lr_mask)
            total = total + cfg.consistency_weight * consistency
            components["consistency"] = float(consistency.detach())
        components["total"] = float(total.detach())
        return LossOutput(total=total, components=components)
