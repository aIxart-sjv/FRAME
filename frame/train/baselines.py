"""Reference upsamplers evaluated through the same loop as the trained model (Phase 4).

    bicubic   the naive baseline, in FRAME's existing convention (bicubic with antialiasing, negatives clamped:
              frame.validation.bicubic_upsample). No parameters.
    lite      the published SEN2SR-Lite inference model (with its low-frequency hard constraint), loaded from the local
              cache; NEVER downloaded here. Its constraint mask is fixed at 512 x 512, so it only accepts 128 x 128 LR tiles.

They exist so that a tiny experiment reports comparable numbers, not so that anything is declared a winner. The trained
model, bicubic and Lite are also different kinds of object (Lite carries a hard constraint and pretrained weights that
the small smoke model does not), which every report has to say.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Callable, Dict, Optional, Sequence, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from frame.train.errors import ModelBuildError, TrainError

LITE_NATIVE_LR = 128
BASELINE_NAMES = ("bicubic", "lite")


class BicubicUpsampler(nn.Module):
    def __init__(self, scale: int = 4):
        super().__init__()
        self.scale = scale

    def forward(self, lr: torch.Tensor) -> torch.Tensor:
        return F.interpolate(lr.float(), scale_factor=self.scale, mode="bicubic", antialias=True).clamp(min=0.0)


class _LiteBaseline(nn.Module):
    def __init__(self, compiled: nn.Module):
        super().__init__()
        self.compiled = compiled

    def forward(self, lr: torch.Tensor) -> torch.Tensor:
        if lr.shape[-2:] != (LITE_NATIVE_LR, LITE_NATIVE_LR):
            raise TrainError(
                f"The Lite baseline only accepts {LITE_NATIVE_LR}x{LITE_NATIVE_LR} LR tiles (its hard-constraint mask is 512x512), "
                f"got {tuple(lr.shape[-2:])}; set data.val_lr_patch to {LITE_NATIVE_LR} or drop 'lite' from baselines."
            )
        return self.compiled(lr)


def load_lite_baseline(device: Union[str, torch.device] = "cpu", weights_dir: Optional[Path] = None) -> nn.Module:
    directory = Path(weights_dir or os.environ.get("SEN2SR_BASELINE_WEIGHTS_DIR") or Path.home() / ".cache" / "sen2sr_baseline" / "SEN2SRLite_RGBN")
    if not (directory / "mlm.json").is_file() or not (directory / "model.safetensor").is_file():
        raise ModelBuildError(f"SEN2SR-Lite weights not found in {directory}. Nothing is downloaded automatically (the API downloads them on first use).")
    import mlstac

    return _LiteBaseline(mlstac.load(str(directory)).compiled_model(device=str(device))).eval()


def build_baselines(names: Sequence[str], *, device: Union[str, torch.device], scale: int) -> Dict[str, Callable[[torch.Tensor], torch.Tensor]]:
    built: Dict[str, Callable[[torch.Tensor], torch.Tensor]] = {}
    for name in names:
        if name == "bicubic":
            built[name] = BicubicUpsampler(scale).to(device)
        elif name == "lite":
            built[name] = load_lite_baseline(device)
        else:
            raise ModelBuildError(f"Unknown baseline {name!r}; known: {BASELINE_NAMES}.")
    return built
