"""Model registry for training (Phase 4): everything is a plain ``sr = model(lr)`` ``nn.Module``.

The training loop never learns which architecture it is driving and never touches the Mamba worker
protocol (frame.models.mamba_client is an INFERENCE bridge to another process; a trainer needs the module
in-process, with gradients).

    tiny_cnn       a small residual CNN over FRAME's bicubic base. Its output head is zero-initialised, so before
                   any update it IS the bicubic baseline (same operator as frame.validation.bicubic_upsample): every later gain is attributable to training. Used for
                   the smoke experiments; has no pretrained weights.
    sen2sr_lite    SEN2SR-Lite's CNN (``CNNSR(4, 4, 24, 4, True, True, 6)``, 572,336 parameters of which 472,496
                   are trainable; the published inference build uses ``train_mode=False``, which cannot be
                   fine-tuned -- see `_LITE_ARCHITECTURE`), built from the vendored sen2sr package. ``pretrained=True`` loads the cached published weights (never
                   downloads). This is the ``sr_model`` WITHOUT the low-frequency hard constraint: the
                   constraint's mask is fixed at 512 x 512, so it cannot be applied to small training crops.
    sen2sr_mamba   ``MambaSR`` (13,759,444 parameters). Needs ``mamba_ssm`` and a CUDA GPU, which exist only in
                   the isolated Mamba environment; ``activation_checkpointing: true`` (frame.train.memory) is what
                   makes it fit a 4 GB card at the native tile; in the main environment building it fails with a clear error
                   (docs/TRAINING.md). Same constraint note as Lite.

Adding a model means adding a name to `frame.train.config.MODEL_NAMES` and a builder here.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from frame.train.errors import ModelBuildError

# The published inference artifact is built with train_mode=False (fused convs). That mode detaches the fused weights
# from their branches on every forward, so it cannot be fine-tuned meaningfully; train_mode=True has the same state-dict
# keys and computes the same function through the trainable branches (tested against the published model).
_LITE_ARCHITECTURE = dict(in_channels=4, out_channels=4, feature_channels=24, upscale=4, bias=True, train_mode=True, num_blocks=6)
LITE_WEIGHTS_FILE = "model.safetensor"


def count_parameters(model: nn.Module, *, trainable_only: bool = False) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad or not trainable_only)


def parameter_megabytes(model: nn.Module) -> float:
    """Memory held by the model's parameters (MiB), at their actual dtypes."""
    return sum(p.numel() * p.element_size() for p in model.parameters()) / 2**20


def _only(params: Mapping[str, Any], allowed: Mapping[str, Any], model: str) -> Dict[str, Any]:
    unknown = sorted(set(params) - set(allowed))
    if unknown:
        raise ModelBuildError(f"{model}: unknown parameter(s) {unknown}; valid: {sorted(allowed)}.")
    return {**allowed, **params}


# ---------------------------------------------------------------------------------------------------------------
# tiny_cnn
# ---------------------------------------------------------------------------------------------------------------

class TinyResidualCNN(nn.Module):
    def __init__(self, in_channels: int = 4, width: int = 16, depth: int = 3, scale: int = 4):
        super().__init__()
        self.scale = scale
        # LeakyReLU, not ReLU: with ReLU, some initialisations left ~94% of the last layer's channels dead and training stayed at exactly
        # the bicubic starting point (measured; see test_train_models.py). 2 of 6 seeds stalled at batch 1, 3 of 6 at batch 4.
        layers = [nn.Conv2d(in_channels, width, 3, padding=1), nn.LeakyReLU(0.1, inplace=True)]
        for _ in range(depth - 1):
            layers += [nn.Conv2d(width, width, 3, padding=1), nn.LeakyReLU(0.1, inplace=True)]
        self.body = nn.Sequential(*layers)
        self.head = nn.Conv2d(width, in_channels * scale * scale, 3, padding=1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)
        self.shuffle = nn.PixelShuffle(scale)

    def forward(self, lr: torch.Tensor) -> torch.Tensor:
        # FRAME's bicubic convention (frame.validation.bicubic_upsample): antialiased kernel, negatives clamped. Torch's default
        # bicubic uses a different cubic coefficient, so using it here would make "step 0 == the bicubic baseline" false.
        base = F.interpolate(lr, scale_factor=self.scale, mode="bicubic", antialias=True).clamp(min=0.0)
        return base + self.shuffle(self.head(self.body(lr)))


def _build_tiny(params: Mapping[str, Any], pretrained: bool, seed: int) -> nn.Module:
    if pretrained:
        raise ModelBuildError("tiny_cnn has no pretrained weights; set model.pretrained to false.")
    p = _only(params, {"in_channels": 4, "width": 16, "depth": 3, "scale": 4}, "tiny_cnn")
    for key, value in p.items():
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            raise ModelBuildError(f"tiny_cnn: {key} must be a positive integer, got {value!r} (width must be > 0).")
    return TinyResidualCNN(**p)


# ---------------------------------------------------------------------------------------------------------------
# sen2sr_lite
# ---------------------------------------------------------------------------------------------------------------

def _weights_dir(params: Mapping[str, Any], env: str, default: Path) -> Path:
    return Path(params.get("weights_dir") or os.environ.get(env) or default)


def _build_lite(params: Mapping[str, Any], pretrained: bool, seed: int) -> nn.Module:
    p = _only(params, {"weights_dir": None}, "sen2sr_lite")
    try:
        from sen2sr.models.opensr_baseline.cnn import CNNSR
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ModelBuildError(f"The vendored sen2sr package could not be imported ({exc}); run from the repository root.") from exc
    model = CNNSR(**_LITE_ARCHITECTURE)
    if pretrained:
        import safetensors.torch as st

        directory = _weights_dir(p, "SEN2SR_BASELINE_WEIGHTS_DIR", Path.home() / ".cache" / "sen2sr_baseline" / "SEN2SRLite_RGBN")
        weights = directory / LITE_WEIGHTS_FILE
        if not weights.is_file():
            raise ModelBuildError(f"SEN2SR-Lite weights not found: {weights}. Nothing is downloaded automatically.")
        model.load_state_dict(st.load_file(str(weights)))
    return model


# ---------------------------------------------------------------------------------------------------------------
# sen2sr_mamba
# ---------------------------------------------------------------------------------------------------------------

def _build_mamba(params: Mapping[str, Any], pretrained: bool, seed: int) -> nn.Module:
    from frame.models import config as models_config

    p = _only(params, {"weights_dir": None, "activation_checkpointing": False}, "sen2sr_mamba")
    try:
        from sen2sr.models.opensr_baseline.mamba import MambaSR
    except ImportError as exc:
        raise ModelBuildError(
            "sen2sr_mamba needs mamba_ssm and CUDA, which exist only in the isolated Mamba environment "
            f"(sen2sr_mamba_venv); run frame.train from there. Import failed: {type(exc).__name__}: {exc}"
        ) from exc
    model = MambaSR(**models_config.MAMBA_ARCHITECTURE.kwargs())
    if pretrained:
        import safetensors.torch as st

        directory = Path(p.get("weights_dir") or models_config.MAMBA_WEIGHTS_DIR)
        weights = directory / models_config.MAMBA_SR_WEIGHTS_FILENAME
        if not weights.is_file():
            raise ModelBuildError(f"SEN2SR-Mamba weights not found: {weights}.")
        model.load_state_dict(st.load_file(str(weights)), strict=True)
    if p["activation_checkpointing"]:
        from frame.train.memory import enable_block_checkpointing

        enable_block_checkpointing(model)
    return model


_BUILDERS: Dict[str, Callable[[Mapping[str, Any], bool, int], nn.Module]] = {
    "tiny_cnn": _build_tiny,
    "sen2sr_lite": _build_lite,
    "sen2sr_mamba": _build_mamba,
}


def build_model(config: Any, *, seed: int) -> nn.Module:
    """Construct the model described by ``config`` (a `ModelConfig`), deterministically for ``seed``, on the CPU.

    The global RNG is restored afterwards, so building a model never perturbs the training data order.
    """
    try:
        builder = _BUILDERS[config.name]
    except KeyError:
        raise ModelBuildError(f"Unknown model {config.name!r}; known: {sorted(_BUILDERS)}.") from None
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        return builder(config.params, bool(config.pretrained), seed)
