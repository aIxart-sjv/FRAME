"""The RGBN model I/O contract, checked in code (Phase 1).

    input   float32, shape (4, 128, 128) or (B, 4, 128, 128)
            channel order B04, B03, B02, B08 (see config.RGBN_BAND_ORDER)
            surface reflectance as a fraction (0..~1; never raw DN)
    output  float32, shape (4, 512, 512) or (B, 4, 512, 512), same channel
            order, same value scale, finite

Nothing here silently repairs input: a wrong channel count, size, dtype or
value range raises `ModelContractError` with a message saying what was
wrong, rather than relying on whatever the network happens to do with it.
"""

from __future__ import annotations

from typing import Tuple

import torch

from frame.models import config
from frame.models.errors import ModelContractError


def expected_output_shape(input_shape: Tuple[int, ...]) -> Tuple[int, ...]:
    """Output shape for a (valid) input shape."""
    return tuple(input_shape[:-2]) + (config.OUTPUT_SIZE, config.OUTPUT_SIZE)


def validate_input(x: object) -> None:
    """Raise `ModelContractError` unless ``x`` satisfies the input contract."""
    if not isinstance(x, torch.Tensor):
        raise ModelContractError(f"Model input must be a torch.Tensor, got {type(x).__name__}.")

    if x.ndim not in (3, 4):
        raise ModelContractError(
            f"Model input must have shape (4, {config.INPUT_SIZE}, {config.INPUT_SIZE}) or "
            f"(B, 4, {config.INPUT_SIZE}, {config.INPUT_SIZE}); got {x.ndim}-D shape {tuple(x.shape)}."
        )

    channels, height, width = x.shape[-3], x.shape[-2], x.shape[-1]
    if channels != config.INPUT_CHANNELS:
        raise ModelContractError(
            f"Model input must have {config.INPUT_CHANNELS} channels in the order "
            f"{'/'.join(config.RGBN_BAND_ORDER)}; got {channels} (shape {tuple(x.shape)})."
        )
    if (height, width) != (config.INPUT_SIZE, config.INPUT_SIZE):
        raise ModelContractError(
            f"Model input must be {config.INPUT_SIZE}x{config.INPUT_SIZE} pixels (the size the model was "
            f"trained on; scenes of any size are split into tiles of this size by frame.tiling before they reach "
            f"the model); got {height}x{width}."
        )
    if x.ndim == 4 and x.shape[0] < 1:
        raise ModelContractError("Model input batch is empty.")

    if x.dtype != torch.float32:
        raise ModelContractError(f"Model input must be float32, got {x.dtype}.")

    if not bool(torch.isfinite(x).all()):
        raise ModelContractError("Model input contains NaN or Inf; it must be finite reflectance.")

    # Compare against the bounds *as float32*: the input is float32, so e.g. a
    # DN of 65535 -> 6.5535 is stored as 6.5535002, which would exceed the
    # double-precision bound and wrongly reject a legitimate maximum value.
    lo, hi = float(x.min()), float(x.max())
    lo_bound = float(torch.tensor(config.MIN_REFLECTANCE, dtype=torch.float32))
    hi_bound = float(torch.tensor(config.MAX_REFLECTANCE, dtype=torch.float32))
    if lo < lo_bound or hi > hi_bound:
        raise ModelContractError(
            f"Model input values span [{lo:.4g}, {hi:.4g}], outside the reflectance range "
            f"[{config.MIN_REFLECTANCE}, {config.MAX_REFLECTANCE:.4f}]. Input must be surface reflectance as a "
            "fraction (raw digital numbers must be divided by 10000 first)."
        )


def validate_output(y: object, *, input_shape: Tuple[int, ...]) -> None:
    """Raise `ModelContractError` unless ``y`` is a valid output for an input
    of ``input_shape``: 4x the spatial size, same channels, float32, finite."""
    if not isinstance(y, torch.Tensor):
        raise ModelContractError(f"Model output must be a torch.Tensor, got {type(y).__name__}.")

    expected = expected_output_shape(input_shape)
    if tuple(y.shape) != expected:
        raise ModelContractError(f"Model output has shape {tuple(y.shape)}, expected {expected}.")
    if y.dtype != torch.float32:
        raise ModelContractError(f"Model output must be float32, got {y.dtype}.")
    if not bool(torch.isfinite(y).all()):
        raise ModelContractError("Model output contains NaN or Inf.")
