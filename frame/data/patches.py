"""Paired patch extraction (Phase 3).

This is TRAINING/EVALUATION patch extraction, not inference tiling (frame.tiling): it
cuts small aligned LR/HR windows out of a pair, never pads, never blends, and never
crops LR and HR independently. The single rule everything else relies on:

    HR window = LR window x scale        (origin and size)

so an LR patch of 128 x 128 at (r, c) goes with the HR patch of 512 x 512 at (4r, 4c) for a
x4 pair, and 64 x 64 / 128 x 128 for a x2 pair whose LR patch is 32 -- whatever the pair's
own scale is. That holds only if the HR grid is an exact refinement of the LR grid, which
`check_alignment` (and frame.data.geo for georeferenced pairs) verify first.

Defaults come from Phase 2's definitions (`frame.tiling.plan.DEFAULT_TILE_SIZE` = 128 for the
model's LR input, `DEFAULT_SCALE` = 4) so there is one place those numbers live. The dense
grid reuses `frame.tiling.plan.axis_starts` and then clamps the last window inside the raster
instead of padding it.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

from frame.data.contract import PairedSample, PatchCoords
from frame.data.errors import PatchError
from frame.tiling.plan import DEFAULT_SCALE, DEFAULT_TILE_SIZE, axis_starts

DEFAULT_LR_PATCH = DEFAULT_TILE_SIZE  # 128: the LR input size of the SR models
DEFAULT_PAIR_SCALE = DEFAULT_SCALE    # 4


def _pair(value: Tuple[int, int] | int) -> Tuple[int, int]:
    return (value, value) if isinstance(value, int) else (int(value[0]), int(value[1]))


def check_alignment(lr_hw: Tuple[int, int], hr_hw: Tuple[int, int], scale: int) -> None:
    """Raise `PatchError` unless the HR raster is exactly ``scale`` x the LR raster."""
    if tuple(hr_hw) != (lr_hw[0] * scale, lr_hw[1] * scale):
        raise PatchError(f"HR {tuple(hr_hw)} is not LR {tuple(lr_hw)} x scale {scale}; cannot extract aligned patches.")


def _axis_origins(size: int, patch: int, stride: int) -> List[int]:
    if patch > size:
        raise PatchError(f"A patch of {patch} does not fit in a raster axis of {size}.")
    if stride < 1:
        raise PatchError(f"stride must be >= 1, got {stride}.")
    starts = []
    for start in axis_starts(size, patch, stride):
        clamped = min(start, size - patch)  # clamp the last window inside the raster: no padding
        if clamped not in starts:
            starts.append(clamped)
    return starts


def dense_origins(
    lr_height: int, lr_width: int, patch: Tuple[int, int] | int = DEFAULT_LR_PATCH, stride: Optional[Tuple[int, int] | int] = None
) -> List[Tuple[int, int]]:
    """Row-major LR origins of a regular patch grid covering the raster (last row/column clamped inside).

    ``stride`` defaults to the patch size (non-overlapping). Deterministic; no randomness.
    """
    ph, pw = _pair(patch)
    sh, sw = _pair(stride) if stride is not None else (ph, pw)
    rows = _axis_origins(lr_height, ph, sh)
    cols = _axis_origins(lr_width, pw, sw)
    return [(r, c) for r in rows for c in cols]


def random_origin(lr_height: int, lr_width: int, patch: Tuple[int, int] | int, rng: np.random.Generator) -> Tuple[int, int]:
    """A uniformly random LR origin that keeps the whole patch inside the raster."""
    ph, pw = _pair(patch)
    if ph > lr_height or pw > lr_width:
        raise PatchError(f"Patch {(ph, pw)} does not fit in raster {(lr_height, lr_width)}.")
    return int(rng.integers(0, lr_height - ph + 1)), int(rng.integers(0, lr_width - pw + 1))


def centred_origin(lr_height: int, lr_width: int, patch: Tuple[int, int] | int) -> Tuple[int, int]:
    """The origin of the patch centred in the raster (for example the 128x128 core of a 130x130 SEN2NAIPv2 tile)."""
    ph, pw = _pair(patch)
    if ph > lr_height or pw > lr_width:
        raise PatchError(f"Patch {(ph, pw)} does not fit in raster {(lr_height, lr_width)}.")
    return (lr_height - ph) // 2, (lr_width - pw) // 2


def make_coords(
    sample: PairedSample, lr_row: int, lr_col: int, patch: Tuple[int, int] | int = DEFAULT_LR_PATCH
) -> PatchCoords:
    """Validated coordinates of an LR window inside ``sample`` (and its HR twin)."""
    ph, pw = _pair(patch)
    height, width = sample.lr.shape[-2:]
    if lr_row < 0 or lr_col < 0 or lr_row + ph > height or lr_col + pw > width:
        raise PatchError(f"LR window ({lr_row}, {lr_col}) size {(ph, pw)} falls outside the LR raster {(height, width)}.")
    return PatchCoords(lr_row=lr_row, lr_col=lr_col, lr_height=ph, lr_width=pw, scale=sample.scale_factor)


def extract_patch(sample: PairedSample, coords: PatchCoords) -> PairedSample:
    """Crop the LR window and its exact HR twin (and the masks) out of ``sample``."""
    if coords.scale != sample.scale_factor:
        raise PatchError(f"Patch scale {coords.scale} differs from the pair's scale {sample.scale_factor}.")
    lr_h, lr_w = sample.lr.shape[-2:]
    if coords.lr_row + coords.lr_height > lr_h or coords.lr_col + coords.lr_width > lr_w:
        raise PatchError(f"Patch {coords.to_dict()} falls outside the LR raster {(lr_h, lr_w)}.")

    lr_window = (slice(coords.lr_row, coords.lr_row + coords.lr_height), slice(coords.lr_col, coords.lr_col + coords.lr_width))
    hr = hr_mask = None
    if sample.hr is not None:
        check_alignment((lr_h, lr_w), tuple(sample.hr.shape[-2:]), sample.scale_factor)
        hr_window = (
            slice(coords.hr_row, coords.hr_row + coords.hr_height),
            slice(coords.hr_col, coords.hr_col + coords.hr_width),
        )
        hr = sample.hr[:, hr_window[0], hr_window[1]].contiguous()
        hr_mask = sample.hr_mask[hr_window[0], hr_window[1]].contiguous() if sample.hr_mask is not None else None

    return PairedSample(
        record=sample.record,
        lr=sample.lr[:, lr_window[0], lr_window[1]].contiguous(),
        hr=hr,
        lr_bands=sample.lr_bands,
        hr_bands=sample.hr_bands,
        lr_mask=sample.lr_mask[lr_window[0], lr_window[1]].contiguous() if sample.lr_mask is not None else None,
        hr_mask=hr_mask,
        patch=coords,
    )


def valid_fraction(sample: PairedSample) -> float:
    """The smaller of the LR and HR valid-pixel fractions (1.0 where no mask exists)."""
    fractions = [float(m.float().mean()) for m in (sample.lr_mask, sample.hr_mask) if m is not None]
    return min(fractions) if fractions else 1.0
