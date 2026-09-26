"""Overlap blending (Phase 2): a separable linear weighting window and a
weighted accumulator.

Each tile's SR output is multiplied by a weight window, summed into a canvas
together with the weights themselves, and the canvas is divided by the summed
weights at the end. Every output pixel is therefore a convex combination of the
predictions that cover it -- no brightness change, no last-write-wins.

Window (per axis, on the SR grid): weight 1 in the tile's interior, and a linear
ramp ``(k + 0.5) / R`` over the ``R = overlap * scale`` pixels on each side that
borders another tile. The two ramps of neighbouring tiles cover the same pixels
in opposite directions and sum to exactly 1 there. A side that lies on the
scene boundary gets no ramp (weight stays 1), and every weight is strictly
positive, so the division is always defined. The ramp sits on the tile's
border zone -- where a convolutional/attention model is least reliable -- which
is the point of overlapping in the first place.

``overlap = 0``: ``R = 0``, all weights are 1, tiles abut and nothing is
blended (documented; a seam is possible).

Deterministic: fixed tile order, float32 arithmetic, no randomness.
"""

from __future__ import annotations

import numpy as np
import torch

from frame.tiling.errors import ReconstructionError
from frame.tiling.plan import BLEND_LINEAR, TileSpec, TilingConfig


def axis_window(valid_sr: int, ramp_sr: int, ramp_start: bool, ramp_end: bool) -> np.ndarray:
    """1-D weights (float64) for one tile axis of ``valid_sr`` SR pixels."""
    weights = np.ones(valid_sr, dtype=np.float64)
    if ramp_sr <= 0:
        return weights
    ramp = (np.arange(ramp_sr, dtype=np.float64) + 0.5) / ramp_sr
    if ramp_start:
        n = min(ramp_sr, valid_sr)
        weights[:n] = np.minimum(weights[:n], ramp[:n])
    if ramp_end:
        n = min(ramp_sr, valid_sr)
        weights[valid_sr - n :] = np.minimum(weights[valid_sr - n :], ramp[:n][::-1])
    return weights


def tile_window(tile: TileSpec, config: TilingConfig) -> torch.Tensor:
    """The (sr_valid_height, sr_valid_width) float32 weight window for ``tile``."""
    if config.blend_mode != BLEND_LINEAR:
        raise ValueError(f"Unsupported blend_mode {config.blend_mode!r}.")
    ramp = config.overlap * config.scale
    rows = axis_window(tile.sr_valid_height, ramp, tile.overlaps_top, tile.overlaps_bottom)
    cols = axis_window(tile.sr_valid_width, ramp, tile.overlaps_left, tile.overlaps_right)
    return torch.from_numpy(np.outer(rows, cols)).to(torch.float32)


class WeightedCanvas:
    """Accumulates weighted tile predictions and normalises them at the end."""

    def __init__(self, channels: int, height: int, width: int):
        self._acc = torch.zeros((channels, height, width), dtype=torch.float32)
        self._weights = torch.zeros((height, width), dtype=torch.float32)

    @property
    def shape(self) -> tuple:
        return tuple(self._acc.shape)

    def add(self, values: torch.Tensor, weights: torch.Tensor, row: int, col: int) -> None:
        """Add ``values`` (C, h, w) with ``weights`` (h, w) at SR position (row, col)."""
        _, h, w = values.shape
        if tuple(weights.shape) != (h, w):
            raise ReconstructionError(f"Weights {tuple(weights.shape)} do not match tile values {(h, w)}.")
        if row < 0 or col < 0 or row + h > self._acc.shape[1] or col + w > self._acc.shape[2]:
            raise ReconstructionError(
                f"Tile at SR position ({row}, {col}) size {(h, w)} falls outside the canvas {tuple(self._acc.shape[1:])}."
            )
        self._acc[:, row : row + h, col : col + w] += values * weights
        self._weights[row : row + h, col : col + w] += weights

    def finalize(self) -> torch.Tensor:
        """Divide by the accumulated weights. Raises `ReconstructionError` if any
        pixel received no weight (an incomplete reconstruction) -- never returns a
        partially filled raster."""
        uncovered = int((self._weights <= 0).sum())
        if uncovered:
            raise ReconstructionError(
                f"Reconstruction is incomplete: {uncovered} of {self._weights.numel()} output pixels "
                "were not covered by any tile."
            )
        self._acc /= self._weights
        return self._acc
