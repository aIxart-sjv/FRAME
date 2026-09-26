"""Edge-tile padding (Phase 2).

A tile that extends past the scene edge is filled to the full tile size by
MIRRORING the scene about its edge, without repeating the edge pixel
(``a b c`` -> ``a b c b a b c ...``): "reflect" padding.

Why reflect, for satellite imagery feeding these models:

* zero-fill invents a hard black border; the models' Fourier hard constraint
  works on the whole tile, so a hard edge rings across it;
* edge-replication makes streaks that read as a real (constant) surface;
* reflect keeps the local statistics, texture and spectral values of the real
  neighbourhood, so the model sees plausible imagery, and the padded strip is
  cropped away after inference -- the final raster never contains it.

The mirror is periodic, so it works for any scene size, including a scene
smaller than one tile (a 1-pixel axis simply repeats that pixel).
"""

from __future__ import annotations

import torch

from frame.tiling.plan import PADDING_REFLECT, TileSpec


def reflect_indices(valid: int, total: int, device: torch.device | str = "cpu") -> torch.Tensor:
    """Indices into an axis of length ``valid`` that yield ``total`` samples,
    mirroring about the edge without repeating it. ``total == valid`` gives
    the identity."""
    positions = torch.arange(total, device=device)
    if valid == total:
        return positions
    if valid == 1:
        return torch.zeros_like(positions)
    period = 2 * (valid - 1)
    folded = positions % period
    return torch.where(folded < valid, folded, period - folded)


def extract_tile(scene: torch.Tensor, tile: TileSpec, padding_mode: str = PADDING_REFLECT) -> torch.Tensor:
    """The tile's pixels from ``scene`` (C, H, W) as a (C, padded_height,
    padded_width) tensor on the scene's device. Only the part inside the scene
    is read; the rest of an edge tile is padding."""
    if padding_mode != PADDING_REFLECT:
        raise ValueError(f"Unsupported padding_mode {padding_mode!r}.")

    valid = scene[:, tile.row_start : tile.row_end, tile.col_start : tile.col_end]
    if not tile.is_padded:
        return valid.contiguous()

    rows = reflect_indices(tile.valid_height, tile.padded_height, scene.device)
    cols = reflect_indices(tile.valid_width, tile.padded_width, scene.device)
    return valid[:, rows][:, :, cols].contiguous()
