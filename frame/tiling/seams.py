"""Tile-seam diagnostic (Phase 2): how much do neighbouring tiles disagree?

For every pair of horizontally or vertically adjacent tiles, compare their two
SR predictions on the pixels they share -- BEFORE blending, using the real tile
outputs -- and pool the results. A large disagreement means the model gives
noticeably different answers for the same pixel depending on which tile it sat
in, i.e. a tiling artefact (a seam) is likely.

This is a tiling/seam consistency diagnostic. It says nothing about accuracy
against any reference. It is also biased upward by construction: the shared
pixels are each tile's border zone, where the model is least reliable. Compare
runs with the same overlap, not across overlaps.

The pairwise comparison itself is `frame.consistency.tiles.compare_tile_overlap`
(written in an earlier phase for exactly this, and waiting for a FRAME-owned
tiler to feed it); this module only feeds it and pools the results. Not
computable (and reported as such, never as 0) for a single tile or overlap 0.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Dict, Optional, Tuple

import torch

from frame.consistency.status import ComputationStatus
from frame.consistency.tiles import compare_tile_overlap
from frame.tiling.plan import TilePlan, TileSpec

SEAM_DIAGNOSTIC_DEFINITION = (
    "Disagreement between adjacent tiles' predictions on their shared pixels, measured before blending. "
    "A tiling/seam consistency diagnostic, not an accuracy measure against any reference."
)


@dataclass(frozen=True)
class SeamDiagnostic:
    status: str                              # ComputationStatus value
    pair_count: int
    overlap_pixel_count: int
    mean_abs_difference: Optional[float]
    rmse: Optional[float]
    max_abs_difference: Optional[float]
    note: str = SEAM_DIAGNOSTIC_DEFINITION

    def as_dict(self) -> Dict:
        return asdict(self)


NOT_COMPUTABLE_NO_OVERLAP = SeamDiagnostic(
    status=ComputationStatus.NOT_COMPUTABLE.value,
    pair_count=0,
    overlap_pixel_count=0,
    mean_abs_difference=None,
    rmse=None,
    max_abs_difference=None,
    note="No overlapping tiles (a single tile, or overlap = 0), so there is nothing to compare. " + SEAM_DIAGNOSTIC_DEFINITION,
)


class _Pool:
    """Pixel-count-weighted pooling of per-pair overlap statistics."""

    def __init__(self) -> None:
        self.pairs = 0
        self.pixels = 0
        self.sum_abs = 0.0
        self.sum_sq = 0.0
        self.max_abs = 0.0

    def add(self, pixels: int, mean_abs: float, rmse: float, max_abs: float) -> None:
        self.pairs += 1
        self.pixels += pixels
        self.sum_abs += mean_abs * pixels
        self.sum_sq += rmse * rmse * pixels
        self.max_abs = max(self.max_abs, max_abs)

    def result(self) -> SeamDiagnostic:
        if self.pixels == 0:
            return NOT_COMPUTABLE_NO_OVERLAP
        return SeamDiagnostic(
            status=ComputationStatus.COMPUTABLE.value,
            pair_count=self.pairs,
            overlap_pixel_count=self.pixels,
            mean_abs_difference=self.sum_abs / self.pixels,
            rmse=math.sqrt(self.sum_sq / self.pixels),
            max_abs_difference=self.max_abs,
        )


class SeamAccumulator:
    """Feed it each tile's cropped SR output in row-major order; read `result()` at the end.

    Only the previous tile row and the current one are kept, so memory stays
    bounded by two rows of tile outputs regardless of scene height.
    """

    def __init__(self, plan: TilePlan):
        self._plan = plan
        self._recent: Dict[Tuple[int, int], Tuple[TileSpec, "object"]] = {}
        self._pool = _Pool()

    def observe(self, tile: TileSpec, values: torch.Tensor) -> None:
        array = values.detach().cpu().numpy()
        for neighbour_index in ((tile.row_index, tile.col_index - 1), (tile.row_index - 1, tile.col_index)):
            neighbour = self._recent.get(neighbour_index)
            if neighbour is None:
                continue
            other, other_array = neighbour
            outcome = compare_tile_overlap(
                array,
                (tile.sr_row_start, tile.sr_col_start),
                other_array,
                (other.sr_row_start, other.sr_col_start),
            )
            if outcome.status == ComputationStatus.COMPUTABLE:
                self._pool.add(outcome.valid_pixel_count, outcome.mean_abs_discrepancy, outcome.rmse, outcome.max_abs_discrepancy)

        self._recent[(tile.row_index, tile.col_index)] = (tile, array)
        for stale in [key for key in self._recent if key[0] < tile.row_index - 1]:
            del self._recent[stale]

    def result(self) -> SeamDiagnostic:
        return self._pool.result()


def pool_diagnostics(diagnostics) -> SeamDiagnostic:
    """Pool several `SeamDiagnostic`s (e.g. one per ensemble pass) into one."""
    pool = _Pool()
    for d in diagnostics:
        if d is not None and d.status == ComputationStatus.COMPUTABLE.value:
            pool.pairs += d.pair_count
            pool.pixels += d.overlap_pixel_count
            pool.sum_abs += d.mean_abs_difference * d.overlap_pixel_count
            pool.sum_sq += d.rmse * d.rmse * d.overlap_pixel_count
            pool.max_abs = max(pool.max_abs, d.max_abs_difference)
    return pool.result()
