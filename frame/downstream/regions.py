"""Fixed regions for the downstream analysis (Phase 7).

A region is one of FRAME's existing fixed square cells on the aligned HR grid (the cells frame.reliability already uses): ``size`` = 4 HR px is 10 m, the footprint of one Sentinel-2 pixel
(the requirements' "10 m pixel = 16 cells at 2.5 m"), and 16 HR px is 40 m. The grid is placed so that every region starts on a multiple of ``size`` in PREDICTION (SR) coordinates, whatever
the whole-pixel translation crop the reference gate applied: a 10 m region is therefore one Sentinel-2 pixel, not a window that straddles two. A ragged remainder at the far edges is not
covered (and is counted, not silently averaged in). Regions are disjoint by construction; nothing is resampled.

For every region that has enough valid pixels the table keeps its location in the aligned frame, in the prediction grid and in the original reference grid, its valid-pixel count, the mean of
every covariate map and of every NDVI source over its valid pixels, and, for each declared vegetation threshold, the vegetation counts of the reference and the confusion counts
(true positive / false positive / false negative; true negatives follow) of every system against the reference. A region below the minimum valid fraction has NO values (excluded and
counted with its reason), never a mean of nothing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

EXCLUSION_INSUFFICIENT = "region_insufficient_valid_pixels"


@dataclass(frozen=True)
class RegionGrid:
    size: int
    height: int                # of the aligned frame
    width: int
    sr_row_start: int          # position of the aligned frame's first row / column in the prediction (SR) grid
    sr_col_start: int
    hr_row_start: int          # ... in the original reference grid
    hr_col_start: int
    origin_row: int            # aligned-frame position of the first region (a multiple of size in prediction coordinates)
    origin_col: int
    n_rows: int
    n_cols: int

    @property
    def n_cells(self) -> int:
        return self.n_rows * self.n_cols


def make_grid(shape: Tuple[int, int], size: int, correction: Sequence[int]) -> RegionGrid:
    """The region grid of an aligned frame of ``shape``. ``correction`` is the recorded whole-pixel translation (dy, dx) of the gate: the prediction window starts at ``max(-dy, 0)`` and the
    reference window at ``max(dy, 0)`` (see frame.evaluate.shift.displace_pair)."""
    height, width = int(shape[0]), int(shape[1])
    cy, cx = int(correction[0]), int(correction[1])
    sr_r, sr_c, hr_r, hr_c = max(-cy, 0), max(-cx, 0), max(cy, 0), max(cx, 0)
    origin_r, origin_c = (-sr_r) % size, (-sr_c) % size
    return RegionGrid(size=size, height=height, width=width, sr_row_start=sr_r, sr_col_start=sr_c, hr_row_start=hr_r, hr_col_start=hr_c, origin_row=origin_r, origin_col=origin_c,
                      n_rows=max(0, (height - origin_r) // size), n_cols=max(0, (width - origin_c) // size))


def candidate_region_count(shape: Tuple[int, int], size: int) -> int:
    """Number of cells the tile's grid would have with no correction: geometry only, computed without any value of the reference (used to count the candidates of excluded tiles)."""
    return make_grid(shape, size, (0, 0)).n_cells


def _blocks(a: np.ndarray, g: RegionGrid) -> np.ndarray:
    core = a[g.origin_row: g.origin_row + g.n_rows * g.size, g.origin_col: g.origin_col + g.n_cols * g.size]
    return core.reshape(g.n_rows, g.size, g.n_cols, g.size)


def _block_sum(a: np.ndarray, g: RegionGrid) -> np.ndarray:
    return _blocks(a, g).sum(axis=(1, 3))


def _threshold_key(th: float) -> str:
    return f"{th:g}"


def extract_regions(
    maps: Mapping[str, np.ndarray],
    valid: np.ndarray,
    grid: RegionGrid,
    *,
    min_valid_fraction: float,
    thresholds: Sequence[float],
    systems: Sequence[str],
    reference: str,
    ndvi: Optional[Mapping[str, np.ndarray]] = None,
) -> Dict[str, Any]:
    """The region table of one aligned tile (see the module docstring). ``maps`` are covariate maps (texture, stability, ...), ``ndvi`` maps the reference and each system to its NDVI map."""
    v = np.asarray(valid, dtype=bool)
    ndvi = dict(ndvi or {})
    n_candidate = grid.n_cells
    out: Dict[str, Any] = {"size": grid.size, "n_candidate": int(n_candidate), "excluded": {EXCLUSION_INSUFFICIENT: 0}, "grid": {k: int(getattr(grid, k)) for k in
                           ("size", "sr_row_start", "sr_col_start", "hr_row_start", "hr_col_start", "origin_row", "origin_col", "n_rows", "n_cols")}}
    if n_candidate == 0:
        for k in ("row", "col", "sr_row", "sr_col", "hr_row", "hr_col", "n_valid"):
            out[k] = np.zeros(0, dtype=np.int32)
        return out
    count = _block_sum(v.astype(np.int64), grid)
    usable = count >= max(1.0, min_valid_fraction * grid.size ** 2)
    out["excluded"][EXCLUSION_INSUFFICIENT] = int((~usable).sum())
    rows, cols = np.nonzero(usable)
    out["row"], out["col"] = rows.astype(np.int32), cols.astype(np.int32)
    out["sr_row"] = (grid.sr_row_start + grid.origin_row + rows * grid.size).astype(np.int32)
    out["sr_col"] = (grid.sr_col_start + grid.origin_col + cols * grid.size).astype(np.int32)
    out["hr_row"] = (grid.hr_row_start + grid.origin_row + rows * grid.size).astype(np.int32)
    out["hr_col"] = (grid.hr_col_start + grid.origin_col + cols * grid.size).astype(np.int32)
    out["n_valid"] = count[usable].astype(np.int32)

    def mean_of(a: np.ndarray) -> np.ndarray:
        finite = v & np.isfinite(a)
        n = _block_sum(finite.astype(np.int64), grid)
        total = _block_sum(np.where(finite, a, 0.0), grid)
        with np.errstate(invalid="ignore", divide="ignore"):
            m = np.where(n > 0, total / np.maximum(n, 1), np.nan)
        return m[usable].astype(np.float32)

    for name, a in maps.items():
        out[f"mean:{name}"] = mean_of(np.asarray(a, dtype=np.float64))
    for name, a in ndvi.items():
        out[f"mean:{name}"] = mean_of(np.asarray(a, dtype=np.float64))
    if ndvi and thresholds:
        ref = np.asarray(ndvi[reference], dtype=np.float64)
        for th in thresholds:
            key = _threshold_key(th)
            ref_veg = v & (ref >= th)
            out[f"veg:{reference}@{key}"] = _block_sum(ref_veg.astype(np.int64), grid)[usable].astype(np.int32)
            for s in systems:
                sy = np.asarray(ndvi[s], dtype=np.float64)
                sys_veg = v & (sy >= th)
                out[f"veg:{s}@{key}"] = _block_sum(sys_veg.astype(np.int64), grid)[usable].astype(np.int32)
                out[f"tp:{s}@{key}"] = _block_sum((sys_veg & ref_veg).astype(np.int64), grid)[usable].astype(np.int32)
                out[f"fp:{s}@{key}"] = _block_sum((sys_veg & ~ref_veg).astype(np.int64), grid)[usable].astype(np.int32)
                out[f"fn:{s}@{key}"] = _block_sum((~sys_veg & ref_veg).astype(np.int64) * v, grid)[usable].astype(np.int32)
    return out


def stack_tables(tables: Sequence[Tuple[Mapping[str, Any], Mapping[str, Any]]]) -> Dict[str, Any]:
    """Concatenate region tables of several tiles; every region keeps the identity (dataset, scene unit, tile) of the tile it came from, so regions of one scene are never mistaken for
    independent observations."""
    if not tables:
        return {"n_candidate": 0, "excluded": {}}
    array_keys = [k for k, v in tables[0][1].items() if isinstance(v, np.ndarray)]
    out: Dict[str, Any] = {k: np.concatenate([t[k] for _, t in tables]) for k in array_keys}
    for meta_key in tables[0][0]:
        out[meta_key] = np.concatenate([np.full(len(t["row"]), meta[meta_key], dtype=object) for meta, t in tables])
    sizes = {int(t["size"]) for _, t in tables}
    if len(sizes) != 1:
        raise ValueError(f"tables of different region sizes cannot be stacked: {sorted(sizes)}")
    out["size"] = sizes.pop()
    out["n_candidate"] = int(sum(t["n_candidate"] for _, t in tables))
    excluded: Dict[str, int] = {}
    for _, t in tables:
        for k, n in t["excluded"].items():
            excluded[k] = excluded.get(k, 0) + int(n)
    out["excluded"] = excluded
    return out
