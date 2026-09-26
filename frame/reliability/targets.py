"""Error targets and trivial predictors of error (Phase 6). Pure numpy; every function takes arrays on the SR grid, already cropped to the aligned overlap.

The stability being validated is the deployed one: the per-pixel standard deviation of the six de-transformed TTA predictions, averaged over the four bands
(``overall_std`` of the API; :func:`stability_map`). What it is compared with:

Error targets (against the independent HR reference, on the strict valid mask only)
    ``abs_error``   band-mean absolute reflectance error                       (E1, radiometric)   pixel / 10 m cell / 40 m cell / tile / scene
    ``sq_error``    band-mean squared reflectance error                        (E1)                for RMSE aggregation
    ``sam_deg``     spectral angle between the predicted and reference spectra (E2, spectral)      NaN where a spectrum has no direction (never 0)
    tile RMSE / MAE / SAM / ERGAS       the Phase 5 definitions (frame.evaluate.metrics), computed here without the SSIM the tile-level correlation does not need
    detail fractions  per-pixel share of the four bands labelled supported synthesis / unsupported detail / omission / neutral by the Phase 5 element analysis (E4 / E5)

Trivial predictors (what stability must beat to be worth anything)
    ``texture_baseline``       Sobel gradient of the band-mean bicubic input: where there is structure to reconstruct, known without any model
    ``added_detail_baseline``  |prediction - bicubic|: how far the model moved away from the interpolation, known from a single pass

Both correlate with error for reasons that have nothing to do with ensemble disagreement; a stability signal that only re-encodes them adds nothing.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Sequence, Tuple

import numpy as np
from scipy import ndimage

from frame.evaluate import metrics as M


def _check(a: np.ndarray, b: np.ndarray, name_a: str, name_b: str) -> None:
    if a.shape != b.shape:
        raise ValueError(f"{name_a} and {name_b} must have the same shape, got {a.shape} and {b.shape}")


def stability_map(std: np.ndarray) -> np.ndarray:
    """The deployed per-pixel stability: the ensemble standard deviation averaged over bands, (C, H, W) -> (H, W)."""
    return np.asarray(std, dtype=np.float64).mean(axis=0)


def pixel_error_maps(pred: np.ndarray, hr: np.ndarray, mask: np.ndarray) -> Dict[str, np.ndarray]:
    """Per-pixel error maps of ``pred`` against ``hr`` (C, H, W). Pixels outside ``valid`` (the mask, and both images finite) are NaN."""
    p, h = np.asarray(pred, dtype=np.float64), np.asarray(hr, dtype=np.float64)
    _check(p, h, "prediction", "reference")
    m = np.asarray(mask, dtype=bool)
    if m.shape != p.shape[-2:]:
        raise ValueError(f"mask shape {m.shape} does not match the image shape {p.shape[-2:]}")
    valid = m & np.isfinite(p).all(axis=0) & np.isfinite(h).all(axis=0)
    diff = np.where(valid[None], p - h, np.nan)
    abs_error = np.abs(diff).mean(axis=0)
    sq_error = (diff ** 2).mean(axis=0)
    dot = np.where(valid[None], p * h, np.nan).sum(axis=0)
    denom = np.sqrt(np.where(valid[None], p ** 2, np.nan).sum(axis=0)) * np.sqrt(np.where(valid[None], h ** 2, np.nan).sum(axis=0))
    with np.errstate(invalid="ignore", divide="ignore"):
        cos = np.where(denom > 0, dot / denom, np.nan)
    sam = np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))
    return {"abs_error": abs_error, "sq_error": sq_error, "sam_deg": sam, "valid": valid}


def texture_baseline(bicubic: np.ndarray) -> np.ndarray:
    """Sobel gradient magnitude of the band-mean bicubic upsampling of the input: a model-free measure of local structure."""
    mean = np.asarray(bicubic, dtype=np.float64).mean(axis=0)
    return np.hypot(ndimage.sobel(mean, axis=0, mode="reflect"), ndimage.sobel(mean, axis=1, mode="reflect"))


def added_detail_baseline(pred: np.ndarray, bicubic: np.ndarray) -> np.ndarray:
    """Band-mean |prediction - bicubic|: the amount of change the model made, known from one forward pass."""
    p, b = np.asarray(pred, dtype=np.float64), np.asarray(bicubic, dtype=np.float64)
    _check(p, b, "prediction", "bicubic")
    return np.abs(p - b).mean(axis=0)


def detail_label_maps(pred: np.ndarray, bicubic: np.ndarray, hr: np.ndarray, tau: float) -> Dict[str, np.ndarray]:
    """Per-pixel share of the four bands in each Phase 5 element category (supported synthesis, unsupported detail, omission, neutral)."""
    labels = M.hallucination_labels(bicubic, pred, hr, tau)
    return {k: labels[k].astype(np.float64).mean(axis=0) for k in ("supported_synthesis", "unsupported_detail", "omission", "neutral")}


def block_reduce(values: np.ndarray, valid: np.ndarray, *, size: int, min_valid_fraction: float) -> Tuple[np.ndarray, np.ndarray]:
    """Mean of ``values`` over the valid pixels of every ``size`` x ``size`` block (the ragged edge is cropped). A block with fewer than ``min_valid_fraction`` valid
    pixels is unusable: it is marked so and its value is NaN, never 0."""
    v, ok = np.asarray(values, dtype=np.float64), np.asarray(valid, dtype=bool) & np.isfinite(values)
    height, width = v.shape
    if height < size or width < size:
        raise ValueError(f"block size {size} is larger than the image {v.shape}")
    h, w = height // size, width // size
    v, ok = v[: h * size, : w * size], ok[: h * size, : w * size]
    blocks = lambda a: a.reshape(h, size, w, size)                                            # noqa: E731
    count = blocks(ok).sum(axis=(1, 3))
    total = np.where(blocks(ok), blocks(v), 0.0).sum(axis=(1, 3))
    usable = count >= max(1.0, min_valid_fraction * size * size)
    cells = np.where(usable, total / np.maximum(count, 1), np.nan)
    return cells, usable


def cell_arrays(maps: Dict[str, np.ndarray], valid: np.ndarray, *, size: int, min_valid_fraction: float) -> Dict[str, np.ndarray]:
    """Flattened per-cell values of every map over the cells that are usable under ``valid`` (same cells for every map), plus each cell's ``row`` and ``col``."""
    _, usable = block_reduce(np.zeros(np.asarray(valid).shape), valid, size=size, min_valid_fraction=min_valid_fraction)
    reduced = {name: block_reduce(m, valid, size=size, min_valid_fraction=min_valid_fraction)[0] for name, m in maps.items()}
    rows, cols = np.nonzero(usable)
    out = {name: cells[usable] for name, cells in reduced.items()}
    out["row"], out["col"] = rows.astype(np.int32), cols.astype(np.int32)
    return out


def tile_targets(pred: np.ndarray, hr: np.ndarray, mask: np.ndarray, band_names: Sequence[str], *, scale: int) -> Dict[str, Any]:
    """RMSE, MAE, SAM and ERGAS of the tile under the strict mask (the Phase 5 definitions); missing, not zero, when nothing is valid."""
    maps = pixel_error_maps(pred, hr, mask)
    valid = maps["valid"]
    n = int(valid.sum())
    if n == 0:
        return {"rmse": None, "mae": None, "sam_degrees": None, "ergas": None, "n_valid_pixels": 0, "n_sam_undefined": 0}
    p, h = np.asarray(pred, dtype=np.float64)[:, valid], np.asarray(hr, dtype=np.float64)[:, valid]
    err = p - h
    rmse_band = np.sqrt((err ** 2).mean(axis=1))
    mean_hr = h.mean(axis=1)
    ergas = 100.0 / scale * math.sqrt(float(np.mean((rmse_band / np.where(mean_hr != 0, mean_hr, np.nan)) ** 2))) if (mean_hr != 0).all() else None
    sam = maps["sam_deg"][valid]
    finite_sam = np.isfinite(sam)
    return {"rmse": float(math.sqrt(float((err ** 2).mean()))), "mae": float(np.abs(err).mean()), "sam_degrees": float(sam[finite_sam].mean()) if finite_sam.any() else None,
            "ergas": ergas, "n_valid_pixels": n, "n_sam_undefined": int((~finite_sam).sum())}
