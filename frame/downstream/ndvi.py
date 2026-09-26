"""NDVI for the downstream analysis (Phase 7): NDVI = (NIR - Red) / (NIR + Red) = (B08 - B04) / (B08 + B04) on reflectance fractions.

The core division is the one FRAME already uses (``frame.consistency.spectral_ratios.compute_ndvi``: a pixel whose denominator is at or below a numerical floor is NOT computable, never
divided into a fabricated near-infinite value). Bands are selected BY NAME from the stack, never by position. A pixel is scored only when it is valid for the reference AND computable for
every system, so all systems are always compared on identical pixels; which pixels are dark, or nodata, is decided from the reference alone (Phase 5 rule: Red + NIR reflectance > 0.02).
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import numpy as np

from frame.consistency.spectral_ratios import compute_ndvi
from frame.downstream.config import NdviSpec
from frame.downstream.errors import DownstreamError


def band_index(band_names: Sequence[str], name: str) -> int:
    names = list(band_names)
    if len(set(names)) != len(names):
        raise DownstreamError(f"duplicate band names {names}: bands are selected by name")
    if name not in names:
        raise DownstreamError(f"band {name} is not in the stack (bands: {names})")
    return names.index(name)


def _bands(stack: np.ndarray, band_names: Sequence[str], spec: NdviSpec) -> Tuple[np.ndarray, np.ndarray]:
    s = np.asarray(stack, dtype=np.float64)
    if s.ndim != 3 or s.shape[0] != len(band_names):
        raise DownstreamError(f"the stack has {s.shape[0] if s.ndim == 3 else 'no'} channels but {len(band_names)} band names were given")
    return s[band_index(band_names, spec.red_band)], s[band_index(band_names, spec.nir_band)]


def ndvi_map(stack: np.ndarray, band_names: Sequence[str], spec: NdviSpec, valid: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
    """``(ndvi, computable)`` for a ``(C, H, W)`` reflectance stack. NDVI is NaN wherever it is not computable: outside ``valid``, non-finite inputs, or a denominator at or below the floor."""
    red, nir = _bands(stack, band_names, spec)
    finite = np.isfinite(red) & np.isfinite(nir)
    ok = finite if valid is None else finite & np.asarray(valid, dtype=bool)
    ndvi = np.full(red.shape, np.nan, dtype=np.float64)
    r, n = np.where(ok, red, 0.0), np.where(ok, nir, 0.0)
    raw, computable = compute_ndvi(n, r)
    computable = computable & ok & (np.abs(n + r) > spec.denominator_epsilon)
    ndvi[computable] = raw[computable]
    return ndvi, computable


def reference_valid_mask(reference: np.ndarray, band_names: Sequence[str], strict_mask: np.ndarray, spec: NdviSpec) -> np.ndarray:
    """Pixels the REFERENCE alone admits: strictly valid, finite, and Red + NIR reflectance above the minimum sum (dark pixels have an unstable NDVI)."""
    red, nir = _bands(reference, band_names, spec)
    return np.asarray(strict_mask, dtype=bool) & np.isfinite(red) & np.isfinite(nir) & ((red + nir) > spec.min_reflectance_sum)


def common_valid_mask(reference_valid: np.ndarray, computable_by_system: Sequence[np.ndarray]) -> np.ndarray:
    """The pixels scored for every system: valid for the reference and NDVI-computable for each system."""
    out = np.asarray(reference_valid, dtype=bool).copy()
    for c in computable_by_system:
        out &= np.asarray(c, dtype=bool)
    return out
