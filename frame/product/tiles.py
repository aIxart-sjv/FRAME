"""A minimal, real COG -> PNG XYZ tile renderer (Requirements 142 Requirement 11, Step I).

This project has no `titiler`/`rio-tiler` dependency (see docs/OPERATIONAL_DELIVERY.md for why
one was not added at this stage). This module is deliberately small and does exactly one thing:
read the requested Web Mercator XYZ tile window from a real COG via `rasterio.warp.reproject`
(the same reprojection/resampling primitive titiler itself is built on) and encode it as a PNG.
It is not a general tile server (no caching, no tile-matrix-set beyond XYZ/EPSG:3857, no vector
tiles) -- labelled here and in the API response as a minimal, in-process renderer, not "TiTiler".

Bands are 1-indexed (rasterio convention) throughout, matching how `frame.geospatial.write_cog`
lays a product's bands out (B04, B03, B02, B08 in that order for the RGBN SR product).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, Tuple, Union

import numpy as np
import rasterio
from rasterio.io import MemoryFile
from rasterio.transform import from_bounds
from rasterio.warp import Resampling, reproject

TILE_SIZE = 256
WEB_MERCATOR = "EPSG:3857"
_WEB_MERCATOR_HALF_EXTENT = 20037508.342789244  # metres; standard EPSG:3857 world extent / 2


class TileOutOfBoundsError(Exception):
    """Raised for a z/x/y that is not a valid Web Mercator XYZ tile index (never silently
    clamped -- a caller asking for tile (0, -1, 0) gets a clear error, not a black tile)."""


def tile_bounds_3857(z: int, x: int, y: int) -> Tuple[float, float, float, float]:
    """(minx, miny, maxx, maxy) of XYZ tile (z, x, y) in EPSG:3857 metres."""
    if z < 0 or not (0 <= x < 2**z) or not (0 <= y < 2**z):
        raise TileOutOfBoundsError(f"z={z}, x={x}, y={y} is not a valid XYZ tile index (0 <= x, y < 2**z).")
    n = 2**z
    tile_size_m = 2 * _WEB_MERCATOR_HALF_EXTENT / n
    minx = x * tile_size_m - _WEB_MERCATOR_HALF_EXTENT
    maxx = minx + tile_size_m
    maxy = _WEB_MERCATOR_HALF_EXTENT - y * tile_size_m
    miny = maxy - tile_size_m
    return minx, miny, maxx, maxy


def _percentile_stretch_to_uint8(band: np.ndarray, valid: np.ndarray, *, low: float = 2.0, high: float = 98.0) -> np.ndarray:
    if not valid.any():
        return np.zeros(band.shape, dtype="uint8")
    lo, hi = np.percentile(band[valid], [low, high])
    if hi <= lo:
        hi = lo + 1e-6
    safe_band = np.where(valid, band, lo)  # NaN outside the valid footprint must never reach the uint8 cast below
    stretched = np.clip((safe_band - lo) / (hi - lo), 0.0, 1.0)
    out = (stretched * 255.0).astype("uint8")
    out[~valid] = 0
    return out


def _reproject_bands(cog_path: Union[str, Path], band_indexes: Sequence[int], z: int, x: int, y: int, resampling: Resampling) -> Tuple[np.ndarray, np.ndarray]:
    minx, miny, maxx, maxy = tile_bounds_3857(z, x, y)
    dst_transform = from_bounds(minx, miny, maxx, maxy, TILE_SIZE, TILE_SIZE)
    with rasterio.open(cog_path) as src:
        dst = np.zeros((len(band_indexes), TILE_SIZE, TILE_SIZE), dtype="float32")
        reproject(
            source=rasterio.band(src, list(band_indexes)), destination=dst,
            src_transform=src.transform, src_crs=src.crs,
            dst_transform=dst_transform, dst_crs=WEB_MERCATOR,
            resampling=resampling, src_nodata=src.nodata, dst_nodata=np.nan,
        )
    valid = np.isfinite(dst).all(axis=0) & (dst != 0).any(axis=0)
    return dst, valid


def render_rgb_tile(cog_path: Union[str, Path], z: int, x: int, y: int, *, band_indexes: Sequence[int] = (1, 2, 3)) -> bytes:
    """RGB PNG tile from three bands of a real COG (default: bands 1,2,3 = B04,B03,B02 -- true
    colour -- for FRAME's RGBN product layout). Pixels with no data in the source are rendered
    black (this minimal renderer has no alpha channel -- see the module docstring)."""
    data, valid = _reproject_bands(cog_path, band_indexes, z, x, y, Resampling.bilinear)
    rgb = np.stack([_percentile_stretch_to_uint8(data[i], valid) for i in range(3)])
    return _encode_png(rgb)


def render_single_band_tile(cog_path: Union[str, Path], z: int, x: int, y: int, *, band_index: int = 1) -> bytes:
    """Greyscale PNG tile from one band (e.g. the overall TTA-stability band of an uncertainty
    GeoTIFF). Not a colour ramp -- brighter pixel = larger value in this band, nothing more."""
    data, valid = _reproject_bands(cog_path, [band_index], z, x, y, Resampling.bilinear)
    band = _percentile_stretch_to_uint8(data[0], valid)
    return _encode_png(np.stack([band, band, band]))


def _encode_png(rgb_uint8: np.ndarray) -> bytes:
    with MemoryFile() as memfile:
        with memfile.open(driver="PNG", height=rgb_uint8.shape[1], width=rgb_uint8.shape[2], count=rgb_uint8.shape[0], dtype="uint8") as dst:
            dst.write(rgb_uint8)
        return bytes(memfile.read())
