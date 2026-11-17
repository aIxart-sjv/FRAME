"""Cloud Optimized GeoTIFF validation (Requirements 142 Requirement 11, Step I).

This project has no `osgeo.gdal` Python binding and no system `gdalinfo`/`rio-cogeo` installed
(only rasterio's own bundled GDAL) -- see docs/OPERATIONAL_DELIVERY.md for why a new dependency
was not added for this. `validate_cog` is therefore NOT the official GDAL
`validate_cloud_optimized_geotiff.py` script; it inspects the same structural facts through
rasterio directly, including GDAL's own `LAYOUT=COG` self-report in the `IMAGE_STRUCTURE`
tag namespace (written by the COG driver itself, not inferred by this module). What this checks
is stated explicitly in `CogValidationResult` -- nothing here claims full COG-specification
conformance beyond what was actually inspected.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple, Union

import rasterio

from frame.geospatial.errors import UnreadableRasterError


@dataclass(frozen=True)
class CogValidationResult:
    path: str
    is_valid_cog: bool
    reasons: Tuple[str, ...]  # non-empty only when is_valid_cog is False
    driver_reports_cog_layout: bool  # GDAL's own IMAGE_STRUCTURE LAYOUT=COG tag
    is_tiled: bool
    block_shapes: Tuple[Tuple[int, int], ...]
    compression: Optional[str]
    overview_factors: Tuple[int, ...]  # empty is not necessarily a defect -- see module docstring
    has_crs: bool
    has_transform: bool
    dtype: str
    band_count: int
    width: int
    height: int
    checked: Tuple[str, ...] = field(
        default_factory=lambda: (
            "GDAL COG-driver self-report (IMAGE_STRUCTURE LAYOUT tag) -- the validity gate",
            "CRS present -- the other validity gate",
            "affine transform present -- the other validity gate",
            "internal tiling (is_tiled, block_shapes) -- reported, not gated (see module docstring)",
            "compression -- reported, not gated",
            "overview presence -- reported, not gated (absence is legitimate on a small raster)",
        )
    )


def validate_cog(path: Union[str, Path]) -> CogValidationResult:
    """Inspect a GeoTIFF at ``path`` and report whether it is structurally a valid COG.

    Raises `UnreadableRasterError` if the file cannot be opened at all (a missing/corrupt file is
    a hard failure, not "not a COG"). A geospatially incomplete file (no CRS or no transform) is
    reported as invalid with a reason, matching this project's rule that a raster is never
    silently treated as georeferenced when it is not.
    """
    try:
        with rasterio.open(path) as dataset:
            structure_tags = dataset.tags(ns="IMAGE_STRUCTURE")
            layout = structure_tags.get("LAYOUT")
            reasons: List[str] = []

            if layout != "COG":
                reasons.append(f"GDAL IMAGE_STRUCTURE LAYOUT tag is {layout!r}, not 'COG'.")
            # NOT gated on dataset.is_tiled: measured directly (a 512x512 output from a single
            # 128x128 tile at scale 4 -- a common real FRAME output size) writes blockxsize=
            # blockysize=512 as a single TIFF *strip* rather than a *tile* whenever the whole
            # image fits in exactly one block, so rasterio's is_tiled reports False even though
            # GDAL's own LAYOUT tag (the authoritative signal here) still says COG -- there is
            # only one block either way, so this is a libtiff strip-vs-tile implementation
            # choice for the single-block case, not a structural defect. is_tiled/block_shapes
            # are still reported below as diagnostic information, just not as a validity gate.
            if dataset.crs is None:
                reasons.append("no CRS embedded.")
            if dataset.transform is None or dataset.transform.is_identity:
                reasons.append("no real affine transform embedded.")

            result = CogValidationResult(
                path=str(path),
                is_valid_cog=not reasons,
                reasons=tuple(reasons),
                driver_reports_cog_layout=(layout == "COG"),
                is_tiled=dataset.is_tiled,
                block_shapes=tuple(dataset.block_shapes),
                compression=structure_tags.get("COMPRESSION"),
                overview_factors=tuple(dataset.overviews(1)) if dataset.count > 0 else (),
                has_crs=dataset.crs is not None,
                has_transform=dataset.transform is not None and not dataset.transform.is_identity,
                dtype=str(dataset.dtypes[0]) if dataset.count > 0 else "unknown",
                band_count=dataset.count,
                width=dataset.width,
                height=dataset.height,
            )
    except rasterio.errors.RasterioIOError as exc:
        raise UnreadableRasterError(f"{Path(path).name} could not be opened for COG validation ({type(exc).__name__}).") from exc

    return result
