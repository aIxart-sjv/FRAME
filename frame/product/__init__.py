"""GIS-ready product packaging (Requirements 142 Requirement 11, Step I).

    frame.product.stac    -- build_stac_item, validate_stac_item, RECONSTRUCTED_GRID_STATEMENT
    frame.product.tiles   -- render_rgb_tile, render_single_band_tile, tile_bounds_3857 (Step I)
    frame.product.errors  -- ProductError, MissingGeospatialContextError

COG writing/validation lives in `frame.geospatial` (write_cog, validate_cog) -- that is ordinary
raster I/O, not product packaging. This package is the one place a completed SR job's real
metadata (model, checkpoint, provenance, bands, resolution) is turned into a STAC Item, and the
one place a COG is rendered into an XYZ map tile; it never computes SR, never touches the model,
and never invents a field or a pixel it was not given.
"""

from frame.product.errors import MissingGeospatialContextError, ProductError
from frame.product.stac import (
    RECONSTRUCTED_GRID_STATEMENT,
    STAC_VERSION,
    build_stac_item,
    validate_stac_item,
)
from frame.product.tiles import (
    TILE_SIZE,
    WEB_MERCATOR,
    TileOutOfBoundsError,
    render_rgb_tile,
    render_single_band_tile,
    tile_bounds_3857,
)

__all__ = [
    "ProductError",
    "MissingGeospatialContextError",
    "build_stac_item",
    "validate_stac_item",
    "RECONSTRUCTED_GRID_STATEMENT",
    "STAC_VERSION",
    "render_rgb_tile",
    "render_single_band_tile",
    "tile_bounds_3857",
    "TileOutOfBoundsError",
    "TILE_SIZE",
    "WEB_MERCATOR",
]
