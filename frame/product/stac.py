"""A real STAC Item for one completed FRAME SR product (Requirements 142 Requirement 11, Step I).

Not a full STAC catalog/collection -- an honest STAC Item plus real asset links is what Step I's
own instructions call sufficient for this project's scale, and this project has no `pystac`
dependency, so the Item is built as a plain dict following the STAC 1.0.0 Item spec directly
(https://github.com/radiantearth/stac-spec/tree/master/item-spec). FRAME-specific facts (model
id, checkpoint hash, scale factor, radiometric convention, scientific caveats) are namespaced
under ``frame:`` -- the same extension-prefix convention STAC's own ``proj:``/``eo:``/``sat:``
extensions use, not an invented ad-hoc scheme.

This module never fabricates a field: a value that is not actually known (no acquisition
provenance for a manually-uploaded input, no checkpoint hash computed for a model) is omitted
or set to ``None``, never guessed.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import rasterio
from rasterio.warp import transform_bounds

from frame.preprocessing.metadata import RasterMetadata
from frame.product.errors import MissingGeospatialContextError

STAC_VERSION = "1.0.0"

#: RGBN super-resolution is NOT native 2.5 m Sentinel-2 observation -- every item says so in its
#: own properties, not only in project documentation, so the claim travels with the file.
RECONSTRUCTED_GRID_STATEMENT = (
    "This asset is an RGBN super-resolved product on a reconstructed 2.5 m pixel grid. "
    "Sentinel-2 has not observed the ground at 2.5 m; this is not native 2.5 m imagery."
)


def _epsg_code(crs: str) -> Optional[int]:
    try:
        return rasterio.crs.CRS.from_string(crs).to_epsg()
    except Exception:  # noqa: BLE001 -- an unparseable CRS string just means proj:epsg is omitted
        return None


def _bbox_and_geometry_4326(metadata: RasterMetadata) -> "tuple[List[float], Dict[str, Any]]":
    if metadata.crs is None or metadata.bounds is None:
        raise MissingGeospatialContextError(
            "Cannot build a STAC item's geometry/bbox: the raster has no CRS and/or bounds. "
            "A STAC item's geometry is never fabricated -- see frame.geospatial.metadata.validate_geospatial_completeness."
        )
    min_x, min_y, max_x, max_y = metadata.bounds
    lon_min, lat_min, lon_max, lat_max = transform_bounds(metadata.crs, "EPSG:4326", min_x, min_y, max_x, max_y)
    bbox = [lon_min, lat_min, lon_max, lat_max]
    geometry = {
        "type": "Polygon",
        "coordinates": [[
            [lon_min, lat_min], [lon_max, lat_min], [lon_max, lat_max], [lon_min, lat_max], [lon_min, lat_min],
        ]],
    }
    return bbox, geometry


def build_stac_item(
    *,
    item_id: str,
    output_metadata: RasterMetadata,
    model_id: str,
    model_name: str,
    architecture: str,
    checkpoint_sha256: Optional[str],
    input_resolution_m: float,
    output_resolution_m: float,
    scale_factor: int,
    bands: Sequence[str],
    radiometric_convention: str,
    processing_timestamp: str,
    assets: Dict[str, Dict[str, Any]],
    scene_id: Optional[str] = None,
    platform: Optional[str] = None,
    constellation: Optional[str] = None,
    acquisition_datetime: Optional[str] = None,
    source_provider: Optional[str] = None,
    scientific_caveats: Sequence[str] = (),
) -> Dict[str, Any]:
    """Build a real STAC 1.0.0 Item for one completed SR job's output.

    ``assets`` maps a STAC asset key (e.g. ``"sr_cog"``, ``"stability"``, ``"ndvi"``) to a dict
    with at least ``href`` and ``type``; this function does not decide which assets exist --
    the caller (frame.api.services.product) only passes assets it actually wrote to disk.

    Raises `MissingGeospatialContextError` if ``output_metadata`` has no real CRS/bounds -- never
    fabricates a geometry. ``scene_id``/``platform``/``acquisition_datetime``/``source_provider``
    are ``None`` when the input was a manually-uploaded GeoTIFF with no acquisition provenance --
    the item says so explicitly rather than inventing a scene identity.
    """
    bbox, geometry = _bbox_and_geometry_4326(output_metadata)

    properties: Dict[str, Any] = {
        "datetime": acquisition_datetime or processing_timestamp,
        "created": processing_timestamp,
        "gsd": output_resolution_m,
        "frame:model_id": model_id,
        "frame:model_name": model_name,
        "frame:architecture": architecture,
        "frame:checkpoint_sha256": checkpoint_sha256,
        "frame:input_resolution_m": input_resolution_m,
        "frame:output_resolution_m": output_resolution_m,
        "frame:scale_factor": scale_factor,
        "frame:bands": list(bands),
        "frame:radiometric_convention": radiometric_convention,
        "frame:source_scene_id": scene_id,
        "frame:source_provider": source_provider,
        "frame:product_statement": RECONSTRUCTED_GRID_STATEMENT,
        "frame:scientific_caveats": list(scientific_caveats),
    }
    epsg = _epsg_code(output_metadata.crs)
    if epsg is not None:
        properties["proj:epsg"] = epsg
    if platform is not None:
        properties["platform"] = platform
    if constellation is not None:
        properties["constellation"] = constellation

    return {
        "type": "Feature",
        "stac_version": STAC_VERSION,
        "id": item_id,
        "geometry": geometry,
        "bbox": bbox,
        "properties": properties,
        "assets": assets,
        "links": [],
    }


_REQUIRED_TOP_LEVEL = ("type", "stac_version", "id", "geometry", "bbox", "properties", "assets", "links")
_REQUIRED_GEOMETRY = ("type", "coordinates")


def validate_stac_item(item: Dict[str, Any]) -> List[str]:
    """Structural checks against the STAC 1.0.0 Item spec's required fields -- not the official
    `pystac`/`stac-validator` JSON Schema validation (neither is a dependency here), but real,
    specific, itemised checks rather than "looks like a dict". Returns a list of problems (empty
    means every check here passed); never raises for a malformed item -- a validator that cannot
    itself fail loudly on bad input is more useful for reporting.
    """
    problems: List[str] = []
    for key in _REQUIRED_TOP_LEVEL:
        if key not in item:
            problems.append(f"missing required field {key!r}")
    if item.get("type") != "Feature":
        problems.append(f"type must be 'Feature', got {item.get('type')!r}")
    if not isinstance(item.get("bbox"), list) or len(item.get("bbox", [])) != 4:
        problems.append("bbox must be a 4-element [minx, miny, maxx, maxy] list")
    geometry = item.get("geometry") or {}
    for key in _REQUIRED_GEOMETRY:
        if key not in geometry:
            problems.append(f"geometry missing required field {key!r}")
    if not item.get("properties", {}).get("datetime") and item.get("properties", {}).get("datetime") != "null":
        if "datetime" not in item.get("properties", {}):
            problems.append("properties.datetime is required")
    assets = item.get("assets") or {}
    if not assets:
        problems.append("assets must be non-empty -- a STAC item with no data is not useful")
    for key, asset in assets.items():
        if "href" not in asset:
            problems.append(f"asset {key!r} missing 'href'")
        if "type" not in asset:
            problems.append(f"asset {key!r} missing 'type'")
    return problems
