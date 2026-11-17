"""Small, dependency-light geometry helpers shared by the acquisition providers.

An AcquisitionQuery's bbox is always EPSG:4326 degrees (see contract.py); real
pixel retrieval needs a projected CRS with metre-sized pixels so a requested
``resolution_m`` means what it says. This module picks the correct UTM zone
for an AOI and reprojects its bbox there -- nothing here fabricates a
resolution or CRS silently.
"""

from __future__ import annotations

from typing import Iterable, Mapping, Tuple

from rasterio.warp import transform_bounds

from frame.acquisition.contract import BBox
from frame.acquisition.errors import InvalidQueryError


def utm_epsg_for_lonlat(lon: float, lat: float) -> str:
    """The EPSG code of the UTM zone containing (lon, lat) -- ``EPSG:326xx``
    north of the equator, ``EPSG:327xx`` south of it."""
    zone = int((lon + 180.0) // 6.0) + 1
    zone = min(max(zone, 1), 60)
    base = 32600 if lat >= 0 else 32700
    return f"EPSG:{base + zone}"


def _flatten_positions(coordinates) -> Iterable[Tuple[float, float]]:
    """Recursively yield ``(lon, lat)`` pairs from a GeoJSON ``coordinates`` array of
    any nesting depth (Point, Polygon, MultiPolygon, ...) -- a raw coordinate pair is
    a 2-tuple/list of numbers; anything else is a further-nested list to recurse into."""
    if len(coordinates) >= 2 and all(isinstance(v, (int, float)) for v in coordinates[:2]):
        yield float(coordinates[0]), float(coordinates[1])
        return
    for item in coordinates:
        yield from _flatten_positions(item)


def bbox_from_geojson(geojson: Mapping[str, object]) -> BBox:
    """The EPSG:4326 bounding box of a GeoJSON ``Feature``, ``Polygon`` or
    ``MultiPolygon``.

    This reduces an arbitrary AOI polygon to its bounding box -- real
    intersection-based filtering against the polygon's actual shape is not
    implemented (see docs/ACQUISITION.md); every real Sentinel-2 scene this
    bbox returns is a real candidate for the AOI, but a caller expecting the
    search to exclude scenes that only intersect the bbox (not the polygon
    itself) is not yet supported.
    """
    geometry = geojson.get("geometry", geojson) if geojson.get("type") == "Feature" else geojson
    coordinates = geometry.get("coordinates")
    if coordinates is None:
        raise InvalidQueryError(f"GeoJSON has no 'coordinates': {geojson!r}.")

    positions = list(_flatten_positions(coordinates))
    if not positions:
        raise InvalidQueryError(f"GeoJSON has no coordinate positions: {geojson!r}.")

    lons = [p[0] for p in positions]
    lats = [p[1] for p in positions]
    return (min(lons), min(lats), max(lons), max(lats))


def bbox_to_utm(bbox: BBox) -> Tuple[str, Tuple[float, float, float, float]]:
    """Reproject an EPSG:4326 ``bbox`` to its UTM zone.

    Returns ``(utm_crs, utm_bbox)`` where ``utm_bbox`` is
    ``(min_x, min_y, max_x, max_y)`` in metres.
    """
    min_lon, min_lat, max_lon, max_lat = bbox
    center_lon = (min_lon + max_lon) / 2.0
    center_lat = (min_lat + max_lat) / 2.0
    crs = utm_epsg_for_lonlat(center_lon, center_lat)
    utm_bbox = transform_bounds("EPSG:4326", crs, min_lon, min_lat, max_lon, max_lat)
    return crs, utm_bbox
