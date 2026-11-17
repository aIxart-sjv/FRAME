import pytest

from frame.acquisition.errors import InvalidQueryError
from frame.acquisition.geometry import bbox_from_geojson, bbox_to_utm, utm_epsg_for_lonlat


def test_utm_zone_north_of_equator():
    # Chennai, India: ~80.2 E, ~13.0 N -> UTM zone 44N
    assert utm_epsg_for_lonlat(80.2144, 13.0125) == "EPSG:32644"


def test_utm_zone_south_of_equator():
    # Sydney, Australia: ~151.2 E, ~-33.9 -> UTM zone 56S
    assert utm_epsg_for_lonlat(151.2, -33.9) == "EPSG:32756"


def test_bbox_to_utm_preserves_real_area_order_of_magnitude():
    # A ~0.2 x 0.3 degree box near the equator is roughly 20km x 33km -- confirm the
    # reprojected box is the right order of magnitude, not degrees mistaken for metres.
    crs, (min_x, min_y, max_x, max_y) = bbox_to_utm((80.12, 12.85, 80.32, 13.15))
    assert crs == "EPSG:32644"
    width_m = max_x - min_x
    height_m = max_y - min_y
    assert 15_000 < width_m < 30_000
    assert 25_000 < height_m < 40_000


def test_bbox_from_geojson_polygon():
    polygon = {
        "type": "Polygon",
        "coordinates": [[[80.12, 12.85], [80.32, 12.85], [80.32, 13.15], [80.12, 13.15], [80.12, 12.85]]],
    }
    assert bbox_from_geojson(polygon) == (80.12, 12.85, 80.32, 13.15)


def test_bbox_from_geojson_feature_wrapper():
    feature = {
        "type": "Feature",
        "properties": {},
        "geometry": {"type": "Polygon", "coordinates": [[[80.12, 12.85], [80.32, 13.15], [80.12, 13.15]]]},
    }
    assert bbox_from_geojson(feature) == (80.12, 12.85, 80.32, 13.15)


def test_bbox_from_geojson_multipolygon():
    multipolygon = {
        "type": "MultiPolygon",
        "coordinates": [
            [[[80.12, 12.85], [80.20, 12.85], [80.20, 13.00]]],
            [[[80.25, 13.00], [80.32, 13.15], [80.25, 13.15]]],
        ],
    }
    assert bbox_from_geojson(multipolygon) == (80.12, 12.85, 80.32, 13.15)


def test_bbox_from_geojson_without_coordinates_is_invalid():
    with pytest.raises(InvalidQueryError):
        bbox_from_geojson({"type": "Polygon"})
