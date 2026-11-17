import pytest

from frame.acquisition.contract import AcquisitionQuery, SceneCandidate
from frame.acquisition.errors import InvalidQueryError

VALID_BBOX = (80.12, 12.85, 80.32, 13.15)


def test_a_valid_query_is_accepted():
    query = AcquisitionQuery(bbox=VALID_BBOX, start_date="2024-01-01", end_date="2024-04-01")
    assert query.collection == "sentinel-2-l2a"
    assert query.max_cloud_cover == 100.0


def test_center_lonlat_is_the_bbox_midpoint():
    query = AcquisitionQuery(bbox=(0.0, 0.0, 2.0, 4.0), start_date="2024-01-01", end_date="2024-04-01")
    assert query.center_lonlat == (1.0, 2.0)


@pytest.mark.parametrize(
    "bbox",
    [
        (80.32, 12.85, 80.12, 13.15),  # min_lon > max_lon
        (80.12, 13.15, 80.32, 12.85),  # min_lat > max_lat
        (-190.0, 12.85, 80.32, 13.15),  # out of range longitude
        (80.12, 12.85, 80.32, 95.0),  # out of range latitude
    ],
)
def test_an_invalid_bbox_is_rejected(bbox):
    with pytest.raises(InvalidQueryError):
        AcquisitionQuery(bbox=bbox, start_date="2024-01-01", end_date="2024-04-01")


def test_an_inverted_date_range_is_rejected():
    with pytest.raises(InvalidQueryError):
        AcquisitionQuery(bbox=VALID_BBOX, start_date="2024-04-01", end_date="2024-01-01")


def test_an_unparseable_date_is_rejected():
    with pytest.raises(InvalidQueryError):
        AcquisitionQuery(bbox=VALID_BBOX, start_date="not-a-date", end_date="2024-04-01")


@pytest.mark.parametrize("max_cloud_cover", [-1.0, 100.1])
def test_an_out_of_range_cloud_cover_is_rejected(max_cloud_cover):
    with pytest.raises(InvalidQueryError):
        AcquisitionQuery(bbox=VALID_BBOX, start_date="2024-01-01", end_date="2024-04-01", max_cloud_cover=max_cloud_cover)


def test_a_zero_max_results_is_rejected():
    with pytest.raises(InvalidQueryError):
        AcquisitionQuery(bbox=VALID_BBOX, start_date="2024-01-01", end_date="2024-04-01", max_results=0)


def test_scene_candidate_holds_real_fields_verbatim():
    candidate = SceneCandidate(
        id="S2B_MSIL2A_20240330T045659_N0510_R119_T44PMV_20240330T072508",
        provider="copernicus-cdse",
        stac_endpoint="https://stac.dataspace.copernicus.eu/v1/search",
        collection="sentinel-2-l2a",
        datetime="2024-03-30T04:56:59.024000Z",
        cloud_cover=2.43,
        bbox=(80.075, 12.574, 81.090, 13.568),
        geometry=None,
        mgrs_tile="44PMV",
        platform="sentinel-2b",
        processing_baseline="05.10",
        asset_hrefs={"B04_10m": "s3://eodata/..."},
    )
    assert candidate.cloud_cover == 2.43
    assert candidate.asset_hrefs["B04_10m"].startswith("s3://")
