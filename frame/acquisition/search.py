"""Real Sentinel-2 scene discovery via the Copernicus Data Space Ecosystem (CDSE)
STAC API -- the primary, credential-free discovery provider named in Requirements
142 Section 1 and Section 8 (`https://stac.dataspace.copernicus.eu/v1`, collection
`sentinel-2-l2a`; confirmed empirically -- `SENTINEL-2` is not a valid collection id
on this deployment).

This module only *discovers* scenes (id, footprint, cloud cover, datetime, asset
references); it never reads pixels. Pixel retrieval is a separate, explicit step
(frame.acquisition.providers) because CDSE's own STAC asset hrefs point at
``s3://eodata/...`` and require S3-specific credentials this project does not
have -- retrieval goes through the Sentinel Hub Process API or a fallback
provider instead, never by fetching an asset href directly.

``http_post`` is injectable so tests never need real network access or CDSE
credentials; the default is a real ``requests.post``.
"""

from __future__ import annotations

import os
from datetime import datetime
from typing import Callable, List, Optional

import requests

from frame.acquisition.contract import AcquisitionQuery, SceneCandidate, parse_iso_datetime
from frame.acquisition.errors import InvalidQueryError, NoScenesFoundError, ProviderUnavailableError

PROVIDER_NAME = "copernicus-cdse"
DEFAULT_STAC_URL = "https://stac.dataspace.copernicus.eu/v1"

HttpPost = Callable[..., "requests.Response"]


def stac_url() -> str:
    """The configured CDSE STAC endpoint, or the documented default."""
    return os.environ.get("FRAME_CDSE_STAC_URL", DEFAULT_STAC_URL)


def search_scenes(
    query: AcquisitionQuery,
    *,
    stac_endpoint: Optional[str] = None,
    http_post: HttpPost = requests.post,
    timeout_seconds: float = 30.0,
) -> List[SceneCandidate]:
    """Search real Sentinel-2 scenes matching ``query``.

    Raises:
        ProviderUnavailableError: the STAC endpoint could not be reached, or
            responded with a non-2xx status.
        NoScenesFoundError: the request was valid but zero scenes matched --
            never silently returned as an empty list, so a caller cannot
            mistake "no scenes" for "search not yet run".
    """
    if not isinstance(query, AcquisitionQuery):
        raise InvalidQueryError(f"search_scenes requires an AcquisitionQuery, got {type(query).__name__}.")

    endpoint = (stac_endpoint or stac_url()).rstrip("/")
    search_url = f"{endpoint}/search"
    body = {
        "collections": [query.collection],
        "bbox": list(query.bbox),
        "datetime": f"{_rfc3339(query.start_date)}/{_rfc3339(query.end_date, end_of_day=True)}",
        "limit": query.max_results,
    }

    try:
        response = http_post(search_url, json=body, timeout=timeout_seconds)
    except requests.RequestException as exc:
        raise ProviderUnavailableError(f"CDSE STAC search failed: {exc}", provider=PROVIDER_NAME) from exc

    if response.status_code >= 400:
        raise ProviderUnavailableError(
            f"CDSE STAC search returned HTTP {response.status_code}: {response.text[:500]}",
            provider=PROVIDER_NAME,
        )

    payload = response.json()
    features = payload.get("features", [])

    candidates = [
        _candidate_from_feature(feature, search_url, query.collection)
        for feature in features
        if _passes_cloud_filter(feature, query.max_cloud_cover)
    ]

    if not candidates:
        raise NoScenesFoundError(
            f"No Sentinel-2 scenes found for bbox={query.bbox}, "
            f"{query.start_date}/{query.end_date}, max_cloud_cover={query.max_cloud_cover}."
        )
    return candidates


def _rfc3339(value: str, *, end_of_day: bool = False) -> str:
    """CDSE's STAC v1 endpoint requires a full RFC3339 datetime (rejecting a bare
    ``"2024-01-01"`` with a 400 -- confirmed against the real API), while
    `AcquisitionQuery` itself accepts either form for a human-friendly query. A
    date with no time is normalised to midnight (start) or 23:59:59 (end), so an
    end date genuinely includes the whole of that day rather than excluding it."""
    parsed = parse_iso_datetime(value, field_name="date")
    if end_of_day and parsed.time() == datetime.min.time() and len(value.strip()) <= 10:
        parsed = parsed.replace(hour=23, minute=59, second=59)
    return parsed.strftime("%Y-%m-%dT%H:%M:%SZ")


def _passes_cloud_filter(feature: dict, max_cloud_cover: float) -> bool:
    """CDSE's server-side cloud filtering support is not assumed -- every
    candidate is re-checked client-side against the feature's own reported
    ``eo:cloud_cover`` so behaviour never silently depends on an unverified
    server-side query extension. A feature with no reported cloud cover is
    kept (never excluded on a value we do not actually have)."""
    cloud_cover = feature.get("properties", {}).get("eo:cloud_cover")
    return cloud_cover is None or cloud_cover <= max_cloud_cover


def _candidate_from_feature(feature: dict, stac_endpoint: str, collection: str) -> SceneCandidate:
    props = feature.get("properties", {})
    assets = feature.get("assets", {})
    bbox = feature.get("bbox")
    return SceneCandidate(
        id=feature["id"],
        provider=PROVIDER_NAME,
        stac_endpoint=stac_endpoint,
        collection=collection,
        datetime=props.get("datetime", ""),
        cloud_cover=props.get("eo:cloud_cover"),
        bbox=tuple(bbox) if bbox else (0.0, 0.0, 0.0, 0.0),
        geometry=feature.get("geometry"),
        mgrs_tile=props.get("s2:mgrs_tile") or props.get("grid:code"),
        platform=props.get("platform"),
        processing_baseline=props.get("s2:processing_baseline") or props.get("processing:version"),
        asset_hrefs={
            name: asset["href"]
            for name, asset in assets.items()
            if isinstance(asset, dict) and asset.get("href")
        },
    )
