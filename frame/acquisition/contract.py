"""Typed contracts for FRAME's real Sentinel-2 acquisition layer (Requirements 142,
Section 1: Sentinel-2 Data Acquisition).

Three stages, three types:

    AcquisitionQuery   -- what the caller asked for (AOI, date range, cloud limit).
    SceneCandidate      -- one real scene a search provider returned for that query.
    RetrievedScene      -- the actual pixels + provenance for one candidate, after
                           real retrieval from a named provider.

Nothing here reaches the network; this module only defines the shapes that
`search.py` and `retrieve.py` produce and consume, plus the validation that
keeps a query from being silently accepted when it cannot mean anything (an
inverted date range, an out-of-range cloud threshold, a degenerate bbox).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Dict, Mapping, Optional, Sequence, Tuple

from frame.acquisition.errors import InvalidQueryError

#: (min_lon, min_lat, max_lon, max_lat) in EPSG:4326 -- the AOI shape every
#: provider in this package accepts, regardless of what CRS it retrieves in.
BBox = Tuple[float, float, float, float]


def parse_iso_datetime(value: str, *, field_name: str = "date") -> datetime:
    """Parse a date-only (``"2024-01-01"``) or full (``"2024-01-01T00:00:00Z"``)
    ISO 8601 string. Shared by AcquisitionQuery's own validation and by
    frame.acquisition.search, which must send a full RFC3339 datetime to CDSE's
    STAC API regardless of which form the caller used."""
    text = value.strip()
    for candidate_format in ("%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d"):
        try:
            return datetime.strptime(text, candidate_format)
        except ValueError:
            continue
    raise InvalidQueryError(f"{field_name}={value!r} is not a recognised ISO date/datetime.")


@dataclass(frozen=True)
class AcquisitionQuery:
    """A real search request: an AOI, a date range, and a cloud-cover ceiling.

    Dates are ISO 8601 (``"2024-01-01"`` or ``"2024-01-01T00:00:00Z"``); ``bbox`` is
    always EPSG:4326 degrees, independent of which CRS a provider ultimately
    retrieves pixels in. Validated eagerly so an invalid query fails at
    construction, not three network calls later.
    """

    bbox: BBox
    start_date: str
    end_date: str
    max_cloud_cover: float = 100.0
    collection: str = "sentinel-2-l2a"
    max_results: int = 10

    def __post_init__(self) -> None:
        min_lon, min_lat, max_lon, max_lat = self.bbox
        if not (-180.0 <= min_lon < max_lon <= 180.0):
            raise InvalidQueryError(f"bbox longitudes are not a valid, ordered range: {self.bbox}.")
        if not (-90.0 <= min_lat < max_lat <= 90.0):
            raise InvalidQueryError(f"bbox latitudes are not a valid, ordered range: {self.bbox}.")

        start = parse_iso_datetime(self.start_date, field_name="start_date")
        end = parse_iso_datetime(self.end_date, field_name="end_date")
        if not start < end:
            raise InvalidQueryError(f"start_date {self.start_date!r} is not before end_date {self.end_date!r}.")

        if not 0.0 <= self.max_cloud_cover <= 100.0:
            raise InvalidQueryError(f"max_cloud_cover must be in [0, 100], got {self.max_cloud_cover}.")

        if self.max_results < 1:
            raise InvalidQueryError(f"max_results must be >= 1, got {self.max_results}.")

    @property
    def center_lonlat(self) -> Tuple[float, float]:
        min_lon, min_lat, max_lon, max_lat = self.bbox
        return (min_lon + max_lon) / 2.0, (min_lat + max_lat) / 2.0


@dataclass(frozen=True)
class SceneCandidate:
    """One real scene a search provider returned -- every field here is copied
    verbatim from that provider's response, never invented or guessed."""

    id: str
    provider: str
    stac_endpoint: str
    collection: str
    datetime: str
    cloud_cover: Optional[float]
    bbox: BBox
    geometry: Optional[Mapping[str, object]]
    mgrs_tile: Optional[str]
    platform: Optional[str]
    processing_baseline: Optional[str]
    asset_hrefs: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class RetrievalProvenance:
    """Everything needed to answer "where did these pixels actually come from"
    without ambiguity -- see Requirements 142, Section 22 (Provenance)."""

    query: AcquisitionQuery
    candidate_id: str
    search_provider: str
    search_stac_endpoint: str
    retrieval_provider: str
    retrieval_endpoint: str
    matched_acquisition_datetime: str
    retrieved_at_utc: str
    band_sources: Mapping[str, str]
    warnings: Tuple[str, ...] = ()


@dataclass
class RetrievedScene:
    """Real pixels for one candidate, harmonised onto one grid, with full provenance.

    ``bands`` maps a canonical band name (frame.data.bands.L2A_BANDS) to a 2-D
    array on the common ``(height, width)`` grid at ``resolution_m``; ``scl`` is
    the scene classification layer on the same grid when it was requested.
    Bands retrieved natively at a coarser resolution than ``resolution_m`` are
    still present here (resampled), but their true native resolution is
    recorded per-band in ``band_native_resolution_m`` -- this array is never
    presented as if every band were natively acquired at ``resolution_m``.
    """

    candidate: SceneCandidate
    bands: Dict[str, "object"]
    band_native_resolution_m: Dict[str, float]
    scl: Optional["object"]
    crs: str
    transform: Tuple[float, float, float, float, float, float]
    resolution_m: float
    width: int
    height: int
    provenance: RetrievalProvenance
