"""Real, multi-resolution pixel retrieval for one selected scene, harmonised
onto a single grid.

Sentinel-2 L2A bands are natively acquired at three different resolutions
(10 m / 20 m / 60 m) plus a 20 m scene-classification layer (SCL); this
module retrieves each native-resolution group as its own real request (never
requesting a mix of resolutions in one call and pretending the result is
uniform), then resamples every group onto one common grid -- bilinear for
continuous reflectance bands, nearest-neighbour for the categorical SCL
layer, because averaging class codes together would produce meaningless
values.

Provider preference is CDSE (primary, credentialed, Copernicus-native) with
Microsoft Planetary Computer as a fallback (secondary, credential-free, same
underlying archive) -- see frame.acquisition.providers. Retrieval is scoped
to a tight time window around the *selected candidate's own* acquisition
datetime, not the full search date range, so a scene can never be silently
substituted for a different real acquisition than the one the user picked.
"""

from __future__ import annotations

import datetime as dt
from typing import Dict, List, Sequence, Tuple

import numpy as np

from frame.acquisition import geometry
from frame.acquisition.contract import AcquisitionQuery, RetrievalProvenance, RetrievedScene, SceneCandidate
from frame.acquisition.errors import ProviderUnavailableError
from frame.acquisition.providers import cdse as cdse_provider
from frame.acquisition.providers import planetary_computer
from frame.data.bands import canonical_bands, native_resolution_m

#: Sentinel-2's ~5-day revisit means this window contains at most the one
#: real acquisition the candidate names.
_ACQUISITION_WINDOW = dt.timedelta(hours=12)

_SCL_NATIVE_RESOLUTION_M = 20.0


def _parse_candidate_datetime(candidate: SceneCandidate) -> dt.datetime:
    text = candidate.datetime.split(".")[0].rstrip("Z")
    return dt.datetime.strptime(text, "%Y-%m-%dT%H:%M:%S")


def _time_window(candidate: SceneCandidate) -> Tuple[str, str]:
    when = _parse_candidate_datetime(candidate)
    start = (when - _ACQUISITION_WINDOW).strftime("%Y-%m-%dT%H:%M:%SZ")
    end = (when + _ACQUISITION_WINDOW).strftime("%Y-%m-%dT%H:%M:%SZ")
    return start, end


def _group_bands_by_resolution(bands: Sequence[str]) -> Dict[float, List[str]]:
    groups: Dict[float, List[str]] = {}
    for band in canonical_bands(bands):
        resolution = native_resolution_m(band)
        groups.setdefault(resolution, []).append(band)
    return groups


def _resample_to_grid(array: np.ndarray, *, to_height: int, to_width: int, categorical: bool) -> np.ndarray:
    """Resample a (bands, H, W) array onto a (bands, to_height, to_width) grid."""
    from scipy.ndimage import zoom

    _, from_height, from_width = array.shape
    zoom_factors = (1.0, to_height / from_height, to_width / from_width)
    order = 0 if categorical else 1
    return zoom(array, zoom_factors, order=order, mode="nearest")


def _retrieve_group(
    *,
    auth: cdse_provider.CDSEAuth,
    utm_crs: str,
    utm_bbox: Tuple[float, float, float, float],
    time_range: Tuple[str, str],
    bands: Sequence[str],
    resolution_m: float,
    query: AcquisitionQuery,
    candidate: SceneCandidate,
    categorical: bool,
) -> Tuple[np.ndarray, str]:
    """Retrieve one native-resolution band group. Tries CDSE first; falls back to
    Planetary Computer only if CDSE is unconfigured or fails, returning which
    provider actually served the pixels alongside the array."""
    if auth.configured():
        try:
            array, _, _ = cdse_provider.retrieve_bands(
                auth=auth,
                utm_crs=utm_crs,
                utm_bbox=utm_bbox,
                time_range=time_range,
                bands=bands,
                resolution_m=resolution_m,
                max_cloud_cover=query.max_cloud_cover,
                categorical=categorical,
            )
            return array, cdse_provider.PROVIDER_NAME
        except ProviderUnavailableError:
            pass  # fall through to the secondary provider, recorded below

    lon, lat = query.center_lonlat
    min_x, min_y, max_x, max_y = utm_bbox
    extent_m = max(max_x - min_x, max_y - min_y)
    edge_size_px = max(1, round(extent_m / resolution_m))
    data_array = planetary_computer.retrieve_bands(
        lat=lat,
        lon=lon,
        bands=bands,
        resolution_m=resolution_m,
        edge_size_px=edge_size_px,
        start_date=time_range[0],
        end_date=time_range[1],
        max_cloud_cover=query.max_cloud_cover,
        categorical=categorical,
    )
    target_time = np.datetime64(_parse_candidate_datetime(candidate))
    time_diffs = np.abs(data_array["time"].values - target_time)
    nearest_index = int(np.argmin(time_diffs))
    array = data_array.isel(time=nearest_index).values
    return array, planetary_computer.PROVIDER_NAME


def retrieve_scene(
    candidate: SceneCandidate,
    query: AcquisitionQuery,
    *,
    bands: Sequence[str],
    resolution_m: float = 10.0,
    include_scl: bool = True,
    cdse_auth: cdse_provider.CDSEAuth = None,
) -> RetrievedScene:
    """Real multi-resolution pixel retrieval for ``candidate``, harmonised onto a
    single ``resolution_m`` grid covering ``query.bbox``.

    Raises ProviderUnavailableError if a required band group cannot be served
    by either provider -- never a partially-fabricated RetrievedScene.
    """
    auth = cdse_auth if cdse_auth is not None else cdse_provider.CDSEAuth()
    utm_crs, utm_bbox = geometry.bbox_to_utm(query.bbox)
    time_range = _time_window(candidate)
    min_x, min_y, max_x, max_y = utm_bbox
    target_width = max(1, round((max_x - min_x) / resolution_m))
    target_height = max(1, round((max_y - min_y) / resolution_m))

    groups = _group_bands_by_resolution(bands)
    band_arrays: Dict[str, np.ndarray] = {}
    band_native_resolution_m: Dict[str, float] = {}
    band_sources: Dict[str, str] = {}
    warnings: List[str] = []

    for group_resolution in sorted(groups):
        group_bands = groups[group_resolution]
        # Never request more native detail from a real provider than the target grid can use: if the
        # target is coarser than this group's native resolution, fetch directly at the target
        # resolution (the provider does its own real downsampling) rather than fetching the full
        # native-resolution pixel count and discarding the extra detail locally. This keeps a real
        # request's actual size bounded by the analyst's chosen output grid, not by the AOI's raw
        # extent at native resolution -- verified live to matter: an 80 km AOI at native 10 m is a
        # ~7900x7900-pixel real Process API request regardless of how coarse the final grid is.
        fetch_resolution = max(group_resolution, resolution_m)
        array, source = _retrieve_group(
            auth=auth,
            utm_crs=utm_crs,
            utm_bbox=utm_bbox,
            time_range=time_range,
            bands=group_bands,
            resolution_m=fetch_resolution,
            query=query,
            candidate=candidate,
            categorical=False,
        )
        if array.shape[1:] != (target_height, target_width):
            array = _resample_to_grid(array, to_height=target_height, to_width=target_width, categorical=False)
            warnings.append(
                f"{group_bands} resampled from native {group_resolution} m to "
                f"{resolution_m} m grid (bilinear)."
            )
        elif fetch_resolution != group_resolution:
            warnings.append(
                f"{group_bands} requested directly at the coarser {fetch_resolution} m target grid rather "
                f"than their native {group_resolution} m (the provider performed the downsampling, not "
                "this project) -- no finer detail than the target grid was fetched for this AOI."
            )
        if source == planetary_computer.PROVIDER_NAME:
            warnings.append(
                f"{group_bands} served by {planetary_computer.PROVIDER_NAME} using a flat DN/10000 "
                "reflectance conversion (frame.preprocessing.reflectance), which does not apply the "
                "Sentinel-2 processing-baseline radiometric offset that CDSE's own Process API reflectance "
                "units may already include for this scene -- a small, real, disclosed discrepancy versus "
                "bands served directly by CDSE, not a silently absorbed one."
            )
        for index, band in enumerate(group_bands):
            band_arrays[band] = array[index]
            band_native_resolution_m[band] = group_resolution
            band_sources[band] = source

    scl_array = None
    if include_scl:
        scl_fetch_resolution = max(_SCL_NATIVE_RESOLUTION_M, resolution_m)
        scl_raw, scl_source = _retrieve_group(
            auth=auth,
            utm_crs=utm_crs,
            utm_bbox=utm_bbox,
            time_range=time_range,
            bands=["SCL"],
            resolution_m=scl_fetch_resolution,
            query=query,
            candidate=candidate,
            categorical=True,
        )
        scl_array = scl_raw[0]
        if scl_array.shape != (target_height, target_width):
            scl_array = _resample_to_grid(
                scl_raw, to_height=target_height, to_width=target_width, categorical=True
            )[0]
            warnings.append(
                f"SCL resampled from native {_SCL_NATIVE_RESOLUTION_M} m to "
                f"{resolution_m} m grid (nearest-neighbour)."
            )
        elif scl_fetch_resolution != _SCL_NATIVE_RESOLUTION_M:
            warnings.append(
                f"SCL requested directly at the coarser {scl_fetch_resolution} m target grid rather than "
                f"its native {_SCL_NATIVE_RESOLUTION_M} m -- no finer detail than the target grid was "
                "fetched for this AOI."
            )
        band_sources["SCL"] = scl_source

    providers_used = sorted(set(band_sources.values()))
    provenance = RetrievalProvenance(
        query=query,
        candidate_id=candidate.id,
        search_provider=candidate.provider,
        search_stac_endpoint=candidate.stac_endpoint,
        retrieval_provider="+".join(providers_used),
        retrieval_endpoint=cdse_provider.process_url() if cdse_provider.PROVIDER_NAME in providers_used else planetary_computer.STAC_ENDPOINT,
        matched_acquisition_datetime=candidate.datetime,
        retrieved_at_utc=dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        band_sources=dict(band_sources),
        warnings=tuple(warnings),
    )

    output_transform = (resolution_m, 0.0, min_x, 0.0, -resolution_m, max_y)

    return RetrievedScene(
        candidate=candidate,
        bands=band_arrays,
        band_native_resolution_m=band_native_resolution_m,
        scl=scl_array,
        crs=utm_crs,
        transform=output_transform,
        resolution_m=resolution_m,
        width=target_width,
        height=target_height,
        provenance=provenance,
    )
