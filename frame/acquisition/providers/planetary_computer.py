"""Microsoft Planetary Computer -- the secondary, credential-free real
pixel-retrieval provider, used only as an honestly-recorded fallback when
CDSE is not configured or fails.

Backed by `cubo` (already a project dependency; `experiments/baseline/
run_baseline.py`'s `fetch_scene()` is the existing precedent this reuses
rather than duplicates). Verified in this project's own acquisition audit to
return real Sentinel-2 pixels for all three native resolution groups
(10 m/20 m/60 m) plus SCL, with matching timestamps across groups, from the
same underlying archive CDSE catalogues.
"""

from __future__ import annotations

from typing import Callable, Optional, Sequence

from frame.acquisition.errors import ProviderUnavailableError
from frame.preprocessing.reflectance import RAW_DIGITAL_NUMBER, to_reflectance

PROVIDER_NAME = "microsoft-planetary-computer"
STAC_ENDPOINT = "https://planetarycomputer.microsoft.com/api/stac/v1"

CuboCreate = Callable[..., "object"]


def retrieve_bands(
    *,
    lat: float,
    lon: float,
    bands: Sequence[str],
    resolution_m: float,
    edge_size_px: int,
    start_date: str,
    end_date: str,
    max_cloud_cover: float = 100.0,
    categorical: bool = False,
    cubo_create: Optional[CuboCreate] = None,
):
    """Real pixel retrieval for ``bands`` (all requested together at
    ``resolution_m``) centred on ``(lat, lon)``, via `cubo`.

    `cubo`/Planetary Computer returns raw Sentinel-2 digital numbers (verified
    empirically: values in the thousands, not [0, 1] reflectance) -- unlike
    the CDSE provider, which requests reflectance directly. Continuous bands
    (``categorical=False``) are converted here with
    `frame.preprocessing.reflectance.to_reflectance` (this project's one
    existing, tested DN-to-reflectance convention: a flat divide by 10,000,
    with no Sentinel-2 processing-baseline offset correction), so a
    RetrievedScene's bands are always in the same reflectance convention
    regardless of which provider actually served them. Categorical bands
    (SCL) are left as their raw integer class codes, unscaled.

    This flat conversion does not apply the baseline-04.00+ radiometric
    offset that CDSE's own Process API reflectance units may already
    include -- a real, small, documented discrepancy between providers for
    scenes processed under that baseline, disclosed by the caller
    (frame.acquisition.retrieve) as a provenance warning whenever this
    provider serves a reflectance band, never silently absorbed.

    Returns the `xarray.DataArray` `cubo.create` produces (dims
    ``time, band, y, x``); the caller selects the matching timestep. Raises
    ProviderUnavailableError if `cubo` is not installed, the request fails,
    or zero scenes are returned -- never a fabricated array.
    """
    if cubo_create is None:
        try:
            import cubo
        except ImportError as exc:
            raise ProviderUnavailableError(f"cubo is not installed: {exc}", provider=PROVIDER_NAME) from exc
        cubo_create = cubo.create

    try:
        data_array = cubo_create(
            lat=lat,
            lon=lon,
            collection="sentinel-2-l2a",
            bands=list(bands),
            start_date=start_date,
            end_date=end_date,
            edge_size=edge_size_px,
            resolution=resolution_m,
        )
    except ProviderUnavailableError:
        raise
    except Exception as exc:  # cubo raises assorted library-specific exceptions
        raise ProviderUnavailableError(f"Planetary Computer retrieval failed: {exc}", provider=PROVIDER_NAME) from exc

    if int(data_array.sizes.get("time", 0)) == 0:
        raise ProviderUnavailableError(
            f"Planetary Computer returned zero scenes for bands={list(bands)}, "
            f"{start_date}/{end_date} at ({lat}, {lon}).",
            provider=PROVIDER_NAME,
        )

    cloud_filtered = _filter_by_cloud_cover(data_array, max_cloud_cover)
    if int(cloud_filtered.sizes.get("time", 0)) == 0:
        raise ProviderUnavailableError(
            f"Planetary Computer returned scenes, but none at or below "
            f"max_cloud_cover={max_cloud_cover}.",
            provider=PROVIDER_NAME,
        )

    if categorical:
        return cloud_filtered

    reflectance_values = to_reflectance(cloud_filtered.values, RAW_DIGITAL_NUMBER)
    return cloud_filtered.copy(data=reflectance_values)


def _filter_by_cloud_cover(data_array, max_cloud_cover: float):
    """`cubo`'s own per-timestep cloud-cover attribute varies by collection/version;
    if the array does not expose one, every timestep is kept rather than guessed at.

    Known limitation: as verified in this project's acquisition audit, `cubo`'s
    current output attaches no per-scene `eo:cloud_cover` attribute at all (only
    collection/CRS/AOI metadata), so this filter is a documented no-op on this
    provider today -- cloud filtering is genuinely enforced upstream of retrieval,
    at the CDSE STAC search stage (frame.acquisition.search), not here.
    """
    cloud_cover = getattr(data_array, "attrs", {}).get("eo:cloud_cover")
    if cloud_cover is None:
        return data_array
    try:
        mask = cloud_cover <= max_cloud_cover
    except TypeError:
        return data_array
    return data_array.sel(time=data_array["time"][mask]) if hasattr(mask, "__iter__") else data_array
