"""Copernicus Data Space Ecosystem (CDSE) -- the primary real pixel-retrieval
provider, via the Sentinel Hub Process API.

This is genuinely credentialed, Copernicus-native retrieval, distinct from
the STAC search in `frame.acquisition.search` (whose asset hrefs point at
``s3://eodata/...`` and are not fetchable without separate S3 credentials
this project does not have). Requesting bands through the Process API with
``units: "REFLECTANCE"`` also sidesteps a real pitfall: Sentinel Hub applies
the correct per-processing-baseline radiometric offset itself, so this
provider never has to guess whether a scene needs the post-04.00 "DN/10000
minus offset" correction or the older plain "DN/10000" -- see Requirements
142 Section 14's warning against a single blind DN-scaling formula.

Every request is scoped to one resolution group (bands that share a native
resolution) so the caller decides, explicitly, when bands are being
resampled -- this module never mixes native and resampled pixels in one
response without the caller asking for exactly that.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from io import BytesIO
from typing import Callable, Optional, Sequence, Tuple

import numpy as np
import rasterio
import requests

from frame.acquisition.errors import AuthenticationError, ProviderUnavailableError

PROVIDER_NAME = "copernicus-cdse-process-api"
DEFAULT_TOKEN_URL = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
DEFAULT_PROCESS_URL = "https://sh.dataspace.copernicus.eu/process/v1"

#: Refresh the cached token this many seconds before its real expiry, so a
#: request never races a token that is about to lapse mid-flight.
_TOKEN_SAFETY_MARGIN_SECONDS = 60.0

#: SCL is a classification code, not a continuous value -- requesting it
#: with bilinear resampling would blend adjacent class codes into
#: meaningless numbers. Reflectance bands use BILINEAR; SCL must use NEAREST.
CATEGORICAL_RESAMPLING = "NEAREST"
CONTINUOUS_RESAMPLING = "BILINEAR"

HttpPost = Callable[..., "requests.Response"]


@dataclass
class _CachedToken:
    access_token: str
    expires_at: float


class CDSEAuth:
    """Client-credentials OAuth2 token manager for CDSE.

    The token is cached in memory for its real lifetime (minus a safety
    margin) and is never logged, returned from an API response, or written
    to disk. ``configured()`` lets a caller check availability before
    attempting a request that would otherwise raise.
    """

    def __init__(
        self,
        *,
        client_id: Optional[str] = None,
        client_secret: Optional[str] = None,
        token_url: Optional[str] = None,
        http_post: HttpPost = requests.post,
    ):
        self._client_id = client_id or os.environ.get("FRAME_CDSE_CLIENT_ID")
        self._client_secret = client_secret or os.environ.get("FRAME_CDSE_CLIENT_SECRET")
        self._token_url = token_url or os.environ.get("FRAME_CDSE_TOKEN_URL", DEFAULT_TOKEN_URL)
        self._http_post = http_post
        self._cached: Optional[_CachedToken] = None

    def configured(self) -> bool:
        return bool(self._client_id and self._client_secret)

    def get_token(self) -> str:
        if not self.configured():
            raise ProviderUnavailableError(
                "CDSE OAuth credentials are not configured "
                "(FRAME_CDSE_CLIENT_ID / FRAME_CDSE_CLIENT_SECRET) -- this provider "
                "is unavailable, not silently skipped.",
                provider=PROVIDER_NAME,
            )

        now = time.time()
        if self._cached is not None and now < self._cached.expires_at:
            return self._cached.access_token

        try:
            response = self._http_post(
                self._token_url,
                data={
                    "grant_type": "client_credentials",
                    "client_id": self._client_id,
                    "client_secret": self._client_secret,
                },
                timeout=20,
            )
        except requests.RequestException as exc:
            raise ProviderUnavailableError(f"CDSE token request failed: {exc}", provider=PROVIDER_NAME) from exc

        if response.status_code in (401, 403):
            raise AuthenticationError(
                f"CDSE rejected the configured OAuth2 credentials (HTTP {response.status_code}); "
                "check FRAME_CDSE_CLIENT_ID/SECRET.",
                provider=PROVIDER_NAME,
            )
        if response.status_code >= 400:
            raise ProviderUnavailableError(
                f"CDSE token request returned HTTP {response.status_code}.",
                provider=PROVIDER_NAME,
            )

        body = response.json()
        token = body["access_token"]
        expires_in = float(body.get("expires_in", 0))
        self._cached = _CachedToken(
            access_token=token,
            expires_at=now + max(expires_in - _TOKEN_SAFETY_MARGIN_SECONDS, 0.0),
        )
        return token


def _evalscript(bands: Sequence[str], sample_type: str, units: str) -> str:
    band_list = ", ".join(f'"{b}"' for b in bands)
    output_refs = ", ".join(f"sample.{b}" for b in bands)
    return (
        "//VERSION=3\n"
        "function setup() {\n"
        "  return {\n"
        f'    input: [{{ bands: [{band_list}], units: "{units}" }}],\n'
        f'    output: {{ bands: {len(bands)}, sampleType: "{sample_type}" }}\n'
        "  };\n"
        "}\n"
        "function evaluatePixel(sample) {\n"
        f"  return [{output_refs}];\n"
        "}\n"
    )


def process_url() -> str:
    return os.environ.get("FRAME_CDSE_PROCESS_URL", DEFAULT_PROCESS_URL)


def retrieve_bands(
    *,
    auth: CDSEAuth,
    utm_crs: str,
    utm_bbox: Tuple[float, float, float, float],
    time_range: Tuple[str, str],
    bands: Sequence[str],
    resolution_m: float,
    max_cloud_cover: float = 100.0,
    categorical: bool = False,
    endpoint: Optional[str] = None,
    http_post: HttpPost = requests.post,
    timeout_seconds: float = 60.0,
) -> Tuple[np.ndarray, str, Tuple[float, float, float, float, float, float]]:
    """Real pixel retrieval for ``bands`` (all requested together at ``resolution_m``,
    in ``utm_crs``) via the CDSE Sentinel Hub Process API.

    Set ``categorical=True`` for SCL (nearest-neighbour resampling, integer
    sample type, raw digital numbers) rather than reflectance bands.

    Returns ``(array, crs, transform)`` with ``array`` shaped
    ``(len(bands), height, width)``. Raises ProviderUnavailableError if
    credentials are missing or the API call fails for any reason -- this
    function never returns a partial or fabricated result.
    """
    token = auth.get_token()
    url = endpoint or process_url()

    min_x, min_y, max_x, max_y = utm_bbox
    width = max(1, round((max_x - min_x) / resolution_m))
    height = max(1, round((max_y - min_y) / resolution_m))

    resampling = CATEGORICAL_RESAMPLING if categorical else CONTINUOUS_RESAMPLING
    sample_type = "UINT8" if categorical else "FLOAT32"
    units = "DN" if categorical else "REFLECTANCE"
    epsg_code = utm_crs.split(":")[-1]

    body = {
        "input": {
            "bounds": {
                "bbox": [min_x, min_y, max_x, max_y],
                "properties": {"crs": f"http://www.opengis.net/def/crs/EPSG/0/{epsg_code}"},
            },
            "data": [
                {
                    "type": "sentinel-2-l2a",
                    "dataFilter": {
                        "timeRange": {"from": time_range[0], "to": time_range[1]},
                        "maxCloudCoverage": max_cloud_cover,
                        "mosaickingOrder": "leastCC",
                    },
                    "processing": {"upsampling": resampling, "downsampling": resampling},
                }
            ],
        },
        "output": {
            "width": width,
            "height": height,
            "responses": [{"identifier": "default", "format": {"type": "image/tiff"}}],
        },
        "evalscript": _evalscript(bands, sample_type, units),
    }

    try:
        response = http_post(url, json=body, headers={"Authorization": f"Bearer {token}"}, timeout=timeout_seconds)
    except requests.RequestException as exc:
        raise ProviderUnavailableError(f"CDSE Process API request failed: {exc}", provider=PROVIDER_NAME) from exc

    if response.status_code in (401, 403):
        raise AuthenticationError(
            f"CDSE Process API rejected the request (HTTP {response.status_code}): {response.text[:500]}",
            provider=PROVIDER_NAME,
        )
    if response.status_code >= 400:
        raise ProviderUnavailableError(
            f"CDSE Process API returned HTTP {response.status_code}: {response.text[:500]}",
            provider=PROVIDER_NAME,
        )

    with rasterio.open(BytesIO(response.content)) as src:
        array = src.read()
        out_crs = src.crs.to_string()
        transform = tuple(src.transform)[:6]

    return array, out_crs, transform
