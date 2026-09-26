"""SEN2NAIPv2 adapter (Phase 3) -- documented format, tested on synthetic fixtures ONLY.

HONEST STATUS: no real SEN2NAIPv2 data has been read by FRAME. The dataset is 139.1 GB in the
TACO format, which needs the `tacoreader` package (not installed, and not added: it is not needed
to prove the data layer). This module implements everything that can be implemented from the
dataset's public card without it:

* the profile facts (bands B04,B03,B02,B08; x4; LR 130x130 at 10 m, HR 520x520 at 2.5 m; the
  variants below) -- see frame.data.roles;
* a record builder for an EXPORT of SEN2NAIPv2 as GeoTIFF pairs plus a small table of rows (the
  schema is `record_from_row`'s docstring). The TACO metadata schema was not inspected, so the
  export must supply ``scene_id`` and ``region_id`` explicitly rather than have them guessed;
* reading of such an export with the generic GeoTIFF reader.

Roles (docs/Requirements 142.txt Requirement 3 sections 9-11, 24, 35):

* ``unet`` / ``histmatch``  -- synthetic; the PRIMARY training data. Their LR images are shipped by
  the dataset (an upstream degradation: Gaussian blur + bilinear downsampling, reflectance
  harmonisation by a retrained U-Net or by histogram matching, noise), so the records carry a
  ``DegradationRecord(origin="upstream")`` describing that, not one of FRAME's own.
* ``crosssensor``          -- 8,000 REAL Sentinel-2 / NAIP pairs (same-day, cloud-free): development
  validation, never training.
"""

from __future__ import annotations

from typing import Any, Mapping, Optional, Sequence, Union

from frame.data.adapters.base import DatasetAdapter
from frame.data.adapters.geotiff import read_geotiff_pair
from frame.data.contract import (
    DegradationRecord,
    HRStatus,
    PairedSample,
    PairRecord,
    PairType,
    RasterSpec,
    Split,
)
from frame.data.errors import ContractError
from frame.data.roles import get_profile

DATASET = "sen2naipv2"
_HARMONISATION = {"unet": "unet", "histmatch": "histogram_match"}
_UPSTREAM_STEPS = ["gaussian_blur", "bilinear_downsample", "reflectance_harmonisation", "noise"]


def record_from_row(row: Mapping[str, Any], *, dataset_revision: Optional[str] = None) -> PairRecord:
    """One row of a SEN2NAIPv2 export -> a `PairRecord`.

    Required keys: ``id``, ``variant`` (unet | histmatch | crosssensor), ``scene_id``, ``region_id``, ``split``,
    ``lr_path``, ``hr_path`` (relative GeoTIFF paths). Optional: ``lon``, ``lat``, ``crs``, ``lr_transform``,
    ``hr_transform``, ``source_split``, ``reflectance_scale`` (default 10000; NOT stated on the dataset card, so
    verify it against a real sample), ``provenance`` (dict).
    """
    try:
        variant = str(row["variant"])
        profile = get_profile(DATASET)
        spec = profile.variant(variant)
        scale = float(row.get("reflectance_scale", 10_000.0))
        lr = RasterSpec(band_names=spec.lr_bands, width=spec.lr_size[1], height=spec.lr_size[0], pixel_size_m=spec.lr_pixel_size_m,
                        path=str(row["lr_path"]), dtype=str(row.get("dtype", "uint16")), reflectance_scale=scale,
                        crs=row.get("crs"), transform=tuple(row["lr_transform"]) if row.get("lr_transform") else None)
        hr = RasterSpec(band_names=spec.hr_bands, width=spec.hr_size[1], height=spec.hr_size[0], pixel_size_m=spec.hr_pixel_size_m,
                        path=str(row["hr_path"]), dtype=str(row.get("dtype", "uint16")), reflectance_scale=scale,
                        crs=row.get("crs"), transform=tuple(row["hr_transform"]) if row.get("hr_transform") else None)
        degradation = None
        if spec.pair_type == PairType.SYNTHETIC:
            degradation = DegradationRecord(
                name=f"sen2naipv2-{variant}", version="upstream (dataset release of 2025-02-01)",
                config={"steps": _UPSTREAM_STEPS, "note": "numeric parameters are not published on the dataset card"},
                seed=None, harmonisation=_HARMONISATION[variant], parameters_verified=True, origin="upstream",
            )
        return PairRecord(
            sample_id=f"{DATASET}:{variant}:{row['id']}", dataset=DATASET, variant=variant, dataset_revision=dataset_revision,
            scene_id=str(row["scene_id"]), region_id=str(row["region_id"]), split=Split(row["split"]),
            source_split=row.get("source_split"), pair_type=spec.pair_type, hr_status=HRStatus.AVAILABLE, scale_factor=spec.scale_factor,
            lr=lr, hr=hr, lon=row.get("lon"), lat=row.get("lat"), degradation=degradation, license=profile.license,
            provenance=dict(row.get("provenance", {})),
        )
    except KeyError as exc:
        raise ContractError(f"SEN2NAIPv2 export row is missing key {exc}.", code="missing_field") from None


class Sen2NaipV2Adapter(DatasetAdapter):
    dataset = DATASET

    def load_pair(self, record: Union[PairRecord, str], *, lr_bands: Optional[Sequence[str]] = None,
                  hr_bands: Optional[Sequence[str]] = None) -> PairedSample:
        record = self._resolve(record)
        return read_geotiff_pair(record, self._require_dir(), lr_bands=lr_bands, hr_bands=hr_bands)
