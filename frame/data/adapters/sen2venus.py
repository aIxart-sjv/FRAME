"""SEN2VENuS adapter (Phase 3) -- documented format, tested on synthetic fixtures ONLY.

HONEST STATUS: no real SEN2VENuS data has been read by FRAME (139.5 GB in 29 per-site zips, each
holding nested per-date zips). Implemented from the Zenodo record's own description
(zenodo.org/records/14603764, v2.0.0):

* one same-day Sentinel-2 / VENuS acquisition = a PAIR ID ``{site}_{acquisition_date}_{mgrs_tile}`` (the SCENE);
  the SITE is the REGION -- and the dataset's authors themselves advise keeping separate pairs, even
  sites, for testing;
* two LR->HR relations per pair, both to a 5 m HR:  ``rgbn_10m`` (B2,B3,B4,B8; 128 -> 256 px; x2) and
  ``rededge_20m`` (B5,B6,B7,B8A; 64 -> 256 px; x4). B11/B12 exist LR-only and form no pair;
* file names ``{site}_{idx}_{acquisition_date}_{mgrs_tile}_{bands}_{resolution}.tif``, int16 reflectance x 10000;
* band order in the files is the ASCENDING one (B02,B03,B04,B08), NOT FRAME's RGBN order: use
  ``lr_bands=RGBN_BANDS`` when loading to reorder, explicitly, by name.

Role: SUPPLEMENTARY training (requirements section 25): real S2 LR and a real HR, but at 5 m and for 8 bands, so it
never trains the final 12-band 2.5 m task by itself. Licence: Sentinel-2 files Etalab-2.0, VENuS files
CC-BY-NC-4.0 (non-commercial); the records carry the mixed licence string.

The builder takes explicit relative paths: how a user unpacks the nested zips is up to them.
"""

from __future__ import annotations

from typing import Any, Mapping, Optional, Sequence, Union

from frame.data.adapters.base import DatasetAdapter
from frame.data.adapters.geotiff import read_geotiff_pair
from frame.data.contract import HRStatus, PairedSample, PairRecord, RasterSpec, Split
from frame.data.errors import ContractError
from frame.data.roles import get_profile

DATASET = "sen2venus"
_FILE_TAGS = {"rgbn_10m": ("b2b3b4b8", "10m", "05m"), "rededge_20m": ("b5b6b7b8a", "20m", "05m")}


def file_name(site: str, idx: int, date: str, mgrs: str, bands: str, resolution: str) -> str:
    """The documented patch file name: ``{site}_{idx}_{date}_{mgrs}_{bands}_{resolution}.tif``."""
    return f"{site}_{idx}_{date}_{mgrs}_{bands}_{resolution}.tif"


def expected_file_names(site: str, idx: int, date: str, mgrs: str, variant: str) -> tuple:
    """(LR name, HR name) a pair's patch should have under the documented convention."""
    bands, lr_res, hr_res = _FILE_TAGS[variant]
    return file_name(site, idx, date, mgrs, bands, lr_res), file_name(site, idx, date, mgrs, bands, hr_res)


def record_from_row(row: Mapping[str, Any], *, dataset_revision: Optional[str] = "zenodo-14603764-v2.0.0") -> PairRecord:
    """One SEN2VENuS patch pair -> a `PairRecord`.

    Required keys: ``site``, ``date`` (YYYY-MM-DD), ``mgrs``, ``idx``, ``variant`` (rgbn_10m | rededge_20m), ``split``,
    ``lr_path``, ``hr_path`` (relative). Optional: ``crs``, ``lr_transform``, ``hr_transform``, ``lon``, ``lat``,
    ``source_split``, ``provenance``.
    """
    try:
        variant = str(row["variant"])
        profile = get_profile(DATASET)
        spec = profile.variant(variant)
        site, date, mgrs = str(row["site"]), str(row["date"]), str(row["mgrs"])
        lr = RasterSpec(band_names=spec.lr_bands, width=spec.lr_size[1], height=spec.lr_size[0], pixel_size_m=spec.lr_pixel_size_m,
                        path=str(row["lr_path"]), dtype="int16", reflectance_scale=10_000.0, crs=row.get("crs"),
                        transform=tuple(row["lr_transform"]) if row.get("lr_transform") else None)
        hr = RasterSpec(band_names=spec.hr_bands, width=spec.hr_size[1], height=spec.hr_size[0], pixel_size_m=spec.hr_pixel_size_m,
                        path=str(row["hr_path"]), dtype="int16", reflectance_scale=10_000.0, crs=row.get("crs"),
                        transform=tuple(row["hr_transform"]) if row.get("hr_transform") else None)
        return PairRecord(
            sample_id=f"{DATASET}:{variant}:{site}_{row['idx']}_{date}_{mgrs}", dataset=DATASET, variant=variant,
            dataset_revision=dataset_revision, scene_id=f"{site}_{date}_{mgrs}", region_id=site, split=Split(row["split"]),
            source_split=row.get("source_split"), pair_type=spec.pair_type, hr_status=HRStatus.AVAILABLE,
            scale_factor=spec.scale_factor, lr=lr, hr=hr, lon=row.get("lon"), lat=row.get("lat"), license=profile.license,
            provenance={"site": site, "acquisition_date": date, "mgrs_tile": mgrs, "patch_index": int(row["idx"]),
                        "lr_processing": "Theia L2A (not ESA L2A)", **dict(row.get("provenance", {}))},
        )
    except KeyError as exc:
        raise ContractError(f"SEN2VENuS row is missing key {exc}.", code="missing_field") from None


class Sen2VenusAdapter(DatasetAdapter):
    dataset = DATASET

    def load_pair(self, record: Union[PairRecord, str], *, lr_bands: Optional[Sequence[str]] = None,
                  hr_bands: Optional[Sequence[str]] = None) -> PairedSample:
        record = self._resolve(record)
        return read_geotiff_pair(record, self._require_dir(), lr_bands=lr_bands, hr_bands=hr_bands)
