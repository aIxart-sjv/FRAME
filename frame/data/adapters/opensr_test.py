"""OpenSR-Test adapter (Phase 3) -- REAL: built on the existing frame.validation adapter and its cached data.

Role: independent benchmark (never training). The existing Phase 4 adapter
(`frame.validation.opensr_test_adapter`) is kept exactly as it is; this module puts the same
data behind the common `DatasetAdapter` interface:

* records are built from a loaded subset (the metadata table: ROI, GEE id, CRS, HR affine), with
  the LR transform derived from the HR one by the same relationship frame.data.geo validates;
* pixels are read from the loaded subset dict. The LR keeps ALL 12 L2A bands (the existing
  `extract_sample` reduces to RGBN); a test asserts that selecting RGBN reproduces
  `extract_sample`'s tensors exactly, so the two paths cannot drift apart.

Supported subsets are the existing adapter's: spot, spain_crops, spain_urban (LR 128x128, x4).
`naip` (LR 121x121) and `venus` (x2) are not supported, as before.

Provenance caveat: the HR band order ('RGBNIR' on the dataset card) is read as B04,B03,B02,B08 by
naming convention -- weaker evidence than the LR band table (see the existing adapter's docstring).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Union

import torch

from frame.data.adapters.base import DatasetAdapter
from frame.data.bands import L2A_BANDS, canonical_bands, select_bands
from frame.data.contract import HRStatus, PairedSample, PairRecord, PairType, RasterSpec, Split
from frame.data.errors import ContractError
from frame.data.geo import centroid_lonlat, lr_transform_from_hr
from frame.data.manifest import read_manifest
from frame.preprocessing import RGBN_BANDS

DATASET = "opensr_test"
HR_BAND_ORDER_NOTE = "HR 'RGBNIR' read as B04,B03,B02,B08 by naming convention (not table-confirmed)"


def _parse_affine(text: str):
    values = tuple(float(v) for v in str(text).split(","))
    if len(values) != 6:
        raise ContractError(f"Expected a 6-value affine, got {text!r}.", code="invalid_transform")
    return values


def records_from_loaded(
    loaded: Dict[str, Any], subset: str, *, dataset_revision: Optional[str] = None, hr_variant: str = "HRharm"
) -> List[PairRecord]:
    """Records for every sample of a loaded opensr-test subset (test split)."""
    from frame.validation.opensr_test_adapter import SUPPORTED_SUBSETS

    if subset not in SUPPORTED_SUBSETS:
        raise ContractError(f"opensr-test subset {subset!r} is not supported; supported: {SUPPORTED_SUBSETS}.", code="unknown_variant")
    l2a, hr, table = loaded["L2A"], loaded[hr_variant], loaded["metadata"]
    n, _, lr_h, lr_w = l2a.shape
    scale = hr.shape[-1] // lr_w
    rois = [str(v) for v in table["roi"]] if "roi" in table.columns else [f"{i:04d}" for i in range(n)]
    unique_rois = len(set(rois)) == n

    records: List[PairRecord] = []
    for i in range(n):
        row = table.iloc[i]
        hr_transform = _parse_affine(row["affine"]) if "affine" in table.columns else None
        crs = str(row["crs"]) if "crs" in table.columns else None
        lr_transform = lr_transform_from_hr(hr_transform, scale) if hr_transform else None
        hr_pixel = abs(hr_transform[0]) if hr_transform else None
        lr_spec = RasterSpec(band_names=L2A_BANDS, width=lr_w, height=lr_h, pixel_size_m=hr_pixel * scale if hr_pixel else None,
                             dtype="uint16", reflectance_scale=10_000.0, crs=crs, transform=lr_transform)
        hr_spec = RasterSpec(band_names=RGBN_BANDS, width=hr.shape[-1], height=hr.shape[-2], pixel_size_m=hr_pixel,
                             dtype="uint16", reflectance_scale=10_000.0, crs=crs, transform=hr_transform)
        lonlat = centroid_lonlat(lr_spec)
        roi = rois[i]
        provenance = {"locator": {"kind": "opensr_test_cache", "subset": subset, "index": i, "hr_variant": hr_variant},
                      "hr_band_order_note": HR_BAND_ORDER_NOTE}
        for column in ("lr_gee_id", "hr_file", "reflectance", "spectral", "spatial"):
            if column in table.columns:
                provenance[column] = row[column] if isinstance(row[column], str) else float(row[column])
        records.append(PairRecord(
            sample_id=f"{DATASET}:{subset}:{roi}" if unique_rois else f"{DATASET}:{subset}:{roi}:{i:04d}",
            dataset=DATASET, variant=subset, dataset_revision=dataset_revision, scene_id=f"{subset}_{roi}", region_id=None,
            split=Split.TEST, pair_type=PairType.REAL_CROSS_SENSOR, hr_status=HRStatus.AVAILABLE, scale_factor=scale,
            lr=lr_spec, hr=hr_spec, lon=lonlat[0] if lonlat else None, lat=lonlat[1] if lonlat else None,
            license="MIT (dataset repository)", provenance=provenance,
        ))
    return records


class OpenSRTestAdapter(DatasetAdapter):
    """Manifest-backed; pixels come from the cached opensr-test subset (no network beyond the first download)."""

    dataset = DATASET

    def __init__(self, records: Sequence[PairRecord], *, data_root=None, loaded_subsets: Optional[Dict[str, Any]] = None):
        super().__init__(records, data_root=data_root)
        self._loaded: Dict[str, Any] = dict(loaded_subsets or {})

    @classmethod
    def from_manifest(cls, path, *, data_root=None, loaded_subsets: Optional[Dict[str, Any]] = None) -> "OpenSRTestAdapter":
        contents = read_manifest(path)
        return cls([r for r in contents.records if r.dataset == cls.dataset], data_root=data_root, loaded_subsets=loaded_subsets)

    def _subset(self, name: str) -> Any:
        if name not in self._loaded:
            from frame.validation import load_subset  # the existing, unmodified loader

            self._loaded[name] = load_subset(name)
        return self._loaded[name]

    def load_pair(self, record: Union[PairRecord, str], *, lr_bands: Optional[Sequence[str]] = None,
                  hr_bands: Optional[Sequence[str]] = None) -> PairedSample:
        record = self._resolve(record)
        locator = record.provenance.get("locator", {})
        if locator.get("kind") != "opensr_test_cache":
            raise ContractError(f"{record.sample_id} does not point into the opensr-test cache.", code="missing_path")
        loaded = self._subset(locator["subset"])
        index, hr_variant = int(locator["index"]), locator.get("hr_variant", "HRharm")

        lr_raw = torch.from_numpy(loaded["L2A"][index].astype("float32")) / 10_000.0
        hr_raw = torch.from_numpy(loaded[hr_variant][index].astype("float32")) / 10_000.0
        lr_names = canonical_bands(lr_bands) if lr_bands is not None else record.lr.band_names
        hr_names = canonical_bands(hr_bands) if hr_bands is not None else record.hr.band_names
        return PairedSample(
            record=record, lr=select_bands(lr_raw, record.lr.band_names, lr_names).contiguous(),
            hr=select_bands(hr_raw, record.hr.band_names, hr_names).contiguous(), lr_bands=lr_names, hr_bands=hr_names,
        )
