"""Reading LR/HR GeoTIFF pairs into `PairedSample`s (Phase 3).

The generic reader behind every dataset whose pairs are two GeoTIFFs (SEN2NEON as published;
SEN2NAIPv2 / SEN2VENuS after extraction; an Indian scene). What it does, and does not do:

* Bands are selected BY NAME from the band list the record declares; nothing is reordered unless
  ``lr_bands`` / ``hr_bands`` ask for it. Asking for a band the file does not hold is an error.
* Stored values are converted to float32 reflectance by the record's own ``reflectance_scale``.
* Nodata is detected on the RAW stored values with `ValidityMask.from_nodata` (a pixel is nodata
  only if every band equals the nodata value), returned as an explicit mask, and set to 0.0 in
  the tensor. A masked pixel is never confused with a genuine zero: the mask travels with it.
* NaN/Inf and out-of-range values are NOT replaced (no `nan_to_num`): frame.data.qc reports them.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence, Tuple

import numpy as np
import torch

from frame.data.bands import canonical_bands, select_bands
from frame.data.contract import PairedSample, PairRecord, RasterSpec
from frame.data.errors import ContractError, DatasetUnavailableError
from frame.preprocessing.masks import ValidityMask


def read_raster(
    path: Path, spec: RasterSpec, bands: Optional[Sequence[str]]
) -> Tuple[torch.Tensor, torch.Tensor, Tuple[str, ...]]:
    """(reflectance (C, H, W) float32, valid mask (H, W) bool, band names) for one raster."""
    import rasterio

    if not path.is_file():
        raise DatasetUnavailableError(f"File not found: {path}")
    with rasterio.open(path) as src:
        raw = src.read()
    if raw.shape[0] != len(spec.band_names):
        raise ContractError(f"{path.name} has {raw.shape[0]} band(s) but the record lists {len(spec.band_names)}.", code="band_count_mismatch")
    if raw.shape[1:] != (spec.height, spec.width):
        raise ContractError(f"{path.name} is {raw.shape[1]}x{raw.shape[2]} but the record says {spec.height}x{spec.width}.", code="shape_mismatch")

    valid = (ValidityMask.from_nodata(raw, spec.nodata).array if spec.nodata is not None else np.ones(raw.shape[1:], dtype=bool))
    values = raw.astype(np.float32)
    if spec.reflectance_scale is not None:
        values = values / np.float32(spec.reflectance_scale)
    values[:, ~valid] = 0.0

    names = canonical_bands(bands) if bands is not None else spec.band_names
    tensor = select_bands(torch.from_numpy(values), spec.band_names, names)
    return tensor.contiguous(), torch.from_numpy(valid), names


def read_geotiff_pair(
    record: PairRecord,
    dataset_dir: Path,
    *,
    lr_bands: Optional[Sequence[str]] = None,
    hr_bands: Optional[Sequence[str]] = None,
) -> PairedSample:
    """Load ``record``'s LR (and HR, if it has one) from ``dataset_dir / spec.path``."""
    for name, spec in (("LR", record.lr), ("HR", record.hr)):
        if spec is not None and spec.path is None:
            raise ContractError(f"{record.sample_id}: the {name} raster has no path; this reader needs GeoTIFF files.", code="missing_path")

    lr, lr_mask, lr_names = read_raster(dataset_dir / record.lr.path, record.lr, lr_bands)
    if record.hr is None:
        return PairedSample(record=record, lr=lr, hr=None, lr_bands=lr_names, hr_bands=None, lr_mask=lr_mask)
    hr, hr_mask, hr_names = read_raster(dataset_dir / record.hr.path, record.hr, hr_bands)
    return PairedSample(record=record, lr=lr, hr=hr, lr_bands=lr_names, hr_bands=hr_names, lr_mask=lr_mask, hr_mask=hr_mask)
