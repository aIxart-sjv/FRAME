"""Shared fixtures for the paired-data tests (Phase 3): small, coherent, synthetic, no network.

`write_pair_files` writes an LR/HR GeoTIFF pair that describes ONE scene: the HR is a smooth random
field and the LR is the exact block mean of it, so the two genuinely correspond (geometry, radiometry,
alignment). `tiny_env` registers a small dataset profile ("tinyset": 4 RGBN bands, LR 32x32, HR 128x128,
x4) so QC's profile checks apply to fixtures too, and yields a root directory plus its records.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pytest
import rasterio
import torch
import torch.nn.functional as F
from rasterio.transform import Affine

from frame.data.adapters.base import DatasetAdapter
from frame.data.adapters.geotiff import read_geotiff_pair
from frame.data.contract import (
    DatasetRole,
    HRStatus,
    PairedSample,
    PairRecord,
    PairType,
    RasterSpec,
    Split,
)
from frame.data.roles import DATASET_PROFILES, DatasetProfile, VariantSpec
from frame.preprocessing import RGBN_BANDS

TINY = "tinyset"
LR_HW = (32, 32)
SCALE = 4
LR_PIXEL_M = 10.0
CRS = "EPSG:32630"


def tiny_profile(*, splits=frozenset({Split.TRAIN, Split.VAL}), pair_type=PairType.REAL_CROSS_SENSOR, role=DatasetRole.TRAINING_PRIMARY,
                 scale: int = SCALE, lr_hw: Tuple[int, int] = LR_HW) -> DatasetProfile:
    variant = VariantSpec(
        name="default", role=role, pair_type=pair_type, allowed_splits=splits, lr_bands=RGBN_BANDS, hr_bands=RGBN_BANDS,
        scale_factor=scale, lr_pixel_size_m=LR_PIXEL_M, hr_pixel_size_m=LR_PIXEL_M / scale, lr_size=lr_hw,
        hr_size=(lr_hw[0] * scale, lr_hw[1] * scale),
    )
    return DatasetProfile(dataset=TINY, title="tiny fixture dataset", variants={"default": variant}, scene_unit="fixture scene",
                          region_unit="fixture region", license="CC0-1.0", status="fixture", source="tests", requirements_ref="tests")


class TinyAdapter(DatasetAdapter):
    dataset = TINY

    def load_pair(self, record: Union[PairRecord, str], *, lr_bands=None, hr_bands=None) -> PairedSample:
        record = self._resolve(record)
        return read_geotiff_pair(record, self._require_dir(), lr_bands=lr_bands, hr_bands=hr_bands)


def smooth_field(bands: int, height: int, width: int, seed: int) -> np.ndarray:
    """A deterministic smooth reflectance field in about [0.05, 0.45], shape (bands, height, width)."""
    rng = np.random.default_rng(seed)
    coarse = torch.from_numpy(rng.random((1, bands, max(2, height // 8), max(2, width // 8))).astype("float32"))
    field = F.interpolate(coarse, size=(height, width), mode="bicubic", align_corners=False)[0]
    return (0.05 + 0.4 * field.clamp(0, 1)).numpy()


def block_mean(hr: np.ndarray, scale: int) -> np.ndarray:
    c, h, w = hr.shape
    return hr.reshape(c, h // scale, scale, w // scale, scale).mean(axis=(2, 4))


def write_raster(path: Path, data: np.ndarray, *, transform: Tuple[float, ...], crs: str, dtype: str, nodata: Optional[float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(path, "w", driver="GTiff", height=data.shape[1], width=data.shape[2], count=data.shape[0], dtype=dtype,
                       crs=crs, transform=Affine(*transform), nodata=nodata) as dst:
        dst.write(data.astype(dtype))


def write_pair_files(
    root: Path,
    dataset: str,
    rel_lr: str,
    rel_hr: str,
    *,
    bands: Sequence[str] = RGBN_BANDS,
    lr_hw: Tuple[int, int] = LR_HW,
    scale: int = SCALE,
    seed: int = 0,
    dtype: str = "uint16",
    reflectance_scale: float = 10_000.0,
    crs: str = CRS,
    origin: Tuple[float, float] = (500_000.0, 4_000_000.0),
    lr_pixel: float = LR_PIXEL_M,
    lr_nodata: Optional[float] = None,
    hr_nodata: Optional[float] = None,
    nodata_box: Optional[Tuple[int, int, int, int]] = None,   # (row0, row1, col0, col1) in LR pixels, applied to both LR and HR
    hr_origin_shift: Tuple[float, float] = (0.0, 0.0),
) -> Tuple[RasterSpec, RasterSpec]:
    """Write a coherent LR/HR pair under ``root/dataset``; return the two `RasterSpec`s describing the files."""
    hr = smooth_field(len(bands), lr_hw[0] * scale, lr_hw[1] * scale, seed)
    lr = block_mean(hr, scale)
    lr_dn, hr_dn = np.round(lr * reflectance_scale), np.round(hr * reflectance_scale)
    if nodata_box is not None:
        r0, r1, c0, c1 = nodata_box
        if lr_nodata is not None:
            lr_dn[:, r0:r1, c0:c1] = lr_nodata
        if hr_nodata is not None:
            hr_dn[:, r0 * scale : r1 * scale, c0 * scale : c1 * scale] = hr_nodata
    x0, y0 = origin
    lr_t = (lr_pixel, 0.0, x0, 0.0, -lr_pixel, y0)
    hr_t = (lr_pixel / scale, 0.0, x0 + hr_origin_shift[0], 0.0, -lr_pixel / scale, y0 + hr_origin_shift[1])
    write_raster(root / dataset / rel_lr, lr_dn, transform=lr_t, crs=crs, dtype=dtype, nodata=lr_nodata)
    write_raster(root / dataset / rel_hr, hr_dn, transform=hr_t, crs=crs, dtype=dtype, nodata=hr_nodata)
    lr_spec = RasterSpec(band_names=tuple(bands), width=lr_hw[1], height=lr_hw[0], pixel_size_m=lr_pixel, path=rel_lr, dtype=dtype,
                         reflectance_scale=reflectance_scale, nodata=lr_nodata, crs=crs, transform=lr_t)
    hr_spec = RasterSpec(band_names=tuple(bands), width=lr_hw[1] * scale, height=lr_hw[0] * scale, pixel_size_m=lr_pixel / scale,
                         path=rel_hr, dtype=dtype, reflectance_scale=reflectance_scale, nodata=hr_nodata, crs=crs, transform=hr_t)
    return lr_spec, hr_spec


def make_record(
    sample_id: str, *, scene: str, region: Optional[str], split: Split = Split.TRAIN, dataset: str = TINY, lr: RasterSpec, hr: Optional[RasterSpec],
    scale: int = SCALE, pair_type: PairType = PairType.REAL_CROSS_SENSOR, lon: Optional[float] = None, lat: Optional[float] = None, **extra,
) -> PairRecord:
    return PairRecord(
        sample_id=sample_id, dataset=dataset, scene_id=scene, region_id=region, split=split, pair_type=pair_type,
        hr_status=HRStatus.AVAILABLE if hr is not None else HRStatus.UNKNOWN, scale_factor=scale, lr=lr, hr=hr, lon=lon, lat=lat, **extra,
    )


@dataclass
class TinyEnv:
    root: Path
    records: List[PairRecord]

    def adapter(self) -> TinyAdapter:
        return TinyAdapter(self.records, data_root=self.root)


def build_tiny_records(root: Path, layout: Dict[str, Dict[str, int]], *, splits: Optional[Dict[str, Split]] = None, seed0: int = 0) -> List[PairRecord]:
    """``layout = {region: {scene: n_tiles}}`` -> written files + records (region-level splits from ``splits`` or all train)."""
    records: List[PairRecord] = []
    n = 0
    for region_index, (region, scenes) in enumerate(sorted(layout.items())):
        for scene_index, (scene, tiles) in enumerate(sorted(scenes.items())):
            for tile in range(tiles):
                rel = f"tiles/{region}_{scene}_{tile}"
                lr, hr = write_pair_files(root, TINY, f"lr/{rel}.tif", f"hr/{rel}.tif", seed=seed0 + n,
                                          origin=(500_000.0 + 1000.0 * (region_index * 10 + scene_index * 3 + tile), 4_000_000.0))
                records.append(make_record(f"{TINY}:{region}:{scene}:{tile}", scene=f"{region}_{scene}", region=region,
                                           split=(splits or {}).get(region, Split.TRAIN), lr=lr, hr=hr,
                                           lon=-3.0 + 0.5 * region_index, lat=40.0 + 0.5 * region_index))
                n += 1
    return records


@pytest.fixture()
def tiny_profile_installed(monkeypatch):
    """Register the tiny profile so profile-aware QC accepts fixture records."""
    monkeypatch.setitem(DATASET_PROFILES, TINY, tiny_profile())
    return DATASET_PROFILES[TINY]


@pytest.fixture()
def tiny_env(tmp_path, tiny_profile_installed) -> TinyEnv:
    """Two regions x two scenes x two tiles of coherent LR/HR GeoTIFF pairs (region 'A' train, 'B' val)."""
    layout = {"A": {"s1": 2, "s2": 2}, "B": {"s1": 2, "s2": 2}}
    records = build_tiny_records(tmp_path, layout, splits={"A": Split.TRAIN, "B": Split.VAL})
    return TinyEnv(root=tmp_path, records=records)
