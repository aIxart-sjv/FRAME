"""Synthetic smoke dataset (Phase 4): controlled RGBN scenes with a known, recorded LR degradation.

Purpose: let the WHOLE pipeline (adapter -> manifest -> geographic split -> QC -> training -> validation) be exercised
on real GeoTIFF files at a scale a laptop handles in seconds, without any real data and without pretending to be one.
It is not a benchmark and any number measured on it says nothing about real imagery (see its profile).

Scenes (`generate_scene`) are piecewise-constant "land cover" cells (vegetation, crop, bare, built-up, water reflectances in
FRAME's RGBN band order) with smooth shading, mild noise and a few thin bright lines. The sharp edges and lines are the
information the LR loses, so a model can genuinely learn something from them (bicubic cannot recover it).

The LR of a scene is `frame.data.degradation.degrade` (``frame_default_v1``, seeded) applied to the stored, quantised HR, then
quantised the same way. Config, seed and version are stored in the record, so an LR can be regenerated exactly from the
stored HR file and its record (tested). Everything is a pure function of the seed: the same arguments write byte-identical
files and manifests.

Records are written all in ``train``; assign real splits with `python -m frame.data split` (per region).
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import List, Optional, Sequence, Union

import torch
import torch.nn.functional as F

from frame.data.adapters.base import DatasetAdapter
from frame.data.adapters.geotiff import read_geotiff_pair
from frame.data.contract import HRStatus, PairedSample, PairRecord, PairType, RasterSpec, Split
from frame.data.degradation import degrade, frame_default_v1
from frame.preprocessing import RGBN_BANDS

DATASET = "synthetic_smoke"
GENERATOR_VERSION = "frame.data.adapters.synthetic/1"
SCALE = 4
LR_SIZE = 128
HR_SIZE = LR_SIZE * SCALE
CRS = "EPSG:32630"
REFLECTANCE_SCALE = 10_000.0

#: RGBN (B04, B03, B02, B08) reflectance prototypes and how often a cell picks each.
_PALETTE = torch.tensor([
    [0.04, 0.08, 0.03, 0.42],    # vegetation
    [0.09, 0.13, 0.06, 0.36],    # crop
    [0.26, 0.22, 0.17, 0.31],    # bare soil
    [0.21, 0.20, 0.19, 0.23],    # built-up
    [0.03, 0.05, 0.07, 0.02],    # water
])
_WEIGHTS = torch.tensor([0.30, 0.30, 0.15, 0.15, 0.10])
_ROAD = torch.tensor([0.28, 0.27, 0.25, 0.30])


def generate_scene(seed: int, size: int = HR_SIZE) -> torch.Tensor:
    """One (4, size, size) float32 reflectance scene, a pure function of ``seed``."""
    g = torch.Generator().manual_seed(int(seed))
    n_cells = 48
    seeds = torch.rand(n_cells, 2, generator=g) * size
    ys, xs = torch.meshgrid(torch.arange(size, dtype=torch.float32), torch.arange(size, dtype=torch.float32), indexing="ij")
    grid = torch.stack([ys.reshape(-1), xs.reshape(-1)], dim=1)
    labels = torch.cdist(grid, seeds).argmin(dim=1)

    classes = torch.multinomial(_WEIGHTS, n_cells, replacement=True, generator=g)
    jitter = (1.0 + 0.15 * torch.randn(n_cells, 1, generator=g)).clamp(0.6, 1.4)
    img = (_PALETTE[classes] * jitter)[labels].reshape(size, size, 4).permute(2, 0, 1).contiguous()

    shading = F.interpolate(torch.rand(1, 1, 8, 8, generator=g), size=(size, size), mode="bicubic", align_corners=False)[0]
    img = img * (1.0 + 0.16 * (shading - 0.5))
    img = img * (1.0 + 0.02 * torch.randn(4, size, size, generator=g))

    for _ in range(4):                                                    # thin bright lines ("roads")
        angle = float(torch.rand(1, generator=g)) * math.pi
        centre = torch.rand(2, generator=g) * size
        normal = torch.tensor([math.cos(angle), math.sin(angle)])
        distance = ((grid - centre) @ normal).abs().reshape(size, size)
        img = torch.where((distance < 1.5)[None], _ROAD[:, None, None].expand_as(img), img)
    return img.clamp(0.001, 1.0).to(torch.float32)


def _quantise(x: torch.Tensor) -> torch.Tensor:
    """Round to whole digital numbers exactly as the on-disk uint16 rasters hold them (returned as DN, float32)."""
    return torch.round(x * REFLECTANCE_SCALE).clamp(0, 65535)


def build_synthetic_dataset(root: Union[str, Path], *, n_regions: int, scenes_per_region: int, seed: int) -> List[PairRecord]:
    """Write ``n_regions x scenes_per_region`` LR/HR GeoTIFF pairs under ``root/synthetic_smoke`` and return their records."""
    if not isinstance(n_regions, int) or n_regions < 1:
        raise ValueError(f"n_regions must be an integer >= 1, got {n_regions!r}.")
    if not isinstance(scenes_per_region, int) or scenes_per_region < 1:
        raise ValueError(f"scenes_per_region must be an integer >= 1, got {scenes_per_region!r}.")
    import rasterio
    from rasterio.transform import Affine

    directory = Path(root) / DATASET
    config = frame_default_v1(SCALE, harmonisation="none")
    records: List[PairRecord] = []
    for region in range(n_regions):
        for scene in range(scenes_per_region):
            name = f"R{region:02d}_S{scene:02d}"
            hr_seed = seed * 1_000_003 + region * 1009 + scene
            hr_dn = _quantise(generate_scene(hr_seed))
            hr = hr_dn / REFLECTANCE_SCALE                                 # exactly what a reader recovers from the file
            lr, degradation = degrade(hr, config, seed=hr_seed + 1)
            lr_dn = _quantise(lr)

            x0, y0 = 400_000.0 + region * 60_000.0 + scene * 1_500.0, 4_400_000.0
            lr_t, hr_t = (10.0, 0.0, x0, 0.0, -10.0, y0), (2.5, 0.0, x0, 0.0, -2.5, y0)
            for rel, data, transform in ((f"lr/{name}.tif", lr_dn, lr_t), (f"hr/{name}.tif", hr_dn, hr_t)):
                path = directory / rel
                path.parent.mkdir(parents=True, exist_ok=True)
                with rasterio.open(path, "w", driver="GTiff", height=data.shape[1], width=data.shape[2], count=data.shape[0], dtype="uint16",
                                   crs=CRS, transform=Affine(*transform)) as dst:
                    dst.write(data.numpy().astype("uint16"))

            def spec(size: int, pixel: float, rel: str, transform) -> RasterSpec:
                return RasterSpec(band_names=RGBN_BANDS, width=size, height=size, pixel_size_m=pixel, path=rel, dtype="uint16",
                                  reflectance_scale=REFLECTANCE_SCALE, nodata=None, crs=CRS, transform=transform)

            records.append(PairRecord(
                sample_id=f"{DATASET}:{name}", dataset=DATASET, scene_id=name, region_id=f"R{region:02d}", split=Split.TRAIN,
                pair_type=PairType.SYNTHETIC, hr_status=HRStatus.AVAILABLE, scale_factor=SCALE,
                lr=spec(LR_SIZE, 10.0, f"lr/{name}.tif", lr_t), hr=spec(HR_SIZE, 2.5, f"hr/{name}.tif", hr_t),
                lon=-3.0 + 0.5 * region + 0.015 * scene, lat=40.0 + 0.015 * scene, degradation=degradation, license="CC0-1.0",
                provenance={"generator": GENERATOR_VERSION, "hr_seed": hr_seed, "note": "synthetic scene; not evidence about real imagery"},
            ))
    return records


class SyntheticSmokeAdapter(DatasetAdapter):
    dataset = DATASET

    def load_pair(self, record: Union[PairRecord, str], *, lr_bands: Optional[Sequence[str]] = None,
                  hr_bands: Optional[Sequence[str]] = None) -> PairedSample:
        record = self._resolve(record)
        return read_geotiff_pair(record, self._require_dir(), lr_bands=lr_bands, hr_bands=hr_bands)
