"""A small end-to-end smoke test of the downstream pipeline that needs no dataset, no weights and no GPU (``python -m frame.downstream smoke``).

Synthetic reflectance scenes with vegetated (NIR-bright) patches, a low-resolution version made by block averaging, and two toy x4 models (a smooth interpolator and an asymmetric sharpener,
so the six TTA views genuinely differ) run through exactly the same gate, TTA, NDVI, regions, metrics, association and risk-coverage code as a real run. The numbers mean nothing scientifically:
it only proves the pipeline runs and writes its record. Nothing here is a benchmark.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage

from frame.downstream.config import DownstreamConfig
from frame.downstream.runner import run_downstream
from frame.evaluate.datasets import DatasetValidation, EvalSample
from frame.evaluate.systems import CallableSystem
from frame.tiling.plan import TilingConfig

BANDS = ("B04", "B03", "B02", "B08")


class _Smooth(torch.nn.Module):
    def forward(self, x):
        return F.interpolate(x, scale_factor=4, mode="bicubic", antialias=True).clamp(min=0)


class _Sharpen(torch.nn.Module):
    def forward(self, x):
        up = F.interpolate(x, scale_factor=4, mode="bicubic", antialias=True).clamp(min=0)
        return (up + 0.5 * (up - torch.roll(up, shifts=(1, 2), dims=(-2, -1)))).clamp(min=0)


def synthetic_sample(index: int, size: int = 128) -> EvalSample:
    rng = np.random.default_rng(1000 + index)
    texture = ndimage.gaussian_filter(rng.standard_normal((4, size, size)), (0, 1.2, 1.2))
    base = (0.22 + 0.05 * texture / texture.std()).clip(0.02, 0.9)
    patch = ndimage.gaussian_filter(rng.standard_normal((size, size)), 9)
    patch = (patch - patch.min()) / (patch.max() - patch.min())
    base[3] = base[3] + 0.4 * patch                                                             # NIR-bright patches: vegetation
    hr = torch.from_numpy(base.astype("float32"))
    lr = F.avg_pool2d(hr[None], 4)[0]
    mask = torch.ones(size, size, dtype=torch.bool)
    quality = {"hr_valid_fraction": 1.0, "hr_dataset_nodata_fraction": 0.0, "hr_partial_nodata_fraction": 0.0, "hr_nonfinite_fraction": 0.0, "hr_valid_pixels": size * size,
               "rule": "smoke", "lr_valid_fraction": 1.0, "lr_valid_pixels": (size // 4) ** 2}
    return EvalSample(sample_id=f"smoke:s{index}", dataset="smoke", scene_group=f"scene{index}", category=None, split="test", evidence_class="synthetic", lr=lr, hr=hr,
                      lr_mask=torch.ones(size // 4, size // 4, dtype=torch.bool), hr_mask=mask, bands=BANDS, scale=4, lr_pixel_m=10.0, hr_pixel_m=2.5, quality=quality, provenance={"dataset": "smoke"})


class SmokeDataset:
    """Just enough of EvalDataset for the runner."""

    kind, role, evidence_class = "sen2neon", "smoke_test", "synthetic"
    name = "smoke"

    def __init__(self, n: int = 6):
        self._samples = {f"smoke:s{i}": synthetic_sample(i) for i in range(n)}
        self.records = [SimpleNamespace(sample_id=k, scene_id=s.scene_group) for k, s in self._samples.items()]
        self.ignored, self.manifest_digest, self.info = {}, "0" * 64, {"note": "in-memory synthetic scenes; not a dataset"}
        self.spec = SimpleNamespace(split="test", subset=None, hr_variant=None)

    def validate(self, *, check_files: bool = True) -> DatasetValidation:
        return DatasetValidation(valid_records=list(self.records), invalid=[])

    def load(self, record: Any) -> EvalSample:
        return self._samples[record.sample_id]


def run_smoke(output_dir: Path, n_scenes: int = 6, progress: Optional[Any] = None) -> Dict[str, Any]:
    """Run the pipeline on synthetic scenes and toy models; returns the run summary."""
    config = DownstreamConfig.from_dict({
        "name": "smoke", "output_dir": str(output_dir), "device": "cpu", "systems": [{"name": "toy_smooth", "kind": "lite"}, {"name": "toy_sharpen", "kind": "lite"}],
        "datasets": [{"name": "smoke", "kind": "sen2neon", "manifest": "unused.jsonl"}], "bootstrap": {"n_boot": 100}, "analysis": {"pooled_regions_per_tile": 300},
        "tiling": {"tile_size": 128, "overlap": 32}})
    tiling = TilingConfig(tile_size=128, overlap=32)
    systems = [CallableSystem("toy_smooth", _Smooth(), tiling, hard_constraint=False, provenance={"model_name": "toy interpolator", "weights": None}, device="cpu", kind="lite"),
               CallableSystem("toy_sharpen", _Sharpen(), tiling, hard_constraint=False, provenance={"model_name": "toy asymmetric sharpener", "weights": None}, device="cpu", kind="lite")]
    return run_downstream(config, systems=systems, datasets=[SmokeDataset(n_scenes)], progress=progress)
