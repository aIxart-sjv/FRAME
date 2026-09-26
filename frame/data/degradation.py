"""Synthetic LR generation (Phase 3): HR -> Sentinel-2-like LR.

Grounding
---------
docs/Requirements 142.txt (Requirement 3, sections 6, 9, 30-31; lines 2741-2757, 2820-2836,
3333-3350) and the SEN2NAIPv2 dataset card give the SEQUENCE FRAME follows:

    HR (NAIP-like)
      -> Gaussian blur                    (the sensor point-spread function)
      -> bilinear downsampling  x scale   (10 m from 2.5 m)
      -> reflectance harmonisation        (make the radiometry Sentinel-2-like)
      -> noise                            (signal-dependent sensor noise)
      -> S2-like LR

and the requirements' instruction is to reproduce the established SEN2NAIPv2 degradation
first and NOT to invent a new one (section 30). What this module does and does not do:

* The sequence and the alignment convention are implemented as documented.
* The NUMERIC parameters of the published process (kernel width, noise model, the trained
  U-Net) are not in the requirements or on the dataset card, and are not reproduced here.
  `frame_default_v1` supplies clearly-labelled FRAME defaults, and every record carries
  ``parameters_verified = False`` so nothing downstream can mistake it for the published
  SEN2NAIPv2 process. SEN2NAIPv2's own ready-made LR images are described by
  ``origin="upstream"`` records instead (frame.data.adapters.sen2naipv2).
* Harmonisation offers ``"none"`` and ``"histogram_match"`` (the SEN2NAIPv2 ``histmatch``
  variant's method, given a reference LR image). The learned U-Net variant needs trained
  weights and is not available.

Reproducibility: the only randomness is the noise, drawn from a CPU generator seeded by
``seed``; blur, downsampling and harmonisation are deterministic. The config, the seed and
the version go into the sample's `DegradationRecord`, so a synthetic LR can be regenerated
from its record.

Alignment: bilinear sampling with ``align_corners=False`` puts LR pixel i at HR position
``i * scale + (scale - 1) / 2``, the centre of the HR block it covers, so LR and HR share a
grid with no shift (tested with an impulse).
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Optional, Tuple

import torch
import torch.nn.functional as F

from frame.data.contract import DegradationRecord, HRStatus, PairedSample, PairRecord, PairType, RasterSpec
from frame.data.errors import DegradationError
from frame.data.geo import lr_transform_from_hr

DEGRADATION_NAME = "frame-sen2naipv2-style"
DEGRADATION_VERSION = "frame-degradation/1"
HARMONISATIONS = ("none", "histogram_match")


@dataclass(frozen=True)
class DegradationConfig:
    scale: int = 4
    blur_sigma: float = 2.0       # Gaussian sigma in HR pixels; 0 disables the blur
    harmonisation: str = "none"
    noise_a: float = 0.0          # noise variance = a * reflectance + b
    noise_b: float = 0.0
    clip_min: float = 0.0         # reflectance cannot be negative

    def __post_init__(self) -> None:
        if isinstance(self.scale, bool) or not isinstance(self.scale, int) or self.scale < 1:
            raise DegradationError(f"scale must be a positive integer, got {self.scale!r}.")
        for name in ("blur_sigma", "noise_a", "noise_b"):
            value = getattr(self, name)
            if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise DegradationError(f"{name} must be a finite number >= 0, got {value!r}.")
        if self.harmonisation not in HARMONISATIONS:
            raise DegradationError(f"harmonisation must be one of {HARMONISATIONS}, got {self.harmonisation!r}.")


def frame_default_v1(scale: int = 4, harmonisation: str = "none") -> DegradationConfig:
    """FRAME's default degradation. NOT the published SEN2NAIPv2 parameters (see the module docstring).

    Rationale for the numbers (all unverified against SEN2NAIPv2): the blur sigma is half the
    scale in HR pixels, so the kernel's FWHM (~1.18 x scale HR pixels) is about one LR pixel;
    the noise variance ``a*x + b`` gives a standard deviation of about 0.2 % reflectance at
    x = 0.2, small next to the signal but not zero.
    """
    return DegradationConfig(scale=scale, blur_sigma=0.5 * scale, harmonisation=harmonisation, noise_a=2e-5, noise_b=1e-6)


# ---------------------------------------------------------------------------------------------------------------
# steps
# ---------------------------------------------------------------------------------------------------------------

def _gaussian_kernel(sigma: float) -> torch.Tensor:
    radius = max(1, math.ceil(3.0 * sigma))
    x = torch.arange(-radius, radius + 1, dtype=torch.float64)
    kernel = torch.exp(-0.5 * (x / sigma) ** 2)
    return (kernel / kernel.sum()).to(torch.float32)


def gaussian_blur(image: torch.Tensor, sigma: float) -> torch.Tensor:
    """Separable Gaussian blur of a (C, H, W) tensor with reflect padding (sigma in pixels)."""
    if sigma == 0:
        return image
    kernel = _gaussian_kernel(sigma)
    radius = (kernel.numel() - 1) // 2
    _, height, width = image.shape
    if radius >= min(height, width):
        raise DegradationError(f"Blur radius {radius} px needs an image larger than {min(height, width)} px.")
    channels = image.shape[0]
    x = image[None]
    horizontal = kernel.view(1, 1, 1, -1).repeat(channels, 1, 1, 1)
    vertical = kernel.view(1, 1, -1, 1).repeat(channels, 1, 1, 1)
    x = F.conv2d(F.pad(x, (radius, radius, 0, 0), mode="reflect"), horizontal, groups=channels)
    x = F.conv2d(F.pad(x, (0, 0, radius, radius), mode="reflect"), vertical, groups=channels)
    return x[0]


def bilinear_downsample(image: torch.Tensor, scale: int) -> torch.Tensor:
    """Bilinear downsampling of a (C, H, W) tensor by an integer factor (no antialiasing: the blur came first)."""
    _, height, width = image.shape
    if height % scale or width % scale:
        raise DegradationError(f"HR size {(height, width)} is not divisible by scale {scale}.")
    return F.interpolate(image[None], size=(height // scale, width // scale), mode="bilinear", align_corners=False, antialias=False)[0]


def histogram_match(source: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    """Per-band quantile mapping of ``source`` (C, h, w) onto the value distribution of ``reference`` (C, h', w').

    Deterministic (stable sort). Used for the SEN2NAIPv2 ``histmatch`` style of harmonisation.
    """
    if source.shape[0] != reference.shape[0]:
        raise DegradationError(f"Reference has {reference.shape[0]} band(s), source has {source.shape[0]}.")
    out = torch.empty_like(source)
    for band in range(source.shape[0]):
        flat = source[band].reshape(-1)
        ref_sorted = torch.sort(reference[band].reshape(-1)).values
        order = torch.sort(flat, stable=True).indices
        positions = torch.linspace(0, ref_sorted.numel() - 1, flat.numel(), dtype=torch.float64)
        lower = positions.floor().long()
        upper = (lower + 1).clamp(max=ref_sorted.numel() - 1)
        fraction = (positions - lower).to(torch.float32)
        matched = ref_sorted[lower] * (1 - fraction) + ref_sorted[upper] * fraction
        result = torch.empty_like(flat)
        result[order] = matched
        out[band] = result.view(source.shape[1:])
    return out


def add_signal_dependent_noise(image: torch.Tensor, a: float, b: float, seed: int, clip_min: float) -> torch.Tensor:
    """Gaussian noise with variance ``a * x + b`` (x = reflectance, floored at 0), seeded and reproducible."""
    if a == 0 and b == 0:
        return image
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    noise = torch.randn(image.shape, generator=generator, dtype=torch.float32)
    std = torch.sqrt((a * image.clamp(min=0.0) + b).clamp(min=0.0))
    return (image + std * noise).clamp(min=clip_min)


# ---------------------------------------------------------------------------------------------------------------
# the pipeline
# ---------------------------------------------------------------------------------------------------------------

def degrade(
    hr: torch.Tensor, config: DegradationConfig, *, seed: int, reference: Optional[torch.Tensor] = None
) -> Tuple[torch.Tensor, DegradationRecord]:
    """HR (C, H, W float reflectance) -> (LR (C, H/scale, W/scale), the record that reproduces it)."""
    if not isinstance(hr, torch.Tensor) or hr.ndim != 3 or not hr.is_floating_point():
        raise DegradationError("hr must be a floating-point (channels, height, width) tensor.")
    if not bool(torch.isfinite(hr).all()):
        raise DegradationError("hr contains NaN or Inf; the degradation refuses to run on corrupt input.")
    if config.harmonisation == "histogram_match" and reference is None:
        raise DegradationError("harmonisation='histogram_match' needs a reference LR image.")

    lr = bilinear_downsample(gaussian_blur(hr.to(torch.float32), config.blur_sigma), config.scale)
    if config.harmonisation == "histogram_match":
        lr = histogram_match(lr, reference.to(torch.float32))  # type: ignore[union-attr]
    lr = add_signal_dependent_noise(lr, config.noise_a, config.noise_b, seed, config.clip_min)

    record = DegradationRecord(
        name=DEGRADATION_NAME, version=DEGRADATION_VERSION, config=asdict(config), seed=int(seed),
        harmonisation=config.harmonisation, parameters_verified=False, origin="frame",
    )
    return lr.to(torch.float32), record


def synthesize_pair(
    hr: torch.Tensor,
    *,
    sample_id: str,
    dataset: str,
    scene_id: str,
    split,
    hr_spec: RasterSpec,
    config: DegradationConfig,
    seed: int,
    variant: str = "default",
    region_id: Optional[str] = None,
    reference: Optional[torch.Tensor] = None,
    hr_mask: Optional[torch.Tensor] = None,
    license: Optional[str] = None,
    provenance: Optional[dict] = None,
) -> PairedSample:
    """Generate the LR of ``hr`` and return a complete synthetic `PairedSample` with full provenance.

    The LR raster description is derived from ``hr_spec``: same bands and CRS, pixel size x scale,
    transform via the same LR/HR relationship frame.data.geo validates.
    """
    lr, degradation = degrade(hr, config, seed=seed, reference=reference)
    lr_transform = lr_transform_from_hr(hr_spec.transform, config.scale) if hr_spec.transform is not None else None
    lr_spec = RasterSpec(
        band_names=hr_spec.band_names, width=lr.shape[-1], height=lr.shape[-2],
        pixel_size_m=hr_spec.pixel_size_m * config.scale if hr_spec.pixel_size_m is not None else None,
        path=None, dtype="float32", reflectance_scale=None, nodata=None, crs=hr_spec.crs, transform=lr_transform,
    )
    record = PairRecord(
        sample_id=sample_id, dataset=dataset, scene_id=scene_id, split=split, pair_type=PairType.SYNTHETIC,
        hr_status=HRStatus.AVAILABLE, scale_factor=config.scale, lr=lr_spec, hr=hr_spec, variant=variant,
        region_id=region_id, degradation=degradation, license=license, provenance=dict(provenance or {}),
    )
    lr_mask = None
    if hr_mask is not None:
        lr_mask = (F.avg_pool2d(hr_mask.float()[None, None], config.scale)[0, 0] == 1.0)
    return PairedSample(
        record=record, lr=lr, hr=hr.to(torch.float32), lr_bands=hr_spec.band_names, hr_bands=hr_spec.band_names,
        lr_mask=lr_mask, hr_mask=hr_mask,
    )
