"""Mask-aware metrics for evaluating an SR output against an independent HR reference (Phase 5).

Every function takes ``sr`` and ``hr`` as (C, H, W) arrays or tensors on the HR grid and a boolean ``mask`` (H, W) that is True where the HR reference is a
real observation. **A pixel outside the mask is never scored**: it contributes to no sum, mean, correlation or window, and windowed metrics (SSIM, the
high-frequency filter, gradients) additionally exclude a border of the mask's own eroded width so their windows never straddle nodata. A tile with no valid
pixel yields ``None`` values (never 0.0), and an infinite PSNR is reported as ``None``.

Groups (kept separate everywhere, never merged into one score)
-------------------------------------------------------------
reference_accuracy / per_band_metrics / index_metrics / band_ratio_metrics
    SR against the independent HR reference. PSNR, RMSE, SAM and ERGAS reuse frame.validation.compute_reference_metrics (which reuses opensr-test's own
    distances); MAE, per-band values and SSIM with an eroded mask are added here. PSNR/SSIM use data range 1.0 (reflectance as a fraction, FRAME's convention;
    reflectance may exceed 1.0). Valid-pixel rule: the HR mask. The index and ratio rules additionally exclude pixels whose HR reflectance sum/values are below
    a floor (noise-dominated denominators), decided from the HR reference ONLY so the same pixels are scored for every system; the excluded fraction is reported.
spatial_detail_metrics / seam_error / hallucination_analysis
    Whether apparent fine detail is supported by reference structure. Definitions are in each docstring.
self_consistency
    SR -> area-average downsample -> compare with the LR observation (frame.consistency). This is NOT accuracy against an HR reference.
opensr_native
    opensr-test's own consistency / synthesis / hallucination / omission / improvement values, as its own group.
data_quality
    valid and nodata fractions.

The nodata behaviour differs by evidence class: for synthetic pairs the reference is the ground truth the LR was made from; for real cross-sensor pairs (SEN2NEON,
OpenSR-Test) the reference is a DIFFERENT sensor with its own radiometry and registration error, so no metric here is an absolute error against "truth".
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from scipy import ndimage
from scipy import stats as scipy_stats

from frame.evaluate.errors import ReferenceMismatchError

METRICS_VERSION = "frame-eval-metrics/1"
Array = Union[np.ndarray, torch.Tensor]


@dataclass(frozen=True)
class MetricConfig:
    hf_sigma: float = 2.0                                    # HR pixels: detail finer than ~half an LR pixel (an LR pixel is 4 HR pixels)
    ssim_window: int = 7                                     # skimage default
    hallucination_taus: Tuple[float, ...] = (0.005, 0.0025, 0.01)     # reflectance; the FIRST is the primary threshold, the others are sensitivity checks
    index_min_sum: float = 0.02                              # HR reflectance sum below which a normalised difference is noise-dominated
    ratio_floor: float = 0.005                               # HR band reflectance below which a band ratio is not scored
    phase_upsample: int = 10                                 # sub-pixel resolution of the registration estimate (1/10 HR pixel)
    version: str = METRICS_VERSION

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["hallucination_taus"] = list(self.hallucination_taus)
        return d


CFG = MetricConfig()


# ---------------------------------------------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------------------------------------------

def _np(x: Array, dtype=None) -> np.ndarray:
    a = x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else np.asarray(x)
    return a.astype(dtype) if dtype is not None else a


def _prepare(sr: Array, hr: Array, mask: Array) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    sr_a, hr_a, mask_a = _np(sr, np.float64), _np(hr, np.float64), _np(mask).astype(bool)
    if sr_a.ndim != 3 or sr_a.shape != hr_a.shape:
        raise ReferenceMismatchError(f"SR shape {tuple(sr_a.shape)} does not equal the HR reference shape {tuple(hr_a.shape)} (both must be (bands, H, W)).")
    if mask_a.shape != hr_a.shape[1:]:
        raise ReferenceMismatchError(f"mask shape {tuple(mask_a.shape)} does not match the image shape {tuple(hr_a.shape[1:])}.")
    return sr_a, hr_a, mask_a


def erode(mask: np.ndarray, radius: int) -> np.ndarray:
    """Erode a boolean mask by ``radius`` pixels (square). The image border counts as invalid, so window filters never see edge padding."""
    if radius <= 0:
        return mask.copy()
    return ndimage.binary_erosion(mask, structure=np.ones((3, 3), bool), iterations=int(radius), border_value=0)


def _finite(value: Any) -> Optional[float]:
    if value is None:
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def _pearson(a: np.ndarray, b: np.ndarray) -> Optional[float]:
    if a.size < 2 or float(a.std()) == 0.0 or float(b.std()) == 0.0:
        return None
    return _finite(np.corrcoef(a, b)[0, 1])


def _psnr(mse: float) -> Optional[float]:
    return None if mse <= 0 else 10.0 * math.log10(1.0 / mse)


def ssim_maps(sr: np.ndarray, hr: np.ndarray, cfg: MetricConfig = CFG) -> np.ndarray:
    """Per-band SSIM maps, (C, H, W) (skimage, data range 1.0). Computed once and shared by `reference_accuracy` and `per_band_metrics`."""
    from skimage.metrics import structural_similarity

    maps = []
    for b in range(hr.shape[0]):
        _, m = structural_similarity(hr[b], sr[b], data_range=1.0, full=True, win_size=cfg.ssim_window)
        maps.append(m)
    return np.stack(maps)


# ---------------------------------------------------------------------------------------------------------------
# reference accuracy
# ---------------------------------------------------------------------------------------------------------------

def reference_accuracy(sr: Array, hr: Array, mask: Array, band_names: Sequence[str], scale: int, cfg: MetricConfig = CFG, *,
                       ssim_map: Optional[np.ndarray] = None) -> Dict[str, Any]:
    """PSNR, SSIM, RMSE, MAE, SAM, ERGAS of ``sr`` against ``hr`` over the valid pixels (pooled over bands)."""
    sr_a, hr_a, mask_a = _prepare(sr, hr, mask)
    n_valid = int(mask_a.sum())
    out: Dict[str, Any] = {"n_valid_pixels": n_valid, "valid_fraction": n_valid / mask_a.size}
    keys = ("psnr_db", "ssim", "rmse", "mae", "sam_degrees", "ergas")
    if n_valid == 0:
        return {**out, **{k: None for k in keys}}

    from frame.validation import compute_reference_metrics

    ref = compute_reference_metrics(torch.from_numpy(sr_a.astype(np.float32)), torch.from_numpy(hr_a.astype(np.float32)), mask_a,
                                    band_names=list(band_names), scale_factor=int(scale))
    err = sr_a[:, mask_a] - hr_a[:, mask_a]
    ssim_valid = erode(mask_a, cfg.ssim_window // 2)
    maps = ssim_map if ssim_map is not None else ssim_maps(sr_a, hr_a, cfg)
    out.update(
        psnr_db=_finite(ref.psnr_db), rmse=_finite(ref.rmse), mae=float(np.abs(err).mean()), sam_degrees=_finite(ref.sam_degrees), ergas=_finite(ref.ergas),
        ssim=_finite(maps[:, ssim_valid].mean()) if ssim_valid.any() else None, ssim_n_valid_pixels=int(ssim_valid.sum()),
    )
    return out


def per_band_metrics(sr: Array, hr: Array, mask: Array, band_names: Sequence[str], cfg: MetricConfig = CFG, *,
                     ssim_map: Optional[np.ndarray] = None) -> Dict[str, Dict[str, Any]]:
    """RMSE, MAE, bias, percent bias, PSNR, Pearson r, SSIM and the reference mean, for every band."""
    sr_a, hr_a, mask_a = _prepare(sr, hr, mask)
    if len(band_names) != hr_a.shape[0]:
        raise ReferenceMismatchError(f"{len(band_names)} band name(s) for {hr_a.shape[0]} band(s).")
    ssim_valid = erode(mask_a, cfg.ssim_window // 2)
    maps = None
    out: Dict[str, Dict[str, Any]] = {}
    for i, name in enumerate(band_names):
        if not mask_a.any():
            out[name] = {k: None for k in ("rmse", "mae", "bias", "pbias_percent", "psnr_db", "pearson_r", "ssim", "hr_mean")}
            continue
        s, h = sr_a[i][mask_a], hr_a[i][mask_a]
        err = s - h
        mse = float(np.mean(err ** 2))
        if maps is None and ssim_valid.any():
            maps = ssim_map if ssim_map is not None else ssim_maps(sr_a, hr_a, cfg)
        hr_mean = float(h.mean())
        out[name] = {
            "rmse": math.sqrt(mse), "mae": float(np.abs(err).mean()), "bias": float(err.mean()),
            "pbias_percent": 100.0 * float(err.mean()) / hr_mean if hr_mean > 0 else None, "psnr_db": _psnr(mse), "pearson_r": _pearson(s, h),
            "ssim": _finite(maps[i][ssim_valid].mean()) if maps is not None else None, "hr_mean": hr_mean,
        }
    return out


# ---------------------------------------------------------------------------------------------------------------
# spectral indices and band ratios
# ---------------------------------------------------------------------------------------------------------------

def _nd(a: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Normalised difference (a - b) / (a + b) and its element-wise sum."""
    total = a + b
    with np.errstate(divide="ignore", invalid="ignore"):
        return (a - b) / total, total


#: name -> (bands required, function of {band: array} returning (index, denominator))
_INDICES: Dict[str, Tuple[Tuple[str, ...], Any]] = {
    "NDVI": (("B08", "B04"), lambda b: _nd(b["B08"], b["B04"])),
    "NDWI": (("B03", "B08"), lambda b: _nd(b["B03"], b["B08"])),                                           # McFeeters: (green - NIR) / (green + NIR)
    "MNDWI": (("B03", "B11"), lambda b: _nd(b["B03"], b["B11"])),
    "NDBI": (("B11", "B08"), lambda b: _nd(b["B11"], b["B08"])),
    "BSI": (("B11", "B04", "B08", "B02"), lambda b: _nd(b["B11"] + b["B04"], b["B08"] + b["B02"])),
}


def index_metrics(sr: Array, hr: Array, mask: Array, band_names: Sequence[str], cfg: MetricConfig = CFG) -> Dict[str, Any]:
    """Reference-based derived-index accuracy: index(SR) vs index(HR) -- MAE, RMSE, bias, Pearson r, Wasserstein-1 distance of the two distributions.

    Indices whose bands are absent from ``band_names`` (MNDWI, NDBI, BSI need SWIR B11, which RGBN systems do not produce) are listed under ``skipped``
    with the reason; nothing is approximated or invented. The scored pixels are the mask AND HR reflectance sum > ``cfg.index_min_sum`` AND a finite SR index.
    """
    sr_a, hr_a, mask_a = _prepare(sr, hr, mask)
    index_of = {n: i for i, n in enumerate(band_names)}
    out: Dict[str, Any] = {"skipped": {}}
    for name, (needed, fn) in _INDICES.items():
        missing = [b for b in needed if b not in index_of]
        if missing:
            out["skipped"][name] = f"missing band(s) {missing} (e.g. B11 is SWIR and is not produced by the RGBN systems)"
            continue
        sr_idx, sr_sum = fn({b: sr_a[index_of[b]] for b in needed})
        hr_idx, hr_sum = fn({b: hr_a[index_of[b]] for b in needed})
        n_mask = int(mask_a.sum())
        valid = mask_a & (hr_sum > cfg.index_min_sum) & np.isfinite(sr_idx) & np.isfinite(hr_idx) & (np.abs(sr_sum) > 1e-6)
        n_valid = int(valid.sum())
        entry: Dict[str, Any] = {"n_valid": n_valid, "excluded_fraction": (1 - n_valid / n_mask) if n_mask else None}
        if n_valid == 0:
            entry.update({k: None for k in ("mae", "rmse", "bias", "pearson_r", "wasserstein")})
        else:
            d = sr_idx[valid] - hr_idx[valid]
            entry.update(mae=float(np.abs(d).mean()), rmse=float(np.sqrt(np.mean(d ** 2))), bias=float(d.mean()), pearson_r=_pearson(sr_idx[valid], hr_idx[valid]),
                         wasserstein=float(scipy_stats.wasserstein_distance(sr_idx[valid], hr_idx[valid])))
        out[name] = entry
    return out


_RATIOS = (("B08", "B04"), ("B03", "B04"))


def band_ratio_metrics(sr: Array, hr: Array, mask: Array, band_names: Sequence[str], cfg: MetricConfig = CFG) -> Dict[str, Dict[str, Any]]:
    """Band-ratio consistency against the reference: error of log(SR_a / SR_b) relative to log(HR_a / HR_b) (bias, mean and median absolute).

    A log ratio makes a 10% band error the same size everywhere. Pixels are scored where both HR bands exceed ``cfg.ratio_floor``; SR values are floored at the same
    value before the log (an SR reflectance <= 0 is an error to be seen, not a NaN to be dropped).
    """
    sr_a, hr_a, mask_a = _prepare(sr, hr, mask)
    index_of = {n: i for i, n in enumerate(band_names)}
    out: Dict[str, Dict[str, Any]] = {}
    for num, den in _RATIOS:
        key = f"{num}/{den}"
        if num not in index_of or den not in index_of:
            out[key] = {"status": "skipped", "reason": f"missing band(s) among {num}, {den}"}
            continue
        hn, hd = hr_a[index_of[num]], hr_a[index_of[den]]
        valid = mask_a & (hn > cfg.ratio_floor) & (hd > cfg.ratio_floor)
        if not valid.any():
            out[key] = {"status": "not_computable", "n_valid": 0, "bias_log_ratio": None, "mae_log_ratio": None, "median_abs_log_ratio_error": None}
            continue
        sn = np.maximum(sr_a[index_of[num]], cfg.ratio_floor)[valid]
        sd = np.maximum(sr_a[index_of[den]], cfg.ratio_floor)[valid]
        d = np.log(sn / sd) - np.log(hn[valid] / hd[valid])
        out[key] = {"status": "ok", "n_valid": int(valid.sum()), "bias_log_ratio": float(d.mean()), "mae_log_ratio": float(np.abs(d).mean()),
                    "median_abs_log_ratio_error": float(np.median(np.abs(d)))}
    return out


# ---------------------------------------------------------------------------------------------------------------
# spatial / detail
# ---------------------------------------------------------------------------------------------------------------

def _highpass(x: np.ndarray, sigma: float) -> np.ndarray:
    return x - ndimage.gaussian_filter(x, sigma=(0, sigma, sigma), mode="reflect", truncate=3.0)


def _phase_shift(sr_mean: np.ndarray, hr_mean: np.ndarray, mask: np.ndarray, upsample: int) -> Dict[str, Any]:
    """Global displacement of the SR relative to the reference by cross-correlation of the band-mean images (masked when there is nodata).

    The correlation is NOT phase-whitened (``normalization=None``): whitening gives every frequency the same weight, and a blurry SR has no energy where a sharper reference does,
    so those frequencies become noise and the peak is lost. Measured on real SEN2NEON tiles it reported ~0 for tiles whose best whole-pixel alignment (a brute-force
    search) was 2-6 pixels away, while the un-whitened estimate agreed with the search within one pixel on all 39 tiles tested. A (nearly) constant image cannot be registered
    and is reported as not computable rather than given the arbitrary shift the correlation returns for it.
    """
    from skimage.registration import phase_cross_correlation

    for label, image in (("SR", sr_mean), ("reference", hr_mean)):
        if not mask.any() or float(np.std(image[mask])) < 1e-6:
            return {"status": "not_computable", "reason": f"the {label} is constant over the valid pixels; there is nothing to register", "dy": None, "dx": None, "magnitude": None}
    try:
        if mask.all():
            shift, error, _ = phase_cross_correlation(hr_mean - hr_mean.mean(), sr_mean - sr_mean.mean(), upsample_factor=upsample, normalization=None)
        else:
            shift, error, _ = phase_cross_correlation(hr_mean, sr_mean, upsample_factor=upsample, reference_mask=mask, moving_mask=mask, normalization=None)
    except Exception as exc:  # degenerate (fully masked) images: report it, do not guess
        return {"status": "not_computable", "reason": f"{type(exc).__name__}: {exc}", "dy": None, "dx": None, "magnitude": None}
    dy, dx = -float(shift[0]), -float(shift[1])        # skimage returns the shift that registers SR onto HR; the SR displacement is its negative
    return {"status": "ok", "dy": dy, "dx": dx, "magnitude": math.hypot(dy, dx), "registration_error": _finite(error)}


def spatial_detail_metrics(sr: Array, hr: Array, mask: Array, band_names: Sequence[str], cfg: MetricConfig = CFG) -> Dict[str, Any]:
    """Is the reconstructed fine detail the reference's detail?

    High-frequency detail is ``x - gaussian_blur(x, hf_sigma)`` (sigma in HR pixels), per band, scored on the mask eroded by the filter radius so no window
    reaches nodata:

    * ``hf_correlation``      mean over bands of the Pearson correlation between SR and reference detail (1 = same structure in the same places).
    * ``hf_energy_ratio``     sum(SR_hf^2) / sum(HR_hf^2): < 1 the SR has less detail than the reference (smoother), > 1 more. It says HOW MUCH detail, not whether it is right.
    * ``hf_relative_error``   sqrt(sum((SR_hf - HR_hf)^2) / sum(HR_hf^2)): 0 = perfect, exactly 1 = as good as adding NO detail at all, > 1 = worse than adding none
                              (detail that is wrong or invented). This is the number to read against ``hf_energy_ratio``.
    * ``gradient_correlation``  Pearson correlation of Sobel gradient magnitudes of the band-mean images (edge structure).
    * ``phase_shift_px``      global displacement (dy, dx) of the SR relative to the reference in HR pixels, by phase correlation of the band-mean images with
                              masked correlation when there is nodata. Non-zero means misregistration of the SR and/or of the reference itself.
    """
    sr_a, hr_a, mask_a = _prepare(sr, hr, mask)
    radius = int(cfg.hf_sigma * 3.0 + 0.5)
    valid = erode(mask_a, radius)
    out: Dict[str, Any] = {"hf_sigma_px": cfg.hf_sigma, "hf_valid_pixels": int(valid.sum())}
    if not valid.any():
        return {**out, "hf_correlation": None, "hf_energy_ratio": None, "hf_relative_error": None, "gradient_correlation": None,
                "phase_shift_px": {"status": "not_computable", "reason": "no valid pixels", "dy": None, "dx": None, "magnitude": None}, "per_band": {}}
    sr_hf, hr_hf = _highpass(sr_a, cfg.hf_sigma), _highpass(hr_a, cfg.hf_sigma)
    per_band, corrs = {}, []
    for i, name in enumerate(band_names):
        s, h = sr_hf[i][valid], hr_hf[i][valid]
        r = _pearson(s, h)
        corrs.append(r)
        h_energy = float(np.sum(h ** 2))
        per_band[name] = {"hf_correlation": r, "hf_energy_ratio": float(np.sum(s ** 2) / h_energy) if h_energy > 0 else None,
                          "hf_relative_error": float(math.sqrt(np.sum((s - h) ** 2) / h_energy)) if h_energy > 0 else None}
    s_all, h_all = sr_hf[:, valid], hr_hf[:, valid]
    h_energy_all = float(np.sum(h_all ** 2))
    finite_corrs = [c for c in corrs if c is not None]
    out.update(
        hf_correlation=float(np.mean(finite_corrs)) if finite_corrs else None,
        hf_energy_ratio=float(np.sum(s_all ** 2) / h_energy_all) if h_energy_all > 0 else None,
        hf_relative_error=float(math.sqrt(np.sum((s_all - h_all) ** 2) / h_energy_all)) if h_energy_all > 0 else None,
        per_band=per_band,
    )
    sr_mean, hr_mean = sr_a.mean(axis=0), hr_a.mean(axis=0)
    grad = lambda x: np.hypot(ndimage.sobel(x, axis=0), ndimage.sobel(x, axis=1))
    gvalid = erode(mask_a, 2)
    out["gradient_correlation"] = _pearson(grad(sr_mean)[gvalid], grad(hr_mean)[gvalid]) if gvalid.any() else None
    out["phase_shift_px"] = _phase_shift(sr_mean, hr_mean, mask_a, cfg.phase_upsample)
    return out


def seam_error(sr: Array, hr: Array, mask: Array, *, seam_rows: Sequence[int], seam_cols: Sequence[int], half_width: int) -> Dict[str, Any]:
    """Reference-based tile-seam error: mean |SR - HR| within ``half_width`` pixels of a seam line versus the rest of the valid image.

    Seam positions come from the tile plan (frame.tiling). A ratio near 1 means seams are no worse than the interior; well above 1 means tiling artefacts.
    Complements frame.tiling's no-reference seam diagnostic (which compares overlapping tile predictions with each other).
    """
    sr_a, hr_a, mask_a = _prepare(sr, hr, mask)
    if not seam_rows and not seam_cols:
        return {"status": "no_seams", "reason": "the scene was reconstructed from a single tile", "seam_mae": None, "interior_mae": None, "ratio": None, "n_seam_pixels": 0}
    seam = np.zeros(mask_a.shape, dtype=bool)
    for c in seam_cols:
        seam[:, max(0, c - half_width): c + half_width] = True
    for r in seam_rows:
        seam[max(0, r - half_width): r + half_width, :] = True
    err = np.abs(sr_a - hr_a).mean(axis=0)
    seam_v, interior_v = mask_a & seam, mask_a & ~seam
    seam_mae = float(err[seam_v].mean()) if seam_v.any() else None
    interior_mae = float(err[interior_v].mean()) if interior_v.any() else None
    ratio = seam_mae / interior_mae if seam_mae is not None and interior_mae not in (None, 0.0) else None
    return {"status": "ok", "seam_mae": seam_mae, "interior_mae": interior_mae, "ratio": ratio, "n_seam_pixels": int(seam_v.sum()), "half_width_px": int(half_width)}


# ---------------------------------------------------------------------------------------------------------------
# improvement / supported synthesis / omission / unsupported detail
# ---------------------------------------------------------------------------------------------------------------

def hallucination_labels(baseline: Array, sr: Array, hr: Array, tau: float) -> Dict[str, np.ndarray]:
    """Per-element boolean maps of the decomposition documented in :func:`hallucination_analysis` (no mask applied: the caller restricts them to the valid pixels).

    Shared with the reliability analysis (frame.reliability), which needs the same categories per pixel rather than as fractions; ``hallucination_analysis``
    is computed from these maps, so the two can never disagree.
    """
    base_a, sr_a, hr_a = _np(baseline, np.float64), _np(sr, np.float64), _np(hr, np.float64)
    if not (base_a.shape == sr_a.shape == hr_a.shape):
        raise ReferenceMismatchError(f"baseline {tuple(base_a.shape)}, SR {tuple(sr_a.shape)} and reference {tuple(hr_a.shape)} must have the same shape.")
    A, T = sr_a - base_a, hr_a - base_a
    err_sr, err_base = np.abs(sr_a - hr_a), np.abs(base_a - hr_a)
    added, needed = np.abs(A) > tau, np.abs(T) > tau
    same_sign = np.sign(A) == np.sign(T)
    improved = (err_base - err_sr) > 1e-9
    supported = added & needed & same_sign & improved
    return {"supported_synthesis": supported, "unsupported_detail": added & ~supported, "omission": needed & ~added, "neutral": ~needed & ~added,
            "unsupported_wrong_direction": added & needed & ~same_sign, "unsupported_overshoot": added & needed & same_sign & ~improved,
            "unsupported_where_reference_has_none": added & ~needed, "added": added, "needed": needed, "improved": improved}


def hallucination_analysis(baseline: Array, sr: Array, hr: Array, mask: Array, cfg: MetricConfig = CFG) -> Dict[str, Dict[str, Any]]:
    """Transparent element-wise decomposition of what an SR system ADDED relative to a baseline, judged against the HR reference.

    For every valid (band, pixel) element: ``A = SR - baseline`` (detail the system added), ``T = HR - baseline`` (detail the reference has that the baseline
    lacks), and a reflectance threshold ``tau`` below which a difference is treated as absent (noise floor). ``added = |A| > tau``, ``needed = |T| > tau``,
    ``improved = |SR - HR| < |baseline - HR|``. The valid elements are partitioned into exactly four categories:

    * ``supported_synthesis``  added, needed, same sign as T, and it reduced the error: detail the reference confirms.
    * ``unsupported_detail``   added but NOT confirmed. Sub-breakdown (a subset, not part of the partition): ``unsupported_wrong_direction`` (needed, opposite
                               sign), ``unsupported_overshoot`` (needed, right sign, but the error grew), ``unsupported_where_reference_has_none`` (not needed).
    * ``omission``             the reference has detail (needed) and nothing was added.
    * ``neutral``              nothing needed, nothing added.

    Also ``improved_fraction`` and ``mean_error_reduction`` (mean of |baseline - HR| - |SR - HR|, positive = closer to the reference than the baseline) and
    ``mse_skill_vs_baseline`` = 1 - MSE_SR / MSE_baseline (1 = perfect, 0 = no better than the baseline, negative = worse).

    Caveats to keep in view: the HR reference is not ground truth (a different sensor with its own radiometry and registration error), so "unsupported" means
    "not confirmed by THIS reference". The categories are element-wise and use no semantics: no land-cover labels are invented. Results are given at several
    ``tau`` values (the first in ``cfg.hallucination_taus`` is primary) so their sensitivity to that choice is visible. The baseline is the bicubic upsampling
    of the same LR, so this analysis is not defined for the baseline itself.
    """
    base_a, sr_a, mask_a = _np(baseline, np.float64), _np(sr, np.float64), _np(mask).astype(bool)
    _, hr_a, mask_a = _prepare(sr_a, hr, mask_a)
    if base_a.shape != hr_a.shape:
        raise ReferenceMismatchError(f"baseline shape {tuple(base_a.shape)} does not equal the reference shape {tuple(hr_a.shape)}.")
    valid = np.broadcast_to(mask_a[None], hr_a.shape)
    err_sr, err_base = np.abs(sr_a - hr_a)[valid], np.abs(base_a - hr_a)[valid]
    n = int(err_sr.size)
    out: Dict[str, Dict[str, Any]] = {}
    for tau in cfg.hallucination_taus:
        if n == 0:
            out[str(tau)] = {"tau": tau, "n_valid_elements": 0}
            continue
        lab = {k: v[valid] for k, v in hallucination_labels(base_a, sr_a, hr_a, tau).items()}
        supported, unsupported, omission, neutral = lab["supported_synthesis"], lab["unsupported_detail"], lab["omission"], lab["neutral"]
        wrong_dir, overshoot, no_ref = lab["unsupported_wrong_direction"], lab["unsupported_overshoot"], lab["unsupported_where_reference_has_none"]
        added, needed, improved = lab["added"], lab["needed"], lab["improved"]
        mse_base = float(np.mean((base_a - hr_a)[valid] ** 2))
        out[str(tau)] = {
            "tau": tau, "n_valid_elements": n,
            "supported_synthesis": float(supported.mean()), "unsupported_detail": float(unsupported.mean()), "omission": float(omission.mean()), "neutral": float(neutral.mean()),
            "unsupported_wrong_direction": float(wrong_dir.mean()), "unsupported_overshoot": float(overshoot.mean()),
            "unsupported_where_reference_has_none": float(no_ref.mean()),
            "added_fraction": float(added.mean()), "needed_fraction": float(needed.mean()),
            "improved_fraction": float(improved.mean()), "mean_error_reduction": float(np.mean(err_base - err_sr)),
            "mse_skill_vs_baseline": (1.0 - float(np.mean((sr_a - hr_a)[valid] ** 2)) / mse_base) if mse_base > 0 else None,
        }
    return out


# ---------------------------------------------------------------------------------------------------------------
# self-consistency (NOT reference accuracy), opensr-test native, data quality
# ---------------------------------------------------------------------------------------------------------------

SELF_CONSISTENCY_DEFINITION = ("SR -> area-average downsample to the LR grid -> compare with the LR observation it was made from. "
                               "This is NOT accuracy against an HR reference: an SR can average back to the LR perfectly and still be far from the truth.")


def _disc(b) -> Dict[str, Any]:
    return {"status": b.status.value, "n_samples": b.sample_count, "mae": b.mean_abs_error, "rmse": b.rmse, "max_abs_error": b.max_abs_error, "normalized_rmse": b.normalized_rmse}


def _idx(c) -> Dict[str, Any]:
    return {"status": c.status.value, "n_valid": c.valid_pixel_count, "mae": c.mean_abs_discrepancy, "rmse": c.rmse, "max_abs": c.max_abs_discrepancy}


def self_consistency(lr: Array, sr: Array, lr_mask: Array, band_names: Sequence[str], scale: int) -> Dict[str, Any]:
    """Downsample-consistency of ``sr`` against ``lr`` (frame.consistency): per band, pooled, NDVI and B08/B04 ratio. Uses the LR validity mask only."""
    from frame.consistency import run_consistency_diagnostics

    d = run_consistency_diagnostics(torch.from_numpy(_np(lr, np.float32)), torch.from_numpy(_np(sr, np.float32)), _np(lr_mask).astype(bool),
                                    band_names=list(band_names), scale_factor=int(scale))
    return {
        "definition": SELF_CONSISTENCY_DEFINITION, "downsample_method": d.downsample_consistency.downsample_method, "lr_valid_pixels": d.downsample_consistency.valid_pixel_count,
        "overall": _disc(d.downsample_consistency.overall), "per_band": {n: _disc(b) for n, b in d.downsample_consistency.per_band.items()},
        "ndvi": _idx(d.ndvi_comparison), "b08_b04_ratio": _idx(d.b08_b04_ratio_comparison),
    }


OPENSR_DEFINITION = ("opensr-test's own metric vocabulary (reflectance / spectral / spatial consistency, synthesis, hallucination, omission, improvement), computed by the "
                     "installed opensr-test package with its default Config. Not FRAME's definitions; not comparable with the element-wise analysis.")


def opensr_native(lr: Array, sr: Array, hr: Array, *, hr_mask: Array) -> Dict[str, Any]:
    """opensr-test's own values (frame.validation wrapper). Refuses data with nodata: its metrics have no mask and would score the zeros."""
    if not _np(hr_mask).astype(bool).all():
        return {"status": "not_computed", "reason": "the reference has nodata pixels and opensr-test's metrics take no mask; not scored rather than scoring zeros as data",
                "definition": OPENSR_DEFINITION}
    from frame.validation import compute_opensr_test_metrics

    m = compute_opensr_test_metrics(torch.from_numpy(_np(lr, np.float32)), torch.from_numpy(_np(sr, np.float32)), torch.from_numpy(_np(hr, np.float32)))
    values = {k: getattr(m, k) for k in ("reflectance", "spectral", "spatial", "synthesis", "hallucination", "omission", "improvement")}
    return {"status": "computed", "values": values, "opensr_test_version": m.opensr_test_version, "config": m.config_summary, "definition": OPENSR_DEFINITION}


def data_quality(hr_mask: Array, lr_mask: Optional[Array] = None) -> Dict[str, Any]:
    hm = _np(hr_mask).astype(bool)
    out = {"hr_valid_pixels": int(hm.sum()), "hr_total_pixels": int(hm.size), "hr_valid_fraction": float(hm.mean()), "hr_nodata_fraction": float(1 - hm.mean())}
    if lr_mask is not None:
        lm = _np(lr_mask).astype(bool)
        out.update(lr_valid_pixels=int(lm.sum()), lr_total_pixels=int(lm.size), lr_valid_fraction=float(lm.mean()), lr_nodata_fraction=float(1 - lm.mean()))
    return out
