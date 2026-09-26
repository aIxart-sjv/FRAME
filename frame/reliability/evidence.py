"""One tile's evidence (Phase 6): the deployed stability, the error targets, the trivial baselines, and every within-tile association, from arrays already in memory.

Input: the TTA ensemble of one tile (mean, per-band spread, the identity-member prediction), the bicubic baseline, the reference and its strict mask, and the eligibility gate that
admitted the tile. Everything is computed on the ALIGNED overlap: the whole-pixel correction recorded by the gate (an integer crop of every grid, no resampling) is applied to all
grids before any error is formed. Output: a JSON-ready result row, and a dict of compact arrays (a seeded subsample of the tile's cells and pixels) from which the pooled analyses
(risk-coverage, detection, calibration) are built without repeating the ensemble.

Associations are computed WITHIN the tile (pixels and cells of one tile are spatially autocorrelated, so they are summarised per tile and aggregated over scene units by the runner,
never pooled as independent samples). Undefined results (a constant stability map, a zero-error prediction) are reported as such, never as 0.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple, Union

import numpy as np

from frame.evaluate.datasets import MASK_RULE
from frame.evaluate.metrics import METRICS_VERSION
from frame.evaluate.shift import displace_pair
from frame.reliability import targets as T
from frame.reliability.alignment import apply_correction
from frame.reliability.association import correlation, partial_spearman
from frame.reliability.calibration import interval_coverage, scale_ratio
from frame.reliability.config import ReliabilityConfig
from frame.reliability.eligibility import check_prediction, exclusion_row
from frame.uncertainty.report import SCALAR_SUMMARY_DEFINITION

RELIABILITY_VERSION = "frame-reliability/1"
TARGET_DEFINITION = ("product = the TTA ensemble mean (what the pipeline returns); single_pass = the identity-orientation member alone. rmse/mae over bands and valid pixels, sam_degrees = mean per-pixel "
                     "spectral angle (undefined spectra excluded and counted), ergas = 100/scale * sqrt(mean_b (rmse_b / mean(HR_b))^2); all on the aligned overlap under the strict mask")
CELL_VARIABLES = ("stability", "abs_error", "sq_error", "sam_deg", "texture", "added_detail", "unsupported", "omission", "supported")


@dataclass
class EnsembleArrays:
    mean: np.ndarray                     # (C, H, W) TTA mean prediction
    std: np.ndarray                      # (C, H, W) per-band spread across the members
    member0: np.ndarray                  # (C, H, W) the identity-orientation member (a single pass)
    n_members: int
    transform_names: Tuple[str, ...]
    seed: int
    seconds_total: float
    seconds_per_member: Tuple[float, ...]


@dataclass
class TileInputs:
    dataset: str
    system: str
    sample_id: str
    scene_unit: str
    category: Optional[str]
    bands: Tuple[str, ...]
    scale: int
    bicubic: np.ndarray                  # (C, H, W) bicubic upsampling of the same LR
    hr: np.ndarray                       # (C, H, W) reference
    mask: np.ndarray                     # (H, W) strict evaluation mask
    gate: Dict[str, Any]                 # eligibility.evaluate_reference output for this tile (status eligible)
    ensemble: EnsembleArrays


def _rng(config: ReliabilityConfig, inp: TileInputs, tag: str) -> np.random.Generator:
    digest = hashlib.sha256(f"{config.bootstrap.seed}|{inp.dataset}|{inp.sample_id}|{tag}".encode("utf-8")).digest()
    return np.random.default_rng(int.from_bytes(digest[:8], "little"))


def _subsample(n: int, cap: int, rng: np.random.Generator) -> np.ndarray:
    return np.arange(n) if n <= cap else np.sort(rng.choice(n, size=cap, replace=False))


def _corr(x: np.ndarray, y: np.ndarray, kind: str = "spearman") -> Dict[str, Any]:
    return correlation(x, y, kind)


def _associations(stab: np.ndarray, err: np.ndarray, sam: np.ndarray, texture: np.ndarray, added: np.ndarray) -> Dict[str, Any]:
    return {
        "n": int(stab.size),
        "spearman_abs_error": _corr(stab, err), "pearson_abs_error": _corr(stab, err, "pearson"), "spearman_sam": _corr(stab, sam),
        "spearman_texture_abs_error": _corr(texture, err), "spearman_added_detail_abs_error": _corr(added, err),
        "partial_spearman_abs_error_given_baselines": partial_spearman(stab, err, [texture, added]),
    }


def compute_tile_evidence(inp: TileInputs, config: ReliabilityConfig) -> Union[Tuple[Dict[str, Any], Dict[str, np.ndarray]], Dict[str, Any]]:
    """Evidence of one admitted tile, or the machine-readable exclusion row if the PREDICTION cannot be analysed (see :func:`frame.reliability.eligibility.check_prediction`)."""
    ens, gate = inp.ensemble, inp.gate
    alignment = gate["alignment"]
    C = ens.mean.shape[0]
    stab_full = T.stability_map(ens.std)
    stack = np.concatenate([np.asarray(ens.mean, np.float64), np.asarray(ens.member0, np.float64), stab_full[None], np.asarray(inp.bicubic, np.float64), np.asarray(ens.std, np.float64)], axis=0)
    stack, hr_c, mask_c = apply_correction(stack, np.asarray(inp.hr, np.float64), np.asarray(inp.mask, bool), alignment["correction"])
    pred, single, stab, bic, std = stack[:C], stack[C:2 * C], stack[2 * C], stack[2 * C + 1:3 * C + 1], stack[3 * C + 1:]

    problem = check_prediction(pred, std, mask_c, n_members=ens.n_members, min_members=config.eligibility.min_members) or check_prediction(single, std, mask_c, n_members=ens.n_members,
                                                                                                                                              min_members=config.eligibility.min_members)
    if problem is not None:
        return exclusion_row(dataset=inp.dataset, system=inp.system, sample_id=inp.sample_id, scene_unit=inp.scene_unit, category=inp.category, reason=problem["reason"], detail=problem["detail"])

    maps = T.pixel_error_maps(pred, hr_c, mask_c)
    valid = maps["valid"] & np.isfinite(stab)
    tau = float(config.metric_config().hallucination_taus[0])
    texture, added = T.texture_baseline(bic), T.added_detail_baseline(pred, bic)
    labels = T.detail_label_maps(pred, bic, hr_c, tau)
    n_valid = int(valid.sum())

    # ---- tile-level summaries
    v_stab = stab[valid]
    stability = {"mean": float(v_stab.mean()), "median": float(np.median(v_stab)), "p90": float(np.percentile(v_stab, 90)), "definition": SCALAR_SUMMARY_DEFINITION,
                 "band_mean_std": {b: float(std[i][valid].mean()) for i, b in enumerate(inp.bands)}}
    targets = {"product": T.tile_targets(pred, hr_c, mask_c, inp.bands, scale=inp.scale), "single_pass": T.tile_targets(single, hr_c, mask_c, inp.bands, scale=inp.scale), "definition": TARGET_DEFINITION}
    detail = {k: float(labels[k][valid].mean()) for k in ("supported_synthesis", "unsupported_detail", "omission", "neutral")}
    detail["tau"] = tau

    # ---- pixel level (within the tile)
    err, sam = maps["abs_error"][valid], maps["sam_deg"][valid]
    pixel = _associations(v_stab, err, sam, texture[valid], added[valid])
    pixel["per_band"] = {b: _corr(std[i][valid], np.abs(pred[i] - hr_c[i])[valid]) for i, b in enumerate(inp.bands)}
    sweep: Dict[str, Any] = {}
    for s in config.analysis.displacement_sweep_hr_px:
        if s == 0:
            sweep["0"] = pixel["spearman_abs_error"]
            continue
        shifted, hr_s, mask_s = displace_pair(np.concatenate([pred, stab[None]], axis=0), hr_c, mask_c, 0, int(s))
        ms = T.pixel_error_maps(shifted[:C], hr_s, mask_s)
        vs = ms["valid"] & np.isfinite(shifted[C])
        sweep[str(int(s))] = _corr(shifted[C][vs], ms["abs_error"][vs])
    pixel["by_reference_displacement"] = sweep

    arrays: Dict[str, np.ndarray] = {}
    pix_idx = _subsample(n_valid, config.analysis.pooled_pixels_per_tile, _rng(config, inp, "pixels"))
    for name, values in (("stability", v_stab), ("abs_error", err), ("sam_deg", sam), ("texture", texture[valid]), ("added_detail", added[valid])):
        arrays[f"p_{name}"] = values[pix_idx].astype(np.float32)

    # ---- cells (10 m = 4 HR px, 40 m = 16 HR px, ...): within-tile associations on all cells, pooled arrays on a seeded subsample
    cell_maps = {"stability": stab, "abs_error": maps["abs_error"], "sq_error": maps["sq_error"], "sam_deg": maps["sam_deg"], "texture": texture, "added_detail": added,
                 "unsupported": labels["unsupported_detail"], "omission": labels["omission"], "supported": labels["supported_synthesis"]}
    cells_out: Dict[str, Any] = {}
    for size in config.analysis.cell_sizes_hr_px:
        cells = T.cell_arrays(cell_maps, valid, size=int(size), min_valid_fraction=config.analysis.min_cell_valid_fraction)
        n_cells = int(len(cells["stability"]))
        record = {"n_cells": n_cells, "size_hr_px": int(size)}
        if n_cells >= 3:
            record.update(_associations(cells["stability"], cells["abs_error"], cells["sam_deg"], cells["texture"], cells["added_detail"]))
            record["spearman_unsupported"] = _corr(cells["stability"], cells["unsupported"])
            record["spearman_omission"] = _corr(cells["stability"], cells["omission"])
            record["spearman_supported"] = _corr(cells["stability"], cells["supported"])
        cells_out[str(int(size))] = record
        idx = _subsample(n_cells, config.analysis.pooled_cells_per_tile, _rng(config, inp, f"cells{size}"))
        for name in CELL_VARIABLES:
            arrays[f"c{int(size)}_{name}"] = cells[name][idx].astype(np.float32)

    row: Dict[str, Any] = {
        "type": "tile_result", "status": "eligible", "dataset": inp.dataset, "system": inp.system, "sample_id": inp.sample_id, "scene_unit": inp.scene_unit, "category": inp.category,
        "evidence_level": gate["evidence_level"], "alignment": alignment, "valid_fraction": gate["valid_fraction"], "valid_fraction_after_alignment": gate["valid_fraction_after_alignment"],
        "valid_pixel_rule": MASK_RULE + "; then restricted to the aligned overlap and to pixels where the prediction and the spread are finite", "metric_version": METRICS_VERSION,
        "reliability_version": RELIABILITY_VERSION,
        "tta": {"n_members": int(ens.n_members), "transforms": list(ens.transform_names), "seed": int(ens.seed), "total_seconds": float(ens.seconds_total),
                "seconds_per_member": [float(s) for s in ens.seconds_per_member], "single_pass_seconds": float(np.median(ens.seconds_per_member)) if ens.seconds_per_member else None},
        "stability": stability, "targets": targets, "detail": detail,
        "baselines": {"texture_mean": float(texture[valid].mean()), "added_detail_mean": float(added[valid].mean()),
                      "definition": "texture = Sobel gradient of the band-mean bicubic input; added_detail = band-mean |prediction - bicubic|"},
        "pixel": pixel, "cells": cells_out, "calibration_raw": _raw_calibration(pred, std, hr_c, valid),
    }
    return row, arrays


def _raw_calibration(pred: np.ndarray, std: np.ndarray, hr: np.ndarray, valid: np.ndarray) -> Dict[str, Any]:
    """Is the raw ensemble spread a calibrated interval? Coverage of pred +/- k*std over the valid (band, pixel) elements against the Gaussian nominal level, and the scale ratio."""
    p, s, r = pred[:, valid].ravel(), std[:, valid].ravel(), hr[:, valid].ravel()
    return {"n_elements": int(p.size), "coverage": {f"k{k}": interval_coverage(p, s, r, k=float(k)) for k in (1, 2)}, "scale": scale_ratio(s, np.abs(p - r))}


# ---------------------------------------------------------------------------------------------------------------
# the cache (outside the repository): the per-tile row and its pooled arrays, so the analysis can be repeated without repeating the ensemble
# ---------------------------------------------------------------------------------------------------------------

def save_evidence(stem: Union[str, Path], row: Dict[str, Any], arrays: Dict[str, np.ndarray]) -> None:
    stem = str(stem)
    Path(stem).parent.mkdir(parents=True, exist_ok=True)
    Path(stem + ".json").write_text(json.dumps(row, sort_keys=True, allow_nan=False), encoding="utf-8")
    np.savez_compressed(stem + ".npz", **arrays)


def load_evidence(stem: Union[str, Path]) -> Tuple[Dict[str, Any], Dict[str, np.ndarray]]:
    stem = str(stem)
    jp, np_ = Path(stem + ".json"), Path(stem + ".npz")
    if not jp.is_file() or not np_.is_file():
        raise FileNotFoundError(f"no cached evidence at {stem}(.json/.npz)")
    with np.load(np_) as data:
        arrays = {k: data[k] for k in data.files}
    return json.loads(jp.read_text(encoding="utf-8")), arrays
