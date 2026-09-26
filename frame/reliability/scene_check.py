"""The multi-tile / rectangular-scene check of the stability map (Phase 6): does the deployed TTA-over-tiling pipeline give a complete, correctly oriented, seam-free, correctly
georeferenced stability map on a scene that is neither square nor a multiple of the tile size?

Checks, each with its numbers (a boolean is only ever a summary of numbers that are recorded next to it):

* **coverage**      the spread and the mean cover the full SR output, and are finite everywhere;
* **orientation**   every ensemble member, after de-transformation, is much closer to the identity view than the same member mis-oriented would be (a wrong inverse puts a member on the
                    wrong side of the scene and is caught); a member view of a rectangular scene has swapped sides (rot90 / rot270) and its own tile grid;
* **seams**         the mean spread within a few pixels of the tile seams divided by the mean spread elsewhere, for the canonical grid and for the grid of the 90-degree-rotated view
                    (mapped back to canonical coordinates): the members disagree about where the seams are, so a seam artefact would show as a ridge in the spread;
* **georeferencing** the spread GeoTIFF is written with the SR output's grid (frame.geospatial.derive_output_metadata) and read back: same CRS, same origin and bounds, a quarter of the pixel
                    size, identical values.
"""

from __future__ import annotations

import dataclasses
import statistics
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import numpy as np
import torch

from frame.evaluate.systems import seam_lines
from frame.geospatial import RGBN_SCALE_FACTOR, derive_output_metadata, read_geotiff, write_geotiff
from frame.preprocessing.metadata import RasterMetadata
from frame.tiling import TilingConfig, TiledModel, plan_tiles
from frame.uncertainty.ensemble import run_tta_ensemble
from frame.uncertainty.transforms import DEFAULT_TRANSFORMS, Transform

SEAM_TOLERANCE = 1.10          # a spread up to 10% higher within the seam neighbourhood than elsewhere is accepted as noise; declared before any result
ORIENTATION_RATIO = 0.5        # a member is aligned if its distance to the identity view is at most half the distance a mis-oriented copy would have


def seam_mask(plan: Any, tiling: TilingConfig, *, half_width: int) -> np.ndarray:
    """Boolean SR-grid mask of the neighbourhood (+/- ``half_width`` px) of every seam line, i.e. the middle of each blend overlap between consecutive tiles."""
    height, width = plan.sr_shape
    rows, cols = seam_lines(plan, tiling)
    mask = np.zeros((height, width), dtype=bool)
    for r in rows:
        mask[max(0, r - half_width): r + half_width + 1, :] = True
    for c in cols:
        mask[:, max(0, c - half_width): c + half_width + 1] = True
    return mask


def seam_statistics(std_map: np.ndarray, mask: np.ndarray, *, tolerance: float) -> Dict[str, Any]:
    """Mean spread on the seam neighbourhood over the mean spread elsewhere. No seams (a single tile) is 'nothing to compare', not a pass."""
    seam, rest = mask, ~mask
    if not seam.any() or not rest.any():
        return {"seam_over_interior": None, "within_tolerance": None, "n_seam_pixels": int(seam.sum()), "n_interior_pixels": int(rest.sum()), "note": "no seam neighbourhood to compare"}
    interior = float(std_map[rest].mean())
    ratio = float(std_map[seam].mean()) / interior if interior > 0 else 1.0
    return {"seam_over_interior": ratio, "within_tolerance": bool(ratio <= tolerance), "n_seam_pixels": int(seam.sum()), "n_interior_pixels": int(rest.sum()), "tolerance": tolerance}


def check_scene(
    model: Any,
    lr: torch.Tensor,
    input_metadata: RasterMetadata,
    tiling: TilingConfig,
    *,
    work_dir: Optional[Path],
    transforms: Sequence[Transform] = DEFAULT_TRANSFORMS,
    seed: int = 42,
    seam_half_width: int = 8,
    seam_tolerance: float = SEAM_TOLERANCE,
) -> Dict[str, Any]:
    """Run the deployed pipeline (single-tile ``model`` -> tile engine -> TTA ensemble) on one scene and check the resulting stability map."""
    if work_dir is None:
        raise ValueError("work_dir is required: the georeferenced spread is written and read back")
    height, width = int(lr.shape[-2]), int(lr.shape[-1])
    tiled = TiledModel(model, tiling, collect_seam_diagnostic=True)
    result = run_tta_ensemble(tiled, lr, list(transforms), seed=seed, keep_per_member_predictions=True)
    mean, std = result.mean_prediction, result.std_prediction
    sr_shape = [int(mean.shape[0]), height * RGBN_SCALE_FACTOR, width * RGBN_SCALE_FACTOR]
    plan = plan_tiles(height, width, tiling)

    # ---- scene and coverage
    view_grids = {}
    for t in transforms:
        vh, vw = (int(v) for v in t.forward(lr).shape[-2:])
        p = plan_tiles(vh, vw, tiling)
        view_grids[t.name] = {"view_shape": [vh, vw], "grid": [p.n_rows, p.n_cols], "tile_count": p.tile_count}
    scene = {"lr_shape": [int(v) for v in lr.shape], "sr_shape": sr_shape, "grid": [plan.n_rows, plan.n_cols], "tile_count": plan.tile_count, "view_grids": {k: v["grid"] for k, v in view_grids.items()},
             "view_shapes": {k: v["view_shape"] for k, v in view_grids.items()}}
    coverage = {"std_shape_matches_output": list(std.shape) == sr_shape and list(mean.shape) == sr_shape, "non_finite_std_pixels": int((~torch.isfinite(std)).sum()),
                "non_finite_mean_pixels": int((~torch.isfinite(mean)).sum())}
    stab = std.mean(dim=0).numpy().astype(np.float64)
    stability = {"mean_std": float(stab.mean()), "median_std": float(np.median(stab)), "max_std": float(stab.max()), "min_std": float(stab.min())}

    # ---- orientation: each de-transformed member against the identity view
    members = result.per_member_predictions
    names = list(result.transform_names)
    identity = members[names.index("identity")] if "identity" in names else members[0]
    per_member: Dict[str, Any] = {}
    for name, member in zip(names, members):
        mae = float((member - identity).abs().mean())
        misoriented = float((torch.rot90(member, k=2, dims=(-2, -1)) - identity).abs().mean())
        per_member[name] = {"mae_vs_identity": mae, "mae_if_misoriented": misoriented, "aligned": bool(mae <= ORIENTATION_RATIO * misoriented) if misoriented > 0 else True}
    orientation = {"members": per_member, "all_members_aligned": all(m["aligned"] for m in per_member.values()),
                   "criterion": f"aligned if MAE to the identity view <= {ORIENTATION_RATIO} x the MAE the same member rotated by 180 degrees would have"}

    # ---- seams, on the canonical grid and on the grid of the 90-degree-rotated view mapped back to canonical coordinates
    seams: Dict[str, Any] = {"half_width_px": seam_half_width, "tolerance": seam_tolerance}
    seams["canonical_grid"] = seam_statistics(stab, seam_mask(plan, tiling, half_width=seam_half_width), tolerance=seam_tolerance)
    rot = next((t for t in transforms if t.name == "rot90"), None)
    if rot is not None:
        vh, vw = (int(v) for v in rot.forward(lr).shape[-2:])
        rot_mask = torch.from_numpy(seam_mask(plan_tiles(vh, vw, tiling), tiling, half_width=seam_half_width)[None].astype(np.float32))
        canonical_mask = rot.inverse(rot_mask)[0].numpy() > 0.5
        seams["rot90_grid"] = seam_statistics(stab, canonical_mask, tolerance=seam_tolerance)
    checked = [v for v in (seams["canonical_grid"], seams.get("rot90_grid")) if v is not None and v["within_tolerance"] is not None]
    seams["no_seam_introduced"] = bool(all(v["within_tolerance"] for v in checked)) if checked else None
    seams["definition"] = "mean spread within +/- half_width px of the seam lines (the middle of each blend overlap) over the mean spread elsewhere"

    # ---- georeferencing: write the spread with the SR grid and read it back
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    output_metadata = derive_output_metadata(input_metadata, scale_factor=RGBN_SCALE_FACTOR, output_band_names=input_metadata.band_names)
    band_names = tuple(f"{b}_std" for b in input_metadata.band_names) + ("overall_std",)
    array = np.concatenate([std.numpy(), stab[None].astype(np.float32)], axis=0).astype(np.float32)
    path = work_dir / "stability_check.tif"
    write_geotiff(path, array, dataclasses.replace(output_metadata, band_names=band_names, nodata_value=None))
    back, meta = read_geotiff(path)
    georef = {"crs_equal": meta.crs == input_metadata.crs, "origin_equal": bool(np.isclose(meta.transform[2], input_metadata.transform[2]) and np.isclose(meta.transform[5], input_metadata.transform[5])),
              "pixel_size_m": float(abs(meta.transform[0])), "pixel_size_expected_m": float(abs(input_metadata.transform[0]) / RGBN_SCALE_FACTOR),
              "bounds_equal": bool(np.allclose(meta.bounds, input_metadata.bounds)), "shape": [int(meta.height), int(meta.width)], "array_round_trips_exactly": bool(np.array_equal(back, array)),
              "band_names": list(meta.band_names), "path": str(path.name)}

    seconds = list(result.per_member_inference_seconds)
    single = statistics.median(seconds)
    timing = {"n_members": int(result.n), "tta_seconds": float(result.total_seconds), "single_pass_seconds_median": float(single), "tta_over_single_pass": float(result.total_seconds / single) if single > 0 else None,
              "seconds_per_member": {n: float(s) for n, s in zip(names, seconds)}}
    return {"scene": scene, "coverage": coverage, "stability": stability, "orientation": orientation, "seams": seams, "georeferencing": georef, "timing": timing}
