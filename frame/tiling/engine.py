"""The FRAME tile engine (Phase 2): any-size scene -> 4x SR raster, through any
tile model.

    scene (C, H, W)
      -> plan_tiles                    frame.tiling.plan      (explicit tile list)
      -> for each tile, row-major:
           extract_tile (+ reflect pad) frame.tiling.padding
           model(tile[None])            <- the caller's callable: Lite, Mamba, or a test fake
           crop to the valid SR extent
           weighted accumulate          frame.tiling.blend
      -> normalise                      (raises if any pixel was never covered)
      -> SR raster (C_out, scale*H, scale*W)

The engine is model-agnostic by construction: it only ever calls
``model(x) -> y`` with ``x`` of shape (1, C, tile_size, tile_size) on the
scene's device and expects ``y`` of shape (1, C_out, scale*tile_size,
scale*tile_size). Choosing Lite or Mamba is the caller's job
(frame.api.services.model); nothing here knows which one it got. For Mamba the
callable is the long-lived `MambaWorkerClient`, so one worker serves every tile
of the scene.

Tiles are processed one at a time, in a fixed order. There is deliberately no
batching: peak GPU memory stays at the single-tile figure.

A failure on any tile aborts the whole scene: the model's own typed error
(`ModelWorkerError`, `ModelInferenceError`, ...) propagates unchanged, and no
partially reconstructed raster is ever returned.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from statistics import fmean
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch

from frame.models.errors import ModelInferenceError
from frame.tiling.blend import WeightedCanvas, tile_window
from frame.tiling.errors import InvalidSceneError, ReconstructionError
from frame.tiling.padding import extract_tile
from frame.tiling.plan import TilePlan, TileSpec, TilingConfig, plan_tiles
from frame.tiling.seams import SEAM_DIAGNOSTIC_DEFINITION, SeamAccumulator, SeamDiagnostic, pool_diagnostics

TileModel = Callable[[torch.Tensor], torch.Tensor]
TileCallback = Callable[[int, int, TileSpec], None]


@dataclass(frozen=True)
class TiledResult:
    sr: torch.Tensor                    # (C_out, scale*H, scale*W), CPU float32
    plan: TilePlan
    tile_seconds: Tuple[float, ...]     # per tile, in processing order
    total_seconds: float
    seam_diagnostic: Optional[SeamDiagnostic]


def _validate_scene(scene: object) -> None:
    if not isinstance(scene, torch.Tensor):
        raise InvalidSceneError(f"Scene must be a torch.Tensor, got {type(scene).__name__}.")
    if scene.ndim != 3:
        raise InvalidSceneError(f"Scene must have shape (channels, height, width), got {scene.ndim}-D shape {tuple(scene.shape)}.")
    channels, height, width = scene.shape
    if channels < 1:
        raise InvalidSceneError(f"Scene has no channels (shape {tuple(scene.shape)}).")
    if height < 1 or width < 1:
        raise InvalidSceneError(f"Scene is empty (shape {tuple(scene.shape)}).")
    if not scene.is_floating_point():
        raise InvalidSceneError(f"Scene must be a floating-point tensor, got {scene.dtype}.")
    # One cheap pass now beats discovering it tile by tile: the models' Fourier hard constraint turns
    # a single NaN into a NaN band over the whole tile, and the Mamba path would only reject it after
    # minutes of tiles already spent.
    non_finite = int((~torch.isfinite(scene)).sum())
    if non_finite:
        raise InvalidSceneError(
            f"Scene contains {non_finite} non-finite (NaN/Inf) values; fill or mask nodata before super-resolution."
        )


def _tile_output(y: object, tile: TileSpec, config: TilingConfig, channels: Optional[int]) -> torch.Tensor:
    """Validate one tile's model output; return it as (C_out, sr_padded_h, sr_padded_w)."""
    where = f"tile (row {tile.row_index}, col {tile.col_index})"
    expected_size = config.tile_size * config.scale
    if not isinstance(y, torch.Tensor):
        raise ModelInferenceError(f"The model returned {type(y).__name__} instead of a tensor for {where}.")
    if y.ndim != 4 or y.shape[0] != 1 or tuple(y.shape[2:]) != (expected_size, expected_size):
        raise ModelInferenceError(
            f"The model returned shape {tuple(y.shape)} for {where}; expected (1, C, {expected_size}, {expected_size})."
        )
    if channels is not None and y.shape[1] != channels:
        raise ModelInferenceError(f"The model returned {y.shape[1]} channels for {where}; earlier tiles had {channels}.")
    if not bool(torch.isfinite(y).all()):
        raise ModelInferenceError(f"The model returned NaN or Inf for {where}.")
    return y[0]


def run_tiled(
    model: TileModel,
    scene: torch.Tensor,
    config: TilingConfig = TilingConfig(),
    *,
    collect_seam_diagnostic: bool = True,
    on_tile: Optional[TileCallback] = None,
) -> TiledResult:
    """Super-resolve ``scene`` (C, H, W) tile by tile; see the module docstring.

    Args:
        model: ``model(x) -> y``; x (1, C, tile, tile) on the scene's device.
        scene: floating-point (C, H, W) tensor, any H and W >= 1.
        config: tiling configuration (default: 128 px tiles, 32 px overlap, x4).
        collect_seam_diagnostic: also compute the tile-seam diagnostic.
        on_tile: optional ``callback(done, total, tile)`` after each tile
            (progress reporting hook; not used for control flow).
    """
    _validate_scene(scene)
    _, height, width = scene.shape
    plan = plan_tiles(int(height), int(width), config)

    seams = SeamAccumulator(plan) if collect_seam_diagnostic else None
    canvas: Optional[WeightedCanvas] = None
    channels_out: Optional[int] = None
    tile_seconds: List[float] = []
    started = time.perf_counter()

    with torch.no_grad():
        for done, tile in enumerate(plan.tiles, start=1):
            tile_started = time.perf_counter()

            x = extract_tile(scene, tile, config.padding_mode)
            y = _tile_output(model(x[None]), tile, config, channels_out)
            if canvas is None:
                channels_out = int(y.shape[0])
                canvas = WeightedCanvas(channels_out, *plan.sr_shape)

            # crop to the part of the output that lies inside the scene, then to CPU
            valid = y[:, : tile.sr_valid_height, : tile.sr_valid_width].detach().to("cpu", torch.float32)
            canvas.add(valid, tile_window(tile, config), tile.sr_row_start, tile.sr_col_start)
            if seams is not None:
                seams.observe(tile, valid)

            tile_seconds.append(time.perf_counter() - tile_started)
            if on_tile is not None:
                on_tile(done, plan.tile_count, tile)

    sr = canvas.finalize()  # raises ReconstructionError on any uncovered pixel
    expected = (channels_out, *plan.sr_shape)
    if tuple(sr.shape) != expected:
        raise ReconstructionError(f"Reconstructed raster has shape {tuple(sr.shape)}, expected {expected}.")

    return TiledResult(
        sr=sr,
        plan=plan,
        tile_seconds=tuple(tile_seconds),
        total_seconds=time.perf_counter() - started,
        seam_diagnostic=seams.result() if seams is not None else None,
    )


@dataclass(frozen=True)
class _RunRecord:
    """What `TiledModel` remembers about one pass over a scene (never the raster itself)."""

    scene_shape: Tuple[int, int]
    sr_shape: Tuple[int, int]
    grid: Tuple[int, int]
    tile_count: int
    padded_tiles: int
    tile_seconds: Tuple[float, ...]
    seam: Optional[SeamDiagnostic]


class TiledModel:
    """Adapter: a tile model that accepts a whole scene.

    ``TiledModel(model)(x)`` maps (1, C, H, W) -> (1, C_out, scale*H, scale*W)
    (CPU), so anything written against the single-tile ``model(x[None]) -> y``
    convention -- notably frame.uncertainty's test-time-augmentation ensemble --
    runs on scenes of any size unchanged. Every call is one full tiled pass; the
    ensemble makes several, and `summary()` describes all of them.
    """

    def __init__(self, model: TileModel, config: TilingConfig = TilingConfig(), *, collect_seam_diagnostic: bool = True):
        self._model = model
        self.config = config
        self._collect_seam_diagnostic = collect_seam_diagnostic
        self._runs: List[_RunRecord] = []

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if not isinstance(x, torch.Tensor) or x.ndim != 4 or x.shape[0] != 1:
            shape = tuple(x.shape) if isinstance(x, torch.Tensor) else type(x).__name__
            raise InvalidSceneError(f"TiledModel expects one scene of shape (1, C, H, W), got {shape}.")
        result = run_tiled(self._model, x[0], self.config, collect_seam_diagnostic=self._collect_seam_diagnostic)
        plan = result.plan
        self._runs.append(
            _RunRecord(
                scene_shape=(plan.scene_height, plan.scene_width),
                sr_shape=plan.sr_shape,
                grid=(plan.n_rows, plan.n_cols),
                tile_count=plan.tile_count,
                padded_tiles=sum(t.is_padded for t in plan.tiles),
                tile_seconds=result.tile_seconds,
                seam=result.seam_diagnostic,
            )
        )
        return result.sr[None]

    @property
    def runs(self) -> Tuple[_RunRecord, ...]:
        return tuple(self._runs)

    def summary(self) -> Dict[str, Any]:
        """Human-readable record of the reconstruction, for result provenance."""
        summary: Dict[str, Any] = dict(self.config.describe())
        if not self._runs:
            return summary
        first = self._runs[0]
        all_tile_seconds = [s for run in self._runs for s in run.tile_seconds]
        summary.update(
            {
                "scene_shape": list(first.scene_shape),
                "sr_shape": list(first.sr_shape),
                "tile_grid": list(first.grid),
                "tile_count": first.tile_count,
                "padded_tiles": first.padded_tiles,
                "scene_passes": len(self._runs),
                "tile_inferences": len(all_tile_seconds),
                "mean_tile_seconds": round(fmean(all_tile_seconds), 4),
                "tile_seconds_total": round(sum(all_tile_seconds), 3),
                "seam_diagnostic": (
                    pool_diagnostics(run.seam for run in self._runs).as_dict() if self._collect_seam_diagnostic else None
                ),
                "seam_diagnostic_definition": SEAM_DIAGNOSTIC_DEFINITION,
            }
        )
        return summary
