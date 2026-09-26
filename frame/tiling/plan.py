"""Tile planning: scene size + configuration -> an explicit list of tiles.

All coordinate arithmetic for tiling lives in this module. Everything else
(padding, blending, the engine, the seam diagnostic) reads `TileSpec` fields
rather than recomputing positions.

Geometry
--------
Tiles are ``tile_size`` x ``tile_size`` input pixels. Consecutive tiles along
an axis start ``stride = tile_size - overlap`` pixels apart, beginning at 0,
and planning stops at the first tile that reaches the scene edge:

    starts = 0, stride, 2*stride, ...   until  start + tile_size >= size

so every scene pixel is inside at least one tile and no tile starts beyond the
scene. The last tile along an axis may extend past the scene edge; the part
inside the scene is its *valid* extent (``valid_height`` / ``valid_width``),
the rest is padding added before inference and cropped away afterwards (see
frame.tiling.padding). Two properties follow directly from the rule above and
are relied on by the blender (and asserted in tests):

* every tile that has a successor is fully valid (``valid == tile_size``);
* the overlap between two consecutive tiles is exactly ``overlap`` pixels, and
  the last tile's valid extent is always greater than ``overlap``.

Coordinates are in INPUT pixels; multiply by ``scale`` for the SR grid
(``TileSpec.sr_*``). ``*_end`` are exclusive.

Why not upstream's plan (sen2sr/utils.py `define_iteration` +
`fix_lastchunk`): that code clamps edge tiles inward and then crops with
input/output units mixed and rows/columns swapped; on a perfect model it
leaves 23% of a single 128x128 tile unfilled at overlap 32 and crashes on
every non-square input (measured; see docs/TILING.md). This planner keeps
positions on a regular grid and leaves border handling to explicit padding.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

from frame.tiling.errors import InvalidSceneError, InvalidTilingConfigError

DEFAULT_TILE_SIZE = 128     # the models' native input size (frame.models.config.INPUT_SIZE)
DEFAULT_OVERLAP = 32        # upstream's predict_large default; see docs/TILING.md for the check
DEFAULT_SCALE = 4           # 10 m -> 2.5 m
PADDING_REFLECT = "reflect"
BLEND_LINEAR = "linear"
SUPPORTED_PADDING_MODES = (PADDING_REFLECT,)
SUPPORTED_BLEND_MODES = (BLEND_LINEAR,)


def _require_int(name: str, value: object, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise InvalidTilingConfigError(f"{name} must be an integer, got {value!r}.")
    if value < minimum:
        raise InvalidTilingConfigError(f"{name} must be >= {minimum}, got {value}.")
    return value


@dataclass(frozen=True)
class TilingConfig:
    """How a scene is tiled and reconstructed. Validated on construction."""

    tile_size: int = DEFAULT_TILE_SIZE
    overlap: int = DEFAULT_OVERLAP
    scale: int = DEFAULT_SCALE
    padding_mode: str = PADDING_REFLECT
    blend_mode: str = BLEND_LINEAR

    def __post_init__(self) -> None:
        _require_int("tile_size", self.tile_size, 1)
        _require_int("overlap", self.overlap, 0)
        _require_int("scale", self.scale, 1)
        if self.overlap >= self.tile_size:
            raise InvalidTilingConfigError(
                f"overlap ({self.overlap}) must be smaller than tile_size ({self.tile_size}); "
                f"otherwise the stride tile_size - overlap = {self.tile_size - self.overlap} is not positive."
            )
        if self.overlap > self.tile_size // 2:
            raise InvalidTilingConfigError(
                f"overlap ({self.overlap}) must be at most half the tile_size ({self.tile_size // 2}): beyond that the "
                "blend ramps of the two neighbouring tiles meet inside a single tile and the tile count grows "
                "quadratically (stride 1 on a 300x500 scene would be ~64,000 tiles)."
            )
        if self.padding_mode not in SUPPORTED_PADDING_MODES:
            raise InvalidTilingConfigError(
                f"Unsupported padding_mode {self.padding_mode!r}; supported: {', '.join(SUPPORTED_PADDING_MODES)}."
            )
        if self.blend_mode not in SUPPORTED_BLEND_MODES:
            raise InvalidTilingConfigError(
                f"Unsupported blend_mode {self.blend_mode!r}; supported: {', '.join(SUPPORTED_BLEND_MODES)}."
            )

    @property
    def stride(self) -> int:
        return self.tile_size - self.overlap

    @property
    def sr_tile_size(self) -> int:
        return self.tile_size * self.scale

    def describe(self) -> dict:
        return {
            "tile_size": self.tile_size,
            "overlap": self.overlap,
            "stride": self.stride,
            "scale": self.scale,
            "padding_mode": self.padding_mode,
            "blend_mode": self.blend_mode,
        }


@dataclass(frozen=True)
class TileSpec:
    """One tile: where it sits in the scene, how much of it is real, and
    which of its sides border a neighbouring tile."""

    row_index: int
    col_index: int
    row_start: int
    row_end: int            # exclusive; row_start + valid_height
    col_start: int
    col_end: int            # exclusive; col_start + valid_width
    valid_height: int
    valid_width: int
    padded_height: int      # always the configured tile_size
    padded_width: int
    overlaps_top: bool      # a previous tile row overlaps this tile's top edge
    overlaps_bottom: bool
    overlaps_left: bool
    overlaps_right: bool
    scale: int

    @property
    def is_padded(self) -> bool:
        return self.valid_height < self.padded_height or self.valid_width < self.padded_width

    # ---- the same tile on the SR grid --------------------------------------------------
    @property
    def sr_row_start(self) -> int:
        return self.row_start * self.scale

    @property
    def sr_col_start(self) -> int:
        return self.col_start * self.scale

    @property
    def sr_valid_height(self) -> int:
        return self.valid_height * self.scale

    @property
    def sr_valid_width(self) -> int:
        return self.valid_width * self.scale

    @property
    def sr_row_end(self) -> int:
        return self.sr_row_start + self.sr_valid_height

    @property
    def sr_col_end(self) -> int:
        return self.sr_col_start + self.sr_valid_width


@dataclass(frozen=True)
class TilePlan:
    scene_height: int
    scene_width: int
    config: TilingConfig
    row_starts: Tuple[int, ...]
    col_starts: Tuple[int, ...]
    tiles: Tuple[TileSpec, ...]     # row-major: the deterministic processing order

    @property
    def n_rows(self) -> int:
        return len(self.row_starts)

    @property
    def n_cols(self) -> int:
        return len(self.col_starts)

    @property
    def tile_count(self) -> int:
        return len(self.tiles)

    @property
    def sr_shape(self) -> Tuple[int, int]:
        return (self.scene_height * self.config.scale, self.scene_width * self.config.scale)

    def tile_at(self, row_index: int, col_index: int) -> TileSpec:
        return self.tiles[row_index * self.n_cols + col_index]


def axis_starts(size: int, tile_size: int, stride: int) -> Tuple[int, ...]:
    """Tile start positions along one axis (see the module docstring)."""
    starts = [0]
    while starts[-1] + tile_size < size:
        starts.append(starts[-1] + stride)
    return tuple(starts)


def plan_tiles(height: int, width: int, config: TilingConfig = TilingConfig()) -> TilePlan:
    """Plan the tiles for a ``height`` x ``width`` scene.

    Raises `InvalidSceneError` for a non-positive or non-integer size.
    """
    for name, value in (("height", height), ("width", width)):
        if isinstance(value, bool) or not isinstance(value, int):
            raise InvalidSceneError(f"Scene {name} must be an integer, got {value!r}.")
        if value < 1:
            raise InvalidSceneError(f"Scene {name} must be >= 1, got {value} (empty raster).")

    row_starts = axis_starts(height, config.tile_size, config.stride)
    col_starts = axis_starts(width, config.tile_size, config.stride)

    tiles = []
    for row_index, row_start in enumerate(row_starts):
        valid_height = min(config.tile_size, height - row_start)
        for col_index, col_start in enumerate(col_starts):
            valid_width = min(config.tile_size, width - col_start)
            tiles.append(
                TileSpec(
                    row_index=row_index,
                    col_index=col_index,
                    row_start=row_start,
                    row_end=row_start + valid_height,
                    col_start=col_start,
                    col_end=col_start + valid_width,
                    valid_height=valid_height,
                    valid_width=valid_width,
                    padded_height=config.tile_size,
                    padded_width=config.tile_size,
                    overlaps_top=row_index > 0,
                    overlaps_bottom=row_index < len(row_starts) - 1,
                    overlaps_left=col_index > 0,
                    overlaps_right=col_index < len(col_starts) - 1,
                    scale=config.scale,
                )
            )

    return TilePlan(
        scene_height=height,
        scene_width=width,
        config=config,
        row_starts=row_starts,
        col_starts=col_starts,
        tiles=tuple(tiles),
    )
