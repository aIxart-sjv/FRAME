"""FRAME tile engine (Phase 2): arbitrary-size scenes through the 128x128 SR models.

    plan_tiles    scene size + config -> explicit `TileSpec` list        (plan.py)
    extract_tile  edge tiles reflect-padded to full size                 (padding.py)
    blend         separable linear weights + weighted accumulator        (blend.py)
    seams         adjacent-tile disagreement diagnostic                  (seams.py)
    run_tiled     the engine: any model callable, any scene size         (engine.py)
    TiledModel    makes the engine look like a single-tile model         (engine.py)

Model-agnostic: it takes a callable and never inspects it. See docs/TILING.md.
"""

from frame.tiling.engine import TiledModel, TiledResult, run_tiled
from frame.tiling.errors import InvalidSceneError, InvalidTilingConfigError, ReconstructionError, TilingError
from frame.tiling.plan import (
    BLEND_LINEAR,
    DEFAULT_OVERLAP,
    DEFAULT_SCALE,
    DEFAULT_TILE_SIZE,
    PADDING_REFLECT,
    TilePlan,
    TileSpec,
    TilingConfig,
    plan_tiles,
)
from frame.tiling.seams import SeamDiagnostic

__all__ = [
    "BLEND_LINEAR",
    "DEFAULT_OVERLAP",
    "DEFAULT_SCALE",
    "DEFAULT_TILE_SIZE",
    "InvalidSceneError",
    "InvalidTilingConfigError",
    "PADDING_REFLECT",
    "ReconstructionError",
    "SeamDiagnostic",
    "TilePlan",
    "TileSpec",
    "TiledModel",
    "TiledResult",
    "TilingConfig",
    "TilingError",
    "plan_tiles",
    "run_tiled",
]
