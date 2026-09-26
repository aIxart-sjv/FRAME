"""Exceptions for FRAME's tile engine (Phase 2).

Model failures are NOT wrapped here: a worker crash, timeout, CUDA OOM or an
invalid tile output raises the `frame.models.errors` type that describes it
(`ModelWorkerError`, `ModelInferenceError`, ...), unchanged, so callers and
the API handle one taxonomy. These errors cover only what the tiler itself can
get wrong: a bad configuration, a bad scene, or a reconstruction that did not
come out complete.
"""

from __future__ import annotations


class TilingError(Exception):
    """Base class for every error raised by frame.tiling itself."""


class InvalidTilingConfigError(TilingError, ValueError):
    """tile_size / overlap / scale / padding_mode / blend_mode is not valid
    (for example an overlap that would produce a zero or negative stride)."""


class InvalidSceneError(TilingError, ValueError):
    """The scene is not a non-empty (channels, height, width) float tensor."""


class ReconstructionError(TilingError):
    """The reconstructed raster is not complete or not the expected shape.
    Raised instead of returning a partially reconstructed result."""
