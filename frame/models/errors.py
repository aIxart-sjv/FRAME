"""Exceptions for FRAME's model layer (Phase 1: SEN2SR Mamba integration).

Kept dependency-free (stdlib only) because this module is imported on BOTH
sides of the process boundary: by the main FRAME process and by the
isolated Mamba worker.

Two kinds of message live on these exceptions and are deliberately kept
apart:

* ``str(exc)`` -- always safe to show a user. It never names the low-level
  runtime libraries the model happens to be built on.
* ``technical_detail`` -- the underlying cause (an ImportError text, a
  worker stderr tail, ...). For server-side logs only; the API layer never
  puts it in a response body.
"""

from __future__ import annotations


class ModelError(Exception):
    """Base class for every error raised by frame.models."""

    def __init__(self, message: str, *, technical_detail: str = ""):
        super().__init__(message)
        self.technical_detail = technical_detail


class UnknownModelError(ModelError, ValueError):
    """A model id other than one of the supported ids was requested."""


class ModelContractError(ModelError, ValueError):
    """A tensor violates the model's input/output contract (channels, size,
    dtype, value range, NaN/Inf)."""


class ModelUnavailableError(ModelError):
    """The model cannot be used on this machine as configured (no CUDA GPU,
    weights or isolated runtime not found, ...). Not a bug: an environment
    condition the caller can report clearly."""


class ModelLoadError(ModelError):
    """The weights were found but do not match the expected architecture."""


class ModelWorkerError(ModelError):
    """The isolated worker process died, timed out, or spoke a malformed
    protocol."""


class ModelInferenceError(ModelError):
    """The worker was healthy but a forward pass failed (CUDA OOM,
    non-finite output, ...)."""
