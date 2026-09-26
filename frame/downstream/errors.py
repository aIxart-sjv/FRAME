"""Exceptions of the downstream layer (Phase 7); they extend the Phase 5 evaluation errors so the CLI exit codes and messages behave the same way."""

from __future__ import annotations

from frame.evaluate.errors import EvalConfigError, EvaluationError


class DownstreamError(EvaluationError):
    """Base class of everything raised by frame.downstream."""


class DownstreamConfigError(EvalConfigError):
    """A malformed downstream configuration; ``field`` names the dotted config key."""
