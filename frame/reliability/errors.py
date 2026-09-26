"""Exceptions of the reliability layer (Phase 6). They extend the Phase 5 evaluation errors so the CLI exit codes and messages behave the same way."""

from __future__ import annotations

from frame.evaluate.errors import EvalConfigError, EvaluationError


class ReliabilityError(EvaluationError):
    """Base class of everything raised by frame.reliability."""


class ReliabilityConfigError(EvalConfigError):
    """A malformed reliability configuration; ``field`` names the dotted config key."""
