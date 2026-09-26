"""Errors of the evaluation layer (Phase 5). Stdlib only."""

from __future__ import annotations

from typing import Optional, Tuple


class EvaluationError(Exception):
    """Base class for every error raised by frame.evaluate."""


class EvalConfigError(EvaluationError, ValueError):
    """An invalid or unparseable evaluation configuration; ``field`` names the offending (dotted) key."""

    def __init__(self, message: str, *, field: Optional[str] = None):
        super().__init__(f"{field}: {message}" if field else message)
        self.field = field


class ReferenceMismatchError(EvaluationError):
    """An SR output and its HR reference (or an LR and its HR) do not describe the same ground, grid or shape."""


class RoleSafetyError(EvaluationError):
    """An evaluation was refused because it would misuse a dataset role (e.g. a benchmark record labelled train) or contaminate a comparison."""

    def __init__(self, message: str, *, codes: Tuple[str, ...] = ()):
        super().__init__(message)
        self.codes = tuple(codes)


class SystemUnavailableError(EvaluationError):
    """A system to be evaluated (a model, its weights, a worker) cannot be loaded in this environment."""
