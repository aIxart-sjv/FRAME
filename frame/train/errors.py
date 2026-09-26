"""Errors of the training layer (Phase 4). Stdlib only."""

from __future__ import annotations

from typing import Optional


class TrainError(Exception):
    """Base class for every error raised by frame.train."""


class ConfigError(TrainError, ValueError):
    """An invalid or unparseable training configuration; ``field`` names the offending (dotted) key."""

    def __init__(self, message: str, *, field: Optional[str] = None):
        super().__init__(f"{field}: {message}" if field else message)
        self.field = field


class TrainDataError(TrainError):
    """The manifest cannot be used for training (unreadable, empty split, failed QC)."""


class LeakageGuardError(TrainDataError):
    """Training was refused because the manifest fails the geographic-split or dataset-role rules."""

    def __init__(self, message: str, *, codes: tuple = ()):
        super().__init__(message)
        self.codes = tuple(codes)


class ModelBuildError(TrainError):
    """A requested model cannot be constructed in this environment."""


class CheckpointError(TrainError):
    """A checkpoint is missing, corrupt, or incompatible with the run that wants to resume it."""


class NonFiniteLossError(TrainError):
    """The loss became NaN/Inf; training stops instead of corrupting the weights."""


class LossInputError(TrainError, ValueError):
    """A loss was called with tensors of the wrong rank, shape or mask."""
