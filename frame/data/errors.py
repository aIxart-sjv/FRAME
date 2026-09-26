"""Exceptions for FRAME's paired-data layer (Phase 3).

Every check in `frame.data` either raises one of these (for a single sample
that is handled directly) or reports it as a `QCIssue` (for a whole
manifest, where one bad sample must not hide the rest). Nothing here repairs
data: a corrupt sample is reported, never "fixed".
"""

from __future__ import annotations


class DataError(Exception):
    """Base class for every error raised by frame.data."""


class ContractError(DataError, ValueError):
    """A record, sample or configuration does not satisfy the paired-data contract
    (missing field, unknown enum value, absolute path, wrong band count, ...).

    ``code`` is a short machine-readable reason (``missing_hr``, ``absolute_path``, ...)
    that QC reports use to say *why* a record was rejected."""

    def __init__(self, message: str, *, code: str = "invalid_record"):
        super().__init__(message)
        self.code = code


class GeoPairError(DataError):
    """An LR/HR pair is not geometrically consistent."""


class CRSMismatchError(GeoPairError):
    """LR and HR are in different coordinate reference systems."""


class ResolutionRatioError(GeoPairError):
    """LR pixel size is not (scale factor) x HR pixel size."""


class FootprintMismatchError(GeoPairError):
    """LR and HR do not cover the same physical footprint (origin shifted or extent differs)."""


class DimensionMismatchError(GeoPairError):
    """HR raster size is not (scale factor) x LR raster size."""


class PatchError(DataError, ValueError):
    """Invalid patch geometry (patch larger than the raster, misaligned window, ...)."""


class DegradationError(DataError, ValueError):
    """Invalid degradation configuration or input."""


class ManifestError(DataError):
    """A manifest file is unreadable, of an unknown version, or malformed."""


class SplitError(DataError):
    """Base class for split-integrity failures."""


class SplitLeakageError(SplitError):
    """The same scene (or region) contributes samples to more than one split."""


class RoleViolationError(SplitError):
    """A dataset was placed in a split its documented role forbids (for example a
    benchmark used for training)."""


class DatasetUnavailableError(DataError):
    """The dataset files, or the optional reader/dependency needed to read them, are not available."""
