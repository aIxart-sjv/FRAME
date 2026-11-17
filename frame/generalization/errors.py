"""Exceptions for frame.generalization (Step H, Requirements 142 Requirement 12).

Follows the same convention as every other layer (frame.evaluate.errors,
frame.reliability.errors, frame.downstream.errors): a real, named failure --
never a silent fallback to fabricated data.
"""

from __future__ import annotations


class GeneralizationError(Exception):
    """Base class for this package's errors."""


class NoIndianSceneFoundError(GeneralizationError):
    """The AOI/date/cloud query returned no real Sentinel-2 candidate. Never answered by
    substituting a non-Indian scene or a synthetic one."""


class UnsupportedProtocolStageError(GeneralizationError):
    """A caller asked this module to compute a protocol stage the available data does not
    support (e.g. reference-based fidelity when no HR reference exists) -- raised rather than
    silently skipped, so a caller cannot accidentally read a None as "ran and found nothing"."""
