"""Exceptions for FRAME's real Sentinel-2 acquisition layer.

Every failure mode here is named and raised explicitly -- a missing
credential, an empty search result, or a provider outage is never turned
into a fabricated scene or silently swapped for a different provider
without that swap being recorded in provenance (see contract.py).
"""

from __future__ import annotations


class AcquisitionError(Exception):
    """Base class for every error raised by frame.acquisition."""


class InvalidQueryError(AcquisitionError, ValueError):
    """An AcquisitionQuery's fields do not form a valid search (bad bbox ordering,
    an empty/inverted date range, or a cloud-cover threshold outside [0, 100])."""


class ProviderUnavailableError(AcquisitionError):
    """A named provider could not be used for this request -- missing credentials,
    a network failure, or a non-2xx API response. Never used to mean "silently try
    something else without saying so"; callers that fall back to another provider
    must record the fallback in provenance."""

    def __init__(self, message: str, *, provider: str):
        super().__init__(message)
        self.provider = provider


class AuthenticationError(ProviderUnavailableError):
    """A provider rejected the request because its credentials were invalid or
    expired (an HTTP 401/403), as opposed to being unreachable, misconfigured,
    or genuinely down. Distinguished from ProviderUnavailableError because the
    fix is different (check the credentials, not "wait and retry")."""


class NoScenesFoundError(AcquisitionError):
    """A search request was valid but matched zero real scenes."""


class SceneNotFoundError(AcquisitionError):
    """A referenced scene id was not returned by any provider queried for it."""


class RetrievalMismatchError(AcquisitionError):
    """A retrieval provider's actual returned acquisition (timestamp, footprint) does
    not match the selected candidate closely enough to be trusted as the same
    real-world acquisition -- raised instead of silently accepting a near-miss."""


class IncompleteL2ABandsError(AcquisitionError, ValueError):
    """A real band set does not cover all 12 required Sentinel-2 L2A bands
    (frame.data.bands.L2A_BANDS; B10 is correctly never one of them). Raised by
    frame.acquisition.harmonise.validate_full_l2a_bands."""
