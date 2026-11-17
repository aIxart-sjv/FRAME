"""Real Sentinel-2 acquisition: AOI + date range + cloud threshold -> real STAC
scene discovery -> real, provenance-tracked multi-resolution pixel retrieval.

    frame.acquisition.contract   -- AcquisitionQuery / SceneCandidate / RetrievedScene
    frame.acquisition.search     -- real CDSE STAC scene discovery
    frame.acquisition.retrieve   -- real multi-resolution pixel retrieval + harmonisation
    frame.acquisition.providers  -- the two real, named pixel-retrieval providers
    frame.acquisition.geometry   -- AOI/UTM helpers shared by the providers

See docs/FRAME_TECHNICAL_SPEC.md and docs/CLAIMS.md: acquisition never
fabricates a scene, a provider, or a resolution -- an unavailable provider or
empty search result is a named error, not a silent substitution.
"""

from frame.acquisition.contract import (
    AcquisitionQuery,
    RetrievalProvenance,
    RetrievedScene,
    SceneCandidate,
)
from frame.acquisition.errors import (
    AcquisitionError,
    IncompleteL2ABandsError,
    InvalidQueryError,
    NoScenesFoundError,
    ProviderUnavailableError,
    RetrievalMismatchError,
    SceneNotFoundError,
)
from frame.acquisition.harmonise import HarmonisedProduct, harmonise, select_rgbn_adapter, validate_full_l2a_bands

__all__ = [
    "AcquisitionQuery",
    "SceneCandidate",
    "RetrievedScene",
    "RetrievalProvenance",
    "AcquisitionError",
    "InvalidQueryError",
    "ProviderUnavailableError",
    "NoScenesFoundError",
    "SceneNotFoundError",
    "RetrievalMismatchError",
    "IncompleteL2ABandsError",
    "HarmonisedProduct",
    "harmonise",
    "select_rgbn_adapter",
    "validate_full_l2a_bands",
]
