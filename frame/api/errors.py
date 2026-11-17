"""API-level exceptions and their HTTP status-code mapping (Phase 7).

Every `frame.*` package (`frame.preprocessing`, `frame.geospatial`,
`frame.consistency`, `frame.uncertainty`, `frame.analysis`) already raises
specific, named exceptions for malformed input -- this module does not
duplicate their validation logic, only maps their existing exception
families (and a few API-specific ones, e.g. "no such job") onto HTTP status
codes, so routes.py never has to construct an HTTP response by hand for an
error case.

Unrecognized exceptions (anything not covered below) map to 500 -- a
genuine internal failure, never shown to the caller with its traceback
(see frame/api/app.py's generic exception handler).
"""

from __future__ import annotations

from frame.acquisition.errors import (
    AcquisitionError,
    AuthenticationError as AcquisitionAuthenticationError,
    InvalidQueryError as AcquisitionInvalidQueryError,
    ProviderUnavailableError as AcquisitionProviderUnavailableError,
    RetrievalMismatchError as AcquisitionRetrievalMismatchError,
    SceneNotFoundError as AcquisitionSceneNotFoundError,
)
from frame.analysis.errors import AnalysisError
from frame.consistency.errors import ConsistencyError
from frame.geospatial.errors import GeospatialError
from frame.models.errors import (
    ModelContractError,
    ModelError,
    ModelInferenceError,
    ModelLoadError,
    ModelUnavailableError,
    ModelWorkerError,
    UnknownModelError,
)
from frame.preprocessing.errors import PreprocessingError
from frame.tiling.errors import InvalidSceneError
from frame.uncertainty.errors import UncertaintyError


class APIError(Exception):
    """Base class for every API-specific error (as opposed to a `frame.*`
    package error, which is mapped separately -- see `status_code_for`)."""

    code = "api_error"
    status_code = 400

    def __init__(self, message: str):
        super().__init__(message)
        self.message = message


class JobNotFoundError(APIError):
    code = "job_not_found"
    status_code = 404

    def __init__(self, message: str, *, job_id: str = ""):
        super().__init__(f"{message} (job_id={job_id!r})" if job_id else message)
        self.job_id = job_id


class UploadNotFoundError(APIError):
    code = "upload_not_found"
    status_code = 404

    def __init__(self, message: str, *, upload_id: str = ""):
        super().__init__(f"{message} (upload_id={upload_id!r})" if upload_id else message)
        self.upload_id = upload_id


class UnsupportedFileError(APIError):
    code = "unsupported_file"
    status_code = 400


class UnsupportedTileLayerError(APIError):
    code = "unsupported_tile_layer"
    status_code = 400


class AsyncJobNotFoundError(APIError):
    code = "async_job_not_found"
    status_code = 404

    def __init__(self, message: str, *, async_job_id: str = ""):
        super().__init__(f"{message} (async_job_id={async_job_id!r})" if async_job_id else message)
        self.async_job_id = async_job_id


class SceneTooLargeError(APIError):
    code = "scene_too_large"
    status_code = 413


class ValidationFailedError(APIError):
    code = "validation_failed"
    status_code = 422


class AcquisitionArtifactNotAvailableError(APIError):
    code = "acquisition_artifact_not_available"
    status_code = 404


class AcquisitionBandsInsufficientForModelError(APIError):
    """A retrieval's real bands do not cover what the requested SR model needs (today, always the
    4-band RGBN set -- see docs/CLAIMS.md: no model here accepts or produces genuine 12-band SR)."""

    code = "acquisition_bands_insufficient_for_model"
    status_code = 422


# Every `frame.*` package's own error family -> the HTTP status it means
# here. All are caller-input problems (missing band, bad shape, missing
# CRS, ...) -- 400, not 500.
_FRAME_ERROR_STATUS = 400


def model_error_code(exc: ModelError) -> str:
    """Stable machine-readable code for a `frame.models` error."""
    if isinstance(exc, UnknownModelError):
        return "unknown_model"
    if isinstance(exc, ModelUnavailableError):
        return "model_unavailable"
    if isinstance(exc, ModelContractError):
        return "model_input_invalid"
    return "model_runtime_error"


def acquisition_error_code(exc: AcquisitionError) -> str:
    """Stable machine-readable code for a `frame.acquisition` error -- named
    distinctly (unlike the generic "frame_error" every other `frame.*` package
    gets below) because the acquisition/map UI needs to branch on exactly which
    of these happened: a bad AOI/query, an unreachable provider, a rejected
    credential, a scene id that no longer exists, or a retrieval that did not
    match the requested scene."""
    if isinstance(exc, AcquisitionAuthenticationError):
        return "acquisition_authentication_failed"
    if isinstance(exc, AcquisitionProviderUnavailableError):
        return "acquisition_provider_unavailable"
    if isinstance(exc, AcquisitionInvalidQueryError):
        return "invalid_acquisition_query"
    if isinstance(exc, AcquisitionSceneNotFoundError):
        return "acquisition_scene_not_found"
    if isinstance(exc, AcquisitionRetrievalMismatchError):
        return "acquisition_retrieval_mismatch"
    return "acquisition_error"


def status_code_for(exc: Exception) -> int:
    """The HTTP status code `exc` should be reported as."""
    if isinstance(exc, APIError):
        return exc.status_code
    # AuthenticationError is a ProviderUnavailableError subclass -- check it first.
    if isinstance(exc, AcquisitionAuthenticationError):
        return 401
    if isinstance(exc, AcquisitionProviderUnavailableError):
        return 503
    if isinstance(exc, AcquisitionInvalidQueryError):
        return 400
    if isinstance(exc, AcquisitionSceneNotFoundError):
        return 404
    if isinstance(exc, AcquisitionRetrievalMismatchError):
        return 502  # the retrieval provider's own response did not match the requested scene
    if isinstance(exc, AcquisitionError):
        return 400
    if isinstance(exc, (PreprocessingError, GeospatialError, ConsistencyError, UncertaintyError, AnalysisError)):
        return _FRAME_ERROR_STATUS
    if isinstance(exc, InvalidSceneError):
        return 400  # an empty / malformed scene reached the tile engine
    if isinstance(exc, UnknownModelError):
        return 400
    if isinstance(exc, ModelContractError):
        return 422  # the input did not satisfy the model's contract (e.g. wrong value scale)
    if isinstance(exc, (ModelUnavailableError, ModelWorkerError, ModelLoadError)):
        return 503  # the requested model cannot serve right now
    if isinstance(exc, ModelInferenceError):
        return 500
    return 500
