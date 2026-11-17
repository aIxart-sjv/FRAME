"""Tests for frame.api.errors -- the API-level exception taxonomy and its
mapping onto HTTP status codes. Pure logic, no FastAPI app needed.
"""

import pytest

from frame.acquisition.errors import (
    AuthenticationError as AcquisitionAuthenticationError,
    InvalidQueryError as AcquisitionInvalidQueryError,
    ProviderUnavailableError as AcquisitionProviderUnavailableError,
    RetrievalMismatchError,
    SceneNotFoundError as AcquisitionSceneNotFoundError,
)
from frame.api.errors import (
    APIError,
    JobNotFoundError,
    UnsupportedFileError,
    ValidationFailedError,
    acquisition_error_code,
    status_code_for,
)
from frame.consistency.errors import ShapeMismatchError as ConsistencyShapeMismatchError
from frame.geospatial.errors import MissingCRSError
from frame.preprocessing.errors import UnsupportedBandsError


def test_job_not_found_maps_to_404():
    assert status_code_for(JobNotFoundError("no such job")) == 404


def test_unsupported_file_maps_to_400():
    assert status_code_for(UnsupportedFileError("bad file")) == 400


def test_validation_failed_maps_to_422():
    assert status_code_for(ValidationFailedError("bad request")) == 422


def test_frame_preprocessing_error_maps_to_400():
    assert status_code_for(UnsupportedBandsError("missing band")) == 400


def test_frame_geospatial_error_maps_to_400():
    assert status_code_for(MissingCRSError("no crs")) == 400


def test_frame_consistency_error_maps_to_400():
    assert status_code_for(ConsistencyShapeMismatchError("bad shape")) == 400


def test_unrecognized_exception_maps_to_500():
    assert status_code_for(RuntimeError("something broke")) == 500


def test_api_error_carries_a_machine_readable_code():
    err = JobNotFoundError("no such job", job_id="abc123")
    assert err.code == "job_not_found"
    assert "abc123" in str(err)


def test_every_api_error_subclass_is_an_api_error():
    assert issubclass(JobNotFoundError, APIError)
    assert issubclass(UnsupportedFileError, APIError)
    assert issubclass(ValidationFailedError, APIError)


@pytest.mark.parametrize(
    "exc,expected_status,expected_code",
    [
        (AcquisitionInvalidQueryError("bad bbox"), 400, "invalid_acquisition_query"),
        (AcquisitionProviderUnavailableError("down", provider="x"), 503, "acquisition_provider_unavailable"),
        (AcquisitionAuthenticationError("bad creds", provider="x"), 401, "acquisition_authentication_failed"),
        (AcquisitionSceneNotFoundError("no such scene"), 404, "acquisition_scene_not_found"),
        (RetrievalMismatchError("datetime mismatch"), 502, "acquisition_retrieval_mismatch"),
    ],
)
def test_acquisition_errors_map_to_distinct_status_and_code(exc, expected_status, expected_code):
    assert status_code_for(exc) == expected_status
    assert acquisition_error_code(exc) == expected_code


def test_authentication_error_is_not_mistaken_for_generic_provider_unavailable():
    # AuthenticationError subclasses ProviderUnavailableError -- the more specific
    # mapping (401) must win, not the generic one (503).
    exc = AcquisitionAuthenticationError("bad creds", provider="copernicus-cdse")
    assert isinstance(exc, AcquisitionProviderUnavailableError)
    assert status_code_for(exc) == 401
    assert acquisition_error_code(exc) == "acquisition_authentication_failed"
