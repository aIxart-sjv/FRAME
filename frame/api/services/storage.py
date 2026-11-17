"""In-memory upload/job/analysis registry for the FRAME API (Phase 7).

Deliberately a plain in-process dict, not a database or a distributed job
queue -- this is an SIH prototype/demo backend, and the task explicitly
asks not to invent Celery/Redis/queues unless absolutely necessary. State
is lost on server restart; generated artifacts on disk (under
`frame.api.config.WORKSPACE_DIR`) persist independently of this registry.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from frame.acquisition.contract import AcquisitionQuery, RetrievalProvenance, SceneCandidate
from frame.api.errors import APIError, JobNotFoundError, UploadNotFoundError
from frame.preprocessing.metadata import RasterMetadata


class AnalysisNotFoundError(APIError):
    code = "analysis_not_found"
    status_code = 404

    def __init__(self, message: str, *, analysis_id: str = ""):
        super().__init__(f"{message} (analysis_id={analysis_id!r})" if analysis_id else message)
        self.analysis_id = analysis_id


class AcquisitionSearchNotFoundError(APIError):
    code = "acquisition_search_not_found"
    status_code = 404

    def __init__(self, message: str, *, search_id: str = ""):
        super().__init__(f"{message} (search_id={search_id!r})" if search_id else message)
        self.search_id = search_id


class AcquisitionRetrievalNotFoundError(APIError):
    code = "acquisition_retrieval_not_found"
    status_code = 404

    def __init__(self, message: str, *, retrieval_id: str = ""):
        super().__init__(f"{message} (retrieval_id={retrieval_id!r})" if retrieval_id else message)
        self.retrieval_id = retrieval_id


@dataclass
class UploadRecord:
    upload_id: str
    filename: str
    file_path: Path
    band_names: Tuple[str, ...]
    width: int
    height: int
    crs: Optional[str]
    resolution_m: Optional[float]
    input_scale: str
    created_at: str
    # None for a manually-uploaded file (the overwhelming majority of records); set to a real
    # AcquisitionRetrievalRecord.retrieval_id only when this upload was derived from a real
    # frame.acquisition retrieval (frame.api.services.acquisition.prepare_sr_input) -- the one
    # link between an SR job and the real scene/provider it actually came from.
    source_acquisition_retrieval_id: Optional[str] = None


@dataclass
class SRJobRecord:
    job_id: str
    status: str  # "completed" | "failed"
    upload_id: str
    input_tensor_path: Path
    sr_mean_path: Path  # GeoTIFF
    sr_std_path: Path  # GeoTIFF
    sr_mean_tensor_path: Path  # .pt, reused by /analysis/ndvi
    sr_std_tensor_path: Path  # .pt, reused by /analysis/ndvi
    input_metadata: RasterMetadata  # native 10 m grid -- reused by /analysis/ndvi to export NDVI GeoTIFFs
    output_metadata: RasterMetadata  # SR 2.5 m pixel grid -- ditto
    result: Dict[str, Any]
    error: Optional[str]
    created_at: str


@dataclass
class AnalysisRecord:
    analysis_id: str
    job_id: str
    native_ndvi_path: Path  # single-band GeoTIFF, native 10 m grid
    sr_ndvi_path: Path  # single-band GeoTIFF, SR 2.5 m pixel grid
    ndvi_diff_path: Path  # single-band GeoTIFF, native 10 m grid
    result: Dict[str, Any]
    created_at: str


@dataclass
class AcquisitionSearchRecord:
    """One real `POST /acquisition/search` call's query and results, cached so a
    later `/acquisition/retrieve` call can look a selected scene back up by id
    without the frontend re-sending the full STAC item. Real content only --
    ``candidates`` holds exactly what `frame.acquisition.search.search_scenes`
    returned (possibly empty)."""

    search_id: str
    query: AcquisitionQuery
    candidates: Dict[str, SceneCandidate]  # keyed by SceneCandidate.id
    created_at: str


@dataclass
class AcquisitionRetrievalRecord:
    """One real `POST /acquisition/retrieve` call's result: real harmonised
    bands + provenance, plus the real GeoTIFF artifact(s) written to disk."""

    retrieval_id: str
    search_id: str
    scene_id: str
    bands_geotiff_path: Path
    scl_geotiff_path: Optional[Path]
    provenance: RetrievalProvenance
    result: Dict[str, Any]
    created_at: str


class JobStore:
    """A small in-memory registry. One instance per running API process."""

    def __init__(self) -> None:
        self._uploads: Dict[str, UploadRecord] = {}
        self._jobs: Dict[str, SRJobRecord] = {}
        self._analyses: Dict[str, AnalysisRecord] = {}
        self._acquisition_searches: Dict[str, AcquisitionSearchRecord] = {}
        self._acquisition_retrievals: Dict[str, AcquisitionRetrievalRecord] = {}

    @staticmethod
    def generate_id() -> str:
        return uuid.uuid4().hex

    def put_upload(self, record: UploadRecord) -> None:
        self._uploads[record.upload_id] = record

    def get_upload(self, upload_id: str) -> UploadRecord:
        try:
            return self._uploads[upload_id]
        except KeyError:
            raise UploadNotFoundError("No such upload", upload_id=upload_id) from None

    def put_job(self, record: SRJobRecord) -> None:
        self._jobs[record.job_id] = record

    def get_job(self, job_id: str) -> SRJobRecord:
        try:
            return self._jobs[job_id]
        except KeyError:
            raise JobNotFoundError("No such SR job", job_id=job_id) from None

    def put_analysis(self, record: AnalysisRecord) -> None:
        self._analyses[record.analysis_id] = record

    def get_analysis(self, analysis_id: str) -> AnalysisRecord:
        try:
            return self._analyses[analysis_id]
        except KeyError:
            raise AnalysisNotFoundError("No such analysis", analysis_id=analysis_id) from None

    def get_analyses_for_job(self, job_id: str) -> "list[AnalysisRecord]":
        """Every analysis run against ``job_id`` so far (Step I `GET /products/{id}/analysis`),
        oldest first. Empty, never an error, if none exist yet -- a job with no analysis run is a
        normal state, not a failure."""
        return [a for a in self._analyses.values() if a.job_id == job_id]

    def put_acquisition_search(self, record: AcquisitionSearchRecord) -> None:
        self._acquisition_searches[record.search_id] = record

    def get_acquisition_search(self, search_id: str) -> AcquisitionSearchRecord:
        try:
            return self._acquisition_searches[search_id]
        except KeyError:
            raise AcquisitionSearchNotFoundError("No such acquisition search", search_id=search_id) from None

    def put_acquisition_retrieval(self, record: AcquisitionRetrievalRecord) -> None:
        self._acquisition_retrievals[record.retrieval_id] = record

    def get_acquisition_retrieval(self, retrieval_id: str) -> AcquisitionRetrievalRecord:
        try:
            return self._acquisition_retrievals[retrieval_id]
        except KeyError:
            raise AcquisitionRetrievalNotFoundError("No such acquisition retrieval", retrieval_id=retrieval_id) from None
