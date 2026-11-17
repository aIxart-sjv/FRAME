"""Pydantic request/response schemas for the FRAME API (Phase 7; wording updated in Phase 8).

Terminology constants
----------------------
Every response that describes the SR output or its uncertainty uses these
exact phrases -- never "native 2.5 m Sentinel-2" / "true 2.5 m image" for
the former, never "confidence" or "calibrated" for the latter (Phase 6 measured
the signal as weakly informative and uncalibrated, so it is labelled a
"TTA stability -- reconstruction-variation diagnostic"; the JSON field is still
called ``uncertainty`` for API compatibility), and LAM
(explainability/sensitivity, upstream `sen2sr/xai/lam.py`, not exposed by
this phase's endpoints at all) is never called "uncertainty" anywhere in
this codebase. `frame/tests/test_api_terminology.py` enforces this
structurally.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from frame.models import config as models_cfg
from frame.models.selection import normalize_model_name

SR_PRODUCT_DESCRIPTION = "SR-derived product — 2.5 m pixel grid"
UNCERTAINTY_LABEL = "TTA stability — reconstruction-variation diagnostic"
UNCERTAINTY_DISCLAIMER = (
    "This is a relative, architecture-conditioned model-stability diagnostic: how much the reconstruction varies "
    "under test-time perturbation ensembling (TTA). It is NOT a calibrated probability of "
    "error, NOT a confidence interval, and NOT a physically rigorous "
    "uncertainty bound. In FRAME's own validation on registration-checked reference data (docs/RELIABILITY.md) it was "
    "only weakly associated with reconstruction error, about as much as image texture alone, and was not shown to identify "
    "high-error regions reliably; treat it as something to inspect, not as a reliability score. "
    "It is also NOT the upstream LAM explainability tool "
    "(sen2sr/xai/lam.py) -- LAM answers a different question (which input "
    "pixels influence the output) via a different mechanism (gradients on "
    "blurred input copies) and is not exposed by this API."
)
NDVI_STABILITY_CAVEAT = (
    "The stability used in the uncertainty-weighted NDVI summary is the TTA stability diagnostic (see the stability caveat above), "
    "not a calibrated probability of NDVI error."
)
NDVI_DEMONSTRATION_NOTE = (
    "This NDVI view is a downstream analytical demonstration on one scene: it compares NDVI from the SR product with NDVI "
    "from its own low-resolution input, so it is not a reference-based accuracy test. In FRAME's reference-based tests "
    "(docs/DOWNSTREAM.md) super-resolution changed region-level NDVI and a fixed vegetation-threshold decision only "
    "slightly, with a sign that depended on the dataset; no consistent downstream advantage over bicubic was established."
)
GROUND_TRUTH_DISCLAIMER = (
    "Sentinel-2's finest native band resolution is 10 m -- it has never "
    "observed the ground at 2.5 m. The SR-derived product is a learned "
    "statistical inference resampled onto a 2.5 m pixel grid, not a "
    "directly observed 2.5 m measurement."
)

SCIENTIFIC_CAVEATS: List[str] = [
    GROUND_TRUTH_DISCLAIMER,
    UNCERTAINTY_DISCLAIMER,
    "Self-consistency diagnostics measure agreement with the model's own LR "
    "input, not ground-truth accuracy.",
]


class ErrorResponse(BaseModel):
    error: str
    code: str
    detail: str


# ---------------------------------------------------------------------------
# GET /health
# ---------------------------------------------------------------------------

class ModelAvailability(BaseModel):
    """One selectable SR model and whether it can run on this server right now."""

    model_config = ConfigDict(protected_namespaces=())

    id: str
    label: str
    model_name: str
    available: bool
    reason: Optional[str] = Field(None, description="User-facing explanation when available is false")


class ModelCapability(BaseModel):
    """Real, verified capability metadata for one selectable SR model (Step G, Requirements 142
    Requirement 4) -- GET /models. Every field is a fact about the actual checkpoint/executable
    (frame.models.selection.MODEL_SPECS), never an aspiration: no model here claims a 12-band
    input/output or a resolution finer than 2.5 m unless a real, verified model and weights
    support that exact contract (see docs/CLAIMS.md)."""

    model_config = ConfigDict(protected_namespaces=())

    id: str
    label: str
    model_name: str
    description: str
    architecture: str
    input_bands: List[str]
    output_bands: List[str]
    input_resolution_m: Optional[float]
    output_resolution_m: Optional[float]
    scale_factor: Optional[int]
    uncertainty: bool
    uncertainty_method: Optional[str] = Field(
        None, description="What the uncertainty signal actually is, in this project's own honest terms -- never 'calibrated'."
    )
    parameter_count: Optional[int] = None
    environment: str = Field(description="'main' (loaded in-process) or 'isolated:<worker-module>' (a separate environment/process).")
    available: bool
    reason: Optional[str] = Field(None, description="User-facing explanation when available is false")


class HealthResponse(BaseModel):
    status: str
    frame_version: Optional[str]
    api_version: str
    model_name: str
    default_model: str = models_cfg.DEFAULT_MODEL
    available_models: List[ModelAvailability] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# POST /aoi/preview
# ---------------------------------------------------------------------------

class AOIPreviewRequest(BaseModel):
    lat: float = Field(..., ge=-90.0, le=90.0)
    lon: float = Field(..., ge=-180.0, le=180.0)
    start_date: str = Field(..., description="YYYY-MM-DD")
    end_date: str = Field(..., description="YYYY-MM-DD")
    edge_size: int = Field(128, gt=0, description="LR patch size in pixels; 128 is the only proven value")
    bands: List[str] = Field(default_factory=lambda: ["B04", "B03", "B02", "B08"])


class AOIPreviewResponse(BaseModel):
    valid: bool
    coordinates: Dict[str, float]
    date_window: Dict[str, str]
    edge_size: int
    bands: List[str]
    expected_input_shape: List[int]
    expected_output_shape: List[int]
    native_resolution_m: float
    sr_resolution_m: float
    sr_product_description: str = SR_PRODUCT_DESCRIPTION
    note: str = (
        "This prototype's /aoi/preview validates the request and reports the "
        "shapes/resolutions a run would produce; it does not fetch real "
        "imagery. Use /upload with a real Sentinel-2 L2A GeoTIFF to run the "
        "actual pipeline."
    )


# ---------------------------------------------------------------------------
# POST /acquisition/search, GET /acquisition/scenes/{search_id}/{scene_id},
# POST /acquisition/retrieve, GET /acquisition/download/{retrieval_id}/{bands,scl}
#
# Real Sentinel-2 discovery (frame.acquisition.search, CDSE STAC) and
# retrieval (frame.acquisition.retrieve, CDSE Process API primary /
# Planetary Computer fallback) -- see docs/ACQUISITION.md. Distinct from
# /aoi/preview above (which never fetches real imagery): every field here
# is copied from a real provider response, never fabricated. This endpoint
# set does not yet feed a retrieved scene into /sr/run (see docs/ACQUISITION.md
# §5) and does not yet claim 12-band SR.
# ---------------------------------------------------------------------------

ACQUISITION_RETRIEVAL_NOTE = (
    "These are real retrieved reflectance bands for this scene and AOI, harmonised onto one grid. "
    "They are not yet wired into POST /sr/run -- SR models here only ever accept the 4-band RGBN "
    "input POST /upload validates, and none of them are trained on or claim to produce a genuine "
    "12-band 2.5 m reconstruction. See docs/ACQUISITION.md and docs/CLAIMS.md."
)


class AcquisitionSearchRequest(BaseModel):
    bbox: Optional[List[float]] = Field(
        None, min_length=4, max_length=4, description="[min_lon, min_lat, max_lon, max_lat], EPSG:4326"
    )
    aoi_geojson: Optional[Dict[str, Any]] = Field(
        None,
        description="A GeoJSON Feature/Polygon/MultiPolygon. Reduced to its bounding box for search "
        "(true polygon-intersection filtering is not implemented -- see docs/ACQUISITION.md). "
        "Provide this or bbox, not neither.",
    )
    start_date: str = Field(..., description="YYYY-MM-DD or full ISO 8601")
    end_date: str = Field(..., description="YYYY-MM-DD or full ISO 8601")
    max_cloud_cover: float = Field(100.0, ge=0.0, le=100.0)
    max_results: int = Field(10, ge=1, le=50)

    @model_validator(mode="after")
    def _one_of_bbox_or_geojson(self) -> "AcquisitionSearchRequest":
        if self.bbox is None and self.aoi_geojson is None:
            raise ValueError("Provide either 'bbox' or 'aoi_geojson' to define the search AOI.")
        return self


class SceneCandidateSchema(BaseModel):
    id: str
    provider: str
    collection: str
    datetime: str
    cloud_cover: Optional[float]
    bbox: List[float]
    geometry: Optional[Dict[str, Any]] = None
    mgrs_tile: Optional[str] = None
    platform: Optional[str] = None
    processing_baseline: Optional[str] = None
    asset_count: int = Field(0, description="Number of real STAC asset hrefs this scene has (not returned in full -- see the scene-detail endpoint's note).")


class AcquisitionSearchResponse(BaseModel):
    search_id: str
    query: AcquisitionSearchRequest
    provider: str
    stac_endpoint: str
    scenes: List[SceneCandidateSchema]
    scene_count: int
    note: Optional[str] = None


class AcquisitionSceneDetailResponse(BaseModel):
    search_id: str
    scene: SceneCandidateSchema
    note: str = (
        "Real per-resolution asset references (asset_hrefs) point at CDSE's own s3://eodata/... "
        "objects, which require separate S3 credentials this server does not use. Real pixels are "
        "retrieved via POST /acquisition/retrieve (Sentinel Hub Process API, or Planetary Computer "
        "as a disclosed fallback), not by fetching those hrefs directly."
    )


class AcquisitionRetrieveRequest(BaseModel):
    search_id: str
    scene_id: str
    bands: List[str] = Field(default_factory=lambda: ["B02", "B03", "B04", "B08"], min_length=1, max_length=12)
    resolution_m: float = Field(10.0, ge=1.0, le=100.0, description="A sane bound, not a scientific one: real L2A bands are native at 10/20/60 m.")
    include_scl: bool = True


class AcquisitionRetrieveResponse(BaseModel):
    retrieval_id: str
    search_id: str
    scene_id: str
    bands: List[str]
    band_native_resolution_m: Dict[str, float]
    band_sources: Dict[str, str]
    band_radiometric_method: Dict[str, str] = Field(
        default_factory=dict,
        description="Per band, the real radiometric conversion its actual source provider applied "
        "(frame.acquisition.harmonise) -- never a single assumed formula for the whole stack.",
    )
    is_complete_l2a: bool = Field(
        False, description="Whether this retrieval's bands cover all 12 required Sentinel-2 L2A bands (B10 excluded)."
    )
    validity_mask_coverage: Optional[float] = Field(
        None, description="Fraction of pixels the real SCL-derived validity mask marks valid (null if SCL was not retrieved).",
    )
    scl_included: bool
    crs: str
    resolution_m: float
    width: int
    height: int
    retrieval_provider: str
    retrieval_endpoint: str
    matched_acquisition_datetime: str
    retrieved_at_utc: str
    warnings: List[str]
    artifacts: Dict[str, str]
    note: str = ACQUISITION_RETRIEVAL_NOTE


# ---------------------------------------------------------------------------
# POST /acquisition/retrieve/{retrieval_id}/prepare-sr-input
#
# Step E: the one adapter between a real acquisition retrieval and the
# existing, unmodified /sr/run. Shaped like UploadResponse (same first block
# of fields) plus real acquisition provenance, so the frontend can drop the
# first block straight into the same state a manual /upload response fills.
# ---------------------------------------------------------------------------

class AcquisitionSrInputResponse(BaseModel):
    upload_id: str
    filename: str
    valid: bool
    band_names: List[str]
    width: int
    height: int
    crs: Optional[str]
    resolution_m: Optional[float]
    input_scale: str
    validation_messages: List[str]
    retrieval_id: str
    search_id: str
    scene_id: str
    provider: str
    matched_acquisition_datetime: str
    excluded_bands: List[str] = Field(
        default_factory=list,
        description="Real bands this retrieval had beyond the RGBN set /sr/run uses -- preserved in the "
        "retrieval's own bands GeoTIFF, never discarded, not used for this SR run.",
    )
    note: str


# ---------------------------------------------------------------------------
# POST /upload
# ---------------------------------------------------------------------------

class UploadResponse(BaseModel):
    upload_id: str
    filename: str
    valid: bool
    band_names: List[str]
    width: int
    height: int
    crs: Optional[str]
    resolution_m: Optional[float]
    input_scale: str
    validation_messages: List[str]


# ---------------------------------------------------------------------------
# POST /sr/run
# ---------------------------------------------------------------------------

class SRRunRequest(BaseModel):
    upload_id: str
    seed: Optional[int] = Field(None, description="Overrides the default TTA uncertainty seed (42) if given")
    model: str = Field(
        models_cfg.DEFAULT_MODEL,
        description=f"Which SR model to run: one of {', '.join(models_cfg.SUPPORTED_MODELS)}",
    )

    @field_validator("model")
    @classmethod
    def _known_model(cls, value: str) -> str:
        return normalize_model_name(value)


class ResolutionDescription(BaseModel):
    native_resolution_m: float
    sr_resolution_m: float
    scale_factor: int
    description: str = SR_PRODUCT_DESCRIPTION


class UncertaintySummary(BaseModel):
    label: str = UNCERTAINTY_LABEL
    scalar_summary: float
    scalar_summary_definition: str
    overall_distribution: Dict[str, float]
    n: int
    seed: int
    transform_names: List[str]
    disclaimer: str = UNCERTAINTY_DISCLAIMER


class SelfConsistencySummary(BaseModel):
    downsample_rmse: Optional[float]
    ndvi_discrepancy_mean_abs: Optional[float]
    b08_b04_ratio_discrepancy_mean_abs: Optional[float]
    note: str = "Compares the SR output against its own LR input only -- not a ground-truth accuracy check."


class SRResultResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    job_id: str
    status: str
    upload_id: str
    model_name: str
    model_id: str = models_cfg.DEFAULT_MODEL
    input_shape: List[int]
    output_shape: List[int]
    resolution: ResolutionDescription
    bands: List[str]
    crs: Optional[str]
    uncertainty: UncertaintySummary
    self_consistency: SelfConsistencySummary
    metadata: Dict[str, Any]
    scientific_caveats: List[str] = Field(default_factory=lambda: list(SCIENTIFIC_CAVEATS))
    artifacts: Dict[str, str]
    created_at: str


# ---------------------------------------------------------------------------
# POST /analysis/ndvi
# ---------------------------------------------------------------------------

class NDVIAnalysisRequest(BaseModel):
    job_id: str


class NDVIComparisonSummary(BaseModel):
    status: str
    valid_pixel_count: int
    mean_abs_difference: Optional[float]
    rmse: Optional[float]
    max_abs_difference: Optional[float]
    resampling_method: str


class UncertaintyWeightedSummary(BaseModel):
    status: str
    correlation_uncertainty_vs_abs_diff: Optional[float]
    uncertainty_weighted_mean_abs_diff: Optional[float]
    unweighted_mean_abs_diff: Optional[float]


class NDVIAnalysisResponse(BaseModel):
    analysis_id: str
    job_id: str
    status: str
    ndvi_formula: str
    native_resolution_m: float
    sr_resolution_m: float
    comparison: NDVIComparisonSummary
    uncertainty_weighted_summary: UncertaintyWeightedSummary
    scientific_caveats: List[str]
    metadata: Dict[str, Any]
    artifacts: Dict[str, str]
    created_at: str


# ---------------------------------------------------------------------------
# Products (Requirements 142 Requirement 11, Step I) -- a GIS-ready view of an
# already-completed SR job. product_id == job_id: there is no second identity
# space, see frame.api.services.product's module docstring.
# ---------------------------------------------------------------------------

class CogValidationSummary(BaseModel):
    is_valid_cog: bool
    reasons: List[str]
    driver_reports_cog_layout: bool
    is_tiled: bool
    compression: Optional[str]
    overview_factors: List[int]
    has_crs: bool
    has_transform: bool
    dtype: str
    band_count: int
    width: int
    height: int


class AsyncJobResponse(BaseModel):
    """A minimal local background-job status (Step I Section 6) -- not a durable queue, see
    frame.api.services.async_jobs's module docstring. ``sr_job_id`` is set once ``status`` is
    "completed"; poll `GET /sr/result/{sr_job_id}` or `GET /products/{sr_job_id}` next."""

    async_job_id: str
    status: str  # "queued" | "running" | "completed" | "failed"
    sr_job_id: Optional[str] = None
    error: Optional[str] = None
    created_at: str


class ProductResponse(BaseModel):
    job_id: str
    status: str
    cog_available: bool
    cog_validation: Optional[CogValidationSummary] = None
    evidence_assets: Dict[str, str] = Field(default_factory=dict)
    analyses: List[Dict[str, Any]] = Field(default_factory=list)
    stac_url: Optional[str] = None
