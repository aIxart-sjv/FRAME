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

from pydantic import BaseModel, ConfigDict, Field, field_validator

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
