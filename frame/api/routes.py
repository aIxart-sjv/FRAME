"""HTTP routes for the FRAME API (Phase 7).

Every route is thin HTTP glue: parse the request, call
`frame.api.services.pipeline` (which itself only orchestrates the existing,
already-tested `frame.*` packages), and shape the result into a response
schema. No pipeline logic lives here. Errors are never caught locally --
they propagate to `frame.api.app`'s single generic exception handler, which
maps every `frame.api.errors`/`frame.*` exception to the right HTTP status
via `frame.api.errors.status_code_for` and never leaks a traceback.
"""

from __future__ import annotations

import shutil
import uuid
from pathlib import Path
from typing import Any, Callable, List

import dataclasses

from fastapi import APIRouter, Depends, File, Form, Request, Response, UploadFile
from fastapi.responses import FileResponse, JSONResponse

from frame.api import config, schemas
from frame.api.errors import AcquisitionArtifactNotAvailableError, APIError, UnsupportedFileError, UnsupportedTileLayerError
from frame.acquisition.search import stac_url as acquisition_stac_url
from frame.api.services import acquisition as acquisition_service
from frame.api.services import async_jobs
from frame.api.services import model as model_service
from frame.api.services import pipeline
from frame.api.services import product as product_service
from frame.api.services.storage import JobStore
from frame.models.selection import get_spec
from frame.product import tiles
from frame.tiling import TilingConfig

router = APIRouter()

# A single process-lifetime store -- see frame.api.services.storage's own
# docstring for why this is an in-memory registry, not a database. Tests
# override `get_store` (FastAPI dependency injection) to get an isolated,
# empty store per test.
_default_store = JobStore()
_default_async_registry = async_jobs.AsyncJobRegistry()


def get_store() -> JobStore:
    return _default_store


def get_async_registry() -> async_jobs.AsyncJobRegistry:
    return _default_async_registry


def get_device() -> str:
    return model_service.resolve_device(config.DEVICE)


def get_sr_run_request(request: schemas.SRRunRequest) -> schemas.SRRunRequest:
    """The parsed ``POST /sr/run`` body. It is a dependency (FastAPI caches it
    per request) so the route and model selection share ONE body declaration;
    declaring the model in both places would report every validation error twice."""
    return request


def get_model_callable(
    request: schemas.SRRunRequest = Depends(get_sr_run_request), device: str = Depends(get_device)
) -> Callable:
    """The cached model callable for the model the request selected
    (``request.model``: "lite" by default, or "mamba") -- overridden with a
    small fake callable in every non-integration test (see frame/tests/test_api.py)."""
    return model_service.get_model(device, model_name=request.model)


SUPPORTED_UPLOAD_EXTENSIONS = (".tif", ".tiff")
REQUIRED_BANDS = ["B04", "B03", "B02", "B08"]


# ---------------------------------------------------------------------------
# GET /health
# ---------------------------------------------------------------------------

@router.get("/health", response_model=schemas.HealthResponse)
def health() -> schemas.HealthResponse:
    import frame

    frame_version = getattr(frame, "__version__", None)
    return schemas.HealthResponse(
        status="ok",
        frame_version=frame_version,
        api_version=config.API_VERSION,
        model_name=config.MODEL_NAME,
        available_models=[schemas.ModelAvailability(**entry) for entry in model_service.list_models(config.DEVICE)],
    )


# ---------------------------------------------------------------------------
# GET /models
#
# Step G: the authoritative, real model registry (Requirements 142
# Requirement 4). The frontend must read model capability from here rather
# than hard-coding it -- see docs/REGISTRY.md.
# ---------------------------------------------------------------------------

@router.get("/models", response_model=List[schemas.ModelCapability])
def list_models() -> List[schemas.ModelCapability]:
    return [schemas.ModelCapability(**entry) for entry in model_service.list_model_capabilities(config.DEVICE)]


# ---------------------------------------------------------------------------
# POST /aoi/preview
# ---------------------------------------------------------------------------

@router.post("/aoi/preview", response_model=schemas.AOIPreviewResponse)
def aoi_preview(request: schemas.AOIPreviewRequest) -> schemas.AOIPreviewResponse:
    if request.start_date > request.end_date:
        raise APIError(f"start_date ({request.start_date}) must be <= end_date ({request.end_date}).")
    if request.bands != REQUIRED_BANDS:
        raise APIError(
            f"This prototype only supports bands {REQUIRED_BANDS} (in that order), got {request.bands}."
        )
    scale_factor = 4
    return schemas.AOIPreviewResponse(
        valid=True,
        coordinates={"lat": request.lat, "lon": request.lon},
        date_window={"start": request.start_date, "end": request.end_date},
        edge_size=request.edge_size,
        bands=request.bands,
        expected_input_shape=[len(request.bands), request.edge_size, request.edge_size],
        expected_output_shape=[len(request.bands), request.edge_size * scale_factor, request.edge_size * scale_factor],
        native_resolution_m=10.0,
        sr_resolution_m=10.0 / scale_factor,
    )


# ---------------------------------------------------------------------------
# POST /acquisition/search
# ---------------------------------------------------------------------------

def _scene_candidate_schema(candidate) -> schemas.SceneCandidateSchema:
    return schemas.SceneCandidateSchema(
        id=candidate.id,
        provider=candidate.provider,
        collection=candidate.collection,
        datetime=candidate.datetime,
        cloud_cover=candidate.cloud_cover,
        bbox=list(candidate.bbox),
        geometry=candidate.geometry,
        mgrs_tile=candidate.mgrs_tile,
        platform=candidate.platform,
        processing_baseline=candidate.processing_baseline,
        asset_count=len(candidate.asset_hrefs),
    )


@router.post("/acquisition/search", response_model=schemas.AcquisitionSearchResponse)
def acquisition_search(
    request: schemas.AcquisitionSearchRequest, store: JobStore = Depends(get_store)
) -> schemas.AcquisitionSearchResponse:
    query = acquisition_service.build_query(
        bbox=request.bbox,
        aoi_geojson=request.aoi_geojson,
        start_date=request.start_date,
        end_date=request.end_date,
        max_cloud_cover=request.max_cloud_cover,
        max_results=request.max_results,
    )
    record = acquisition_service.search(store, query)
    scenes = [_scene_candidate_schema(c) for c in record.candidates.values()]
    note = None if scenes else (
        "No real Sentinel-2 scenes matched this AOI/date range/cloud threshold at search time. "
        "This is a genuine zero-result search, not an error -- try a wider date range, a larger "
        "cloud-cover threshold, or a different AOI."
    )
    return schemas.AcquisitionSearchResponse(
        search_id=record.search_id,
        query=request,
        provider="copernicus-cdse",
        stac_endpoint=next(iter(record.candidates.values())).stac_endpoint if scenes else f"{acquisition_stac_url().rstrip('/')}/search",
        scenes=scenes,
        scene_count=len(scenes),
        note=note,
    )


# ---------------------------------------------------------------------------
# GET /acquisition/scenes/{search_id}/{scene_id}
# ---------------------------------------------------------------------------

@router.get("/acquisition/scenes/{search_id}/{scene_id}", response_model=schemas.AcquisitionSceneDetailResponse)
def acquisition_scene_detail(
    search_id: str, scene_id: str, store: JobStore = Depends(get_store)
) -> schemas.AcquisitionSceneDetailResponse:
    candidate = acquisition_service.get_candidate(store, search_id, scene_id)
    return schemas.AcquisitionSceneDetailResponse(search_id=search_id, scene=_scene_candidate_schema(candidate))


# ---------------------------------------------------------------------------
# POST /acquisition/retrieve
# ---------------------------------------------------------------------------

@router.post("/acquisition/retrieve", response_model=schemas.AcquisitionRetrieveResponse)
def acquisition_retrieve(
    request: schemas.AcquisitionRetrieveRequest, store: JobStore = Depends(get_store)
) -> schemas.AcquisitionRetrieveResponse:
    config.ensure_workspace_dirs()
    record = acquisition_service.retrieve(
        store,
        search_id=request.search_id,
        scene_id=request.scene_id,
        bands=request.bands,
        resolution_m=request.resolution_m,
        include_scl=request.include_scl,
        workspace_dir=config.WORKSPACE_DIR / "acquisitions",
    )
    return schemas.AcquisitionRetrieveResponse(**record.result)


# ---------------------------------------------------------------------------
# GET /acquisition/download/{retrieval_id}/{bands,scl}
# ---------------------------------------------------------------------------

@router.get("/acquisition/download/{retrieval_id}/bands")
def download_acquisition_bands(retrieval_id: str, store: JobStore = Depends(get_store)) -> FileResponse:
    record = store.get_acquisition_retrieval(retrieval_id)
    return FileResponse(record.bands_geotiff_path, media_type="image/tiff", filename=f"acquisition_bands_{retrieval_id}.tif")


@router.get("/acquisition/download/{retrieval_id}/scl")
def download_acquisition_scl(retrieval_id: str, store: JobStore = Depends(get_store)) -> FileResponse:
    record = store.get_acquisition_retrieval(retrieval_id)
    if record.scl_geotiff_path is None:
        raise AcquisitionArtifactNotAvailableError(
            f"Retrieval {retrieval_id!r} did not include the SCL layer (include_scl was false)."
        )
    return FileResponse(record.scl_geotiff_path, media_type="image/tiff", filename=f"acquisition_scl_{retrieval_id}.tif")


# ---------------------------------------------------------------------------
# POST /acquisition/retrieve/{retrieval_id}/prepare-sr-input
#
# Step E: builds a real RGBN GeoTIFF from a completed retrieval and registers
# it exactly like a manual /upload -- the returned upload_id is used with the
# existing, completely unmodified POST /sr/run.
# ---------------------------------------------------------------------------

@router.post("/acquisition/retrieve/{retrieval_id}/prepare-sr-input", response_model=schemas.AcquisitionSrInputResponse)
def acquisition_prepare_sr_input(retrieval_id: str, store: JobStore = Depends(get_store)) -> schemas.AcquisitionSrInputResponse:
    config.ensure_workspace_dirs()
    result = acquisition_service.prepare_sr_input(
        store,
        retrieval_id,
        workspace_dir=config.WORKSPACE_DIR / "acquisitions",
        max_input_pixels=config.MAX_INPUT_PIXELS,
    )
    return schemas.AcquisitionSrInputResponse(**result)


# ---------------------------------------------------------------------------
# POST /upload
# ---------------------------------------------------------------------------

@router.post("/upload", response_model=schemas.UploadResponse)
def upload(
    file: UploadFile = File(...),
    input_scale: str = Form("raw_digital_number"),
    store: JobStore = Depends(get_store),
) -> schemas.UploadResponse:
    filename = file.filename or "upload"
    if not filename.lower().endswith(SUPPORTED_UPLOAD_EXTENSIONS):
        raise UnsupportedFileError(
            f"Unsupported file type for {filename!r}; expected a GeoTIFF ({', '.join(SUPPORTED_UPLOAD_EXTENSIONS)})."
        )

    config.ensure_workspace_dirs()
    dest = config.WORKSPACE_DIR / "uploads" / f"{uuid.uuid4().hex}_{filename}"
    with open(dest, "wb") as out:
        shutil.copyfileobj(file.file, out)

    try:
        record = pipeline.process_upload(
            store, dest, filename=filename, input_scale=input_scale, max_input_pixels=config.MAX_INPUT_PIXELS
        )
    except Exception:
        dest.unlink(missing_ok=True)  # do not keep a rejected upload on disk
        raise
    return schemas.UploadResponse(
        upload_id=record.upload_id,
        filename=record.filename,
        valid=True,
        band_names=list(record.band_names),
        width=record.width,
        height=record.height,
        crs=record.crs,
        resolution_m=record.resolution_m,
        input_scale=record.input_scale,
        validation_messages=["Upload accepted: required bands and shape present, CRS valid."],
    )


# ---------------------------------------------------------------------------
# GET /upload/{upload_id}/download
#
# Added for Step E: an acquisition-derived upload (frame.api.services.
# acquisition.prepare_sr_input) has no browser File object the frontend can
# already decode client-side for the "original" comparison view -- a manual
# /upload does (the browser already holds the bytes it sent). This lets the
# frontend fetch the exact same real file /sr/run actually used, so the
# "original" viewer's band order can never mismatch what was really run.
# ---------------------------------------------------------------------------

@router.get("/upload/{upload_id}/download")
def download_upload(upload_id: str, store: JobStore = Depends(get_store)) -> FileResponse:
    record = store.get_upload(upload_id)
    return FileResponse(record.file_path, media_type="image/tiff", filename=record.filename)


# ---------------------------------------------------------------------------
# POST /sr/run
# ---------------------------------------------------------------------------

@router.post("/sr/run", response_model=schemas.SRResultResponse)
def sr_run(
    request: schemas.SRRunRequest = Depends(get_sr_run_request),
    store: JobStore = Depends(get_store),
    model: Callable = Depends(get_model_callable),
    device: str = Depends(get_device),
) -> schemas.SRResultResponse:
    config.ensure_workspace_dirs()
    seed = request.seed if request.seed is not None else config.UNCERTAINTY_SEED
    job = pipeline.run_sr_job(
        store,
        request.upload_id,
        model=model,
        device=device,
        seed=seed,
        workspace_dir=config.WORKSPACE_DIR / "jobs",
        model_id=request.model,
        model_name=get_spec(request.model).model_name,
        tiling=TilingConfig(overlap=config.TILE_OVERLAP),
    )
    return schemas.SRResultResponse(**job.result)


# ---------------------------------------------------------------------------
# POST /jobs/sr, GET /jobs/{async_job_id} (Step I Section 6) -- an ADDITIVE,
# minimal background-job path alongside the unchanged, synchronous /sr/run
# above. See frame.api.services.async_jobs's module docstring.
# ---------------------------------------------------------------------------

@router.post("/jobs/sr", response_model=schemas.AsyncJobResponse)
def submit_sr_job(
    request: schemas.SRRunRequest = Depends(get_sr_run_request),
    store: JobStore = Depends(get_store),
    model: Callable = Depends(get_model_callable),
    device: str = Depends(get_device),
    registry: async_jobs.AsyncJobRegistry = Depends(get_async_registry),
) -> schemas.AsyncJobResponse:
    """Same request/model resolution as `POST /sr/run` -- an unavailable model still fails this
    call immediately (HTTP 503/422), never inside the background thread -- but the actual SR
    inference runs in a background thread and this returns "queued" right away."""
    config.ensure_workspace_dirs()
    seed = request.seed if request.seed is not None else config.UNCERTAINTY_SEED
    model_id, model_name = request.model, get_spec(request.model).model_name

    def work() -> Any:
        return pipeline.run_sr_job(
            store, request.upload_id, model=model, device=device, seed=seed,
            workspace_dir=config.WORKSPACE_DIR / "jobs", model_id=model_id, model_name=model_name,
            tiling=TilingConfig(overlap=config.TILE_OVERLAP),
        )

    record = registry.submit(work)
    return schemas.AsyncJobResponse(async_job_id=record.async_job_id, status=record.status, sr_job_id=record.sr_job_id, error=record.error, created_at=record.created_at)


@router.get("/jobs/{async_job_id}", response_model=schemas.AsyncJobResponse)
def get_async_job(async_job_id: str, registry: async_jobs.AsyncJobRegistry = Depends(get_async_registry)) -> schemas.AsyncJobResponse:
    record = registry.get(async_job_id)
    return schemas.AsyncJobResponse(async_job_id=record.async_job_id, status=record.status, sr_job_id=record.sr_job_id, error=record.error, created_at=record.created_at)


# ---------------------------------------------------------------------------
# POST /analysis/ndvi
# ---------------------------------------------------------------------------

@router.post("/analysis/ndvi", response_model=schemas.NDVIAnalysisResponse)
def analysis_ndvi(request: schemas.NDVIAnalysisRequest, store: JobStore = Depends(get_store)) -> schemas.NDVIAnalysisResponse:
    analysis = pipeline.run_ndvi_analysis_job(store, request.job_id)
    return schemas.NDVIAnalysisResponse(**analysis.result)


# ---------------------------------------------------------------------------
# GET /sr/result/{job_id}
# ---------------------------------------------------------------------------

@router.get("/sr/result/{job_id}", response_model=schemas.SRResultResponse)
def get_sr_result(job_id: str, store: JobStore = Depends(get_store)) -> schemas.SRResultResponse:
    job = store.get_job(job_id)
    return schemas.SRResultResponse(**job.result)


# ---------------------------------------------------------------------------
# GET /sr/download/{job_id}
# ---------------------------------------------------------------------------

@router.get("/sr/download/{job_id}")
def download_sr(job_id: str, store: JobStore = Depends(get_store)) -> FileResponse:
    job = store.get_job(job_id)
    return FileResponse(job.sr_mean_path, media_type="image/tiff", filename=f"sr_{job_id}.tif")


# ---------------------------------------------------------------------------
# GET /uncertainty/download/{job_id}
# ---------------------------------------------------------------------------

@router.get("/uncertainty/download/{job_id}")
def download_uncertainty(job_id: str, store: JobStore = Depends(get_store)) -> FileResponse:
    job = store.get_job(job_id)
    return FileResponse(job.sr_std_path, media_type="image/tiff", filename=f"uncertainty_{job_id}.tif")


# ---------------------------------------------------------------------------
# GET /analysis/{analysis_id}
# ---------------------------------------------------------------------------

@router.get("/analysis/{analysis_id}", response_model=schemas.NDVIAnalysisResponse)
def get_analysis(analysis_id: str, store: JobStore = Depends(get_store)) -> schemas.NDVIAnalysisResponse:
    analysis = store.get_analysis(analysis_id)
    return schemas.NDVIAnalysisResponse(**analysis.result)


# ---------------------------------------------------------------------------
# GET /analysis/download/{analysis_id}/{native-ndvi,sr-ndvi,ndvi-diff}
#
# Phase 8 (frontend) addition: the NDVI view needs the actual rasters, not
# just the scalar summary above -- frame.analysis already computes them
# (frame.api.services.pipeline.run_ndvi_analysis_job writes them out; no
# new scientific computation was added for this). Single-band GeoTIFFs,
# same download shape as /sr/download and /uncertainty/download.
# ---------------------------------------------------------------------------

@router.get("/analysis/download/{analysis_id}/native-ndvi")
def download_native_ndvi(analysis_id: str, store: JobStore = Depends(get_store)) -> FileResponse:
    analysis = store.get_analysis(analysis_id)
    return FileResponse(analysis.native_ndvi_path, media_type="image/tiff", filename=f"ndvi_native_{analysis_id}.tif")


@router.get("/analysis/download/{analysis_id}/sr-ndvi")
def download_sr_ndvi(analysis_id: str, store: JobStore = Depends(get_store)) -> FileResponse:
    analysis = store.get_analysis(analysis_id)
    return FileResponse(analysis.sr_ndvi_path, media_type="image/tiff", filename=f"ndvi_sr_{analysis_id}.tif")


@router.get("/analysis/download/{analysis_id}/ndvi-diff")
def download_ndvi_diff(analysis_id: str, store: JobStore = Depends(get_store)) -> FileResponse:
    analysis = store.get_analysis(analysis_id)
    return FileResponse(analysis.ndvi_diff_path, media_type="image/tiff", filename=f"ndvi_diff_{analysis_id}.tif")


# ---------------------------------------------------------------------------
# Products (Requirements 142 Requirement 11, Step I): a GIS-ready view of an
# already-completed SR job -- product_id == job_id (frame.api.services.product).
# ---------------------------------------------------------------------------

@router.get("/products/{job_id}", response_model=schemas.ProductResponse)
def get_product(job_id: str, request: Request, store: JobStore = Depends(get_store)) -> schemas.ProductResponse:
    view = product_service.build_product_view(store, job_id, base_url=str(request.base_url).rstrip("/"))
    return schemas.ProductResponse(
        job_id=view.job_id,
        status=view.status,
        cog_available=view.cog_path is not None,
        cog_validation=schemas.CogValidationSummary(**{k: v for k, v in dataclasses.asdict(view.cog_validation).items() if k != "path" and k != "checked"})
        if view.cog_validation is not None else None,
        evidence_assets=view.evidence_assets,
        analyses=view.analyses,
        stac_url=f"{str(request.base_url).rstrip('/')}/products/{job_id}/stac" if view.stac_item is not None else None,
    )


@router.get("/products/{job_id}/stac")
def get_product_stac(job_id: str, request: Request, store: JobStore = Depends(get_store)) -> JSONResponse:
    view = product_service.build_product_view(store, job_id, base_url=str(request.base_url).rstrip("/"))
    if view.stac_item is None:
        return JSONResponse({"job_id": job_id, "status": view.status, "detail": "STAC item is only available once the job has completed."}, status_code=200)
    return JSONResponse(view.stac_item)


@router.get("/products/{job_id}/cog")
def download_product_cog(job_id: str, store: JobStore = Depends(get_store)) -> FileResponse:
    job = store.get_job(job_id)
    cog_path = product_service.build_cog_for_job(job)
    return FileResponse(cog_path, media_type="image/tiff; application=geotiff; profile=cloud-optimized", filename=f"sr_{job_id}.cog.tif")


@router.get("/products/{job_id}/stability")
def get_product_stability(job_id: str, store: JobStore = Depends(get_store)) -> FileResponse:
    """Alias of `/uncertainty/download/{job_id}` under the product-registry naming (Step I
    Section 6: "a single existing route can satisfy more than one conceptual requirement")."""
    job = store.get_job(job_id)
    return FileResponse(job.sr_std_path, media_type="image/tiff", filename=f"stability_{job_id}.tif")


@router.get("/products/{job_id}/analysis")
def get_product_analyses(job_id: str, request: Request, store: JobStore = Depends(get_store)) -> JSONResponse:
    view = product_service.build_product_view(store, job_id, base_url=str(request.base_url).rstrip("/"))
    return JSONResponse({"job_id": job_id, "status": view.status, "analyses": view.analyses})


@router.get("/products/{job_id}/tiles/{z}/{x}/{y}.png")
def get_product_tile(job_id: str, z: int, x: int, y: int, layer: str = "rgb", store: JobStore = Depends(get_store)) -> Response:
    """A minimal, real in-process XYZ tile renderer (Step I; not titiler -- see
    frame.product.tiles's module docstring). ``layer``: "rgb" (default, the SR product's true
    colour) or "stability" (the TTA-stability overall band)."""
    job = store.get_job(job_id)
    if layer == "rgb":
        cog_path = product_service.build_cog_for_job(job)
        png = tiles.render_rgb_tile(cog_path, z, x, y)
    elif layer == "stability":
        png = tiles.render_single_band_tile(job.sr_std_path, z, x, y, band_index=5)  # overall_std is the 5th band
    else:
        raise UnsupportedTileLayerError(f"Unknown tile layer {layer!r}; supported layers are 'rgb' and 'stability'.")
    return Response(content=png, media_type="image/png")
