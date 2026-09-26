"""Pipeline orchestration for the FRAME API (Phase 7).

This module does not reimplement Phase 1-6 logic -- it calls the existing,
already-tested modules in sequence and packages their results:

    ingestion (an uploaded GeoTIFF, read via frame.geospatial.read_geotiff)
      -> preprocessing (frame.preprocessing.preprocess_rgbn)
      -> frozen SEN2SR + uncertainty in one call
         (frame.uncertainty.run_stochastic_uncertainty -- its TTA ensemble
          IS the model inference step; running it once gives both the SR
          mean prediction and the model-stability uncertainty, rather than
          calling the model a second time separately)
      -> geospatial attachment (frame.geospatial.derive_output_metadata,
         write_geotiff)
      -> self-consistency (frame.consistency.run_consistency_diagnostics)

`run_ndvi_analysis_job` similarly calls `frame.analysis.run_ndvi_analysis`
directly on a completed SR job's saved tensors.

The actual model object is always injected by the caller (`model=...`,
typically `frame.api.services.model.get_model(...)`) -- this module never
imports `sen2sr` or `mlstac` itself, keeping it testable with a small fake
callable (see frame/tests/test_api_pipeline.py) exactly like every prior
phase's own tests.
"""

from __future__ import annotations

import dataclasses
import time
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import numpy as np
import torch

from frame.analysis import run_ndvi_analysis
from frame.consistency import run_consistency_diagnostics
from frame.geospatial import RGBN_SCALE_FACTOR, derive_output_metadata, peek_geotiff_size, read_geotiff, write_geotiff
from frame.models import config as models_cfg
from frame.models.errors import ModelInferenceError
from frame.preprocessing import RGBN_BANDS, preprocess_rgbn
from frame.preprocessing.validation import validate_bands, validate_shape
from frame.tiling import TiledModel, TilingConfig
from frame.uncertainty import DEFAULT_TRANSFORMS, run_stochastic_uncertainty

from frame.api.errors import SceneTooLargeError
from frame.api.schemas import NDVI_DEMONSTRATION_NOTE, NDVI_STABILITY_CAVEAT, SCIENTIFIC_CAVEATS, SR_PRODUCT_DESCRIPTION, UNCERTAINTY_DISCLAIMER, UNCERTAINTY_LABEL
from frame.api.services.storage import AnalysisRecord, JobStore, SRJobRecord, UploadRecord

NATIVE_RESOLUTION_M = 10.0


def _output_checked(model: Callable[[torch.Tensor], torch.Tensor]) -> Callable[[torch.Tensor], torch.Tensor]:
    """``model`` with its output geometry checked on every call: same channels, exactly 4x the spatial size.
    A model that breaks the x4 contract (the wrong scale, a dropped band) is refused at its first tile with a named
    error, instead of failing later in an unrelated place -- or worse, producing a raster of the wrong size."""

    def checked(x: torch.Tensor) -> torch.Tensor:
        y = model(x)
        expected = (x.shape[0], x.shape[1], x.shape[2] * RGBN_SCALE_FACTOR, x.shape[3] * RGBN_SCALE_FACTOR)
        if not isinstance(y, torch.Tensor) or tuple(y.shape) != expected:
            got = tuple(y.shape) if isinstance(y, torch.Tensor) else type(y).__name__
            raise ModelInferenceError(f"The model returned {got} for an input of shape {tuple(x.shape)}; expected {expected} ({RGBN_SCALE_FACTOR}x, same channels).")
        return y

    return checked


def _preprocess(array: np.ndarray, geo_metadata: Any, *, band_names, input_scale: str):
    """The one preprocessing call of the API path, used at upload (so a bad scene is refused before it is
    stored) and again at run time (so a stale or hand-edited record cannot bypass it). `validate_content` turns on the
    Phase 8 checks: a scene with no valid pixel and a declared scale the values contradict are errors, not results."""
    return preprocess_rgbn(
        array,
        band_names=list(band_names),
        input_scale=input_scale,
        resolution_m=geo_metadata.resolution_m,
        nodata_value=geo_metadata.nodata_value,
        patch_size=None,
        require_square=False,
        crs=geo_metadata.crs,
        transform=geo_metadata.transform,
        bounds=geo_metadata.bounds,
        acquisition_timestamp=geo_metadata.acquisition_timestamp,
        require_geospatial=True,
        validate_content=True,
    )


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


# ---------------------------------------------------------------------------
# POST /upload
# ---------------------------------------------------------------------------

def process_upload(
    store: JobStore,
    file_path: Path,
    *,
    filename: str,
    input_scale: str,
    max_input_pixels: Optional[int] = None,
) -> UploadRecord:
    """Read and validate an uploaded Sentinel-2 L2A GeoTIFF. Does NOT run SR.

    Any height and width are accepted (larger scenes are tiled at run time,
    frame.tiling), up to ``max_input_pixels`` (None or 0: no limit) -- checked
    from the file header, before the pixels are read into memory.

    Raises the underlying `frame.geospatial`/`frame.preprocessing` error
    directly on invalid input (missing CRS, wrong band set, empty shape) --
    frame.api.errors.status_code_for maps these to HTTP 400 at the route
    layer, not duplicated here.
    """
    if max_input_pixels:
        height, width = peek_geotiff_size(file_path)
        if height * width > max_input_pixels:
            raise SceneTooLargeError(
                f"The scene is {width} x {height} = {height * width:,} pixels; this server accepts at most "
                f"{max_input_pixels:,} pixels per scene."
            )

    array, metadata = read_geotiff(file_path, require_crs=True)
    validate_bands(list(metadata.band_names), list(RGBN_BANDS))
    validate_shape(array, expected_size=None, require_square=False)
    _preprocess(array, metadata, band_names=metadata.band_names, input_scale=input_scale)   # raises on a scene with no valid pixel or a contradicted scale

    record = UploadRecord(
        upload_id=store.generate_id(),
        filename=filename,
        file_path=file_path,
        band_names=tuple(metadata.band_names),
        width=metadata.width,
        height=metadata.height,
        crs=metadata.crs,
        resolution_m=metadata.resolution_m,
        input_scale=input_scale,
        created_at=_now_iso(),
    )
    store.put_upload(record)
    return record


# ---------------------------------------------------------------------------
# POST /sr/run
# ---------------------------------------------------------------------------

def run_sr_job(
    store: JobStore,
    upload_id: str,
    *,
    model: Callable[[torch.Tensor], torch.Tensor],
    device: str,
    seed: int,
    workspace_dir: Path,
    model_id: str = models_cfg.MODEL_LITE,
    model_name: str = models_cfg.LITE_MODEL_NAME,
    tiling: TilingConfig = TilingConfig(),
) -> SRJobRecord:
    """Run the full FRAME pipeline for a previously-uploaded input.

    The scene may have any height and width: ``model`` (a single-tile
    ``model(x[None]) -> y`` callable) is wrapped in the tile engine
    (frame.tiling.TiledModel, configured by ``tiling``), which is what the
    uncertainty ensemble then runs. A scene that is exactly one tile is one
    call to ``model``, bit-identical to running it directly.

    ``model_id`` / ``model_name`` say which SR model the injected ``model``
    callable is; they are provenance only (recorded in the result and in the
    output GeoTIFF's ``FRAME_SR_VARIANT`` tag) -- the pipeline itself is
    identical for every model.
    """
    upload = store.get_upload(upload_id)

    array, geo_metadata = read_geotiff(upload.file_path, require_crs=True)
    preprocessed = _preprocess(array, geo_metadata, band_names=upload.band_names, input_scale=upload.input_scale)

    # preprocess_rgbn stamps every raster with the Lite model's name; overwrite it
    # with the model that actually ran so a Mamba result is never mislabelled.
    input_metadata = dataclasses.replace(preprocessed.metadata, sr_variant=model_name)

    X = preprocessed.tensor.to(device)
    t0 = time.time()
    tiled_model = TiledModel(_output_checked(model), tiling)
    # keep_per_member_predictions=False: the API reports mean and std only, and keeping
    # every ensemble member's full-size prediction would multiply memory on large scenes.
    uncertainty_result = run_stochastic_uncertainty(
        tiled_model, X, transforms=DEFAULT_TRANSFORMS, seed=seed, band_names=RGBN_BANDS,
        keep_per_member_predictions=False,
    )
    inference_seconds = time.time() - t0

    output_metadata = derive_output_metadata(
        input_metadata, scale_factor=RGBN_SCALE_FACTOR, output_band_names=input_metadata.band_names
    )

    # Models that can describe themselves (the isolated Mamba worker: weights
    # hash, parameter count, runtime versions, ...) add that to the metadata.
    describe = getattr(model, "describe", None)
    model_runtime = describe() if callable(describe) else None

    job_id = store.generate_id()
    job_dir = workspace_dir / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    mean_prediction = uncertainty_result.mean_prediction
    std_prediction = uncertainty_result.std_prediction

    sr_mean_path = job_dir / "sr_mean.tif"
    write_geotiff(sr_mean_path, mean_prediction.numpy(), output_metadata)

    overall_std = std_prediction.mean(dim=0).numpy()
    uncertainty_band_names = tuple(f"{b}_std" for b in RGBN_BANDS) + ("overall_std",)
    uncertainty_array = np.concatenate([std_prediction.numpy(), overall_std[None]], axis=0)
    uncertainty_metadata = dataclasses.replace(output_metadata, band_names=uncertainty_band_names, nodata_value=None)
    sr_std_path = job_dir / "uncertainty.tif"
    write_geotiff(sr_std_path, uncertainty_array, uncertainty_metadata)

    input_tensor_path = job_dir / "input_tensor.pt"
    sr_mean_tensor_path = job_dir / "sr_mean_tensor.pt"
    sr_std_tensor_path = job_dir / "sr_std_tensor.pt"
    torch.save(preprocessed.tensor.cpu(), input_tensor_path)
    torch.save(mean_prediction, sr_mean_tensor_path)
    torch.save(std_prediction, sr_std_tensor_path)

    consistency = run_consistency_diagnostics(
        lr=preprocessed.tensor,
        sr=mean_prediction,
        mask=preprocessed.mask.array,
        band_names=RGBN_BANDS,
        scale_factor=RGBN_SCALE_FACTOR,
    )

    result: Dict[str, Any] = {
        "job_id": job_id,
        "status": "completed",
        "upload_id": upload_id,
        "model_name": model_name,
        "model_id": model_id,
        "input_shape": list(preprocessed.tensor.shape),
        "output_shape": list(mean_prediction.shape),
        "resolution": {
            "native_resolution_m": NATIVE_RESOLUTION_M,
            "sr_resolution_m": NATIVE_RESOLUTION_M / RGBN_SCALE_FACTOR,
            "scale_factor": RGBN_SCALE_FACTOR,
            "description": SR_PRODUCT_DESCRIPTION,
        },
        "bands": list(RGBN_BANDS),
        "crs": output_metadata.crs,
        "uncertainty": {
            "label": UNCERTAINTY_LABEL,
            "scalar_summary": uncertainty_result.scalar_summary,
            "scalar_summary_definition": uncertainty_result.scalar_summary_definition,
            "overall_distribution": dataclasses.asdict(uncertainty_result.overall_distribution),
            "n": uncertainty_result.n,
            "seed": uncertainty_result.seed,
            "transform_names": list(uncertainty_result.transform_names),
            "disclaimer": UNCERTAINTY_DISCLAIMER,
        },
        "self_consistency": {
            "downsample_rmse": consistency.downsample_consistency.overall.rmse,
            "ndvi_discrepancy_mean_abs": consistency.ndvi_comparison.mean_abs_discrepancy,
            "b08_b04_ratio_discrepancy_mean_abs": consistency.b08_b04_ratio_comparison.mean_abs_discrepancy,
            "note": "Compares the SR output against its own LR input only -- not a ground-truth accuracy check.",
        },
        "metadata": {
            "inference_seconds": round(inference_seconds, 4),
            "device": device,
            "tiling": tiled_model.summary(),
            **({"model_runtime": model_runtime} if model_runtime else {}),
            "output_geospatial": {
                "crs": output_metadata.crs,
                "transform": list(output_metadata.transform),
                "bounds": list(output_metadata.bounds),
                "width": output_metadata.width,
                "height": output_metadata.height,
            },
            "preprocessing_mask_coverage": preprocessed.mask.coverage(),
        },
        "scientific_caveats": list(SCIENTIFIC_CAVEATS),
        "artifacts": {
            "sr_geotiff": str(sr_mean_path),
            "uncertainty_geotiff": str(sr_std_path),
        },
        "created_at": _now_iso(),
    }

    record = SRJobRecord(
        job_id=job_id,
        status="completed",
        upload_id=upload_id,
        input_tensor_path=input_tensor_path,
        sr_mean_path=sr_mean_path,
        sr_std_path=sr_std_path,
        sr_mean_tensor_path=sr_mean_tensor_path,
        sr_std_tensor_path=sr_std_tensor_path,
        input_metadata=input_metadata,
        output_metadata=output_metadata,
        result=result,
        error=None,
        created_at=result["created_at"],
    )
    store.put_job(record)
    return record


# ---------------------------------------------------------------------------
# POST /analysis/ndvi
# ---------------------------------------------------------------------------

def run_ndvi_analysis_job(store: JobStore, job_id: str) -> AnalysisRecord:
    """Run the Phase 6 NDVI analysis against a completed SR job's already-
    saved tensors -- no model inference, no re-fetch."""
    job = store.get_job(job_id)

    lr_reflectance = torch.load(job.input_tensor_path, weights_only=True)
    sr_mean_prediction = torch.load(job.sr_mean_tensor_path, weights_only=True)
    sr_std_prediction = torch.load(job.sr_std_tensor_path, weights_only=True)

    report = run_ndvi_analysis(
        lr_reflectance,
        sr_mean_prediction,
        sr_std_prediction,
        band_names=RGBN_BANDS,
        scale_factor=RGBN_SCALE_FACTOR,
        native_resolution_m=NATIVE_RESOLUTION_M,
    )

    analysis_id = store.generate_id()

    # Single-band NDVI GeoTIFFs -- the frontend's NDVI view needs actual
    # rasters (native NDVI, SR-derived NDVI, their difference), not just the
    # scalar summary above. `report.native_ndvi`/`sr_ndvi`/
    # `ndvi_comparison.absolute_difference_map` already hold these arrays
    # (frame.analysis computed them; nothing is recomputed here). Written
    # into the same per-job directory as the SR artifacts, using the exact
    # geospatial metadata each grid was already derived with.
    job_dir = job.input_tensor_path.parent
    native_ndvi_metadata = dataclasses.replace(job.input_metadata, band_names=("NDVI",), nodata_value=float("nan"))
    sr_ndvi_metadata = dataclasses.replace(job.output_metadata, band_names=("NDVI",), nodata_value=float("nan"))

    native_ndvi_path = job_dir / f"ndvi_native_{analysis_id}.tif"
    write_geotiff(native_ndvi_path, report.native_ndvi.ndvi.numpy()[None], native_ndvi_metadata)

    sr_ndvi_path = job_dir / f"ndvi_sr_{analysis_id}.tif"
    write_geotiff(sr_ndvi_path, report.sr_ndvi.ndvi.numpy()[None], sr_ndvi_metadata)

    ndvi_diff_path = job_dir / f"ndvi_diff_{analysis_id}.tif"
    write_geotiff(ndvi_diff_path, report.ndvi_comparison.absolute_difference_map.numpy()[None], native_ndvi_metadata)

    result: Dict[str, Any] = {
        "analysis_id": analysis_id,
        "job_id": job_id,
        "status": "completed",
        "ndvi_formula": report.metadata["ndvi_formula"],
        "native_resolution_m": report.metadata["native_resolution_m"],
        "sr_resolution_m": report.metadata["sr_resolution_m"],
        "comparison": {
            "status": report.ndvi_comparison.comparison.status.value,
            "valid_pixel_count": report.ndvi_comparison.comparison.valid_pixel_count,
            "mean_abs_difference": report.ndvi_comparison.comparison.mean_abs_difference,
            "rmse": report.ndvi_comparison.comparison.rmse,
            "max_abs_difference": report.ndvi_comparison.comparison.max_abs_difference,
            "resampling_method": report.ndvi_comparison.comparison.resampling_method,
        },
        "uncertainty_weighted_summary": {
            "status": report.uncertainty_weighted_summary.status.value,
            "correlation_uncertainty_vs_abs_diff": report.uncertainty_weighted_summary.correlation_uncertainty_vs_abs_diff,
            "uncertainty_weighted_mean_abs_diff": report.uncertainty_weighted_summary.uncertainty_weighted_mean_abs_diff,
            "unweighted_mean_abs_diff": report.uncertainty_weighted_summary.unweighted_mean_abs_diff,
        },
        # frame.analysis (original Phase 6) still calls the stability "the Phase 5 relative model-stability proxy"; the API says it in its own current words
        "scientific_caveats": [*(NDVI_STABILITY_CAVEAT if c.startswith("Uncertainty here is") else c for c in report.scientific_caveats), NDVI_DEMONSTRATION_NOTE],
        "metadata": report.metadata,
        "artifacts": {
            "native_ndvi_geotiff": str(native_ndvi_path),
            "sr_ndvi_geotiff": str(sr_ndvi_path),
            "ndvi_diff_geotiff": str(ndvi_diff_path),
        },
        "created_at": _now_iso(),
    }

    record = AnalysisRecord(
        analysis_id=analysis_id,
        job_id=job_id,
        native_ndvi_path=native_ndvi_path,
        sr_ndvi_path=sr_ndvi_path,
        ndvi_diff_path=ndvi_diff_path,
        result=result,
        created_at=result["created_at"],
    )
    store.put_analysis(record)
    return record
