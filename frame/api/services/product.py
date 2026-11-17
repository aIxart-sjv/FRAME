"""Product-registry service (Requirements 142 Requirement 11, Step I).

Composes a completed `SRJobRecord`'s already-real data (JobStore) into a GIS-ready product view:
a Cloud Optimized GeoTIFF written next to the job's existing plain GeoTIFF, and a STAC Item
linking to it plus whatever evidence assets already exist for that job (the stability GeoTIFF,
any NDVI analyses already run against it). This module never recomputes SR and never touches a
model -- it only packages what `frame.api.services.pipeline` already produced.

A "product" is not a new identity: `product_id == job_id`. There is no second ID space to keep in
sync with the job store (Step I Section 6: "a single existing route can satisfy more than one
conceptual requirement").
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

from frame.api.services.model import checkpoint_sha256
from frame.api.services.storage import AcquisitionRetrievalRecord, JobStore, SRJobRecord
from frame.geospatial import validate_cog, write_cog
from frame.geospatial.cog import CogValidationResult
from frame.models.selection import get_spec
from frame.product import build_stac_item

RADIOMETRIC_CONVENTION = "Surface reflectance as a fraction (not scaled digital numbers)."


@dataclass(frozen=True)
class AcquisitionProvenanceView:
    scene_id: Optional[str]
    platform: Optional[str]
    constellation: Optional[str]
    acquisition_datetime: Optional[str]
    source_provider: Optional[str]


@dataclass(frozen=True)
class ProductView:
    job_id: str
    status: str
    cog_path: Optional[Path]
    cog_validation: Optional[CogValidationResult]
    stac_item: Optional[Dict[str, Any]]
    evidence_assets: Dict[str, str]  # name -> real file path, only assets that actually exist
    analyses: List[Dict[str, Any]]


def _acquisition_provenance(store: JobStore, job: SRJobRecord) -> AcquisitionProvenanceView:
    """Real scene id/provider/platform/datetime if this job's input came from a real
    frame.acquisition retrieval; every field is ``None`` (never guessed) for a manually-uploaded
    GeoTIFF, which genuinely has no such provenance."""
    upload = store.get_upload(job.upload_id)
    if upload.source_acquisition_retrieval_id is None:
        return AcquisitionProvenanceView(None, None, None, None, None)

    retrieval: AcquisitionRetrievalRecord = store.get_acquisition_retrieval(upload.source_acquisition_retrieval_id)
    providers = sorted(set(retrieval.provenance.band_sources.values()))

    platform = None
    try:
        search = store.get_acquisition_search(retrieval.search_id)
        candidate = search.candidates.get(retrieval.scene_id)
        if candidate is not None:
            platform = candidate.platform
    except Exception:  # noqa: BLE001 -- the search record may have expired; platform stays None, never guessed
        pass

    return AcquisitionProvenanceView(
        scene_id=retrieval.scene_id,
        platform=platform,
        constellation="sentinel-2",
        acquisition_datetime=retrieval.provenance.matched_acquisition_datetime,
        source_provider=", ".join(providers) if providers else None,
    )


def _cog_path_for(job: SRJobRecord) -> Path:
    return job.sr_mean_path.with_name(job.sr_mean_path.stem + ".cog.tif")


def build_cog_for_job(job: SRJobRecord) -> Path:
    """Write (once -- cached on every later call for the same job) the SR mean prediction as a
    COG next to the existing plain GeoTIFF. Reuses the exact same tensor and metadata
    `frame.api.services.pipeline.run_sr_job` already produced and saved; no SR is recomputed."""
    cog_path = _cog_path_for(job)
    if not cog_path.exists():
        mean_prediction = torch.load(job.sr_mean_tensor_path, weights_only=True)
        write_cog(
            cog_path, mean_prediction.numpy(), job.output_metadata,
            extra_tags={"FRAME_MODEL_ID": str(job.result.get("model_id", "")), "FRAME_JOB_ID": job.job_id},
        )
    return cog_path


def build_product_view(store: JobStore, job_id: str, *, base_url: str = "") -> ProductView:
    """The full GIS-ready view of one job: COG (written on first request), its own structural
    validation, a STAC Item, and links to whatever evidence already exists. A job that has not
    completed yet returns a view with everything but ``job_id``/``status`` as ``None``/empty --
    never a fabricated placeholder product."""
    job = store.get_job(job_id)
    if job.status != "completed":
        return ProductView(job_id=job_id, status=job.status, cog_path=None, cog_validation=None, stac_item=None, evidence_assets={}, analyses=[])

    cog_path = build_cog_for_job(job)
    cog_validation = validate_cog(cog_path)

    model_id = job.result["model_id"]
    spec = get_spec(model_id)
    model_runtime = (job.result.get("metadata") or {}).get("model_runtime") or {}
    checkpoint_hash = model_runtime.get("weights_sha256") or checkpoint_sha256(model_id)

    provenance = _acquisition_provenance(store, job)

    evidence_assets: Dict[str, str] = {"stability_geotiff": str(job.sr_std_path)}
    analyses = store.get_analyses_for_job(job_id)
    analysis_summaries: List[Dict[str, Any]] = []
    assets: Dict[str, Any] = {
        "sr_cog": {
            "href": f"{base_url}/products/{job_id}/cog", "type": "image/tiff; application=geotiff; profile=cloud-optimized",
            "roles": ["data"], "title": "RGBN SR product (reconstructed 2.5 m grid)",
        },
        "stability": {
            "href": f"{base_url}/products/{job_id}/stability", "type": "image/tiff; application=geotiff",
            "roles": ["metadata"], "title": "TTA stability -- reconstruction-variation diagnostic (uncalibrated)",
        },
    }
    for a in analyses:
        evidence_assets[f"analysis_{a.analysis_id}_sr_ndvi"] = str(a.sr_ndvi_path)
        evidence_assets[f"analysis_{a.analysis_id}_native_ndvi"] = str(a.native_ndvi_path)
        analysis_summaries.append({"analysis_id": a.analysis_id, "created_at": a.created_at, "result": a.result})
        assets[f"ndvi_{a.analysis_id}"] = {
            "href": f"{base_url}/analysis/{a.analysis_id}", "type": "application/json",
            "roles": ["data"], "title": "NDVI self-consistency analysis (native vs. SR grid)",
        }

    stac_item = build_stac_item(
        item_id=f"frame-sr-{job_id}",
        output_metadata=job.output_metadata,
        model_id=model_id, model_name=job.result["model_name"], architecture=spec.architecture,
        checkpoint_sha256=checkpoint_hash,
        input_resolution_m=job.result["resolution"]["native_resolution_m"],
        output_resolution_m=job.result["resolution"]["sr_resolution_m"],
        scale_factor=job.result["resolution"]["scale_factor"],
        bands=job.result["bands"], radiometric_convention=RADIOMETRIC_CONVENTION,
        processing_timestamp=job.result["created_at"], assets=assets,
        scene_id=provenance.scene_id, platform=provenance.platform, constellation=provenance.constellation,
        acquisition_datetime=provenance.acquisition_datetime, source_provider=provenance.source_provider,
        scientific_caveats=job.result.get("scientific_caveats", []),
    )

    return ProductView(
        job_id=job_id, status=job.status, cog_path=cog_path, cog_validation=cog_validation,
        stac_item=stac_item, evidence_assets=evidence_assets, analyses=analysis_summaries,
    )
