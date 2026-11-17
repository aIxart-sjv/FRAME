"""Acquisition service: thin orchestration between the real `frame.acquisition`
package and the API's JobStore, exactly like `frame.api.services.pipeline` does
for the SR pipeline. No acquisition logic lives here -- only request/response
shaping and the search-result cache that lets a scene selected by id in
`POST /acquisition/search` be retrieved by that same id in
`POST /acquisition/retrieve` without the frontend re-sending the full STAC
item.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from frame.acquisition.contract import AcquisitionQuery, SceneCandidate
from frame.acquisition.errors import InvalidQueryError, NoScenesFoundError, SceneNotFoundError
from frame.acquisition.geometry import bbox_from_geojson
from frame.acquisition.harmonise import harmonise
from frame.acquisition.providers import cdse as cdse_provider
from frame.acquisition.retrieve import retrieve_scene as real_retrieve_scene
from frame.acquisition.search import search_scenes as real_search_scenes
from frame.api.errors import AcquisitionBandsInsufficientForModelError
from frame.api.services import pipeline as pipeline_service
from frame.api.services.storage import (
    AcquisitionRetrievalRecord,
    AcquisitionSearchRecord,
    JobStore,
    UploadRecord,
)
from frame.data.bands import RGBN_BANDS, canonical_bands, native_resolution_m, select_bands
from frame.data.errors import ContractError
from frame.geospatial.geotiff import read_geotiff, write_geotiff
from frame.geospatial.metadata import bounds_from_transform
from frame.preprocessing.metadata import RasterMetadata


def _now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def build_query(
    *,
    bbox: Optional[Sequence[float]],
    aoi_geojson: Optional[Dict[str, Any]],
    start_date: str,
    end_date: str,
    max_cloud_cover: float,
    max_results: int,
) -> AcquisitionQuery:
    """The real `AcquisitionQuery` for a request that gave either a bbox or a
    GeoJSON AOI (never neither -- enforced by the request schema)."""
    resolved_bbox = tuple(bbox) if bbox is not None else bbox_from_geojson(aoi_geojson)
    return AcquisitionQuery(
        bbox=resolved_bbox,
        start_date=start_date,
        end_date=end_date,
        max_cloud_cover=max_cloud_cover,
        max_results=max_results,
    )


def search(store: JobStore, query: AcquisitionQuery) -> AcquisitionSearchRecord:
    """Real CDSE STAC search. A zero-result search is a normal outcome, not an
    error: it is cached and returned as an empty, real search record rather
    than raising `NoScenesFoundError` up to the route."""
    try:
        candidates: List[SceneCandidate] = real_search_scenes(query)
    except NoScenesFoundError:
        candidates = []

    record = AcquisitionSearchRecord(
        search_id=JobStore.generate_id(),
        query=query,
        candidates={candidate.id: candidate for candidate in candidates},
        created_at=_now_iso(),
    )
    store.put_acquisition_search(record)
    return record


def get_candidate(store: JobStore, search_id: str, scene_id: str) -> SceneCandidate:
    record = store.get_acquisition_search(search_id)
    try:
        return record.candidates[scene_id]
    except KeyError:
        raise SceneNotFoundError(
            f"Scene {scene_id!r} was not part of search {search_id!r}'s real results."
        ) from None


def retrieve(
    store: JobStore,
    *,
    search_id: str,
    scene_id: str,
    bands: Sequence[str],
    resolution_m: float,
    include_scl: bool,
    workspace_dir: Path,
) -> AcquisitionRetrievalRecord:
    """Real multi-resolution pixel retrieval for a previously-searched scene,
    written to a real GeoTIFF artifact. Does not register an UploadRecord or
    touch `/sr/run` -- see docs/ACQUISITION.md §5."""
    search_record = store.get_acquisition_search(search_id)
    candidate = get_candidate(store, search_id, scene_id)

    for band in bands:
        try:
            native_resolution_m(band)  # raises for a malformed name or a non-L2A band (e.g. B10)
        except ContractError as exc:
            raise InvalidQueryError(f"Invalid band {band!r} in {list(bands)!r}: {exc}") from exc

    retrieved = real_retrieve_scene(
        candidate,
        search_record.query,
        bands=bands,
        resolution_m=resolution_m,
        include_scl=include_scl,
        cdse_auth=cdse_provider.CDSEAuth(),
    )

    retrieval_id = JobStore.generate_id()
    workspace_dir.mkdir(parents=True, exist_ok=True)

    ordered_bands = list(bands)
    stacked = np.stack([retrieved.bands[band] for band in ordered_bands], axis=0).astype("float32")
    real_bounds = bounds_from_transform(retrieved.transform, retrieved.width, retrieved.height)
    bands_metadata = RasterMetadata(
        crs=retrieved.crs,
        transform=retrieved.transform,
        bounds=real_bounds,
        resolution_m=retrieved.resolution_m,
        width=retrieved.width,
        height=retrieved.height,
        band_names=tuple(ordered_bands),
        acquisition_timestamp=retrieved.provenance.matched_acquisition_datetime,
        nodata_value=None,
        cloud_mask_coverage=None,
        sr_variant="frame.acquisition retrieval -- real reflectance, not an SR product",
    )
    bands_path = workspace_dir / f"{retrieval_id}_bands.tif"
    write_geotiff(bands_path, stacked, bands_metadata)

    scl_path: Optional[Path] = None
    if include_scl and retrieved.scl is not None:
        scl_metadata = RasterMetadata(
            crs=retrieved.crs,
            transform=retrieved.transform,
            bounds=real_bounds,
            resolution_m=retrieved.resolution_m,
            width=retrieved.width,
            height=retrieved.height,
            band_names=("SCL",),
            acquisition_timestamp=retrieved.provenance.matched_acquisition_datetime,
            nodata_value=None,
            cloud_mask_coverage=None,
            sr_variant="frame.acquisition retrieval -- Scene Classification Layer, raw class codes",
        )
        scl_path = workspace_dir / f"{retrieval_id}_scl.tif"
        write_geotiff(scl_path, retrieved.scl[None, :, :].astype("float32"), scl_metadata)

    # Step F: the explicit 12-band harmonisation wrapper (frame.acquisition.harmonise) -- never
    # a second resampling pass (retrieve_scene already did the one real resampling), only a real
    # SCL-derived validity mask and honest per-band radiometric-method provenance, both exposed
    # here since this is the point in the API where "harmonised-input metadata" genuinely exists.
    harmonised = harmonise(retrieved)

    result: Dict[str, Any] = {
        "retrieval_id": retrieval_id,
        "search_id": search_id,
        "scene_id": scene_id,
        "bands": ordered_bands,
        "band_native_resolution_m": dict(retrieved.band_native_resolution_m),
        "band_sources": dict(retrieved.provenance.band_sources),
        "band_radiometric_method": dict(harmonised.band_radiometric_method),
        "is_complete_l2a": harmonised.is_complete_l2a(),
        "validity_mask_coverage": harmonised.validity_mask.coverage() if harmonised.validity_mask is not None else None,
        "scl_included": scl_path is not None,
        "crs": retrieved.crs,
        "resolution_m": retrieved.resolution_m,
        "width": retrieved.width,
        "height": retrieved.height,
        "retrieval_provider": retrieved.provenance.retrieval_provider,
        "retrieval_endpoint": retrieved.provenance.retrieval_endpoint,
        "matched_acquisition_datetime": retrieved.provenance.matched_acquisition_datetime,
        "retrieved_at_utc": retrieved.provenance.retrieved_at_utc,
        "warnings": list(retrieved.provenance.warnings),
        "artifacts": {
            "bands_geotiff": f"/acquisition/download/{retrieval_id}/bands",
            **({"scl_geotiff": f"/acquisition/download/{retrieval_id}/scl"} if scl_path is not None else {}),
        },
    }

    record = AcquisitionRetrievalRecord(
        retrieval_id=retrieval_id,
        search_id=search_id,
        scene_id=scene_id,
        bands_geotiff_path=bands_path,
        scl_geotiff_path=scl_path,
        provenance=retrieved.provenance,
        result=result,
        created_at=_now_iso(),
    )
    store.put_acquisition_retrieval(record)
    return record


def prepare_sr_input(
    store: JobStore,
    retrieval_id: str,
    *,
    workspace_dir: Path,
    max_input_pixels: Optional[int] = None,
) -> Dict[str, Any]:
    """Build a real 4-band RGBN GeoTIFF from a completed acquisition retrieval and register it
    exactly like a manually-uploaded file (`frame.api.services.pipeline.process_upload`) -- the
    existing `/sr/run` then runs completely unmodified on the resulting ``upload_id``. This is the
    only "SR input adapter" in the acquisition path; it does not reimplement upload validation or
    SR inference.

    Raises `AcquisitionRetrievalNotFoundError` (from `store.get_acquisition_retrieval`) for an
    unknown retrieval, `UnreadableRasterError` if its real GeoTIFF cannot be read back, and
    `AcquisitionBandsInsufficientForModelError` if the retrieval's real bands do not cover the RGBN
    set every current SR model requires (frame/models -- no model here accepts or produces genuine
    12-band SR; see docs/CLAIMS.md). Any additional real bands the retrieval has beyond RGBN are
    never silently dropped: they stay in the retrieval's own bands GeoTIFF
    (`GET /acquisition/download/{retrieval_id}/bands`, unaffected by this call) and are named here
    in ``excluded_bands``, preserved for a future multispectral SR pathway.
    """
    retrieval = store.get_acquisition_retrieval(retrieval_id)

    array, metadata = read_geotiff(retrieval.bands_geotiff_path, require_crs=True)
    available = canonical_bands(metadata.band_names)

    missing = [b for b in RGBN_BANDS if b not in available]
    if missing:
        raise AcquisitionBandsInsufficientForModelError(
            f"Retrieval {retrieval_id!r} has real bands {list(available)}, missing {missing} "
            f"required by the current RGBN-only SR models ({list(RGBN_BANDS)}). Retrieve those "
            "bands (POST /acquisition/retrieve) and try again."
        )

    rgbn_array = select_bands(array, available, RGBN_BANDS)
    excluded_bands = [b for b in available if b not in RGBN_BANDS]

    sr_input_metadata = RasterMetadata(
        crs=metadata.crs,
        transform=metadata.transform,
        bounds=metadata.bounds,
        resolution_m=metadata.resolution_m,
        width=metadata.width,
        height=metadata.height,
        band_names=RGBN_BANDS,
        acquisition_timestamp=metadata.acquisition_timestamp,
        nodata_value=metadata.nodata_value,
        cloud_mask_coverage=metadata.cloud_mask_coverage,
        sr_variant="frame.acquisition retrieval -- real reflectance RGBN subset for SR input",
    )
    workspace_dir.mkdir(parents=True, exist_ok=True)
    sr_input_path = workspace_dir / f"{retrieval_id}_sr_input.tif"
    write_geotiff(sr_input_path, rgbn_array, sr_input_metadata)

    upload_record: UploadRecord = pipeline_service.process_upload(
        store,
        sr_input_path,
        filename=f"acquisition_{retrieval.scene_id}.tif",
        # Every acquisition provider returns real reflectance already (CDSE's own REFLECTANCE
        # units, or Planetary Computer's raw DN converted by frame.acquisition.providers) --
        # never raw digital numbers -- so this declaration is a real fact, not a default guess.
        input_scale="reflectance",
        max_input_pixels=max_input_pixels,
        source_acquisition_retrieval_id=retrieval_id,
    )

    note = (
        "Real B04/B03/B02/B08 reflectance retrieved for this scene, used as-is (no synthesis) as "
        "input to the existing RGBN SR pipeline."
    )
    if excluded_bands:
        note += (
            f" {len(excluded_bands)} additional real band(s) also retrieved for this scene "
            f"({', '.join(excluded_bands)}) were not used for this SR run -- they remain in the "
            f"retrieval's own bands GeoTIFF (GET /acquisition/download/{retrieval_id}/bands), not "
            "discarded, pending a future multispectral SR pathway."
        )

    return {
        "upload_id": upload_record.upload_id,
        "filename": upload_record.filename,
        "valid": True,
        "band_names": list(upload_record.band_names),
        "width": upload_record.width,
        "height": upload_record.height,
        "crs": upload_record.crs,
        "resolution_m": upload_record.resolution_m,
        "input_scale": upload_record.input_scale,
        "validation_messages": ["Upload accepted: required RGBN bands present and real (frame.acquisition)."],
        "retrieval_id": retrieval_id,
        "search_id": retrieval.search_id,
        "scene_id": retrieval.scene_id,
        "provider": retrieval.provenance.retrieval_provider,
        "matched_acquisition_datetime": retrieval.provenance.matched_acquisition_datetime,
        "excluded_bands": excluded_bands,
        "note": note,
    }
