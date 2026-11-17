"""Step I Section 20: the final real end-to-end acceptance test.

    real AOI -> real CDSE scene discovery -> real scene selection -> real retrieval
      -> 12-band harmonisation -> RGBN adapter -> real Lite inference (real published weights,
      through the real FastAPI app, not a fake model) -> COG -> COG validation -> STAC generation
      -> STAC validation -> stability/evidence -> NDVI analysis -> API product registration
      -> tile access -> download

Requires real CDSE credentials (FRAME_CDSE_CLIENT_ID/SECRET); this makes real network calls.
Reuses the same real, already-verified Chennai-area AOI as Step F/G/H's own live tests.

No secret ever enters the output record: only ids, counts, shapes, paths and validation results
are written to experiments/final_demo/record.json.

Run: set -a && source .env && set +a && sen2sr_venv/bin/python experiments/final_demo/run_final_demo.py
"""

from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

OUT_DIR = REPO_ROOT / "experiments" / "final_demo"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def main() -> None:
    from fastapi.testclient import TestClient

    from frame.acquisition.contract import AcquisitionQuery
    from frame.acquisition.harmonise import harmonise, select_rgbn_adapter
    from frame.acquisition.providers.cdse import CDSEAuth
    from frame.acquisition.retrieve import retrieve_scene
    from frame.acquisition.search import search_scenes
    from frame.api.app import create_app
    from frame.data.bands import L2A_BANDS
    from frame.geospatial import write_geotiff
    from frame.geospatial.cog import validate_cog
    from frame.preprocessing.metadata import RasterMetadata
    from frame.product.stac import validate_stac_item

    record: Dict[str, Any] = {"created_at": _now_iso(), "steps": []}

    def log(name: str, **fields: Any) -> None:
        entry = {"step": name, **fields}
        record["steps"].append(entry)
        print(f"[{name}] " + ", ".join(f"{k}={v}" for k, v in fields.items()), flush=True)

    aoi_bbox = (80.204, 13.006, 80.2168, 13.0188)  # the same real, already-verified Chennai-area AOI as Step F/G/H
    query = AcquisitionQuery(bbox=aoi_bbox, start_date="2024-01-01", end_date="2024-04-01", max_cloud_cover=20.0)
    log("aoi", bbox=list(aoi_bbox))

    t0 = time.time()
    candidates = search_scenes(query)
    log("real_cdse_discovery", n_candidates=len(candidates), elapsed_s=round(time.time() - t0, 2))
    candidate = candidates[0]
    log("real_scene_selection", scene_id=candidate.id, cloud_cover=candidate.cloud_cover, datetime=candidate.datetime)

    t0 = time.time()
    retrieved = retrieve_scene(candidate, query, bands=L2A_BANDS, resolution_m=10.0, include_scl=True, cdse_auth=CDSEAuth())
    log("real_retrieval", elapsed_s=round(time.time() - t0, 2), grid=f"{retrieved.width}x{retrieved.height}", bands=len(retrieved.bands))

    product = harmonise(retrieved)
    log("harmonisation_12band", is_complete_l2a=product.is_complete_l2a(), validity_mask_coverage=round(product.validity_mask.coverage(), 4) if product.validity_mask else None)

    rgbn_array, band_names, excluded = select_rgbn_adapter(product)
    log("rgbn_adapter", bands=list(band_names), excluded_bands=excluded, shape=list(rgbn_array.shape))

    upload_path = OUT_DIR / "rgbn_input.tif"
    metadata = RasterMetadata(
        crs=product.crs, transform=product.transform, bounds=None, resolution_m=product.resolution_m,
        width=product.width, height=product.height, band_names=band_names,
        acquisition_timestamp=product.provenance.matched_acquisition_datetime, nodata_value=None,
        cloud_mask_coverage=None, sr_variant="frame.acquisition harmonisation -- RGBN adapter output (Step I final demo)",
    )
    write_geotiff(upload_path, rgbn_array, metadata)
    log("rgbn_geotiff_written", path=str(upload_path))

    client = TestClient(create_app())  # the REAL app: real get_model/get_device, no dependency overrides

    with open(upload_path, "rb") as f:
        upload_response = client.post("/upload", files={"file": ("rgbn_input.tif", f, "image/tiff")}, data={"input_scale": "reflectance"})
    assert upload_response.status_code == 200, upload_response.text
    upload_id = upload_response.json()["upload_id"]
    log("real_upload", upload_id=upload_id)

    t0 = time.time()
    sr_response = client.post("/sr/run", json={"upload_id": upload_id, "model": "lite"})
    assert sr_response.status_code == 200, sr_response.text
    sr_result = sr_response.json()
    job_id = sr_result["job_id"]
    log(
        "real_sr_inference", model=sr_result["model_id"], job_id=job_id, elapsed_s=round(time.time() - t0, 2),
        input_shape=sr_result["input_shape"], output_shape=sr_result["output_shape"], crs=sr_result["crs"],
        output_resolution_m=sr_result["resolution"]["sr_resolution_m"],
    )

    analysis_response = client.post("/analysis/ndvi", json={"job_id": job_id})
    assert analysis_response.status_code == 200, analysis_response.text
    analysis = analysis_response.json()
    log("real_ndvi_analysis", analysis_id=analysis["analysis_id"], status=analysis["comparison"]["status"])

    product_response = client.get(f"/products/{job_id}")
    assert product_response.status_code == 200, product_response.text
    product_view = product_response.json()
    log("api_product_registration", status=product_view["status"], cog_available=product_view["cog_available"], cog_valid=product_view["cog_validation"]["is_valid_cog"])

    stac_response = client.get(f"/products/{job_id}/stac")
    assert stac_response.status_code == 200
    stac_item = stac_response.json()
    stac_problems = validate_stac_item(stac_item)
    log("stac_generation_and_validation", stac_id=stac_item["id"], valid=stac_problems == [], problems=stac_problems)

    cog_response = client.get(f"/products/{job_id}/cog")
    assert cog_response.status_code == 200
    downloaded_cog_path = OUT_DIR / "downloaded_product.cog.tif"
    downloaded_cog_path.write_bytes(cog_response.content)
    cog_validation = validate_cog(downloaded_cog_path)
    log("cog_download_and_validation", path=str(downloaded_cog_path), bytes=len(cog_response.content), is_valid_cog=cog_validation.is_valid_cog, driver_layout=cog_validation.driver_reports_cog_layout)

    stability_response = client.get(f"/products/{job_id}/stability")
    assert stability_response.status_code == 200
    log("stability_evidence_download", bytes=len(stability_response.content))

    tile_response = client.get(f"/products/{job_id}/tiles/15/23684/15189.png")
    assert tile_response.status_code == 200
    log("tile_access", content_type=tile_response.headers["content-type"], bytes=len(tile_response.content), note="minimal in-process renderer, not titiler -- see frame.product.tiles")

    analysis_list_response = client.get(f"/products/{job_id}/analysis")
    assert analysis_list_response.status_code == 200
    log("api_analysis_listing", n_analyses=len(analysis_list_response.json()["analyses"]))

    record["summary"] = {
        "aoi_bbox": list(aoi_bbox), "scene_id": candidate.id, "model": sr_result["model_id"],
        "output_shape": sr_result["output_shape"], "output_crs": sr_result["crs"],
        "output_resolution_m": sr_result["resolution"]["sr_resolution_m"], "scale_factor": sr_result["resolution"]["scale_factor"],
        "cog_valid": cog_validation.is_valid_cog, "stac_valid": stac_problems == [],
        "asset_paths": {"rgbn_input": str(upload_path), "downloaded_cog": str(downloaded_cog_path)},
        "no_secrets_recorded": True,
    }
    (OUT_DIR / "record.json").write_text(json.dumps(record, indent=2, sort_keys=True, default=str), encoding="utf-8")
    print(f"\nDONE. Wrote {OUT_DIR / 'record.json'}")


if __name__ == "__main__":
    main()
