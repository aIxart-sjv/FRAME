"""Phase 2 validation -- arbitrary-size tiling, measured on a REAL Sentinel-2 scene.

What this measures (nothing here is an accuracy claim about the SR product):

  A. The overlap default is checked, not copied from upstream: a sweep over overlap
     values on a real rectangular scene, for the fast Lite model at full size and for
     the real Mamba model on a smaller crop. Reported per overlap: tile count, time,
     the tile-seam diagnostic, and how far each reconstruction is from the
     largest-overlap one (the most heavily blended reconstruction available).
  B. The real Mamba model on a real multi-tile rectangular scene through the tile
     engine: scene size, tile count, worker start, per-tile time, total time, peak
     GPU memory, output shape, determinism, hard-constraint consistency.
  C. The same, through the real HTTP API (the ensemble multiplies the tile calls by 6),
     for Mamba and for Lite.
  D. Memory: peak resident memory of the whole pipeline (Lite, so the number is not
     dominated by model time) at growing scene sizes, to justify the API's upload
     guard (FRAME_API_MAX_INPUT_PIXELS).

The real scene is fetched once with `cubo` (the same call as experiments/baseline, at a
larger edge size), cached OUTSIDE the repository, and identified in the metadata by
AOI/date/bounds and a SHA-256 of the cached raster. Sentinel-2 catalogue contents drift
over time (see experiments/end_to_end); the hash is what says whether a re-run used the
same pixels.

Usage
-----
    sen2sr_venv/bin/python experiments/tiling/run_validation.py [--sections A,B,C,D] [--metadata-dir DIR]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))

from frame.consistency.downsample import downsample_to_lr_grid  # noqa: E402
from frame.geospatial import read_geotiff, write_geotiff  # noqa: E402
from frame.models import config as models_cfg  # noqa: E402
from frame.models.mamba_client import MambaWorkerClient, check_mamba_availability  # noqa: E402
from frame.preprocessing import RGBN_BANDS  # noqa: E402
from frame.preprocessing.metadata import RasterMetadata  # noqa: E402
from frame.tiling import TiledModel, TilingConfig, plan_tiles, run_tiled  # noqa: E402

BASE_DIR = Path(__file__).resolve().parent
CACHE_DIR = Path(os.environ.get("FRAME_TILING_CACHE_DIR", Path.home() / ".cache" / "frame_tiling"))

# Same AOI/date window as experiments/baseline, at a larger edge size.
AOI_LAT, AOI_LON = 39.49152740347753, -0.4308725142800361
SCENE_START, SCENE_END = "2023-01-15", "2023-01-16"
FETCH_EDGE_PX = 1024
SCENE_TIME_INDEX = 0
BANDS = ["B04", "B03", "B02", "B08"]
RESOLUTION_M = 10.0

# The rectangular, non-multiple-of-128 crops used below (row offset, col offset, height, width).
CROP_FULL = (20, 10, 511, 777)      # the shape named in the phase brief
CROP_MAMBA_SWEEP = (100, 100, 256, 384)
OVERLAPS_LITE = [0, 8, 16, 32, 48, 64]
OVERLAPS_MAMBA = [0, 16, 32, 64]


# ---------------------------------------------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------------------------------------------

def git_commit() -> Optional[str]:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=_REPO_ROOT, capture_output=True, text=True, timeout=5).stdout.strip() or None
    except Exception:
        return None


def nvidia_smi(query: str) -> str:
    return subprocess.run(["nvidia-smi", f"--query-{query}", "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout.strip()


def process_gpu_mib(pid: int) -> Optional[int]:
    for line in nvidia_smi("compute-apps=pid,used_memory").splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) == 2 and parts[0] == str(pid):
            return int(parts[1])
    return None


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def crop_scene(array: np.ndarray, meta: RasterMetadata, crop: Tuple[int, int, int, int]) -> Tuple[np.ndarray, RasterMetadata]:
    row_off, col_off, h, w = crop
    a, _, c, _, e, f = meta.transform
    x0, y0 = c + col_off * a, f + row_off * e
    cropped = np.ascontiguousarray(array[:, row_off : row_off + h, col_off : col_off + w])
    new_meta = RasterMetadata(
        crs=meta.crs, transform=(a, 0.0, x0, 0.0, e, y0), bounds=(x0, y0 + h * e, x0 + w * a, y0), resolution_m=meta.resolution_m,
        width=w, height=h, band_names=meta.band_names, acquisition_timestamp=meta.acquisition_timestamp,
        nodata_value=meta.nodata_value, cloud_mask_coverage=meta.cloud_mask_coverage, sr_variant=None,
    )
    return cropped, new_meta


def real_scene() -> Tuple[np.ndarray, RasterMetadata, Dict[str, Any]]:
    """The cached real 1024 x 1024 RGBN reflectance scene (fetched once with cubo)."""
    path = CACHE_DIR / f"real_scene_{FETCH_EDGE_PX}.tif"
    identity: Dict[str, Any] = {}
    if not path.exists():
        import cubo

        print(f"[scene] fetching {FETCH_EDGE_PX}x{FETCH_EDGE_PX} px at ({AOI_LAT}, {AOI_LON}) {SCENE_START}..{SCENE_END} via cubo ...", flush=True)
        da = cubo.create(lat=AOI_LAT, lon=AOI_LON, collection="sentinel-2-l2a", bands=BANDS, start_date=SCENE_START,
                         end_date=SCENE_END, edge_size=FETCH_EDGE_PX, resolution=RESOLUTION_M)
        if da.sizes["time"] == 0:
            sys.exit("No Sentinel-2 item found for the configured window; cannot build a real scene.")
        identity["catalogue_items_returned"] = int(da.sizes["time"])
        identity["time_index_used"] = SCENE_TIME_INDEX
        identity["acquisition_timestamp"] = str(da["time"].values[SCENE_TIME_INDEX])
        raw = da[SCENE_TIME_INDEX].compute().to_numpy()
        nan_fraction = float(np.isnan(raw).mean())
        array = np.nan_to_num(raw / 10_000, nan=0.0, posinf=0.0, neginf=0.0).astype("float32")
        epsg = da.attrs["epsg"]
        x, y = da["x"].values, da["y"].values
        origin_x, origin_y = float(x[0]) - RESOLUTION_M / 2, float(y[0]) + RESOLUTION_M / 2
        h, w = len(y), len(x)
        meta = RasterMetadata(
            crs=f"EPSG:{epsg}", transform=(RESOLUTION_M, 0.0, origin_x, 0.0, -RESOLUTION_M, origin_y),
            bounds=(origin_x, origin_y - h * RESOLUTION_M, origin_x + w * RESOLUTION_M, origin_y), resolution_m=RESOLUTION_M,
            width=w, height=h, band_names=tuple(RGBN_BANDS), acquisition_timestamp=identity["acquisition_timestamp"],
            nodata_value=None, cloud_mask_coverage=1.0, sr_variant=None,
        )
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        write_geotiff(path, array, meta)
        identity["nan_fraction_replaced_by_zero"] = nan_fraction
    array, meta = read_geotiff(path, require_crs=True)
    identity.update(
        {
            "source": "Sentinel-2 L2A via cubo (Microsoft Planetary Computer STAC)", "aoi_lat_lon": [AOI_LAT, AOI_LON],
            "date_window": [SCENE_START, SCENE_END], "bands": list(RGBN_BANDS), "size_px": [int(array.shape[1]), int(array.shape[2])],
            "crs": meta.crs, "bounds": list(meta.bounds), "reflectance_min_max": [float(array.min()), float(array.max())],
            "cache_file": str(path), "sha256": sha256_of(path),
            "scale": "DN / 10000, as experiments/baseline (values are already reflectance in the cached file)",
        }
    )
    return array, meta, identity


def rgb_reflectance_ok(array: np.ndarray) -> bool:
    return bool(np.isfinite(array).all() and array.min() >= -0.1 and array.max() <= 6.5535)


def diff_stats(a: torch.Tensor, b: torch.Tensor) -> Dict[str, float]:
    d = (a - b).double()
    return {"rmse": round(float(d.pow(2).mean().sqrt()), 6), "mean_abs": round(float(d.abs().mean()), 6), "max_abs": round(float(d.abs().max()), 6)}


def seam_dict(result) -> Optional[Dict[str, Any]]:
    seam = result.seam_diagnostic
    if seam is None:
        return None
    return {k: (round(v, 6) if isinstance(v, float) else v) for k, v in seam.as_dict().items() if k != "note"}


# ---------------------------------------------------------------------------------------------------------------
# A. overlap verification
# ---------------------------------------------------------------------------------------------------------------

def section_overlap(scene_full: torch.Tensor, scene_sweep: torch.Tensor, with_mamba: bool) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "method": (
            "Each reconstruction is compared with the largest-overlap reconstruction of the same scene (overlap 64), the most heavily "
            "blended one available. 'seam_diagnostic' is the adjacent-tile disagreement on the shared pixels before blending; it is "
            "biased upward (border zone) and only comparable at equal overlap. Neither is an accuracy measure against any reference."
        )
    }
    from frame.api.services import model as model_service

    lite = model_service.get_model("cuda" if torch.cuda.is_available() else "cpu", model_name="lite")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    def sweep(model, scene, overlaps, label, reference_overlap=64):
        runs, recon = {}, {}
        for ov in overlaps:
            res = run_tiled(model, scene.to(device) if label == "lite" else scene, TilingConfig(overlap=ov))
            recon[ov] = res.sr
            runs[ov] = {
                "tile_count": res.plan.tile_count, "grid": [res.plan.n_rows, res.plan.n_cols], "seconds": round(res.total_seconds, 3),
                "seam_diagnostic": seam_dict(res),
            }
        ref = recon[reference_overlap]
        for ov in overlaps:
            runs[ov]["vs_overlap_64"] = diff_stats(recon[ov], ref) if ov != reference_overlap else None
        return {"scene_shape": list(scene.shape[1:]), "reference_overlap": reference_overlap, "by_overlap": {str(k): v for k, v in runs.items()}}

    print("[A] Lite overlap sweep on the full real scene ...", flush=True)
    out["lite"] = sweep(lite, scene_full, OVERLAPS_LITE, "lite")
    if with_mamba:
        print("[A] Mamba overlap sweep on a real crop ...", flush=True)
        with MambaWorkerClient(device="cuda") as client:
            out["mamba"] = sweep(client, scene_sweep, OVERLAPS_MAMBA, "mamba")
    return out


# ---------------------------------------------------------------------------------------------------------------
# B. real Mamba, multi-tile, through the engine
# ---------------------------------------------------------------------------------------------------------------

def section_engine_mamba(scene: torch.Tensor) -> Dict[str, Any]:
    config = TilingConfig()
    plan = plan_tiles(scene.shape[1], scene.shape[2], config)
    out: Dict[str, Any] = {
        "scene_shape_hw": list(scene.shape[1:]), "tile_size": config.tile_size, "overlap": config.overlap, "stride": config.stride,
        "tile_grid": [plan.n_rows, plan.n_cols], "tile_count": plan.tile_count, "padded_tiles": sum(t.is_padded for t in plan.tiles),
    }
    client = MambaWorkerClient(device="cuda")
    try:
        started = time.perf_counter()
        info = client.start()
        out["worker_start_seconds"] = round(time.perf_counter() - started, 3)
        pid = info["worker_pid"]
        out["worker_gpu_memory_after_load_mib"] = process_gpu_mib(pid)
        pids, peaks = [], []

        def on_tile(done, total, tile):
            pids.append(client._proc.pid)
            peaks.append(client.describe()["last_request"]["peak_memory_mib"])

        wall = time.perf_counter()
        result = run_tiled(client, scene, config, on_tile=on_tile)
        out["total_wall_seconds"] = round(time.perf_counter() - wall, 3)
        secs = result.tile_seconds
        out["tile_seconds"] = {"mean": round(statistics.fmean(secs), 4), "median": round(statistics.median(secs), 4), "min": round(min(secs), 4), "max": round(max(secs), 4),
                               "first": round(secs[0], 4)}
        out["engine_overhead_seconds"] = round(result.total_seconds - sum(secs), 3)
        out["one_worker_for_all_tiles"] = len(set(pids)) == 1 and len(pids) == plan.tile_count
        out["peak_gpu_memory_allocated_mib_max_over_tiles"] = max(peaks)
        out["worker_gpu_memory_after_scene_mib"] = process_gpu_mib(pid)
        out["output"] = {"shape": list(result.sr.shape), "dtype": str(result.sr.dtype).replace("torch.", ""), "min": round(float(result.sr.min()), 6),
                         "max": round(float(result.sr.max()), 6), "finite": bool(torch.isfinite(result.sr).all())}
        down = downsample_to_lr_grid(result.sr, 4)
        out["consistency_rmse_sr_downsampled_vs_input"] = round(float(((down - scene) ** 2).mean().sqrt()), 6)
        out["seam_diagnostic"] = seam_dict(result)
        again = run_tiled(client, scene, config)
        out["repeat_run_bit_identical"] = bool(torch.equal(result.sr, again.sr))
        out["repeat_run_max_abs_difference"] = float((result.sr - again.sr).abs().max())
    finally:
        client.close()
    out["phase1_single_tile_warm_seconds_for_reference"] = 1.74
    return out


# ---------------------------------------------------------------------------------------------------------------
# C. through the real HTTP API
# ---------------------------------------------------------------------------------------------------------------

def section_api(array: np.ndarray, meta: RasterMetadata, workdir: Path) -> Dict[str, Any]:
    from fastapi.testclient import TestClient

    from frame.api import config as api_config
    from frame.api.app import create_app
    from frame.api.services import model as model_service

    api_config.WORKSPACE_DIR = workdir / "workspace"
    api_config.DEVICE = "cuda"
    path = workdir / "scene.tif"
    write_geotiff(path, array, meta)
    results: Dict[str, Any] = {"scene_shape_hw": [array.shape[1], array.shape[2]], "input_file_mib": round(path.stat().st_size / 2**20, 2)}
    for model in ("mamba", "lite"):
        model_service.clear_cache()
        client = TestClient(create_app())
        up = client.post("/upload", files={"file": ("scene.tif", path.read_bytes(), "image/tiff")}, data={"input_scale": "reflectance"})
        assert up.status_code == 200, up.text
        started = time.perf_counter()
        response = client.post("/sr/run", json={"upload_id": up.json()["upload_id"], "model": model})
        wall = time.perf_counter() - started
        assert response.status_code == 200, response.text
        body = response.json()
        tiling = body["metadata"]["tiling"]
        sr, sr_meta = read_geotiff(body["artifacts"]["sr_geotiff"], require_crs=True)
        entry = {
            "http_wall_seconds_including_first_model_load": round(wall, 2), "inference_seconds_reported": body["metadata"]["inference_seconds"],
            "tile_count_per_pass": tiling["tile_count"], "scene_passes": tiling["scene_passes"], "tile_inferences": tiling["tile_inferences"],
            "mean_tile_seconds": tiling["mean_tile_seconds"], "output_shape": body["output_shape"],
            "geotiff": {"shape": list(sr.shape), "crs": sr_meta.crs, "resolution_m": sr_meta.resolution_m,
                        "bounds_match_input": bool(np.allclose(sr_meta.bounds, meta.bounds, atol=1e-9)), "sr_variant": sr_meta.sr_variant},
            "seam_diagnostic": {k: v for k, v in tiling["seam_diagnostic"].items() if k != "note"},
            "downsample_rmse_vs_own_input": body["self_consistency"]["downsample_rmse"],
        }
        if model == "mamba":
            pid = body["metadata"]["model_runtime"]["worker_pid"]
            entry["worker_pid"] = pid
            entry["worker_gpu_memory_after_job_mib"] = process_gpu_mib(pid)
        results[model] = entry
    model_service.clear_cache()
    return results


# ---------------------------------------------------------------------------------------------------------------
# D. memory
# ---------------------------------------------------------------------------------------------------------------

def memory_probe(side_h: int, side_w: int) -> None:
    """Runs in a fresh process: the whole API pipeline with a cheap model; prints peak RSS in MiB."""
    import resource
    import shutil
    import tempfile

    from frame.api.services import pipeline
    from frame.api.services.storage import JobStore

    array, meta, _ = real_scene()
    array, meta = crop_scene(array, meta, (0, 0, side_h, side_w))  # every probed size fits inside the cached 1024 x 1024 scene
    tmp = Path(tempfile.mkdtemp(prefix="frame_mem_", dir=os.environ.get("TMPDIR")))
    try:
        path = tmp / "s.tif"
        write_geotiff(path, array, meta)
        store = JobStore()
        upload = pipeline.process_upload(store, path, filename="s.tif", input_scale="reflectance")
        import torch.nn.functional as F

        started = time.perf_counter()
        pipeline.run_sr_job(store, upload.upload_id, model=lambda x: F.interpolate(x, scale_factor=4, mode="bicubic", antialias=True),
                            device="cpu", seed=42, workspace_dir=tmp / "jobs")
        seconds = time.perf_counter() - started
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
        print(json.dumps({"scene": [side_h, side_w], "input_pixels": side_h * side_w, "peak_rss_mib": round(peak), "pipeline_seconds_cheap_model": round(seconds, 1)}))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def section_memory() -> Dict[str, Any]:
    rows = []
    for size in [(256, 256), (511, 777), (768, 768), (1024, 1024)]:
        print(f"[D] memory probe {size} ...", flush=True)
        completed = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--memory-probe", str(size[0]), str(size[1])],
                                   capture_output=True, text=True, env=dict(os.environ))
        line = [ln for ln in completed.stdout.splitlines() if ln.startswith("{")]
        rows.append(json.loads(line[-1]) if line else {"scene": list(size), "error": completed.stderr[-300:]})
    return {"method": "fresh process per size; whole API pipeline (preprocess, tile engine, 6-member uncertainty ensemble, GeoTIFF export, "
                      "consistency) with a cheap bicubic 'model' so the time is not model time; peak resident set size", "rows": rows,
            "total_system_ram_mib": int(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2**20)}


# ---------------------------------------------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sections", default="A,B,C,D")
    parser.add_argument("--metadata-dir", type=Path, default=BASE_DIR / "metadata")
    parser.add_argument("--memory-probe", nargs=2, type=int, metavar=("H", "W"), help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.memory_probe:
        memory_probe(*args.memory_probe)
        return

    sections = set(args.sections.split(","))
    array, meta, identity = real_scene()
    assert rgb_reflectance_ok(array), "the real scene is not valid reflectance"
    crop_a, meta_a = crop_scene(array, meta, CROP_FULL)
    crop_b, meta_b = crop_scene(array, meta, CROP_MAMBA_SWEEP)
    scene_full, scene_sweep = torch.from_numpy(crop_a), torch.from_numpy(crop_b)

    gpu = nvidia_smi("gpu=name,memory.total,driver_version").split(",")
    record: Dict[str, Any] = {
        "phase": "Phase 2 -- arbitrary-size tiling", "git_commit": git_commit(), "date_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "scientific_framing": "Engineering measurements on one machine. Tile-seam numbers are a tiling diagnostic, not accuracy; the SR output is an SR-derived product on a 2.5 m pixel grid.",
        "machine": {"gpu": gpu[0].strip(), "vram_total_mib": int(gpu[1]), "nvidia_driver": gpu[2].strip(), "torch_main_env": torch.__version__},
        "real_scene": identity, "crops": {"full_scene_crop_rowoff_coloff_h_w": list(CROP_FULL), "mamba_sweep_crop": list(CROP_MAMBA_SWEEP)},
        "phase1_reference": {"mamba_warm_tile_seconds": 1.74, "worker_gpu_footprint_mib": 724, "peak_allocator_mib": 605.6},
    }
    mamba_ok = check_mamba_availability("cuda").available
    if not mamba_ok:
        print("[warn] Mamba not runnable here: sections needing it are skipped")
        record["mamba_available"] = False

    if "A" in sections:
        record["A_overlap_verification"] = section_overlap(scene_full, scene_sweep, mamba_ok)
    if "B" in sections and mamba_ok:
        record["B_real_mamba_multi_tile_engine"] = section_engine_mamba(scene_sweep if os.environ.get("FRAME_TILING_SMALL") else scene_full)
    if "C" in sections and mamba_ok:
        import tempfile

        with tempfile.TemporaryDirectory(prefix="frame_tiling_api_", dir=os.environ.get("TMPDIR")) as workdir:
            record["C_api_end_to_end"] = section_api(crop_b, meta_b, Path(workdir))
    if "D" in sections:
        record["D_memory"] = section_memory()

    args.metadata_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.metadata_dir / "run_metadata.json"
    if out_path.exists():  # sections may be run separately: keep the ones not re-run now
        previous = json.loads(out_path.read_text())
        for key, value in previous.items():
            if key.startswith(("A_", "B_", "C_", "D_")) and key not in record:
                record[key] = value
    out_path.write_text(json.dumps(record, indent=2, default=str) + "\n")
    print(json.dumps({k: record[k] for k in record if k.startswith(("A_", "B_", "C_", "D_"))}, indent=1, default=str)[:6000])
    print("wrote", out_path)


if __name__ == "__main__":
    main()
