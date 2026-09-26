"""FRAME end-to-end smoke path (Phase 8): the integrated product, exercised through its real HTTP API, on a deterministic synthetic scene.

    python -m frame.smoke                          # model "toy": a deterministic stand-in x4 model. No weights, no GPU, seconds.
    python -m frame.smoke --model lite             # the real SEN2SR-Lite (needs the cached weights)
    python -m frame.smoke --model mamba            # the real SEN2SR-Mamba (needs a CUDA GPU and the isolated environment)
    python -m frame.smoke --demo-scenes DIR        # write input GeoTIFFs for a live demo (synthetic; real ones if the Baseline 0 / cached scenes exist)

What it exercises, in order, through `frame.api` (FastAPI, in process, no network): the model list (`GET /health`), input validation (three
inputs that must be REFUSED, then the scene), preprocessing and scaling (raw digital numbers, a nodata block), the model, the tile engine
(a rectangular scene that is not a multiple of 128), reconstruction, the TTA stability raster, output validation read back from disk
(exact 4x size, CRS, origin, pixel size, footprint, band names, model tag), the downloads, the NDVI demonstration, and provenance capture.

**A smoke test is not evidence.** With the toy model it says nothing about any real model; with a real model it says the pipeline ran and its invariants
held on one synthetic scene, not that the output is accurate. The synthetic scene is made from smooth random fields, not from Sentinel-2.

The record (``smoke_record.json``) lists every check, the timings, the expected and produced shapes, and what is needed to repeat the run.
Exit codes: 0 every check passed; 1 a check failed; 2 the requested model is unavailable here or the arguments are invalid.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
SMOKE_VERSION = "frame-smoke/1"
DEFAULT_HEIGHT, DEFAULT_WIDTH, DEFAULT_SEED = 200, 300, 42                     # 2 x 3 = 6 tiles per pass: rectangular, not a multiple of 128
ORIGIN_X, ORIGIN_Y, CRS, PIXEL_M = 720285.0, 4375125.0, "EPSG:32630", 10.0
NODATA_BLOCK = (slice(0, 10), slice(0, 12))                                  # 120 pixels the mask must report as invalid
MODELS = ("toy", "lite", "mamba")
NOT_EVIDENCE = (
    "A smoke test, not scientific evidence: the scene is synthetic (smooth random fields) and the checks are pipeline invariants "
    "(sizes, georeferencing, provenance, refusals). It says nothing about the accuracy of any model."
)


class SmokeUnavailable(Exception):
    """The requested model cannot run on this machine (not a failed check)."""


# ------------------------------------------------------------------------------------------------ the scene


def _smooth(rng: np.random.Generator, shape: Sequence[int], radius: int) -> np.ndarray:
    """Seeded smooth noise in [0, 1] (a box blur by cumulative sums: numpy only, no scipy)."""
    field = rng.random(tuple(shape))
    for axis in (-2, -1):
        padded = np.concatenate([field.take(range(radius, 0, -1), axis=axis), field, field.take(range(-2, -radius - 2, -1), axis=axis)], axis=axis)
        csum = np.cumsum(padded, axis=axis, dtype=np.float64)
        upper = np.take(csum, range(2 * radius, padded.shape[axis]), axis=axis)
        lower = np.concatenate([np.zeros_like(np.take(csum, [0], axis=axis)), np.take(csum, range(0, padded.shape[axis] - 2 * radius - 1), axis=axis)], axis=axis)
        field = (upper - lower) / (2 * radius + 1)
    return (field - field.min()) / max(float(field.max() - field.min()), 1e-9)


def synthetic_scene(height: int = DEFAULT_HEIGHT, width: int = DEFAULT_WIDTH, seed: int = DEFAULT_SEED) -> np.ndarray:
    """(4, H, W) float32 raw L2A-style digital numbers in RGBN order (B04, B03, B02, B08), from seeded smooth fields; a 10 x 12 corner is nodata (0)."""
    rng = np.random.default_rng(seed)
    veg = _smooth(rng, (height, width), 9)                     # a vegetation-like field: bright NIR, dark red
    soil = _smooth(rng, (height, width), 5)
    detail = _smooth(rng, (height, width), 1) - 0.5
    red = 0.05 + 0.10 * soil + 0.02 * detail - 0.04 * veg
    green = 0.07 + 0.09 * soil + 0.02 * detail
    blue = 0.04 + 0.07 * soil + 0.015 * detail
    nir = 0.15 + 0.10 * soil + 0.35 * veg + 0.03 * detail
    reflectance = np.clip(np.stack([red, green, blue, nir]), 0.005, 1.0)
    dn = np.round(reflectance * 10000.0).astype("float32")
    dn[(slice(None), *NODATA_BLOCK)] = 0.0
    return dn


def write_scene(path: Path, dn: np.ndarray) -> None:
    from frame.geospatial import write_geotiff
    from frame.preprocessing import RGBN_BANDS
    from frame.preprocessing.metadata import RasterMetadata

    _, h, w = dn.shape
    metadata = RasterMetadata(
        crs=CRS, transform=(PIXEL_M, 0.0, ORIGIN_X, 0.0, -PIXEL_M, ORIGIN_Y), bounds=(ORIGIN_X, ORIGIN_Y - h * PIXEL_M, ORIGIN_X + w * PIXEL_M, ORIGIN_Y), resolution_m=PIXEL_M,
        width=w, height=h, band_names=tuple(RGBN_BANDS), acquisition_timestamp="2023-01-15T10:54:11.024000", nodata_value=0.0, cloud_mask_coverage=1.0, sr_variant=None,
    )
    write_geotiff(path, dn, metadata)


# ------------------------------------------------------------------------------------------------ demo inputs


BASELINE0_SCENE = REPO_ROOT / "experiments" / "baseline" / "outputs" / "input_10m.tif"     # a real 128 x 128 Sentinel-2 RGBN window, reflectance 0-1 (Phase 0)
REAL_SCENE_ENV = "FRAME_TILING_CACHE_DIR"                                                    # the cached real 1024 x 1024 window of experiments/tiling
DEMO_CROP = (300, 400, 200, 300)                                                             # row, column, height, width of the rectangular real crop


def crop_geotiff(source: Path, destination: Path, row: int, col: int, height: int, width: int) -> Path:
    """Write a window of ``source`` as a GeoTIFF with the correct origin and bounds (CRS, band names, timestamp preserved; values untouched)."""
    import dataclasses

    from frame.geospatial import read_geotiff, write_geotiff

    array, meta = read_geotiff(source, require_crs=True)
    if row < 0 or col < 0 or row + height > meta.height or col + width > meta.width or height < 1 or width < 1:
        raise ValueError(f"the window ({row}, {col}, {height} x {width}) does not fit the {meta.height} x {meta.width} source")
    a, b, c, d, e, f = meta.transform
    transform = (a, b, c + a * col + b * row, d, e, f + d * col + e * row)
    bounds = (transform[2], transform[5] + e * height, transform[2] + a * width, transform[5])
    cropped = dataclasses.replace(meta, transform=transform, bounds=bounds, width=width, height=height)
    write_geotiff(destination, array[:, row : row + height, col : col + width], cropped)
    return destination


def write_demo_scenes(output_dir: Path) -> Dict[str, Dict[str, Any]]:
    """Write the inputs for a live demo into ``output_dir`` and say which input scale each needs in the UI.

    * ``synthetic_200x300_raw_dn.tif``: always; digital numbers (upload with "raw digital number"); rectangular, has a nodata corner.
    * ``real_baseline0_128_reflectance.tif``: a copy of the real Baseline 0 window, if it exists (upload with "reflectance"); one tile.
    * ``real_crop_200x300_reflectance.tif``: a rectangular 200 x 300 crop of the cached real 1024 x 1024 scene, if it is cached (upload with "reflectance"); 6 tiles.
    """
    import shutil

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out: Dict[str, Dict[str, Any]] = {}
    synthetic = output_dir / "synthetic_200x300_raw_dn.tif"
    write_scene(synthetic, synthetic_scene(DEFAULT_HEIGHT, DEFAULT_WIDTH, DEFAULT_SEED))
    out["synthetic"] = {"path": str(synthetic), "input_scale": "raw_digital_number", "shape": [4, DEFAULT_HEIGHT, DEFAULT_WIDTH], "sha256": sha256_path(synthetic), "source": "generated (smooth random fields, seed 42)"}
    if BASELINE0_SCENE.is_file():
        real = output_dir / "real_baseline0_128_reflectance.tif"
        shutil.copyfile(BASELINE0_SCENE, real)
        out["real_128"] = {"path": str(real), "input_scale": "reflectance", "shape": [4, 128, 128], "sha256": sha256_path(real), "source": "real Sentinel-2 L2A window, AOI 39.49 / -0.43, 2023-01-15 (experiments/baseline)"}
    cache = Path(os.environ.get(REAL_SCENE_ENV, str(Path.home() / ".cache" / "frame_tiling"))) / "real_scene_1024.tif"
    if cache.is_file():
        crop = crop_geotiff(cache, output_dir / "real_crop_200x300_reflectance.tif", *DEMO_CROP)
        out["real_crop"] = {"path": str(crop), "input_scale": "reflectance", "shape": [4, DEMO_CROP[2], DEMO_CROP[3]], "sha256": sha256_path(crop),
                            "source": f"crop (row {DEMO_CROP[0]}, col {DEMO_CROP[1]}) of the cached real 1024 x 1024 Sentinel-2 window of experiments/tiling"}
    return out


# ------------------------------------------------------------------------------------------------ the toy model


class ToyModel:
    """A deterministic x4 stand-in with the model contract (1, 4, 128, 128) -> (1, 4, 512, 512): antialiased bicubic plus an ASYMMETRIC sharpening,
    so the six TTA views genuinely disagree and the stability raster is not all zeros. It is a test double, not a super-resolution model."""

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        up = F.interpolate(x, scale_factor=4, mode="bicubic", antialias=True).clamp(min=0)
        return up + 0.3 * (up - torch.roll(up, shifts=(1, 2), dims=(-2, -1)))


# ------------------------------------------------------------------------------------------------ helpers


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


class Recorder:
    """Collects named checks and never lets one failure hide the others."""

    def __init__(self) -> None:
        self.checks: List[Dict[str, Any]] = []

    def check(self, name: str, ok: bool, detail: str = "") -> bool:
        self.checks.append({"name": name, "passed": bool(ok), "detail": detail})
        return bool(ok)

    @property
    def passed(self) -> bool:
        return all(c["passed"] for c in self.checks)


def _timed(timings: Dict[str, float], name: str, fn: Callable[[], Any]) -> Any:
    t0 = time.perf_counter()
    try:
        return fn()
    finally:
        timings[name] = round(time.perf_counter() - t0, 4)


def _lite_artifact_files() -> Dict[str, str]:
    """SHA-256 of the cached SEN2SR-Lite files (the API does not record a Lite weights hash; the smoke record does)."""
    from frame.api import config

    out: Dict[str, str] = {}
    for name in ("model.safetensor", "hard_constraint.safetensor", "mlm.json"):
        path = config.MODEL_WEIGHTS_CACHE_DIR / name
        if path.is_file():
            out[name] = sha256_path(path)
    return out


# ------------------------------------------------------------------------------------------------ the run


def run_smoke(
    output_dir: Path,
    *,
    model: str = "toy",
    height: int = DEFAULT_HEIGHT,
    width: int = DEFAULT_WIDTH,
    seed: int = DEFAULT_SEED,
    repeat: Optional[bool] = None,
    workspace_dir: Optional[Path] = None,
    progress: Callable[[str], None] = lambda message: None,
) -> Dict[str, Any]:
    """Run the smoke path and write ``output_dir/smoke_record.json``. Returns the record. Raises `SmokeUnavailable` for a model that cannot run here."""
    if model not in MODELS:
        raise ValueError(f"model must be one of {MODELS}, got {model!r}.")
    from fastapi.testclient import TestClient

    from frame.api import config, routes
    from frame.api.app import create_app
    from frame.api.schemas import NDVI_DEMONSTRATION_NOTE
    from frame.api.services.storage import JobStore
    from frame.geospatial import read_geotiff
    from frame.models.selection import MODEL_SPECS
    from frame.preprocessing import RGBN_BANDS
    from frame.tiling import TilingConfig, plan_tiles
    from frame.train.record import environment_info, git_info

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    repeat = (model == "toy") if repeat is None else repeat
    own_workspace = workspace_dir is None
    workspace = Path(tempfile.mkdtemp(prefix="frame_smoke_")) if own_workspace else Path(workspace_dir)
    workspace.mkdir(parents=True, exist_ok=True)

    rec, timings = Recorder(), {}
    settings = {"model": model, "height": height, "width": width, "seed": seed, "repeat": repeat, "scene": "synthetic_smooth_fields_v1", "overlap": config.TILE_OVERLAP}
    previous_workspace = config.WORKSPACE_DIR
    config.WORKSPACE_DIR = workspace
    try:
        app = create_app()
        store = JobStore()
        app.dependency_overrides[routes.get_store] = lambda: store
        if model == "toy":
            toy = ToyModel()
            app.dependency_overrides[routes.get_model_callable] = lambda: toy
            app.dependency_overrides[routes.get_device] = lambda: "cpu"
        client = TestClient(app, raise_server_exceptions=False)
        spec = MODEL_SPECS["lite" if model == "toy" else model]

        # ---- 1. the model list
        progress("health")
        health = _timed(timings, "health_s", lambda: client.get("/health"))
        listed = {m["id"]: m for m in health.json().get("available_models", [])} if health.status_code == 200 else {}
        rec.check("health_ok_and_lists_both_models", health.status_code == 200 and set(listed) == {"lite", "mamba"}, f"status {health.status_code}, models {sorted(listed)}")
        if model == "mamba" and not listed.get("mamba", {}).get("available", False):
            raise SmokeUnavailable(f"SEN2SR-Mamba is unavailable here: {listed.get('mamba', {}).get('reason')}")

        # ---- 2. the scene, and three inputs that must be refused
        dn = synthetic_scene(height, width, seed)
        scene_path = workspace / "smoke_scene.tif"
        write_scene(scene_path, dn)
        scene_bytes = scene_path.read_bytes()
        progress("input validation")

        def upload(data: bytes, scale: str) -> Any:
            return client.post("/upload", files={"file": ("smoke_scene.tif", data, "image/tiff")}, data={"input_scale": scale})

        junk = upload(b"not a tiff", "raw_digital_number")
        rec.check("refuses_a_file_that_is_not_a_geotiff", junk.status_code == 400 and junk.json().get("code") == "frame_error", f"status {junk.status_code}")
        fractions = np.clip(dn / 10000.0, 0, 1).astype("float32")
        misdeclared = upload(_tiff_bytes(workspace, "fractions", fractions), "raw_digital_number")
        rec.check("refuses_reflectance_declared_as_digital_numbers", misdeclared.status_code == 400 and "reflectance" in str(misdeclared.json().get("detail")), f"status {misdeclared.status_code}")
        empty = upload(_tiff_bytes(workspace, "empty", np.zeros_like(dn)), "raw_digital_number")
        rec.check("refuses_a_scene_with_no_valid_pixel", empty.status_code == 400 and "no valid pixel" in str(empty.json().get("detail")), f"status {empty.status_code}")

        # ---- 3. upload
        progress("upload")
        response = _timed(timings, "upload_s", lambda: upload(scene_bytes, "raw_digital_number"))
        body = response.json() if response.status_code == 200 else {}
        rec.check("accepts_the_rectangular_scene", response.status_code == 200 and (body.get("height"), body.get("width")) == (height, width) and body.get("crs") == CRS,
                  f"status {response.status_code}, {body.get('height')}x{body.get('width')} {body.get('crs')}")
        if response.status_code != 200:
            return _finish(output_dir, rec, timings, settings, None, model, spec, scene_bytes, git_info, environment_info)
        upload_id = body["upload_id"]

        # ---- 4. run
        progress(f"sr/run ({model})")
        run = _timed(timings, "sr_run_s", lambda: client.post("/sr/run", json={"upload_id": upload_id, "model": "lite" if model == "toy" else model, "seed": seed}))
        if run.status_code == 503:
            raise SmokeUnavailable(f"The model is unavailable here: {run.json().get('detail')}")
        rec.check("sr_run_succeeds", run.status_code == 200, f"status {run.status_code}: {str(run.json().get('detail', ''))[:120] if run.status_code != 200 else ''}")
        if run.status_code != 200:
            return _finish(output_dir, rec, timings, settings, None, model, spec, scene_bytes, git_info, environment_info)
        result = run.json()

        # ---- 5. what came out, read back from disk
        expected_shape = [4, 4 * height, 4 * width]
        rec.check("output_shape_is_exactly_4x", result["output_shape"] == expected_shape and result["input_shape"] == [4, height, width], f"expected {expected_shape}, got {result['output_shape']}")
        sr, out = read_geotiff(result["artifacts"]["sr_geotiff"], require_crs=True)
        unc, unc_geo = read_geotiff(result["artifacts"]["uncertainty_geotiff"], require_crs=True)
        rec.check("geotiff_shape_matches", list(sr.shape) == expected_shape, f"{list(sr.shape)}")
        rec.check("crs_preserved", out.crs == CRS and unc_geo.crs == CRS, f"{out.crs}")
        rec.check("origin_preserved_and_pixel_size_is_2.5_m", _close(out.transform, (PIXEL_M / 4, 0.0, ORIGIN_X, 0.0, -PIXEL_M / 4, ORIGIN_Y)), f"{tuple(round(v, 6) for v in out.transform)}")
        rec.check("footprint_equals_the_input_footprint", _close(out.bounds, (ORIGIN_X, ORIGIN_Y - height * PIXEL_M, ORIGIN_X + width * PIXEL_M, ORIGIN_Y)), f"{tuple(out.bounds)}")
        rec.check("band_names_and_model_tag", out.band_names == tuple(RGBN_BANDS) and out.sr_variant == spec.model_name and result["model_name"] == spec.model_name,
                  f"bands {out.band_names}, tag {out.sr_variant!r}")
        rec.check("stability_raster_on_the_same_grid", unc.shape[1:] == sr.shape[1:] and unc_geo.transform == out.transform, f"{list(unc.shape)}")
        rec.check("output_is_finite", bool(np.isfinite(sr).all() and np.isfinite(unc).all()), "")
        expected_cov = 1.0 - (10 * 12) / float(height * width)
        rec.check("nodata_block_reported_in_the_coverage", math.isclose(result["metadata"]["preprocessing_mask_coverage"], expected_cov, abs_tol=1e-9), f"{result['metadata']['preprocessing_mask_coverage']:.6f} vs {expected_cov:.6f}")
        plan = plan_tiles(height, width, TilingConfig(overlap=config.TILE_OVERLAP))
        tiling = result["metadata"]["tiling"]
        rec.check("tiling_matches_the_plan", tiling["tile_grid"] == [plan.n_rows, plan.n_cols] and tiling["tile_count"] == plan.tile_count and plan.tile_count > 1,
                  f"grid {tiling['tile_grid']} x{tiling['tile_count']} tiles per pass, {tiling['tile_inferences']} model calls")
        rec.check("seam_diagnostic_was_computed", tiling["seam_diagnostic"]["status"] == "COMPUTABLE", str(tiling["seam_diagnostic"]["status"]))
        rec.check("stability_is_labelled_a_diagnostic_and_not_calibrated", "diagnostic" in result["uncertainty"]["label"] and "NOT a calibrated" in result["uncertainty"]["disclaimer"], result["uncertainty"]["label"])

        # ---- 6. reproducibility, downloads, NDVI demonstration
        identical: Optional[bool] = None
        if repeat:
            progress("repeat run")
            again = client.post("/sr/run", json={"upload_id": upload_id, "model": "lite" if model == "toy" else model, "seed": seed})
            sr2, _ = read_geotiff(again.json()["artifacts"]["sr_geotiff"], require_crs=True)
            identical = bool(np.array_equal(sr, sr2))
            rec.check("same_seed_same_scene_same_output", identical, "bit-identical" if identical else f"max abs difference {float(np.abs(sr - sr2).max()):.3g}")
        progress("downloads")
        for name, url in (("sr_download", f"/sr/download/{result['job_id']}"), ("stability_download", f"/uncertainty/download/{result['job_id']}")):
            got = client.get(url)
            rec.check(f"{name}_serves_a_geotiff", got.status_code == 200 and got.headers.get("content-type") == "image/tiff" and got.content[:4] in (b"II*\x00", b"MM\x00*"), f"status {got.status_code}, {len(got.content)} bytes")
        progress("ndvi demonstration")
        ndvi = _timed(timings, "ndvi_s", lambda: client.post("/analysis/ndvi", json={"job_id": result["job_id"]}))
        ndvi_body = ndvi.json() if ndvi.status_code == 200 else {}
        rec.check("ndvi_demonstration_runs_and_says_what_it_is", ndvi.status_code == 200 and NDVI_DEMONSTRATION_NOTE in ndvi_body.get("scientific_caveats", []), f"status {ndvi.status_code}")

        extra = {"result": result, "identical_repeat": identical, "output_sha256": sha256_bytes(np.ascontiguousarray(sr).tobytes()), "output_shape": list(sr.shape)}
        return _finish(output_dir, rec, timings, settings, extra, model, spec, scene_bytes, git_info, environment_info)
    finally:
        config.WORKSPACE_DIR = previous_workspace
        if own_workspace:
            import shutil

            shutil.rmtree(workspace, ignore_errors=True)


def _close(actual: Sequence[float], expected: Sequence[float], tol: float = 1e-6) -> bool:
    return len(actual) == len(expected) and all(math.isclose(float(a), float(b), rel_tol=0.0, abs_tol=tol) for a, b in zip(actual, expected))


def _tiff_bytes(workspace: Path, name: str, array: np.ndarray) -> bytes:
    path = workspace / f"{name}.tif"
    write_scene(path, array)
    return path.read_bytes()


def _finish(output_dir: Path, rec: Recorder, timings: Dict[str, float], settings: Dict[str, Any], extra: Optional[Dict[str, Any]], model: str, spec: Any, scene_bytes: bytes,
            git_info: Callable[..., Any], environment_info: Callable[..., Any]) -> Dict[str, Any]:
    result = (extra or {}).get("result") or {}
    provenance: Dict[str, Any] = {
        "smoke_version": SMOKE_VERSION,
        "settings": settings,
        "settings_digest": sha256_bytes(json.dumps(settings, sort_keys=True).encode("utf-8")),
        "scene_sha256": sha256_bytes(scene_bytes),
        "code": {"git": git_info(REPO_ROOT)},
        "environment": environment_info("cpu" if model == "toy" else "auto"),
        "model": {"requested": model, "canonical_name": spec.model_name if model != "toy" else "toy (deterministic test double, not a super-resolution model)", "reported_by_api": result.get("model_name"),
                  "runtime": result.get("metadata", {}).get("model_runtime"), "device": result.get("metadata", {}).get("device"),
                  **({"caveat": "The toy stand-in is served through the 'lite' selection, so the API and the output GeoTIFF tag name SEN2SR-Lite although no SEN2SR model ran."} if model == "toy" else {})},
        "seed": settings["seed"],
        "tta": {"n_members": result.get("uncertainty", {}).get("n"), "transforms": result.get("uncertainty", {}).get("transform_names")},
        "output_sha256": (extra or {}).get("output_sha256"),
    }
    if model == "lite":
        provenance["model"]["artifact_files_sha256"] = _lite_artifact_files()
    record = {
        "status": "passed" if rec.passed else "failed",
        "note": NOT_EVIDENCE,
        "expected_output_shape": [4, 4 * settings["height"], 4 * settings["width"]],
        "produced_output_shape": (extra or {}).get("output_shape"),
        "checks": rec.checks,
        "n_checks": len(rec.checks),
        "n_failed": sum(1 for c in rec.checks if not c["passed"]),
        "timings_seconds": {**timings, "inference_reported_by_api": result.get("metadata", {}).get("inference_seconds")},
        "identical_repeat": (extra or {}).get("identical_repeat"),
        "tiling": result.get("metadata", {}).get("tiling"),
        "provenance": provenance,
    }
    (Path(output_dir) / "smoke_record.json").write_text(json.dumps(record, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    return record


# ------------------------------------------------------------------------------------------------ command line


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m frame.smoke", description="FRAME end-to-end smoke path on a deterministic synthetic scene (not scientific evidence).")
    parser.add_argument("--model", choices=MODELS, default="toy", help="toy (default: no weights, no GPU), lite, or mamba (real models)")
    parser.add_argument("--height", type=int, default=DEFAULT_HEIGHT)
    parser.add_argument("--width", type=int, default=DEFAULT_WIDTH)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--repeat", action="store_true", help="run twice and require identical output (default for toy; slow for mamba)")
    parser.add_argument("--no-repeat", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=None, help="where smoke_record.json is written (default: a temporary directory)")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--demo-scenes", type=Path, default=None, metavar="DIR", help="write the live-demo input GeoTIFFs into DIR (and the input scale each needs), then exit")
    args = parser.parse_args(argv)
    if args.demo_scenes is not None:
        for name, info in write_demo_scenes(args.demo_scenes).items():
            print(f"{name}: {info['path']}  [input scale: {info['input_scale']}; {info['shape']}; {info['source']}]")
        return 0
    if args.height < 1 or args.width < 1:
        print("height and width must be positive", file=sys.stderr)
        return 2
    out = args.output_dir or Path(tempfile.mkdtemp(prefix="frame_smoke_record_"))
    say = (lambda m: None) if args.quiet else (lambda m: print(f"  {m}", flush=True))
    try:
        record = run_smoke(out, model=args.model, height=args.height, width=args.width, seed=args.seed, repeat=True if args.repeat else (False if args.no_repeat else None), progress=say)
    except SmokeUnavailable as exc:
        print(f"unavailable: {exc}", file=sys.stderr)
        return 2
    for c in record["checks"]:
        print(f"[{'ok' if c['passed'] else 'FAIL'}] {c['name']}: {c['detail']}")
    print(f"smoke {record['status']}: {record['n_checks'] - record['n_failed']}/{record['n_checks']} checks; expected output {record['expected_output_shape']}, "
          f"produced {record['produced_output_shape']}; sr/run {record['timings_seconds'].get('sr_run_s')} s; record: {out / 'smoke_record.json'}")
    print(NOT_EVIDENCE)
    return 0 if record["status"] == "passed" else 1


if __name__ == "__main__":
    sys.exit(main())
