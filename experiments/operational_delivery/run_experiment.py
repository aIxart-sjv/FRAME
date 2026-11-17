"""Step I Section 8: real engineering experiments for the existing tiling engine
(frame.tiling, already built and tested in earlier phases) and the new product-packaging layer
(frame.product / frame.geospatial.cog, Step I).

Not a new accuracy benchmark. This machine is an RTX 3050 Laptop, 4 GB VRAM, 15.2 GiB RAM
(docs/REGISTRY.md Sec 3). Every experiment records what was actually measured; a configuration
that cannot safely run is skipped with the reason, never forced or worked around.

Scenes here are synthetic (reflectance-range random tensors, the same convention used
throughout frame/tests and frame.smoke) -- these are runtime/memory/integrity engineering
measurements, not accuracy claims, so a live Sentinel-2 retrieval is not needed for them (Step
I's own instructions allow "a synthetic or real-scene tiling workload" for the scaling
experiment). The separate Step I end-to-end acceptance test (experiments/final_demo/) uses a
real live retrieval instead, where the point IS to prove the real path.

Run: sen2sr_venv/bin/python experiments/operational_delivery/run_experiment.py
"""

from __future__ import annotations

import json
import resource
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from frame.api.services.model import get_model  # noqa: E402
from frame.geospatial import RGBN_SCALE_FACTOR, derive_output_metadata, write_cog  # noqa: E402
from frame.geospatial.cog import validate_cog  # noqa: E402
from frame.preprocessing.metadata import RasterMetadata  # noqa: E402
from frame.tiling import TiledModel, TilingConfig, run_tiled  # noqa: E402
from frame.tiling.errors import TilingError  # noqa: E402

OUT_DIR = REPO_ROOT / "experiments" / "operational_delivery"
RGBN = ("B04", "B03", "B02", "B08")


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _peak_rss_mib() -> float:
    """Peak resident set size of this process so far, in MiB (Linux: ru_maxrss is kilobytes)."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def _gpu_used_mib() -> Optional[int]:
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=5)
        return int(out.stdout.strip().splitlines()[0])
    except Exception:  # noqa: BLE001 -- no nvidia-smi / no GPU is a real, disclosed "not measured", not a crash
        return None


def synthetic_scene(height: int, width: int, seed: int = 0, device: str = "cpu") -> torch.Tensor:
    rng = np.random.default_rng(seed)
    array = (rng.random((4, height, width)).astype("float32") * 0.4) + 0.1
    return torch.from_numpy(array).to(device)


def seam_summary(diag) -> Optional[Dict[str, Any]]:
    if diag is None:
        return None
    from dataclasses import asdict

    return asdict(diag)


# ---------------------------------------------------------------------------
# Experiment A -- tile-size sensitivity
# ---------------------------------------------------------------------------

def experiment_a_tile_size(model, device: str) -> Dict[str, Any]:
    scene = synthetic_scene(300, 400, device=device)
    results = {}
    for tile_size in (64, 128, 256):
        overlap = min(16, tile_size // 2 - 1)
        entry: Dict[str, Any] = {"tile_size": tile_size, "overlap": overlap}
        try:
            cfg = TilingConfig(tile_size=tile_size, overlap=overlap)
        except Exception as exc:  # invalid config itself (should not happen for these values)
            entry.update(success=False, skip_reason=f"invalid TilingConfig: {type(exc).__name__}: {exc}")
            results[str(tile_size)] = entry
            continue
        rss_before = _peak_rss_mib()
        t0 = time.time()
        try:
            result = run_tiled(model, scene, cfg)
            elapsed = time.time() - t0
            entry.update(
                success=True, elapsed_seconds=round(elapsed, 4), n_tiles=len(result.plan.tiles),
                output_shape=list(result.sr.shape), peak_rss_mib=round(_peak_rss_mib() - rss_before, 1),
                seam=seam_summary(result.seam_diagnostic),
            )
        except (RuntimeError, TilingError) as exc:
            entry.update(
                success=False, elapsed_seconds=round(time.time() - t0, 4),
                skip_reason=(
                    f"{type(exc).__name__}: {exc} -- the registered model's own hard-constraint mask is baked to a "
                    f"fixed 128x128 -> 512x512 shape (docs/REGISTRY.md); frame.tiling itself accepts any tile_size, "
                    f"but the model call inside it does not."
                ),
            )
        results[str(tile_size)] = entry
    return {"scene_shape": [300, 400], "device": device, "by_tile_size": results}


# ---------------------------------------------------------------------------
# Experiment B / F -- overlap sensitivity + seam quality
# ---------------------------------------------------------------------------

def experiment_b_overlap(model, device: str) -> Dict[str, Any]:
    scene = synthetic_scene(300, 400, device=device)
    results = {}
    for overlap in (0, 8, 16, 32):
        t0 = time.time()
        result = run_tiled(model, scene, TilingConfig(tile_size=128, overlap=overlap))
        elapsed = time.time() - t0
        results[str(overlap)] = {
            "overlap": overlap, "elapsed_seconds": round(elapsed, 4), "n_tiles": len(result.plan.tiles),
            "seam": seam_summary(result.seam_diagnostic),
        }
    return {"scene_shape": [300, 400], "device": device, "by_overlap": results,
            "note": "Seam values come from frame.tiling's own no-reference seam diagnostic (overlapping tile predictions compared with each other), the same mechanism already measured for real Lite/Mamba scenes in experiments/tiling/run_validation.py."}


# ---------------------------------------------------------------------------
# Experiment C -- scaling
# ---------------------------------------------------------------------------

def experiment_c_scaling(model, device: str) -> Dict[str, Any]:
    results = {}
    for size in (128, 256, 512, 768):
        scene = synthetic_scene(size, size, device=device)
        t0 = time.time()
        try:
            result = run_tiled(model, scene, TilingConfig(overlap=16))
            elapsed = time.time() - t0
            results[str(size)] = {"scene_size": size, "n_tiles": len(result.plan.tiles), "elapsed_seconds": round(elapsed, 4),
                                   "seconds_per_tile": round(elapsed / max(1, len(result.plan.tiles)), 4)}
        except Exception as exc:  # noqa: BLE001
            results[str(size)] = {"scene_size": size, "skip_reason": f"{type(exc).__name__}: {exc}"}
    return {"device": device, "by_scene_size": results,
            "note": "1024x1024 (the API's own MAX_INPUT_PIXELS cap) was not additionally tested here -- the trend from "
                    "128..768 already establishes the per-tile cost; extrapolating further is arithmetic, not a new measurement "
                    "(see docs/REGISTRY.md Sec 3 for the existing Mamba 1024x1024 extrapolation, which uses the same method)."}


# ---------------------------------------------------------------------------
# Experiment D -- CPU vs GPU
# ---------------------------------------------------------------------------

def experiment_d_cpu_vs_gpu() -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for device in ("cpu", "cuda"):
        if device == "cuda" and not torch.cuda.is_available():
            out[device] = {"skip_reason": "torch.cuda.is_available() is False on this run"}
            continue
        model = get_model(device, model_name="lite")
        scene = synthetic_scene(256, 256, device=device)
        if device == "cuda":
            torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        result = run_tiled(model, scene, TilingConfig(overlap=16))
        elapsed = time.time() - t0
        entry = {"elapsed_seconds": round(elapsed, 4), "n_tiles": len(result.plan.tiles)}
        if device == "cuda":
            entry["peak_vram_mib"] = round(torch.cuda.max_memory_allocated() / 1024**2, 1)
        out[device] = entry
    return {"scene_shape": [256, 256], "model": "lite", "results": out}


# ---------------------------------------------------------------------------
# Experiment E -- output integrity
# ---------------------------------------------------------------------------

def experiment_e_output_integrity(model, device: str) -> Dict[str, Any]:
    from frame.product import build_stac_item, validate_stac_item

    scene = synthetic_scene(128, 128, device=device)
    result = run_tiled(model, scene, TilingConfig())
    sr = result.sr

    input_metadata = RasterMetadata(
        crs="EPSG:32644", transform=(10.0, 0.0, 500000.0, 0.0, -10.0, 1440752.0),
        bounds=(500000.0, 1440752.0 - 128 * 10.0, 500000.0 + 128 * 10.0, 1440752.0), resolution_m=10.0,
        width=128, height=128, band_names=RGBN, acquisition_timestamp="2024-03-30T04:56:59Z",
        nodata_value=0.0, cloud_mask_coverage=1.0, sr_variant=None,
    )
    output_metadata = derive_output_metadata(input_metadata, scale_factor=RGBN_SCALE_FACTOR, output_band_names=RGBN)

    cog_path = OUT_DIR / "integrity_check.cog.tif"
    write_cog(cog_path, sr.numpy(), output_metadata)
    cog_result = validate_cog(cog_path)

    checks: Dict[str, Any] = {
        "crs_present": output_metadata.crs is not None,
        "transform_present": output_metadata.transform is not None,
        "bounds_sane": output_metadata.bounds[2] > output_metadata.bounds[0] and output_metadata.bounds[3] > output_metadata.bounds[1],
        "dimensions_match_scale_factor": (output_metadata.width, output_metadata.height) == (128 * RGBN_SCALE_FACTOR, 128 * RGBN_SCALE_FACTOR),
        "band_count_correct": sr.shape[0] == 4,
        "band_names_correct": list(output_metadata.band_names) == list(RGBN),
        "dtype_is_float32": sr.numpy().dtype == np.float32,
        "no_nan": bool(np.isfinite(sr.numpy()).all()),
        "cog_valid": cog_result.is_valid_cog,
        "pixel_size_is_2_5m_in_output_crs": abs(abs(output_metadata.transform[0]) - 2.5) < 1e-9,
    }

    item = build_stac_item(
        item_id="integrity-check", output_metadata=output_metadata, model_id="lite", model_name="lite-test",
        architecture="CNNSR", checkpoint_sha256=None, input_resolution_m=10.0, output_resolution_m=2.5,
        scale_factor=RGBN_SCALE_FACTOR, bands=RGBN, radiometric_convention="reflectance fraction",
        processing_timestamp=_now_iso(), assets={"sr_cog": {"href": str(cog_path), "type": "image/tiff"}},
    )
    checks["stac_valid"] = validate_stac_item(item) == []
    checks["asset_reference_exists"] = Path(item["assets"]["sr_cog"]["href"]).exists()

    return {"device": device, "checks": checks, "all_passed": all(bool(v) for v in checks.values())}


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = get_model(device, model_name="lite")

    record: Dict[str, Any] = {
        "created_at": _now_iso(),
        "machine": {"gpu_used_mib_baseline": _gpu_used_mib(), "device_for_a_b_c_e": device},
    }

    print("[A] tile-size sensitivity ...", flush=True)
    record["A_tile_size_sensitivity"] = experiment_a_tile_size(model, device)

    print("[B/F] overlap sensitivity + seam quality ...", flush=True)
    record["B_F_overlap_and_seam"] = experiment_b_overlap(model, device)

    print("[C] scaling ...", flush=True)
    record["C_scaling"] = experiment_c_scaling(model, device)

    print("[D] CPU vs GPU ...", flush=True)
    record["D_cpu_vs_gpu"] = experiment_d_cpu_vs_gpu()

    print("[E] output integrity ...", flush=True)
    record["E_output_integrity"] = experiment_e_output_integrity(model, device)

    out_path = OUT_DIR / "record.json"
    out_path.write_text(json.dumps(record, indent=2, sort_keys=True, default=str), encoding="utf-8")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
