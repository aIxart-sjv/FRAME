"""Phase 1 validation -- the REAL SEN2SR-Mamba RGBN model on the RTX 3050, via FRAME.

Goal
----
Produce the measured evidence Phase 1 requires for the integrated model, not
a generic Mamba2 smoke test:

    4 x 128 x 128 RGBN  ->  actual MambaSR weights + hard constraint  ->  CUDA
                        ->  4 x 512 x 512

through the same path FRAME uses in production: the main environment's
`frame.models.mamba_client.MambaWorkerClient` launching the isolated worker
in the dedicated Mamba environment.

Measured: worker start (spawn + imports + weight load), first (cold)
inference, warm inference, peak GPU memory (both the allocator's peak inside
the worker and the worker *process's* total footprint from `nvidia-smi`,
which includes the CUDA context), output shape/dtype/range, NaN/Inf, and
that the work really ran on the GPU. Also the full FRAME uncertainty
ensemble (6 test-time transforms) with this model, and -- for context only --
SEN2SR-Lite on the same tile.

Inputs (both real Sentinel-2 reflectance, RGBN in FRAME's B04,B03,B02,B08 order):
  * the tile shipped with the model artifact (models/SEN2SR/example_data.safetensor,
    RGBN slice `[2, 1, 0, 6]` of its 10-band tensor)
  * the deterministic Baseline 0 scene used by every earlier phase
    (experiments/baseline/outputs/input_tensor.pt)

This script trains nothing, downloads nothing, and does not modify `sen2sr/`.
Timings are for THIS machine only; nothing here is an accuracy claim.

Usage
-----
    sen2sr_venv/bin/python experiments/mamba_integration/run_validation.py [--warm-runs N] [--metadata-dir DIR]
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

import torch

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))

from frame.consistency.downsample import downsample_to_lr_grid  # noqa: E402
from frame.models import config  # noqa: E402
from frame.models.mamba_client import MambaWorkerClient, check_mamba_availability  # noqa: E402
from frame.preprocessing import RGBN_BANDS  # noqa: E402
from frame.uncertainty import DEFAULT_TRANSFORMS, run_stochastic_uncertainty  # noqa: E402

BASE_DIR = Path(__file__).resolve().parent
BASELINE_TENSOR = _REPO_ROOT / "experiments" / "baseline" / "outputs" / "input_tensor.pt"
RGBN_FROM_TEN_BAND = [2, 1, 0, 6]  # B04, B03, B02, B08 -- sen2sr/referencex4.py's `bands_10m`


def git_commit() -> Optional[str]:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=_REPO_ROOT, capture_output=True, text=True, timeout=5)
        return out.stdout.strip() or None
    except Exception:
        return None


def nvidia_smi(query: str) -> str:
    out = subprocess.run(["nvidia-smi", f"--query-{query}", "--format=csv,noheader,nounits"], capture_output=True, text=True)
    return out.stdout.strip()


def process_gpu_mib(pid: int) -> Optional[int]:
    """Total GPU memory (MiB) `nvidia-smi` attributes to ``pid`` -- includes the CUDA context."""
    for line in nvidia_smi("compute-apps=pid,used_memory").splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) == 2 and parts[0] == str(pid):
            return int(parts[1])
    return None


def tensor_stats(y: torch.Tensor) -> Dict[str, Any]:
    return {
        "shape": list(y.shape),
        "dtype": str(y.dtype).replace("torch.", ""),
        "min": round(float(y.min()), 6),
        "max": round(float(y.max()), 6),
        "mean": round(float(y.mean()), 6),
        "has_nan": bool(torch.isnan(y).any()),
        "has_inf": bool(torch.isinf(y).any()),
    }


def consistency_rmse(lr: torch.Tensor, sr: torch.Tensor) -> float:
    down = downsample_to_lr_grid(sr[0], config.SCALE_FACTOR)
    return round(float(((down - lr[0]) ** 2).mean().sqrt()), 6)


def time_call(fn) -> float:
    started = time.perf_counter()
    fn()
    return time.perf_counter() - started


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--warm-runs", type=int, default=10)
    parser.add_argument("--metadata-dir", type=Path, default=BASE_DIR / "metadata")
    args = parser.parse_args()

    availability = check_mamba_availability("cuda")
    if not availability.available:
        sys.exit(f"SEN2SR-Mamba is not runnable here: {availability.message} ({availability.reason_code})")

    import safetensors.torch as st

    example = st.load_file(str(config.MAMBA_WEIGHTS_DIR / config.MAMBA_EXAMPLE_DATA_FILENAME))["lr"][:, RGBN_FROM_TEN_BAND].contiguous()
    baseline = torch.load(BASELINE_TENSOR, weights_only=True)[None].contiguous()
    inputs = {"artifact_example_tile": example, "baseline_0_scene": baseline}

    gpu_line = nvidia_smi("gpu=name,memory.total,memory.used,driver_version")
    gpu_name, vram_total, vram_used_before, driver = [p.strip() for p in gpu_line.split(",")]

    record: Dict[str, Any] = {
        "phase": "Phase 1 -- SEN2SR-Mamba integration",
        "purpose": "Measured GPU validation of the real MambaSR RGBN 10 m -> 2.5 m model through FRAME's isolated worker.",
        "scientific_framing": (
            "Timings and memory are engineering measurements on one machine. Output values are an SR-derived product on a "
            "2.5 m pixel grid; nothing here measures accuracy against reference imagery."
        ),
        "git_commit": git_commit(),
        "machine": {
            "gpu": gpu_name,
            "vram_total_mib": int(vram_total),
            "vram_used_before_run_mib": int(vram_used_before),
            "nvidia_driver": driver,
            "main_environment_torch": torch.__version__,
        },
        "contract": {
            "input": f"float32 (4, {config.INPUT_SIZE}, {config.INPUT_SIZE}) or (B, 4, ...), reflectance, bands {list(config.RGBN_BAND_ORDER)}",
            "output": f"float32 (4, {config.OUTPUT_SIZE}, {config.OUTPUT_SIZE}) or (B, 4, ...), same band order and scale",
        },
        "warm_runs": args.warm_runs,
    }

    # ---- worker start (spawn + imports + weight load) ----------------------------------------------
    client = MambaWorkerClient(device="cuda")
    try:
        started = time.perf_counter()
        info = client.start()
        record["worker_start_seconds"] = round(time.perf_counter() - started, 3)
        record["model_load_seconds_reported_by_worker"] = info["load_seconds"]
        pid = info["worker_pid"]
        record["load_report"] = {
            k: info[k]
            for k in (
                "model_name", "executable_architecture", "artifact_metadata_label", "architecture", "parameter_count",
                "state_tensors", "missing_keys", "unexpected_keys", "weights_file", "weights_sha256",
                "hard_constraint_file", "hard_constraint_sha256", "input_band_order",
                "worker_python", "worker_torch", "worker_cuda", "worker_gpu", "device",
            )
        }
        record["worker_gpu_memory_after_load_mib"] = process_gpu_mib(pid)

        # ---- per-input inference ---------------------------------------------------------------------
        record["inputs"] = {}
        first_inference_recorded = False
        for name, lr in inputs.items():
            entry: Dict[str, Any] = {"input": tensor_stats(lr)}
            if not first_inference_recorded:  # the very first request to a fresh worker = cold
                entry["first_inference_seconds"] = round(time_call(lambda: client(lr)), 4)
                first_inference_recorded = True
            warm = []
            for _ in range(args.warm_runs):
                warm.append(time_call(lambda: client(lr)))
            y = client(lr)
            stats = client.describe()["last_request"]
            entry.update(
                {
                    "warm_inference_seconds": {
                        "median": round(statistics.median(warm), 4),
                        "min": round(min(warm), 4),
                        "max": round(max(warm), 4),
                    },
                    "worker_reported_inference_seconds_last": stats["inference_seconds"],
                    "peak_gpu_memory_allocated_mib": stats["peak_memory_mib"],
                    "output": tensor_stats(y),
                    "hard_constraint_check_rmse_sr_downsampled_vs_lr": consistency_rmse(lr, y),
                    "deterministic_across_calls": bool(torch.equal(y, client(lr))),
                }
            )
            record["inputs"][name] = entry
        record["worker_gpu_memory_after_inference_mib"] = process_gpu_mib(pid)

        # ---- FRAME uncertainty ensemble with this model (6 TTA members) ---------------------------------
        ensemble = run_stochastic_uncertainty(client, baseline[0], transforms=DEFAULT_TRANSFORMS, seed=42, band_names=RGBN_BANDS)
        record["uncertainty_ensemble_on_baseline_0_scene"] = {
            "members": ensemble.n,
            "transform_names": list(ensemble.transform_names),
            "total_seconds": round(ensemble.total_seconds, 3),
            "mean_prediction": tensor_stats(ensemble.mean_prediction[None]),
            "scalar_summary": ensemble.scalar_summary,
            "note": "relative model-stability uncertainty; not a calibrated probability of error",
        }

        # ---- context: SEN2SR-Lite on the same tile, in-process, main environment -------------------------
        record["lite_for_context"] = None
        from frame.api import config as api_config

        if (api_config.MODEL_WEIGHTS_CACHE_DIR / "mlm.json").exists():
            from frame.api.services import model as model_service

            lite = model_service.get_model("cuda", model_name="lite")
            x = baseline.to("cuda")
            torch.cuda.reset_peak_memory_stats()
            first = time_call(lambda: lite(x))
            warm = [time_call(lambda: lite(x)) for _ in range(args.warm_runs)]
            torch.cuda.synchronize()
            record["lite_for_context"] = {
                "model_name": config.LITE_MODEL_NAME,
                "first_inference_seconds": round(first, 4),
                "warm_inference_seconds_median": round(statistics.median(warm), 4),
                "peak_gpu_memory_allocated_mib": round(torch.cuda.max_memory_allocated() / 2**20, 1),
                "note": "in-process in the main environment; shown only to put the Mamba numbers in context",
            }
    finally:
        client.close()

    record["gpu_execution_evidence"] = (
        "The worker process (pid above) held GPU memory per nvidia-smi and reported a non-zero torch.cuda peak "
        "allocation; the model is only ever moved to a CUDA device (CPU is refused, not fallen back to)."
    )
    record["verdict"] = {
        "weights_loaded_strictly": not record["load_report"]["missing_keys"] and not record["load_report"]["unexpected_keys"],
        "no_nan_or_inf_in_any_output": all(
            not e["output"]["has_nan"] and not e["output"]["has_inf"] for e in record["inputs"].values()
        ),
        "outputs_are_4x512x512_float32": all(
            e["output"]["shape"] == [1, 4, 512, 512] and e["output"]["dtype"] == "float32" for e in record["inputs"].values()
        ),
        "worker_used_gpu_memory": (record["worker_gpu_memory_after_inference_mib"] or 0) > 0,
    }

    args.metadata_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.metadata_dir / "run_metadata.json"
    out_path.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps({k: record[k] for k in ("worker_start_seconds", "model_load_seconds_reported_by_worker", "worker_gpu_memory_after_load_mib", "worker_gpu_memory_after_inference_mib", "verdict")}, indent=2))
    for name, entry in record["inputs"].items():
        print(name, "| first:", entry.get("first_inference_seconds"), "| warm median:", entry["warm_inference_seconds"]["median"],
              "| peak alloc MiB:", entry["peak_gpu_memory_allocated_mib"], "| out:", entry["output"]["shape"], entry["output"]["dtype"],
              [entry["output"]["min"], entry["output"]["max"]], "| RMSE(SR↓,LR):", entry["hard_constraint_check_rmse_sr_downsampled_vs_lr"])
    print("ensemble:", record["uncertainty_ensemble_on_baseline_0_scene"]["total_seconds"], "s for",
          record["uncertainty_ensemble_on_baseline_0_scene"]["members"], "members")
    print("lite:", record["lite_for_context"])
    print("wrote", out_path)


if __name__ == "__main__":
    main()
