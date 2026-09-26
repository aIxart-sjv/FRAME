"""What is actually reproducible? (Phase 4)

Runs the primary smoke experiment (`experiments/training/smoke_tiny_cnn`) TWICE per setting with the same data, seed and config, through the
real `frame.train.run.execute`, and compares the two runs: the training-loss trajectory, the validation metrics and the final weights
(compared bit for bit and by maximum absolute difference). Settings: CPU, GPU with deterministic kernels requested (the default), GPU with
them not requested, and CPU vs GPU (a different device is a different experiment; reported, not expected to match).

    sen2sr_venv/bin/python experiments/training/reproducibility/compare_runs.py

The documented tolerance is asserted by tests/test_train_cli.py (CPU: exact). This script measures the GPU case instead of assuming it.
"""

from __future__ import annotations

import json
import sys
import tempfile
from dataclasses import replace
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
CONFIG = HERE.parents[0] / "smoke_tiny_cnn" / "config.json"
sys.path.insert(0, str(HERE.parents[2]))

from frame.train import checkpoint as ckpt  # noqa: E402
from frame.train.config import TrainConfig  # noqa: E402
from frame.train.run import execute  # noqa: E402


def run_once(device: str, deterministic: bool):
    base = TrainConfig.load(CONFIG)
    tmp = tempfile.mkdtemp()
    config = replace(base, device=device, deterministic=deterministic, output_dir=tmp, baselines=("bicubic",), evaluate_train=False)
    summary = execute(config, base_dir=CONFIG.parent)
    losses = [json.loads(line)["loss"]["total"] for line in (Path(tmp) / "metrics.jsonl").read_text().splitlines() if '"train"' in line]
    weights = ckpt.load_checkpoint(ckpt.latest_checkpoint(Path(tmp) / "checkpoint"))["model"]
    return {"losses": losses, "val": summary["validation"]["metrics"], "weights": weights}


def compare(a, b):
    diffs = [float((a["weights"][k].float() - b["weights"][k].float()).abs().max()) for k in a["weights"]]
    return {
        "train_loss_trajectory_bitwise_equal": a["losses"] == b["losses"],
        "max_abs_train_loss_diff": max(abs(x - y) for x, y in zip(a["losses"], b["losses"])),
        "val_metrics_bitwise_equal": a["val"] == b["val"],
        "max_abs_val_psnr_diff_db": abs(a["val"]["psnr_db"] - b["val"]["psnr_db"]),
        "weights_bitwise_equal": all(torch.equal(a["weights"][k], b["weights"][k]) for k in a["weights"]),
        "max_abs_weight_diff": max(diffs),
    }


def main() -> int:
    out = {"config": str(CONFIG.relative_to(HERE.parents[2])), "steps": TrainConfig.load(CONFIG).steps, "torch": str(torch.__version__), "comparisons": {}}
    runs = {}
    for label, device, deterministic in (("cpu_a", "cpu", True), ("cpu_b", "cpu", True), ("gpu_det_a", "cuda", True), ("gpu_det_b", "cuda", True),
                                         ("gpu_nondet_a", "cuda", False), ("gpu_nondet_b", "cuda", False)):
        if device == "cuda" and not torch.cuda.is_available():
            continue
        runs[label] = run_once(device, deterministic)
        print(f"ran {label}: val PSNR {runs[label]['val']['psnr_db']:.6f}", flush=True)
    for name, (x, y) in {"cpu_repeat": ("cpu_a", "cpu_b"), "gpu_deterministic_repeat": ("gpu_det_a", "gpu_det_b"),
                         "gpu_default_kernels_repeat": ("gpu_nondet_a", "gpu_nondet_b"), "cpu_vs_gpu": ("cpu_a", "gpu_det_a")}.items():
        if x in runs and y in runs:
            out["comparisons"][name] = compare(runs[x], runs[y])
            print(name, json.dumps(out["comparisons"][name]), flush=True)
    (HERE / "results.json").write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
