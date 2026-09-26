"""Local training feasibility (Phase 4): how much GPU memory and time does one training step really take?

Runs the REAL `frame.train` Trainer (forward, loss, backward, optimizer step, AMP, gradient accumulation) on random tensors of
the requested size, batch by batch size, and records peak VRAM, step time, parameter memory and host RSS. It needs only torch and the
torch-only training core, so it runs in BOTH environments:

    sen2sr_venv/bin/python        experiments/training/feasibility/measure_memory.py --model tiny_cnn --out results/tiny_cnn.json
    sen2sr_venv/bin/python        experiments/training/feasibility/measure_memory.py --model sen2sr_lite ...
    sen2sr_mamba_venv/bin/python  experiments/training/feasibility/measure_memory.py --model sen2sr_mamba --lr-sizes 16 32 64 128 ...

Safety on a 4 GB laptop GPU that also drives the desktop: the process is capped below the free memory it finds at start (so the CUDA
allocator raises a catchable OutOfMemoryError long before the device is exhausted), sizes are tried in ascending order and a group stops
at its first failure, and nothing is measured if free host RAM is low. An OOM is a RESULT ("oom"), not a crash.

Activation checkpointing (``--checkpointing on``) uses `frame.train.memory.enable_block_checkpointing`, which wraps every ``VSSBlock.forward``
of MambaSR from outside. Upstream's own ``use_checkpoint`` flag cannot be used: BasicLayer calls ``checkpoint(blk, x)`` without the required
``x_size`` argument. ``sen2sr/`` is not modified.
"""

from __future__ import annotations

import argparse
import gc
import json
import resource
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List

import torch

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from frame.train.config import ModelConfig, TrainConfig  # noqa: E402
from frame.train.memory import enable_block_checkpointing  # noqa: E402
from frame.train.models import build_model, count_parameters, parameter_megabytes  # noqa: E402
from frame.train.record import environment_info  # noqa: E402
from frame.train.trainer import Trainer  # noqa: E402

DESKTOP_RESERVE_MIB = 350           # GPU memory left for the display server
MIN_HOST_AVAILABLE_GIB = 3.0


class RandomPairs:
    def __init__(self, n: int, lr_size: int, scale: int = 4, seed: int = 0):
        g = torch.Generator().manual_seed(seed)
        self.items = [{"lr": torch.rand(4, lr_size, lr_size, generator=g) * 0.4, "hr": torch.rand(4, lr_size * scale, lr_size * scale, generator=g) * 0.4}
                      for _ in range(n)]

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int) -> Dict[str, torch.Tensor]:
        return self.items[i]


def host_available_gib() -> float:
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) / 2**20
    return float("inf")


def measure_one(model_name: str, lr_size: int, amp: str, checkpointing: bool, batch: int, accum: int, steps: int, pretrained: bool) -> Dict[str, Any]:
    result: Dict[str, Any] = {"model": model_name, "lr_size": lr_size, "hr_size": lr_size * 4, "amp": amp, "checkpointing": checkpointing,
                              "batch_size": batch, "grad_accum": accum, "steps": steps, "status": "ok"}
    trainer = model = None
    try:
        model = build_model(ModelConfig(model_name, {}, pretrained=pretrained), seed=0)
        if checkpointing:
            result["blocks_checkpointed"] = enable_block_checkpointing(model)
        with tempfile.TemporaryDirectory() as tmp:
            config = TrainConfig.from_dict({
                "name": "probe", "model": {"name": model_name}, "data": {"manifest": "unused"}, "steps": steps, "batch_size": batch, "grad_accum": accum,
                "precision": {"amp": amp}, "device": "cuda", "output_dir": tmp, "log_every": 1, "deterministic": False,
                "optim": {"name": "adamw", "lr": 1e-5},
            })
            trainer = Trainer(config, model, RandomPairs(batch * accum * 2, lr_size), manifest_digest="probe", scale=4, output_dir=Path(tmp))
            trainer._save_checkpoint = lambda: None                      # measuring steps, not disk
            summary = trainer.run()
            times = [json.loads(line)["step_seconds"] for line in (Path(tmp) / "metrics.jsonl").read_text().splitlines() if '"train"' in line]
        params = summary["parameters"]
        result.update(
            peak_allocated_mib=round(summary["peak_vram_mib"], 1), peak_reserved_mib=round(torch.cuda.max_memory_reserved() / 2**20, 1),
            step_seconds_first=times[0], step_seconds_steady=round(sum(times[1:]) / max(1, len(times) - 1), 4) if len(times) > 1 else None,
            parameters=params["total"], trainable_parameters=params["trainable"], parameter_mib=params["megabytes"],
            precision_effective=summary["precision"]["effective"], precision_note=summary["precision"]["note"],
            final_loss=summary["final_train_loss"],
        )
    except torch.cuda.OutOfMemoryError as exc:
        result.update(status="oom", error=str(exc).splitlines()[0][:200])
    except Exception as exc:  # AMP / kernel constraints etc. are results too
        result.update(status="error", error=f"{type(exc).__name__}: {str(exc)[:300]}")
    finally:
        del trainer, model
        gc.collect()
        torch.cuda.empty_cache()
    result["host_rss_peak_mib"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, choices=("tiny_cnn", "sen2sr_lite", "sen2sr_mamba"))
    parser.add_argument("--lr-sizes", type=int, nargs="+", default=[32, 64, 128])
    parser.add_argument("--amp", nargs="+", default=["off"], choices=("off", "fp16", "bf16"))
    parser.add_argument("--checkpointing", nargs="+", default=["off"], choices=("off", "on"))
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--accum", type=int, default=1)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--no-pretrained", action="store_true", help="random init instead of the published weights (memory is identical)")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("CUDA is not available; nothing to measure.", file=sys.stderr)
        return 2
    if host_available_gib() < MIN_HOST_AVAILABLE_GIB:
        print(f"Only {host_available_gib():.1f} GiB of host RAM is available (< {MIN_HOST_AVAILABLE_GIB}); refusing to start.", file=sys.stderr)
        return 2
    free, total = torch.cuda.mem_get_info()
    usable_mib = free / 2**20 - DESKTOP_RESERVE_MIB
    torch.cuda.set_per_process_memory_fraction(max(0.05, usable_mib * 2**20 / total))
    header = {"gpu_total_mib": round(total / 2**20), "gpu_free_at_start_mib": round(free / 2**20), "process_cap_mib": round(usable_mib), "desktop_reserve_mib": DESKTOP_RESERVE_MIB}
    print(f"GPU {header['gpu_total_mib']} MiB total, {header['gpu_free_at_start_mib']} MiB free; capping this process at {header['process_cap_mib']} MiB", flush=True)

    results: List[Dict[str, Any]] = []
    for amp in args.amp:
        for ckpt in args.checkpointing:
            failed = False
            for size in sorted(args.lr_sizes):
                if failed:
                    results.append({"model": args.model, "lr_size": size, "amp": amp, "checkpointing": ckpt == "on", "batch_size": args.batch, "status": "skipped",
                                    "error": "a smaller size already failed in this group"})
                    continue
                torch.cuda.reset_peak_memory_stats()
                r = measure_one(args.model, size, amp, ckpt == "on", args.batch, args.accum, args.steps, not args.no_pretrained)
                results.append(r)
                failed = r["status"] != "ok"
                brief = f"peak {r.get('peak_allocated_mib')} MiB, {r.get('step_seconds_steady')} s/step" if r["status"] == "ok" else r.get("error", "")[:90]
                print(f"  {args.model} LR {size:3d} amp={amp} ckpt={ckpt} batch={args.batch}: {r['status']}  {brief}", flush=True)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({"header": header, "environment": environment_info("cuda"), "results": results}, indent=2, sort_keys=True) + "\n")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
