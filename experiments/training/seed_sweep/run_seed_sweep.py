"""How much does the result move with the seed alone? (Phase 4)

Runs the tiny smoke configs over several seeds through the real `frame.train.run.execute` (temporary output directories) and records
validation PSNR / SSIM / RMSE per seed, so a difference between two configs can be judged against the seed-to-seed spread instead of
being read off one run. Same data, same manifest, same everything but ``seed`` (which also reseeds the model init, the data order and
the random patch draws).

    sen2sr_venv/bin/python experiments/training/seed_sweep/run_seed_sweep.py [--seeds 0 1 2 3 4]
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import tempfile
from dataclasses import replace
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2]))

from frame.train.config import TrainConfig  # noqa: E402
from frame.train.run import execute  # noqa: E402

CONFIGS = ("smoke_tiny_cnn", "smoke_tiny_cnn_charbonnier", "smoke_tiny_cnn_spectral")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    args = parser.parse_args()
    results = {}
    for name in CONFIGS:
        config_path = HERE.parent / name / "config.json"
        base = TrainConfig.load(config_path)
        rows = []
        for seed in args.seeds:
            with tempfile.TemporaryDirectory() as tmp:
                config = replace(base, seed=seed, output_dir=tmp, baselines=("bicubic",), evaluate_train=False, checkpoint_every=0, validate_every=0)
                s = execute(config, base_dir=config_path.parent)
            m = s["validation"]["metrics"]
            rows.append({"seed": seed, "psnr_db": m["psnr_db"], "ssim": m["ssim"], "rmse": m["rmse"], "bicubic_psnr_db": s["baselines"]["bicubic"]["metrics"]["psnr_db"]})
            print(f"{name:30s} seed {seed}: PSNR {m['psnr_db']:.3f} dB  SSIM {m['ssim']:.4f}  RMSE {m['rmse']:.5f}", flush=True)
        psnr = [r["psnr_db"] for r in rows]
        results[name] = {"runs": rows, "psnr_db_mean": statistics.mean(psnr), "psnr_db_stdev": statistics.stdev(psnr) if len(psnr) > 1 else None,
                         "psnr_db_min": min(psnr), "psnr_db_max": max(psnr)}
    (HERE / "results.json").write_text(json.dumps({"seeds": args.seeds, "note": "validation split only; single small synthetic set; not a benchmark", "configs": results}, indent=2, sort_keys=True) + "\n")
    for name, r in results.items():
        print(f"{name:30s} PSNR mean {r['psnr_db_mean']:.3f} +- {r['psnr_db_stdev']:.3f} (min {r['psnr_db_min']:.3f}, max {r['psnr_db_max']:.3f}) over {len(args.seeds)} seeds")
    return 0


if __name__ == "__main__":
    sys.exit(main())
