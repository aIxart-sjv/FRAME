"""Train the Phase 4 primary tiny model for several seeds, keeping the checkpoints OUTSIDE the repository (Phase 5).

    sen2sr_venv/bin/python experiments/evaluation/train_tiny_seeds.py [--seeds 0 1 2 3 4]

Runs `frame.train.run.execute` on the committed Phase 4 config (`experiments/training/smoke_tiny_cnn/config.json`: synthetic data, L1, 600 steps) with only ``seed``,
``output_dir`` and the evaluation extras changed. Each run leaves ``summary.json`` (which lists the scenes it trained and validated on, used by the evaluation's train/eval
overlap guard) and ``checkpoint/``. Trains on the synthetic smoke set only: no benchmark data is involved.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))

from frame.train.config import TrainConfig  # noqa: E402
from frame.train.run import execute  # noqa: E402

CONFIG = HERE.parent / "training" / "smoke_tiny_cnn" / "config.json"
OUT = Path.home() / ".cache" / "frame_eval" / "checkpoints"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    args = parser.parse_args()
    base = TrainConfig.load(CONFIG)
    for seed in args.seeds:
        out = OUT / f"tiny_cnn_seed{seed}"
        if (out / "summary.json").is_file():
            print(f"seed {seed}: already trained ({out})")
            continue
        config = replace(base, name=f"tiny_cnn_seed{seed}", seed=seed, output_dir=str(out), baselines=("bicubic",), evaluate_train=False)
        summary = execute(config, base_dir=CONFIG.parent)
        m = summary["validation"]["metrics"]
        print(f"seed {seed}: trained {summary['training']['steps']} steps; validation PSNR {m['psnr_db']:.3f} dB -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
