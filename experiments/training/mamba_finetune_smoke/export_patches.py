"""Step 1 of the Mamba fine-tuning smoke (MAIN environment): export a few real training/validation patches as plain tensors.

The Mamba environment has torch and mamba_ssm but not rasterio / scikit-image, so it cannot import the Phase 3 data layer. This script runs
in the main environment, applies the same manifest -> geographic-split -> leakage gate as any training run (frame.train.data), and writes
tensors that the Mamba environment can load with ``torch.load(weights_only=True)``: N random augmented crops from the TRAIN split and every
validation patch of the VALIDATION split. Test-split records are never read.

    sen2sr_venv/bin/python experiments/training/mamba_finetune_smoke/export_patches.py [--n-train 32] [--lr-patch 64]
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2]))

from frame.data.config import data_root  # noqa: E402
from frame.train.config import TrainConfig  # noqa: E402
from frame.train.data import prepare_training_data  # noqa: E402

SOURCE_CONFIG = HERE.parent / "smoke_tiny_cnn" / "config.json"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-train", type=int, default=32)
    parser.add_argument("--lr-patch", type=int, default=64)
    parser.add_argument("--out", default=str(data_root() / "exports" / "mamba_smoke_patches.pt"))
    args = parser.parse_args()

    base = TrainConfig.load(SOURCE_CONFIG)
    config = replace(base, data=replace(base.data, lr_patch=args.lr_patch, val_lr_patch=128, patches_per_pair=max(1, args.n_train // 6 + 1)))
    data = prepare_training_data(config, base_dir=SOURCE_CONFIG.parent)
    train = [data.train[i] for i in range(min(args.n_train, len(data.train)))]
    val = [data.val[i] for i in range(len(data.val))]
    payload = {
        "train": {"lr": torch.stack([t["lr"] for t in train]), "hr": torch.stack([t["hr"] for t in train])},
        "val": {"lr": torch.stack([v["lr"] for v in val]), "hr": torch.stack([v["hr"] for v in val])},
        "manifest_digest": data.manifest_digest, "scale": data.scale, "bands": list(data.band_names), "lr_patch": args.lr_patch,
        "train_scenes": data.info["train_scenes"], "val_scenes": data.info["val_scenes"], "ignored_records": data.info["ignored_records"],
        "source_config": SOURCE_CONFIG.parent.name,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, out)
    print(f"wrote {out} ({out.stat().st_size / 2**20:.1f} MiB): {len(train)} train patches (LR {args.lr_patch}), {len(val)} validation patches (LR 128); "
          f"manifest digest {data.manifest_digest[:16]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
