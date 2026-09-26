"""Command line for the training layer (Phase 4).

    python -m frame.train run   CONFIG.json [--resume latest|CHECKPOINT] [--output-dir DIR]
    python -m frame.train check CONFIG.json

``check`` validates the config, applies the geographic-leakage gate to the manifest and builds the model, without training.
Relative paths in a config resolve against the directory of the config file.

Exit codes: 0 success; 1 the run failed while training (non-finite loss, out of GPU memory); 2 a refusal or invalid input (bad config,
leakage / role violation, missing or edited manifest, incompatible resume, unbuildable model) -- in which case nothing was trained.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional

import torch

from frame.train.config import TrainConfig
from frame.train.errors import NonFiniteLossError, TrainError

#: Libraries the data layer and evaluation need; absent from the isolated Mamba environment.
DATA_LAYER_LIBRARIES = ("rasterio", "skimage", "opensr_test", "mlstac")


def _load(path: str) -> TrainConfig:
    return TrainConfig.load(path)


def _cmd_check(args: argparse.Namespace) -> int:
    from frame.train.data import prepare_training_data
    from frame.train.models import build_model, count_parameters, parameter_megabytes

    config = _load(args.config)
    data = prepare_training_data(config, base_dir=Path(args.config).resolve().parent)
    model = build_model(config.model, seed=config.seed)
    i = data.info
    print(f"config OK: {config.name} (digest {config.digest()[:16]})")
    print(f"manifest OK: digest {data.manifest_digest}; datasets {i['datasets']}; ignored records {i['ignored_records'] or 'none'}")
    print(f"{i['n_train_pairs']} train pair(s) in {len(i['train_scenes'])} scene(s), {i['n_val_pairs']} validation pair(s) in {len(i['val_scenes'])} scene(s); "
          f"{i['n_train_patches_per_epoch']} train patches/epoch, {i['n_val_patches']} validation patches")
    print(f"model {config.model.name}: {count_parameters(model)} parameters ({parameter_megabytes(model):.3f} MiB), scale x{data.scale}")
    return 0


def _cmd_run(args: argparse.Namespace) -> int:
    from frame.train.run import execute

    config = _load(args.config)
    if args.output_dir:
        config = config.with_output_dir(str(args.output_dir))
    summary = execute(config, base_dir=Path(args.config).resolve().parent, resume=args.resume)
    t, v = summary["training"], summary["validation"] or {}
    print(f"run '{summary['name']}' {summary['status']}: {t['steps']} step(s) on {t['device']} in {t['train_seconds']} s; peak VRAM {t['peak_vram_mib']} MiB")
    if v:
        m = v["metrics"]
        print(f"validation ({v['n_patches']} patches): loss {v['loss']['total']:.5f}, PSNR {m.get('psnr_db')}, SSIM {m.get('ssim')}, RMSE {m.get('rmse')}")
    for name, block in summary["baselines"].items():
        print(f"baseline {name}: loss {block['loss']['total']:.5f}, PSNR {block['metrics'].get('psnr_db')}, RMSE {block['metrics'].get('rmse')}")
    print(f"manifest digest {summary['manifest_digest']}; record in {config.output_dir}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m frame.train", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="train and validate as configured")
    run.add_argument("config")
    run.add_argument("--resume", help="'latest' or a checkpoint file: continue an interrupted run")
    run.add_argument("--output-dir", help="override the config's output directory")
    run.set_defaults(func=_cmd_run)
    check = sub.add_parser("check", help="validate the config and manifest (leakage gate) without training")
    check.add_argument("config")
    check.set_defaults(func=_cmd_check)
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except NonFiniteLossError as exc:
        print(f"error: training failed: {exc}", file=sys.stderr)
        return 1
    except torch.cuda.OutOfMemoryError as exc:
        print(f"error: training failed: out of GPU memory ({exc}); reduce batch_size / data.lr_patch or use a smaller model.", file=sys.stderr)
        return 1
    except TrainError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except ModuleNotFoundError as exc:
        if exc.name in DATA_LAYER_LIBRARIES:
            print(f"error: this command needs the main FRAME environment: the library {exc.name!r} is not installed here. The isolated Mamba "
                  "environment can only run the torch-only training core (docs/TRAINING.md section 2).", file=sys.stderr)
            return 2
        raise
    except Exception as exc:  # frame.data errors (DataError) are refusals too; anything else is a bug and keeps its traceback
        from frame.data.errors import DataError

        if isinstance(exc, DataError):
            print(f"error: {exc}", file=sys.stderr)
            return 2
        raise


if __name__ == "__main__":
    sys.exit(main())
