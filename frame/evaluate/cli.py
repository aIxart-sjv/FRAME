"""Command line for the evaluation layer (Phase 5).

    python -m frame.evaluate check CONFIG.json     validate the config, the datasets (roles, references, geometry, files) and every system's availability; evaluate nothing
    python -m frame.evaluate run   CONFIG.json     run the evaluation and write experiments/evaluation/<name>/
    python -m frame.evaluate shift CONFIG.json --dataset NAME [--systems A B] [--lr-shifts 0 0.25 0.5 1 2] [--output-dir DIR]
                                                    spatial-shift sensitivity (requirements 142 section 27) on one configured dataset

Relative paths in a config resolve against the directory of the config file. Exit codes: 0 success; 1 the run finished but nothing could be evaluated for some
dataset; 2 refused or invalid input (bad config, role violation, train/eval overlap, existing output directory, unreadable manifest).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional

from frame.evaluate.config import EvalConfig
from frame.evaluate.errors import EvaluationError, RoleSafetyError, SystemUnavailableError


def _device(config: EvalConfig) -> str:
    from frame.evaluate.runner import _resolve_device

    return _resolve_device(config.device)


def _cmd_check(args: argparse.Namespace) -> int:
    from frame.evaluate.datasets import build_dataset
    from frame.evaluate.systems import build_system, check_no_training_overlap

    config = EvalConfig.load(args.config)
    base = Path(args.config).resolve().parent
    datasets = [build_dataset(spec, base_dir=base) for spec in config.datasets]
    print(f"config OK: {config.name} (digest {config.digest()[:16]})")
    for ds in datasets:
        v = ds.validate(check_files=True)
        groups = {ds.scene_group_of(r) for r in v.valid_records}
        print(f"dataset {ds.name!r} ({ds.kind}, role {ds.role}, split {ds.spec.split}, evidence {ds.evidence_class}): {len(ds.records)} records, "
              f"{len(v.valid_records)} evaluable, {len(v.invalid)} invalid, {len(groups)} scene units; manifest digest {ds.manifest_digest[:16]}")
        for bad in v.invalid[:5]:
            print(f"  invalid {bad['sample_id']}: {bad['codes']}")
    tiling, device = config.tiling_config(), _device(config)
    for spec in config.systems:
        status = "available"
        try:
            if spec.kind == "checkpoint":
                system = build_system(spec, tiling, device="cpu", base_dir=base)
                for ds in datasets:
                    check_no_training_overlap(system, dataset_kind=ds.kind, scene_ids=[r.scene_id for r in ds.records])
            elif spec.kind == "mamba":
                from frame.models.mamba_client import check_mamba_availability

                a = check_mamba_availability(device)
                status = "available" if a.available else f"UNAVAILABLE ({a.message})"
            elif spec.kind == "lite":
                import os

                d = Path(spec.params.get("weights_dir") or os.environ.get("SEN2SR_BASELINE_WEIGHTS_DIR") or Path.home() / ".cache" / "sen2sr_baseline" / "SEN2SRLite_RGBN")
                status = "available" if (d / "model.safetensor").is_file() else f"UNAVAILABLE (weights not found in {d})"
        except SystemUnavailableError as exc:
            status = f"UNAVAILABLE ({exc})"
        print(f"system {spec.name!r} ({spec.kind}, hard constraint {spec.hard_constraint}): {status}")
    return 0


def _cmd_run(args: argparse.Namespace) -> int:
    from frame.evaluate.runner import run_evaluation

    config = EvalConfig.load(args.config)
    summary = run_evaluation(config, base_dir=Path(args.config).resolve().parent, progress=(lambda m: print(f"  {m}", flush=True)) if args.verbose else None)
    print(f"evaluation '{summary['name']}' {summary['status']}; record in {config.output_dir}")
    empty = []
    for name, d in summary["datasets"].items():
        c = d["counts"]
        print(f"dataset {name}: {c['evaluated']} evaluated, {c['skipped']} skipped, {c['invalid']} invalid, {c['unreadable']} unreadable of {c['total']} ({len(d['scene_units'])} scene units)")
        if c["evaluated"] == 0:
            empty.append(name)
    for u in summary["systems_unavailable"]:
        print(f"system {u['name']} UNAVAILABLE: {u['reason']}")
    for evidence, datasets in summary["sections"].items():
        for name, block in datasets.items():
            for system, row in block["headline"].items():
                psnr, rmse = row.get("reference_accuracy.psnr_db"), row.get("reference_accuracy.rmse")
                if psnr and rmse:
                    print(f"  [{evidence}] {name} / {system}: PSNR {psnr['mean']:.3f} dB, RMSE {rmse['mean']:.5f} (mean over {psnr['n_units']} scene units; descriptive, not a ranking)")
    if empty:
        print(f"error: nothing was evaluated for dataset(s) {empty}; see the skipped/invalid/unreadable lists in summary.json", file=sys.stderr)
        return 1
    return 0


def _cmd_shift(args: argparse.Namespace) -> int:
    from frame.evaluate.shift import SWEEP_LR_PIXELS, run_shift_sensitivity

    config = EvalConfig.load(args.config)
    lr_shifts = tuple(args.lr_shifts) if args.lr_shifts else SWEEP_LR_PIXELS
    summary = run_shift_sensitivity(config, dataset=args.dataset, system_names=args.systems, lr_shifts=lr_shifts, output_dir=args.output_dir, base_dir=Path(args.config).resolve().parent,
                                    progress=(lambda m: print(f"  {m}", flush=True)) if args.verbose else None)
    c = summary["counts"]
    print(f"shift sensitivity '{summary['name']}' {summary['status']} on {summary['dataset']}: {c['evaluated']} evaluated, {c['skipped']} skipped, {c['invalid']} invalid, "
          f"{c['unreadable']} unreadable of {c['total']}; conditions {summary['conditions']}")
    print("An interpretation aid, not a ranking: descriptive, per scene unit; see the README in the output directory.")
    for u in summary["systems_unavailable"]:
        print(f"system {u['name']} UNAVAILABLE: {u['reason']}")
    if c["evaluated"] == 0:
        print("error: nothing was evaluated; see the skipped/invalid/unreadable lists in summary.json", file=sys.stderr)
        return 1
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m frame.evaluate", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="run the evaluation and write its record")
    run.add_argument("config")
    run.add_argument("--verbose", action="store_true", help="print each sample as it is evaluated")
    run.set_defaults(func=_cmd_run)
    check = sub.add_parser("check", help="validate config, datasets and systems; evaluate nothing")
    check.add_argument("config")
    check.set_defaults(func=_cmd_check)
    shift = sub.add_parser("shift", help="spatial-shift sensitivity of the reference metrics on one configured dataset (not a ranking)")
    shift.add_argument("config")
    shift.add_argument("--dataset", required=True, help="name of a dataset in the config")
    shift.add_argument("--systems", nargs="+", help="restrict to these configured systems (default: all)")
    shift.add_argument("--lr-shifts", nargs="+", type=float, help="displacements in LR pixels (default 0 0.25 0.5 1 2); each must be a whole number of HR pixels")
    shift.add_argument("--output-dir", help="where to write the record (default: the config's output_dir)")
    shift.add_argument("--verbose", action="store_true")
    shift.set_defaults(func=_cmd_shift)
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except RoleSafetyError as exc:
        print(f"error: {exc} [{', '.join(exc.codes)}]", file=sys.stderr)
        return 2
    except EvaluationError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:
        from frame.data.errors import DataError

        if isinstance(exc, DataError):
            print(f"error: {exc}", file=sys.stderr)
            return 2
        raise


if __name__ == "__main__":
    sys.exit(main())
