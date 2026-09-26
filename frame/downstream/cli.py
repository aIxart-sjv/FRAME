"""Command line of the downstream layer (Phase 7).

    python -m frame.downstream check CONFIG.json    validate the config, the datasets and the systems, and PREVIEW the reference gate and the region counts on every tile (no model runs, nothing written)
    python -m frame.downstream run   CONFIG.json    run the NDVI downstream analysis and write experiments/downstream/runs/<name>/ (machine-readable JSON and a Markdown README)
    python -m frame.downstream smoke [--output-dir DIR]   a small end-to-end run on synthetic scenes and toy models: no dataset, weights or GPU needed; the numbers mean nothing scientifically

Relative paths in a config resolve against the directory of the config file. Exit codes: 0 success; 1 the run finished but some dataset has no eligible evidence; 2 refused or invalid input.
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path
from typing import List, Optional

from frame.downstream.config import DownstreamConfig
from frame.evaluate.errors import EvaluationError, RoleSafetyError, SystemUnavailableError


def _cmd_check(args: argparse.Namespace) -> int:
    import numpy as np

    from frame.downstream.regions import candidate_region_count
    from frame.evaluate.datasets import build_dataset
    from frame.evaluate.runner import _resolve_device
    from frame.evaluate.systems import BicubicSystem, build_system, check_no_training_overlap
    from frame.reliability.eligibility import GATE_VERSION, evaluate_reference

    config = DownstreamConfig.load(args.config)
    base = Path(args.config).resolve().parent
    device = _resolve_device(config.device)
    print(f"config OK: {config.name} (digest {config.digest()[:16]}); NDVI threshold {config.decision.ndvi_threshold} (sensitivity {list(config.decision.sensitivity_thresholds)}); "
          f"regions {list(config.regions.scales_hr_px)} HR px; gate {GATE_VERSION}")
    datasets = [build_dataset(spec, base_dir=base) for spec in config.datasets]
    for spec in config.systems:
        status = "available"
        try:
            if spec.kind == "checkpoint":
                system = build_system(spec, config.tiling_config(), device="cpu", base_dir=base)
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
        print(f"system {spec.name!r} ({spec.kind}): {status}")
    reference, metric_cfg = BicubicSystem("bicubic"), config.metric_config()
    primary = config.regions.primary_scale_hr_px
    for ds in datasets:
        v = ds.validate(check_files=True)
        levels: dict = {}
        reasons: dict = {}
        candidate = eligible_candidate = 0
        for record in sorted(v.valid_records, key=lambda r: r.sample_id):
            try:
                sample = ds.load(record)
            except Exception as exc:                            # noqa: BLE001
                reasons["sample_unreadable"] = reasons.get("sample_unreadable", 0) + 1
                print(f"  unreadable {record.sample_id}: {type(exc).__name__}: {exc}")
                continue
            n = candidate_region_count(tuple(sample.hr.shape[-2:]), primary)
            gate = evaluate_reference(sample, reference.infer(sample.lr).sr.numpy().astype(np.float64), config.eligibility, config.alignment, metric_cfg)
            candidate += n
            levels[gate["evidence_level"]] = levels.get(gate["evidence_level"], 0) + 1
            if gate["status"] == "eligible":
                eligible_candidate += n
            else:
                reasons[gate["reason"]] = reasons.get(gate["reason"], 0) + 1
            if args.verbose:
                a = gate["alignment"] or {}
                print(f"  {sample.sample_id}: {gate['evidence_level']} ({gate['reason']}); raw displacement {a.get('raw_magnitude')}, correction {a.get('correction')}, residual {a.get('residual_magnitude')}")
        print(f"dataset {ds.name!r} ({ds.kind}, {ds.evidence_class}): {len(ds.records)} records, {len(v.invalid)} invalid; reference gate preview: {dict(sorted(levels.items()))}; "
              f"exclusion reasons: {dict(sorted(reasons.items())) or 'none'}; candidate regions ({primary * 2.5:g} m) {candidate:,}, in eligible tiles {eligible_candidate:,}")
    return 0


def _cmd_run(args: argparse.Namespace) -> int:
    from frame.downstream.runner import run_downstream

    config = DownstreamConfig.load(args.config)
    summary = run_downstream(config, base_dir=Path(args.config).resolve().parent, progress=(lambda m: print(f"  {m}", flush=True)) if args.verbose else None, reuse_regions=args.reuse_regions)
    print(f"downstream run '{summary['name']}' {summary['status']}; record in {config.output_dir}")
    empty = []
    primary = str(config.regions.primary_scale_hr_px)
    for name, d in summary["datasets"].items():
        lv = d["gate"]["by_evidence_level"]
        r = d["regions"][primary]
        print(f"dataset {name}: {d['counts']['total']} records; gate {dict(sorted(lv.items()))}; candidate regions {r['candidate']:,}, analysed {r['analysed']:,}; eligible scene units {len(d['units_eligible'])}/{len(d['scene_units_all'])}")
        ov = summary["overview"][name]
        print(f"  {ov['status']} ({ov['n_tiles']} eligible tiles, {ov['n_units']} scene units{'; descriptive only' if ov.get('descriptive_only') else ''})")
        if ov["status"] != "ok":
            empty.append(name)
    for u in summary["systems_unavailable"]:
        print(f"system {u['name']} UNAVAILABLE: {u['reason']}")
    print(f"{summary['secondary_tasks']['landcover']}; {summary['indian_data']['status']}")
    print("No ranking is made; stability is not a calibrated uncertainty; see README.md in the output directory.")
    if empty:
        print(f"error: no eligible evidence for {empty}; see the gate and exclusion lists in summary.json and tiles.jsonl", file=sys.stderr)
        return 1
    return 0


def _cmd_smoke(args: argparse.Namespace) -> int:
    from frame.downstream.smoke import run_smoke

    out = Path(args.output_dir) if args.output_dir else Path(tempfile.mkdtemp(prefix="frame_downstream_smoke_")) / "run"
    summary = run_smoke(out, n_scenes=args.scenes, progress=(lambda m: print(f"  {m}", flush=True)) if args.verbose else None)
    d = summary["datasets"]["smoke"]
    ov = summary["overview"]["smoke"]
    print(f"smoke run {summary['status']}: {ov['n_tiles']} synthetic scenes, {d['regions']['4']['analysed']:,} regions analysed; record in {out}")
    print("The numbers are synthetic and mean nothing scientifically: this only shows the pipeline runs end to end.")
    return 0 if summary["status"] == "completed" and ov["status"] == "ok" and (out / "README.md").is_file() else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m frame.downstream", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    check = sub.add_parser("check", help="validate config, datasets, systems and preview the reference gate and region counts; run no model")
    check.add_argument("config")
    check.add_argument("--verbose", action="store_true")
    check.set_defaults(func=_cmd_check)
    run = sub.add_parser("run", help="run the downstream NDVI analysis and write its record")
    run.add_argument("config")
    run.add_argument("--verbose", action="store_true")
    run.add_argument("--reuse-regions", action="store_true", help="reuse cached region tables (cache_dir) whose settings, dataset and model digest are unchanged, so only the analysis is repeated")
    run.set_defaults(func=_cmd_run)
    smoke = sub.add_parser("smoke", help="an end-to-end run on synthetic scenes and toy models (no dataset, weights or GPU)")
    smoke.add_argument("--output-dir", help="where to write the record (default: a new temporary directory)")
    smoke.add_argument("--scenes", type=int, default=6)
    smoke.add_argument("--verbose", action="store_true")
    smoke.set_defaults(func=_cmd_smoke)
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
