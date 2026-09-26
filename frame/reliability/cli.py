"""Command line of the reliability layer (Phase 6).

    python -m frame.reliability check CONFIG.json    validate the config, the datasets and the systems, and PREVIEW the reference eligibility gate on every tile (no model runs, nothing written)
    python -m frame.reliability run   CONFIG.json    run the stability-vs-error experiment and write experiments/uncertainty/runs/<name>/
    python -m frame.reliability scene CONFIG.json --system NAME --lr-geotiff FILE [--window R0 R1 C0 C1] --output-dir DIR
                                                     the multi-tile / rectangular-scene check of the stability map (coverage, orientation, seams, georeferencing)

Relative paths in a config resolve against the directory of the config file. Exit codes: 0 success; 1 the run finished but nothing could be analysed for some dataset; 2 refused or invalid input.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional

from frame.evaluate.errors import EvaluationError, RoleSafetyError, SystemUnavailableError
from frame.reliability.config import ReliabilityConfig


def _cmd_check(args: argparse.Namespace) -> int:
    import numpy as np

    from frame.evaluate.datasets import build_dataset
    from frame.evaluate.systems import BicubicSystem, build_system, check_no_training_overlap
    from frame.reliability.eligibility import evaluate_reference
    from frame.evaluate.runner import _resolve_device

    config = ReliabilityConfig.load(args.config)
    base = Path(args.config).resolve().parent
    device = _resolve_device(config.device)
    print(f"config OK: {config.name} (digest {config.digest()[:16]}); alignment tolerance {config.alignment.tolerance_hr_px} HR px, min valid fraction {config.eligibility.min_valid_fraction}")
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
    for ds in datasets:
        v = ds.validate(check_files=True)
        levels: dict = {}
        reasons: dict = {}
        for record in sorted(v.valid_records, key=lambda r: r.sample_id):
            try:
                sample = ds.load(record)
            except Exception as exc:                            # noqa: BLE001
                reasons["sample_unreadable"] = reasons.get("sample_unreadable", 0) + 1
                print(f"  unreadable {record.sample_id}: {type(exc).__name__}: {exc}")
                continue
            gate = evaluate_reference(sample, reference.infer(sample.lr).sr.numpy().astype(np.float64), config.eligibility, config.alignment, metric_cfg)
            levels[gate["evidence_level"]] = levels.get(gate["evidence_level"], 0) + 1
            if gate["reason"]:
                reasons[gate["reason"]] = reasons.get(gate["reason"], 0) + 1
            if args.verbose:
                a = gate["alignment"] or {}
                print(f"  {sample.sample_id}: {gate['evidence_level']} ({gate['reason']}); raw displacement {a.get('raw_magnitude')}, correction {a.get('correction')}, residual {a.get('residual_magnitude')}")
        print(f"dataset {ds.name!r} ({ds.kind}, {ds.evidence_class}): {len(ds.records)} records, {len(v.invalid)} invalid; reference gate preview: {dict(sorted(levels.items()))}; exclusion reasons: {dict(sorted(reasons.items())) or 'none'}")
    return 0


def _cmd_run(args: argparse.Namespace) -> int:
    from frame.reliability.runner import run_reliability

    config = ReliabilityConfig.load(args.config)
    summary = run_reliability(config, base_dir=Path(args.config).resolve().parent, progress=(lambda m: print(f"  {m}", flush=True)) if args.verbose else None, reuse_evidence=args.reuse_evidence)
    print(f"reliability run '{summary['name']}' {summary['status']}; record in {config.output_dir}")
    empty = []
    for name, d in summary["datasets"].items():
        lv = d["gate"]["by_evidence_level"]
        print(f"dataset {name}: {d['counts']['total']} records; gate: {dict(sorted(lv.items()))}; excluded: {dict(sorted(d['gate']['excluded_by_reason'].items())) or 'none'}")
        for system, o in summary["overview"][name].items():
            print(f"  {system}: {o['status']} ({o['n_tiles']} eligible tiles, {o['n_units']} scene units{'; descriptive only' if o.get('descriptive_only') else ''})")
            if o["status"] != "ok":
                empty.append(f"{name}/{system}")
    for u in summary["systems_unavailable"]:
        print(f"system {u['name']} UNAVAILABLE: {u['reason']}")
    print("Stability is a relative model-stability proxy, not a calibrated uncertainty; see README.md in the output directory.")
    if empty:
        print(f"error: no eligible evidence for {empty}; see the gate and exclusion lists in summary.json", file=sys.stderr)
        return 1
    return 0


def _cmd_scene(args: argparse.Namespace) -> int:
    import numpy as np
    import rasterio
    import torch
    from rasterio.windows import Window

    from frame.evaluate.runner import _resolve_device
    from frame.evaluate.systems import build_system
    from frame.preprocessing.metadata import RasterMetadata
    from frame.reliability.scene_check import check_scene

    config = ReliabilityConfig.load(args.config)
    base = Path(args.config).resolve().parent
    spec = next((s for s in config.systems if s.name == args.system), None)
    if spec is None:
        raise EvaluationError(f"system {args.system!r} is not in the config (systems: {[s.name for s in config.systems]})")
    out_dir = Path(args.output_dir)
    if (out_dir / "scene_check.json").exists():
        raise EvaluationError(f"{out_dir} already contains scene_check.json; choose a new directory. Nothing was overwritten.")
    with rasterio.open(args.lr_geotiff) as src:
        window = Window(args.window[2], args.window[0], args.window[3] - args.window[2], args.window[1] - args.window[0]) if args.window else Window(0, 0, src.width, src.height)
        data = src.read([i + 1 for i in args.band_indices], window=window).astype("float32") / float(args.reflectance_scale)
        transform = src.window_transform(window)
        crs = src.crs.to_string() if src.crs is not None else None
    if crs is None:
        raise EvaluationError(f"{args.lr_geotiff} has no CRS: a georeferencing check needs one")
    height, width = int(data.shape[1]), int(data.shape[2])
    tf = (transform.a, transform.b, transform.c, transform.d, transform.e, transform.f)
    metadata = RasterMetadata(crs=crs, transform=tf, bounds=(tf[2], tf[5] + tf[4] * height, tf[2] + tf[0] * width, tf[5]), resolution_m=abs(tf[0]), width=width, height=height,
                              band_names=("B04", "B03", "B02", "B08"), acquisition_timestamp=None, nodata_value=None, cloud_mask_coverage=None, sr_variant=spec.name)
    system = build_system(spec, config.tiling_config(), device=_resolve_device(config.device), base_dir=base)
    try:
        report = check_scene(system.tile_model, torch.from_numpy(data).to(system.device), metadata, config.tiling_config(), work_dir=out_dir, seed=config.tta.seed)
    finally:
        system.close()
    report["provenance"] = {"system": system.provenance(), "input": str(Path(args.lr_geotiff).name), "window_rows_cols": list(args.window) if args.window else None, "band_indices": list(args.band_indices),
                            "reflectance_scale": args.reflectance_scale}
    from frame.evaluate.runner import _clean

    (out_dir / "scene_check.json").write_text(json.dumps(_clean(report), indent=1, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    s, o, g, c = report["seams"], report["orientation"], report["georeferencing"], report["coverage"]
    print(f"scene check {height}x{width} LR px -> {report['scene']['sr_shape'][1:]} SR px in {report['scene']['tile_count']} tiles ({report['scene']['grid']} grid)")
    print(f"  coverage: shape ok {c['std_shape_matches_output']}, non-finite {c['non_finite_std_pixels']}; orientation aligned {o['all_members_aligned']}; no seam introduced {s['no_seam_introduced']} "
          f"(canonical {s['canonical_grid'].get('seam_over_interior')}, rot90 {s.get('rot90_grid', {}).get('seam_over_interior')}); georeferencing crs/origin/bounds/values "
          f"{g['crs_equal']}/{g['origin_equal']}/{g['bounds_equal']}/{g['array_round_trips_exactly']}; TTA x{report['timing']['tta_over_single_pass']:.2f} single pass")
    ok = c["std_shape_matches_output"] and c["non_finite_std_pixels"] == 0 and o["all_members_aligned"] and s["no_seam_introduced"] is not False and g["array_round_trips_exactly"] and g["crs_equal"] and g["origin_equal"]
    return 0 if ok else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m frame.reliability", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    check = sub.add_parser("check", help="validate config, datasets, systems and preview the reference eligibility gate; run no model")
    check.add_argument("config")
    check.add_argument("--verbose", action="store_true", help="print the registration of every tile")
    check.set_defaults(func=_cmd_check)
    run = sub.add_parser("run", help="run the experiment and write its record")
    run.add_argument("config")
    run.add_argument("--verbose", action="store_true")
    run.add_argument("--reuse-evidence", action="store_true", help="reuse cached per-tile evidence (cache_dir) whose settings, dataset and model digest are unchanged, so only the analysis is repeated")
    run.set_defaults(func=_cmd_run)
    scene = sub.add_parser("scene", help="check the stability map of a rectangular multi-tile scene")
    scene.add_argument("config")
    scene.add_argument("--system", required=True)
    scene.add_argument("--lr-geotiff", required=True)
    scene.add_argument("--band-indices", type=int, nargs=4, default=[3, 2, 1, 7], help="0-based indices of B04, B03, B02, B08 in the file (default: 12-band L2A order)")
    scene.add_argument("--window", type=int, nargs=4, metavar=("R0", "R1", "C0", "C1"), help="rows and columns of the LR window (default: the whole file)")
    scene.add_argument("--reflectance-scale", type=float, default=10000.0)
    scene.add_argument("--output-dir", required=True)
    scene.set_defaults(func=_cmd_scene)
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
