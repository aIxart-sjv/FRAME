"""Command-line tools for the paired-data layer (Phase 3).

    python -m frame.data qc MANIFEST [--data-root DIR] [--headers] [--pixels] [--report OUT.json] [--strict]
    python -m frame.data sen2neon-manifest --metadata-csv CSV --out MANIFEST [--ids ID ...] [--revision SHA]
    python -m frame.data split MANIFEST --out OUT [--level region|scene] [--seed N] [--fractions 0.8 0.1 0.1]
    python -m frame.data synthetic --out MANIFEST [--regions N] [--scenes-per-region M] [--seed S]

`qc` prints a summary and can write a machine-readable JSON report; it exits 0 when there is no
error-level finding, 1 otherwise (``--strict`` also fails on warnings). Nothing here downloads or
repairs data.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional

from frame.data.adapters import ADAPTERS
from frame.data.adapters import sen2neon as sen2neon_module
from frame.data.config import data_root
from frame.data.contract import Split
from frame.data.errors import ContractError, DataError
from frame.data.manifest import read_manifest, write_manifest
from frame.data.qc import qc_manifest_file, write_report
from frame.data.splits import apply_splits, assign_splits, check_split_integrity


def _cmd_qc(args: argparse.Namespace) -> int:
    root = Path(args.data_root) if args.data_root else (data_root() if (args.headers or args.pixels) else None)
    loader = None
    if args.pixels:
        contents = read_manifest(args.manifest, strict=False)
        adapters = {d: ADAPTERS[d]([r for r in contents.records if r.dataset == d], data_root=root)
                    for d in {r.dataset for r in contents.records} if d in ADAPTERS}

        def loader(record):
            if record.dataset not in adapters:
                raise ContractError(f"No adapter for dataset {record.dataset!r}; pixels cannot be checked.", code="unknown_dataset")
            return adapters[record.dataset].load_pair(record)
    report = qc_manifest_file(args.manifest, data_root=root, check_headers=args.headers, loader=loader)
    print(report.summary())
    if args.report:
        write_report(report, args.report)
        print(f"report written to {args.report}")
    warnings = any(i.severity == "warning" for i in report.issues)
    return 0 if report.ok and not (args.strict and warnings) else 1


def _cmd_sen2neon(args: argparse.Namespace) -> int:
    records = sen2neon_module.records_from_metadata_csv(args.metadata_csv, dataset_revision=args.revision, ids=args.ids)
    digest = write_manifest(args.out, records, header={"source": "SEN2NEON metadata.csv"})
    print(f"wrote {len(records)} record(s) to {args.out} (digest {digest[:16]})")
    return 0


def _cmd_synthetic(args: argparse.Namespace) -> int:
    from frame.data.adapters.synthetic import build_synthetic_dataset

    records = build_synthetic_dataset(data_root(), n_regions=args.regions, scenes_per_region=args.scenes_per_region, seed=args.seed)
    digest = write_manifest(args.out, records, header={"source": f"frame.data synthetic (seed {args.seed})"})
    print(f"wrote {len(records)} record(s) to {args.out} (digest {digest[:16]}); files under {data_root() / 'synthetic_smoke'}")
    return 0


def _cmd_split(args: argparse.Namespace) -> int:
    contents = read_manifest(args.manifest)
    fractions = dict(zip((Split.TRAIN, Split.VAL, Split.TEST), args.fractions))
    assignment = assign_splits(contents.records, fractions, seed=args.seed, level=args.level)
    records = apply_splits(contents.records, assignment, level=args.level)
    problems = [i for i in check_split_integrity(records, require_region_disjoint=(args.level == "region")) if i.severity == "error"]
    if problems:
        for issue in problems:
            print(f"ERROR {issue.code}: {issue.message}", file=sys.stderr)
        return 1
    digest = write_manifest(args.out, records, header={"split_seed": args.seed, "split_level": args.level, "split_fractions": args.fractions})
    counts = {s.value: sum(1 for r in records if r.split == s) for s in Split}
    print(f"wrote {len(records)} record(s) to {args.out} (digest {digest[:16]}); per split: {counts}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m frame.data", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    qc = sub.add_parser("qc", help="validate a manifest")
    qc.add_argument("manifest")
    qc.add_argument("--data-root", help="dataset root (default: $FRAME_DATA_ROOT) -- enables the file checks")
    qc.add_argument("--headers", action="store_true", help="also compare declared metadata with each file's header")
    qc.add_argument("--pixels", action="store_true", help="also load every pair and check NaN/range/nodata")
    qc.add_argument("--report", help="write the machine-readable JSON report here")
    qc.add_argument("--strict", action="store_true", help="treat warnings as failures")
    qc.set_defaults(func=_cmd_qc)

    neon = sub.add_parser("sen2neon-manifest", help="build a SEN2NEON manifest from its metadata.csv")
    neon.add_argument("--metadata-csv", required=True)
    neon.add_argument("--out", required=True)
    neon.add_argument("--ids", nargs="*", help="only these tile ids (default: all 2,269)")
    neon.add_argument("--revision", help="dataset repository revision to record")
    neon.set_defaults(func=_cmd_sen2neon)

    synthetic = sub.add_parser("synthetic", help="generate the synthetic smoke dataset (files under $FRAME_DATA_ROOT) and an unsplit manifest")
    synthetic.add_argument("--out", required=True)
    synthetic.add_argument("--regions", type=int, default=5)
    synthetic.add_argument("--scenes-per-region", type=int, default=2)
    synthetic.add_argument("--seed", type=int, default=0)
    synthetic.set_defaults(func=_cmd_synthetic)

    split = sub.add_parser("split", help="assign train/val/test by geographic unit")
    split.add_argument("manifest")
    split.add_argument("--out", required=True)
    split.add_argument("--level", choices=("region", "scene"), default="region")
    split.add_argument("--seed", type=int, default=0)
    split.add_argument("--fractions", nargs=3, type=float, default=(0.8, 0.1, 0.1), metavar=("TRAIN", "VAL", "TEST"))
    split.set_defaults(func=_cmd_split)
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except DataError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
