"""Command line for the domain-shift layer (Step H, Requirements 142 Requirement 12).

    python -m frame.generalization run --bbox W S E N --start-date D --end-date D --model lite [--output-dir DIR]

Requires real CDSE credentials (FRAME_CDSE_CLIENT_ID/SECRET) -- this makes a real network call to
discover and retrieve a real Sentinel-2 scene; there is no offline/synthetic mode, since a
domain-shift record about a fabricated scene would say nothing about Requirement 12.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from frame.acquisition.contract import AcquisitionQuery
from frame.generalization.errors import GeneralizationError
from frame.generalization.report import write_record
from frame.generalization.runner import run_domain_shift_experiment


def _cmd_run(args: argparse.Namespace) -> int:
    query = AcquisitionQuery(
        bbox=(args.bbox[0], args.bbox[1], args.bbox[2], args.bbox[3]),
        start_date=args.start_date, end_date=args.end_date, max_cloud_cover=args.max_cloud_cover,
    )
    try:
        record = run_domain_shift_experiment(query, model_name=args.model, device=args.device, seed=args.seed)
    except GeneralizationError as exc:
        print(f"refused: {exc}", file=sys.stderr)
        return 2
    out_dir = Path(args.output_dir)
    write_record(record, out_dir)
    print(f"wrote {out_dir / 'record.json'} and {out_dir / 'README.md'}")
    print(f"scene {record.scene_id}: self-consistency MAE {record.self_consistency_overall['mae']:.4f}, "
          f"stability {record.stability_scalar_summary:.6f}")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="python -m frame.generalization")
    sub = parser.add_subparsers(dest="command", required=True)

    run_p = sub.add_parser("run", help="run one real domain-shift protocol against a real Indian AOI")
    run_p.add_argument("--bbox", type=float, nargs=4, metavar=("W", "S", "E", "N"), required=True)
    run_p.add_argument("--start-date", required=True)
    run_p.add_argument("--end-date", required=True)
    run_p.add_argument("--model", default="lite")
    run_p.add_argument("--device", default="auto")
    run_p.add_argument("--seed", type=int, default=0)
    run_p.add_argument("--max-cloud-cover", type=float, default=20.0)
    run_p.add_argument("--output-dir", required=True)
    run_p.set_defaults(func=_cmd_run)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
