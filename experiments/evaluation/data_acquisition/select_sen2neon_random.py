"""Select the SEN2NEON evaluation tiles by a SEEDED RANDOM draw (Phase 5): no tile is hand-picked, and nothing is filtered.

    sen2sr_venv/bin/python experiments/evaluation/data_acquisition/select_sen2neon_random.py [--n 30 --seed 0 --fetch]

Draws ``n`` tile ids uniformly at random from ALL 2,269 tiles of the dataset's own ``metadata.csv`` (no filtering on nodata, land cover or site), builds the
Phase 3 manifest for them (test split, canonical bands, provenance kept) pinned to the dataset revision, and, with ``--fetch``, downloads exactly those files
and verifies the published LR checksums. Tiles with little valid reference data are NOT removed here: the evaluation lists them as skipped with the rule.
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2]))

from frame.data.adapters import sen2neon  # noqa: E402
from frame.data.config import data_root  # noqa: E402
from frame.data.manifest import write_manifest  # noqa: E402

REVISION = "9f076b4f652aa0253127d382dcb6e611250b2e67"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fetch", action="store_true", help="download the selected files (pinned revision) and verify the LR checksums")
    parser.add_argument("--out", default=str(HERE.parent / "manifests" / "sen2neon_random30_seed0.jsonl"))
    args = parser.parse_args()
    root = data_root()
    csv_path = root / "sen2neon" / "metadata.csv"
    ids = [row["id"] for row in csv.DictReader(open(csv_path, newline=""))]
    picked = sorted(random.Random(args.seed).sample(ids, args.n))
    records = sen2neon.records_from_metadata_csv(csv_path, dataset_revision=REVISION, ids=picked)
    if args.fetch:
        _, size = sen2neon.fetch_files(records, root, revision=REVISION)
        checks = sen2neon.verify_lr_checksums(records, root)
        bad = {k: v for k, v in checks.items() if v != "ok"}
        print(f"fetched/verified {len(records)} tiles ({size / 1e6:.0f} MB incl. metadata); checksum problems: {bad or 'none'}")
    digest = write_manifest(args.out, records, header={"source": "SEN2NEON metadata.csv", "selection": f"seeded uniform random draw of {args.n} of {len(ids)} tiles, seed {args.seed}, no filtering"})
    print(f"wrote {len(records)} records to {args.out} (digest {digest[:16]}); {len({r.scene_id for r in records})} acquisitions, {len({r.region_id for r in records})} sites")
    return 0


if __name__ == "__main__":
    sys.exit(main())
