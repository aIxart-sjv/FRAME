"""Export a small, seeded-random sample of REAL SEN2NAIPv2 pairs as GeoTIFFs (Phase 5). Runs in the ISOLATED export environment.

    ~/.venvs/frame_taco/bin/python experiments/evaluation/data_acquisition/export_naipv2_sample.py \
        --variant unet --n 200 --seed 0 --out $FRAME_DATA_ROOT/sen2naipv2

Why a separate environment: SEN2NAIPv2 ships as TACO containers read by ``tacoreader==0.4.5`` (the version the dataset card uses); that package
is not, and must not become, a dependency of the FRAME or Mamba environments (docs/EVALUATION.md). Nothing here is imported by FRAME.

How it avoids downloading 10-20 GB parts: the TACO metadata (all 61,282 samples: geography, split label, byte range) loads remotely in seconds, and
each pair is then read with GDAL ranged requests (~1.2 MB per pair) and copied, unmodified, to a local GeoTIFF. Sampling is a seeded uniform draw
over ALL samples of the variant (no filtering, no cherry-picking) made on a CANONICAL order: tacoreader 0.4.5 returns the same rows in a DIFFERENT order on every load
(measured), so an index drawn from the loaded order is not reproducible; the rows are therefore sorted by their unique ``tortilla:id`` first. Existing files are kept, so an
interrupted run resumes; ``--shard I --n-shards N`` lets N processes share the (latency-bound) work.

Writes ``<out>/<variant>/lr/<id>.tif``, ``hr/<id>.tif`` and ``<out>/<variant>/rows.jsonl`` (one row per pair: the dataset's own metadata, which the
main environment turns into Phase 3 records).
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import pandas as pd
import rasterio
import rasterio.shutil
import tacoreader


def _readable(path: Path) -> bool:
    """A previously exported file counts only if it opens and its last pixel block reads (a crash can leave a truncated file)."""
    try:
        with rasterio.open(path) as src:
            src.read(window=rasterio.windows.Window(max(0, src.width - 8), max(0, src.height - 8), 8, 8))
        return True
    except Exception:
        return False


def _copy_with_retries(source, target: Path, attempts: int = 5) -> None:
    for attempt in range(1, attempts + 1):
        try:
            temporary = target.with_suffix(".tmp.tif")
            rasterio.shutil.copy(source, str(temporary), driver="GTiff")
            temporary.replace(target)                  # atomic: a crash never leaves a half-written .tif under the final name
            return
        except Exception as exc:                       # transient network errors surface as several different exception types
            if attempt == attempts:
                raise
            print(f"    retry {attempt}/{attempts - 1} after {type(exc).__name__}", flush=True)
            time.sleep(2 * attempt)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", required=True, choices=("unet", "histmatch", "crosssensor"))
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", required=True)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--n-shards", type=int, default=1)
    args = parser.parse_args()

    dataset = tacoreader.load(f"tacofoundation:sen2naipv2-{args.variant}")
    total = len(dataset)
    ids = [str(v) for v in dataset["tortilla:id"]]
    canonical = sorted(range(total), key=lambda i: ids[i])                 # a stable order: sorted by the unique sample id
    chosen_all = sorted(random.Random(args.seed).sample(canonical, args.n), key=lambda i: ids[i])
    chosen = chosen_all[args.shard::args.n_shards]
    root = Path(args.out) / args.variant
    (root / "lr").mkdir(parents=True, exist_ok=True)
    (root / "hr").mkdir(parents=True, exist_ok=True)
    print(f"{args.variant}: {total} samples in the dataset; drawing {args.n} (seed {args.seed}, canonical id order); shard {args.shard}/{args.n_shards} exports {len(chosen)} to {root}", flush=True)

    rows, failures = [], []
    for count, index in enumerate(chosen, start=1):
        meta = dataset.iloc[index]
        sid = str(meta["tortilla:id"])
        try:
            pair = dataset.read(index)
            for role, child in (("lr", 0), ("hr", 1)):
                target = root / role / f"{sid}.tif"
                if not (target.exists() and _readable(target)):
                    _copy_with_retries(pair.read(child), target)
        except Exception as exc:
            failures.append({"id": sid, "dataset_index": int(index), "error": f"{type(exc).__name__}: {exc}"[:300]})
            print(f"  FAILED {sid}: {type(exc).__name__}", flush=True)
            continue
        with rasterio.open(root / "lr" / f"{sid}.tif") as lr, rasterio.open(root / "hr" / f"{sid}.tif") as hr:
            info = {"lr_shape": [lr.count, lr.height, lr.width], "hr_shape": [hr.count, hr.height, hr.width], "lr_dtype": lr.dtypes[0], "hr_dtype": hr.dtypes[0],
                    "lr_nodata": lr.nodata, "hr_nodata": hr.nodata, "lr_transform": list(lr.transform)[:6], "hr_transform": list(hr.transform)[:6],
                    "crs": lr.crs.to_string() if lr.crs else None, "hr_crs": hr.crs.to_string() if hr.crs else None,
                    "lr_descriptions": list(lr.descriptions), "hr_descriptions": list(hr.descriptions)}
        rows.append({"id": sid, "variant": args.variant, "dataset_index": int(index), "dataset_split_label": str(meta["tortilla:data_split"]),
                     "centroid": str(meta["stac:centroid"]), "time_start": float(meta["stac:time_start"]), "days_between": int(meta["days_between"]),
                     "admin0": str(meta["rai:admin0"]), "admin1": str(meta["rai:admin1"]), "admin2": str(meta["rai:admin2"]),
                     "stac_crs": str(meta["stac:crs"]), "stac_geotransform": [float(x) for x in meta["stac:geotransform"]],
                     "byte_length": int(meta["tortilla:length"]), "part": str(meta["internal:subfile"]).rsplit("/", 1)[-1], **info})
        if count % 20 == 0 or count == len(chosen):
            print(f"  {count}/{len(chosen)}", flush=True)
    suffix = f".shard{args.shard}" if args.n_shards > 1 else ""
    (root / f"rows{suffix}.jsonl").write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in rows))
    (root / f"failures{suffix}.json").write_text(json.dumps(failures, indent=1) + "\n")
    print(f"wrote {len(rows)} rows to {root / f'rows{suffix}.jsonl'}; {len(failures)} sample(s) failed after retries (listed in failures{suffix}.json, not silently dropped)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
