"""Verify the exported SEN2NAIPv2 sample, build its Phase 3 manifests, and test the geographic-split workflow on REAL pairs (Phase 5). Main environment.

    sen2sr_venv/bin/python experiments/evaluation/data_acquisition/analyse_naipv2_sample.py

Reads ``$FRAME_DATA_ROOT/sen2naipv2/<variant>/rows*.jsonl`` written by export_naipv2_sample.py (isolated tacoreader environment), then:

* opens every exported LR/HR pair and checks what Phase 3 could only assume: dtype, nodata, the exact x4 geometry (pixel size 10 / 2.5 m, identical origin, 130 -> 520),
  the LR<->HR band correspondence (block-mean of the HR band i against LR band j), and the value scale (percentiles of the stored integers);
* builds Phase 3 records (scene = the NAIP tile id, region = the US STATE from the dataset's own metadata) and runs deep QC (headers + pixels);
* assigns train/val by region with the Phase 3 splitter and runs the leakage checks;
* writes ``naipv2_sample_report.json`` (facts and counts) and the manifests next to it (relative paths only).

No number here is a model result; it is data verification. The sample is a seeded uniform draw over the variant on a canonical (id-sorted) order.
"""

from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import rasterio

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2]))

from frame.data.adapters import Sen2NaipV2Adapter  # noqa: E402
from frame.data.adapters import sen2naipv2  # noqa: E402
from frame.data.config import data_root  # noqa: E402
from frame.data.contract import Split  # noqa: E402
from frame.data.manifest import write_manifest  # noqa: E402
from frame.data.qc import validate_manifest  # noqa: E402
from frame.data.splits import apply_splits, assign_splits, check_split_integrity  # noqa: E402

MANIFESTS = HERE.parent / "manifests"
ORIGIN_TOLERANCE_M = 0.01 * 2.5          # Phase 3's rule: LR and HR origins agree within 0.01 HR pixel


def read_rows(directory: Path):
    rows = []
    for path in sorted(directory.glob("rows*.jsonl")):
        rows += [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return sorted(rows, key=lambda r: r["id"])


def lonlat(centroid: str):
    inner = centroid.replace("POINT (", "").replace(")", "").split()
    return float(inner[0]), float(inner[1])


def inspect(root: Path, rows, max_corr=25):
    lr_stats, hr_stats, corr_diag, corr_full = [], [], [], []
    bad = []
    p99_lr, p99_hr, nod_lr, nod_hr, origin_diffs = [], [], [], [], []
    for i, r in enumerate(rows):
        with rasterio.open(root / r["variant"] / "lr" / f"{r['id']}.tif") as lr, rasterio.open(root / r["variant"] / "hr" / f"{r['id']}.tif") as hr:
            lr_a, hr_a = lr.read(), hr.read()
            geo = {"lr_shape": [lr.count, lr.height, lr.width], "hr_shape": [hr.count, hr.height, hr.width], "px": [lr.transform.a, hr.transform.a],
                   "origin_diff_m": max(abs(lr.transform.c - hr.transform.c), abs(lr.transform.f - hr.transform.f)), "crs_equal": lr.crs == hr.crs,
                   "dtype": [lr.dtypes[0], hr.dtypes[0]], "nodata": [lr.nodata, hr.nodata]}
        origin_diffs.append(geo["origin_diff_m"])
        ok = geo["lr_shape"] == [4, 130, 130] and geo["hr_shape"] == [4, 520, 520] and geo["px"] == [10.0, 2.5] and geo["origin_diff_m"] < ORIGIN_TOLERANCE_M and geo["crs_equal"]
        if not ok:
            bad.append({"id": r["id"], **geo})
        nod_lr.append(float((lr_a == 65535).mean())); nod_hr.append(float((hr_a == 65535).mean()))
        valid_lr, valid_hr = lr_a[lr_a != 65535], hr_a[hr_a != 65535]
        p99_lr.append(float(np.percentile(valid_lr, 99)) if valid_lr.size else None); p99_hr.append(float(np.percentile(valid_hr, 99)) if valid_hr.size else None)
        lr_stats.append(lr_a.reshape(4, -1).astype(np.float64).mean(axis=1)); hr_stats.append(hr_a.reshape(4, -1).astype(np.float64).mean(axis=1))
        if i < max_corr and not (lr_a == 65535).any():
            hr_block = hr_a.astype(np.float64).reshape(4, 130, 4, 130, 4).mean(axis=(2, 4))
            m = np.array([[np.corrcoef(hr_block[a].ravel(), lr_a[b].astype(np.float64).ravel())[0, 1] for b in range(4)] for a in range(4)])
            corr_full.append(m)
    corr = np.nanmean(np.stack(corr_full), axis=0) if corr_full else None
    return {
        "n_pairs": len(rows), "geometry_problems": bad, "max_origin_difference_m": float(max(origin_diffs)), "origin_tolerance_m": ORIGIN_TOLERANCE_M, "lr_nodata_fraction_mean": float(np.mean(nod_lr)), "hr_nodata_fraction_mean": float(np.mean(nod_hr)),
        "pairs_with_any_nodata": int(sum(1 for a, b in zip(nod_lr, nod_hr) if a > 0 or b > 0)),
        "band_means_lr_dn": [float(x) for x in np.mean(lr_stats, axis=0)], "band_means_hr_dn": [float(x) for x in np.mean(hr_stats, axis=0)],
        "p99_dn_lr_median": float(np.nanmedian([v for v in p99_lr if v is not None])), "p99_dn_hr_median": float(np.nanmedian([v for v in p99_hr if v is not None])),
        "hr_blockmean_vs_lr_band_correlation_mean": None if corr is None else [[round(float(v), 3) for v in row] for row in corr],
        "band_correspondence_is_diagonal": None if corr is None else bool(all(int(np.argmax(corr[a])) == a for a in range(4))), "n_pairs_used_for_correlation": len(corr_full),
    }


def main() -> int:
    root = data_root() / "sen2naipv2"
    report = {"note": "data verification of a small seeded sample (canonical id order); not a model result", "variants": {}}
    all_records = {}
    for variant, split_label in (("unet", "train"), ("crosssensor", "val")):
        directory = root / variant
        rows = read_rows(directory)
        if not rows:
            continue
        info = inspect(root, rows)
        records = []
        for r in rows:
            lon, lat = lonlat(r["centroid"])
            state = re.sub(r"[^A-Za-z0-9._-]+", "_", r["admin1"]) if r["admin1"] not in ("missing", "None", "") else "unknown"     # Phase 3 ids allow no spaces
            row = {"id": r["id"], "variant": variant, "scene_id": r["id"], "region_id": state, "split": split_label, "lr_path": f"{variant}/lr/{r['id']}.tif", "hr_path": f"{variant}/hr/{r['id']}.tif",
                   "lon": lon, "lat": lat, "crs": r["crs"], "lr_transform": r["lr_transform"], "hr_transform": r["hr_transform"], "source_split": r["dataset_split_label"],
                   "provenance": {"admin0": r["admin0"], "admin1": r["admin1"], "admin2": r["admin2"], "time_start": r["time_start"], "days_between": r["days_between"], "part": r["part"]}}
            records.append(sen2naipv2.record_from_row(row, dataset_revision="tacofoundation/SEN2NAIPv2 (tacoreader 0.4.5 load)"))
        adapter = Sen2NaipV2Adapter(records, data_root=root.parent)
        qc = validate_manifest(records, data_root=root.parent, check_headers=True, loader=adapter.load_pair, check_split=False)
        info["qc"] = {"ok": qc.ok, "valid": qc.valid, "invalid": qc.invalid, "issue_counts": qc.counts_by_code(), "checks": qc.checks}
        info["geography"] = {"states": len({r["admin1"] for r in rows}), "counties": len({(r["admin1"], r["admin2"]) for r in rows}), "countries": dict(Counter(r["admin0"] for r in rows)),
                             "lon_range": [min(lonlat(r["centroid"])[0] for r in rows), max(lonlat(r["centroid"])[0] for r in rows)],
                             "lat_range": [min(lonlat(r["centroid"])[1] for r in rows), max(lonlat(r["centroid"])[1] for r in rows)],
                             "parts": dict(sorted(Counter(r["part"] for r in rows).items())), "dataset_own_split_labels": dict(Counter(r["dataset_split_label"] for r in rows))}
        info["failures"] = sum((json.loads(p.read_text()) for p in sorted(directory.glob("failures*.json"))), [])
        if variant == "unet":                                   # a synthetic TRAINING variant: split geographically (by state) into train/val
            assignment = assign_splits(records, {Split.TRAIN: 0.8, Split.VAL: 0.2, Split.TEST: 0.0}, seed=0, level="region")
            split_records = apply_splits(records, assignment, level="region")
            issues = [i for i in check_split_integrity(split_records) if i.severity == "error"]
            per = Counter(r.split.value for r in split_records)
            states = {s: sorted({r.region_id for r in split_records if r.split.value == s}) for s in per}
            info["geographic_split"] = {"level": "region = US state", "fractions": {"train": 0.8, "val": 0.2}, "seed": 0, "counts": dict(per), "n_states": {s: len(v) for s, v in states.items()},
                                        "states_disjoint": not (set(states.get("train", [])) & set(states.get("val", []))), "leakage_errors": [i.code for i in issues],
                                        "dataset_labels_of_frame_val": dict(Counter(r.source_split for r in split_records if r.split.value == "val"))}
            records = split_records
        write_manifest(MANIFESTS / f"sen2naipv2_{variant}_sample{len(records)}.jsonl", records, header={"source": "SEN2NAIPv2 exported sample (seeded, canonical id order)", "variant": variant})
        report["variants"][variant] = info
        all_records[variant] = records
    (HERE / "naipv2_sample_report.json").write_text(json.dumps(report, indent=2, sort_keys=True, default=str) + "\n")
    for variant, info in report["variants"].items():
        print(f"{variant}: {info['n_pairs']} pairs; geometry problems {len(info['geometry_problems'])}; QC ok {info['qc']['ok']} ({info['qc']['issue_counts']}); states {info['geography']['states']}; "
              f"max LR/HR origin difference {info['max_origin_difference_m']*1000:.2f} mm; band correspondence diagonal {info['band_correspondence_is_diagonal']}; p99 DN LR/HR {info['p99_dn_lr_median']:.0f}/{info['p99_dn_hr_median']:.0f}")
        if "geographic_split" in info:
            g = info["geographic_split"]
            print(f"  geographic split: {g['counts']} pairs; states {g['n_states']}; disjoint {g['states_disjoint']}; leakage errors {g['leakage_errors'] or 'none'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
