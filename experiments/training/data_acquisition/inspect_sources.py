"""Real training-data sources: what would it cost? (Phase 4) -- METADATA ONLY, nothing is downloaded.

Queries the Hugging Face Hub API (SEN2NAIPv2) and the Zenodo API (SEN2VENuS) for file names, sizes, revision and licence, then writes
`sources.json` next to this script, with estimates of the smallest practical real subsets. Numbers here are measured from the services'
own listings on the day the script ran; sample counts are ESTIMATES scaled from the dataset cards' totals (62,242 unet, 61,282 histmatch,
8,000 crosssensor samples), because the .taco contents were not opened (tacoreader is not installed).

    sen2sr_venv/bin/python experiments/training/data_acquisition/inspect_sources.py
"""

from __future__ import annotations

import datetime
import json
import urllib.request
from pathlib import Path

from huggingface_hub import HfApi

HERE = Path(__file__).resolve().parent
CARD_SAMPLES = {"unet": 62_242, "histmatch": 61_282, "crosssensor": 8_000}


def naip() -> dict:
    info = HfApi().dataset_info("tacofoundation/SEN2NAIPv2", files_metadata=True)
    files = {s.rfilename: s.size or 0 for s in info.siblings if s.rfilename.endswith(".taco")}
    groups = {}
    for variant in CARD_SAMPLES:
        parts = {n: s for n, s in files.items() if f"sen2naipv2-{variant}" in n}
        total = sum(parts.values())
        smallest = min(parts.values())
        groups[variant] = {
            "files": dict(sorted(parts.items())), "total_gb": round(total / 1e9, 2), "card_samples": CARD_SAMPLES[variant],
            "smallest_part_gb": round(smallest / 1e9, 2), "estimated_samples_in_smallest_part": round(CARD_SAMPLES[variant] * smallest / total),
            "estimated_mb_per_sample_compressed": round(total / CARD_SAMPLES[variant] / 1e6, 2),
        }
    return {"repo": "tacofoundation/SEN2NAIPv2", "revision": info.sha, "license": [t for t in info.tags if t.startswith("license:")],
            "total_gb": round(sum(files.values()) / 1e9, 2), "variants": groups,
            "note": "Distributed as multi-part .taco containers of up to 20 GB; the unit of download is one whole part unless the reader supports ranged remote reads (unverified)."}


def venus() -> dict:
    with urllib.request.urlopen("https://zenodo.org/api/records/14603764", timeout=60) as response:
        record = json.load(response)
    files = sorted(((f["key"], f["size"]) for f in record["files"]), key=lambda x: x[1])
    return {"record": "zenodo.org/records/14603764", "version": record["metadata"].get("version"), "license": record["metadata"].get("license", {}).get("id"),
            "n_files": len(files), "total_gb": round(sum(s for _, s in files) / 1e9, 1),
            "site_zips_gb": {name: round(size / 1e9, 2) for name, size in files},
            "smallest_four_sites_gb": round(sum(s for _, s in files[:4]) / 1e9, 2),
            "note": "One zip per SITE (a natural region unit for geographic splits). 'other-nc': the VENuS half is non-commercial. 5 m HR, 8 bands, x2 (RGBN 10 m) and x4 (red-edge 20 m): not the RGBN 10 m -> 2.5 m task."}


def main() -> int:
    out = {"queried_utc": datetime.datetime.now(datetime.timezone.utc).replace(microsecond=0).isoformat(), "nothing_downloaded": True,
           "sen2naipv2": naip(), "sen2venus": venus()}
    (HERE / "sources.json").write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    n = out["sen2naipv2"]["variants"]["unet"]
    print(f"SEN2NAIPv2: {out['sen2naipv2']['total_gb']} GB total; smallest unet part {n['smallest_part_gb']} GB (~{n['estimated_samples_in_smallest_part']} pairs, est.)")
    print(f"SEN2VENuS: {out['sen2venus']['total_gb']} GB total; four smallest sites {out['sen2venus']['smallest_four_sites_gb']} GB; licence {out['sen2venus']['license']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
