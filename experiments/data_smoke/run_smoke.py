"""Phase 3 smoke run on a SMALL REAL sample of SEN2NEON (3 tiles, ~40 MB), no training.

    FRAME_DATA_ROOT=<dir> sen2sr_venv/bin/python experiments/data_smoke/run_smoke.py [--offline]

Steps: fetch (only what is missing, pinned to a dataset revision) -> verify the published LR checksums
-> build the manifest -> deep QC (headers + pixels) -> aligned patches through the DataLoader ->
alignment measured on real data -> one synthetic pair made from a real HR patch.
Everything measured is written to experiments/data_smoke/metadata/smoke_results.json.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))

from frame.data import (  # noqa: E402
    PairedPatchDataset, RGBN_BANDS, collate_pairs, data_root, dataset_dir, degrade, frame_default_v1, manifest_digest, validate_manifest,
    write_manifest,
)
from frame.data.adapters import Sen2NeonAdapter  # noqa: E402
from frame.data.adapters import sen2neon  # noqa: E402
from frame.data.qc import write_report  # noqa: E402

OUT = HERE / "metadata"
REVISION = "9f076b4f652aa0253127d382dcb6e611250b2e67"
TILES = ["2018_MLBS_3__0_2", "2018_MLBS_3__1_1", "2022_KONZ_7__5_3"]   # two acquisitions, two NEON sites


def pearson(a: np.ndarray, b: np.ndarray, valid: np.ndarray) -> float:
    a, b = a[valid], b[valid]
    return float(np.corrcoef(a, b)[0, 1])


def block_mean(x: np.ndarray, s: int) -> np.ndarray:
    h, w = x.shape
    return x.reshape(h // s, s, w // s, s).mean(axis=(1, 3))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--offline", action="store_true", help="do not download; fail if the sample is not already present")
    args = parser.parse_args()
    root = data_root()
    neon_dir = dataset_dir("sen2neon", root)
    results: dict = {"data_root_env": "FRAME_DATA_ROOT", "dataset": "sen2neon", "revision": REVISION, "tiles": TILES}

    # 1. fetch ----------------------------------------------------------------------------------------------
    csv_path = neon_dir / "metadata.csv"
    if not csv_path.is_file():
        if args.offline:
            print("sample not present and --offline given", file=sys.stderr)
            return 2
        from huggingface_hub import hf_hub_download

        hf_hub_download(sen2neon.HF_REPO, "metadata.csv", repo_type="dataset", revision=REVISION, local_dir=str(neon_dir))
    records = sen2neon.records_from_metadata_csv(csv_path, dataset_revision=REVISION, ids=TILES)
    if not args.offline:
        _, size = sen2neon.fetch_files(records, root, revision=REVISION)
    else:
        size = sum((neon_dir / f).stat().st_size for f in sen2neon.relative_files(records))
    results["downloaded_bytes_incl_metadata"] = int(size)
    results["n_files"] = len(sen2neon.relative_files(records)) + 2

    # 2. checksums ------------------------------------------------------------------------------------------
    checks = sen2neon.verify_lr_checksums(records, root)
    results["lr_checksums"] = checks
    assert all(v == "ok" for v in checks.values()), checks

    # 3. manifest -------------------------------------------------------------------------------------------
    manifest_path = OUT / "sen2neon_sample.manifest.jsonl"
    digest = write_manifest(manifest_path, records, header={"source": "SEN2NEON metadata.csv", "note": "Phase 3 smoke sample (3 tiles)"})
    results["manifest"] = {"path": str(manifest_path.relative_to(HERE.parent.parent)), "digest": digest, "n_records": len(records),
                           "digest_matches_recompute": manifest_digest(records) == digest}

    # 4. deep QC --------------------------------------------------------------------------------------------
    adapter = Sen2NeonAdapter(records, data_root=root)
    t0 = time.perf_counter()
    report = validate_manifest(records, data_root=root, check_headers=True, loader=adapter.load_pair)
    results["qc"] = {"seconds": round(time.perf_counter() - t0, 2), **{k: v for k, v in report.to_dict().items() if k != "issues"},
                     "issues": [i.to_dict() for i in report.issues]}
    write_report(report, OUT / "qc_report.json")
    print(report.summary())

    # 5. real pair facts ------------------------------------------------------------------------------------
    pairs = []
    for record in records:
        t0 = time.perf_counter()
        sample = adapter.load_pair(record)
        load_s = time.perf_counter() - t0
        pairs.append({
            "sample_id": record.sample_id, "lr_shape": list(sample.lr.shape), "hr_shape": list(sample.hr.shape),
            "load_seconds": round(load_s, 3), "lr_valid_fraction": round(float(sample.lr_mask.float().mean()), 4),
            "hr_valid_fraction": round(float(sample.hr_mask.float().mean()), 4), "crs": record.lr.crs, "lon_lat": [record.lon, record.lat],
            "lr_range": [round(float(sample.lr[:, sample.lr_mask].min()), 4), round(float(sample.lr[:, sample.lr_mask].max()), 4)],
        })
    results["pairs"] = pairs

    # 6. patches through a DataLoader (evaluation grid, RGBN by name) --------------------------------------
    dataset = PairedPatchDataset(records, adapter.load_pair, lr_patch=128, mode="grid", cache_size=1)
    fractions = [dataset[i]["metadata"]["valid_fraction"] for i in range(len(dataset))]
    results["grid"] = {"lr_patch": 128, "n_patches": len(dataset), "fully_valid_patches": int(sum(f == 1.0 for f in fractions)),
                       "patches_with_any_nodata": int(sum(f < 1.0 for f in fractions)), "patches_with_no_valid_pixel": int(sum(f == 0.0 for f in fractions))}
    rgbn = lambda record: adapter.load_pair(record, lr_bands=RGBN_BANDS, hr_bands=RGBN_BANDS)   # noqa: E731
    train_like = PairedPatchDataset(records, rgbn, lr_patch=128, mode="random", patches_per_pair=4, seed=0, augment=True, cache_size=1)
    batch = next(iter(DataLoader(train_like, batch_size=4, collate_fn=collate_pairs)))
    results["random_rgbn_batch"] = {"lr": list(batch["lr"].shape), "hr": list(batch["hr"].shape), "bands": batch["metadata"][0]["lr_bands"],
                                    "augmentations": [m["augmentation"] for m in batch["metadata"]], "valid_fractions": [round(m["valid_fraction"], 3) for m in batch["metadata"]]}

    # 7. alignment on REAL data: LR should track the block-mean of HR at zero shift, and worse when HR is shifted by one LR pixel
    alignment = []
    for record in records:
        sample = adapter.load_pair(record, lr_bands=["B08"], hr_bands=["B08"])
        lr, hr = sample.lr[0].numpy(), sample.hr[0].numpy()
        valid_lr = sample.lr_mask.numpy() & (block_mean(sample.hr_mask.numpy().astype("float32"), 4) == 1.0)
        base = pearson(lr, block_mean(hr, 4), valid_lr)
        shifted = {}
        for shift in (2, 4, 8):                                # HR pixels = 0.5, 1 and 2 LR pixels
            rolled = np.roll(hr, shift, axis=1)
            shifted[f"{shift / 4:g} LR px"] = round(pearson(lr[:, 3:-3], block_mean(rolled, 4)[:, 3:-3], valid_lr[:, 3:-3]), 4)
        alignment.append({"sample_id": record.sample_id, "band": "B08", "pearson_zero_shift": round(base, 4), "pearson_when_HR_shifted_east": shifted})
    results["alignment_on_real_data"] = alignment

    # 8. a synthetic pair made from a real HR patch, deterministic under a seed --------------------------------
    record = records[0]
    sample = adapter.load_pair(record, lr_bands=RGBN_BANDS, hr_bands=RGBN_BANDS)
    hr_patch = sample.hr[:, 256:768, 256:768].contiguous()
    config = frame_default_v1(4, harmonisation="none")
    a, rec_a = degrade(hr_patch, config, seed=11)
    b, rec_b = degrade(hr_patch, config, seed=11)
    c, _ = degrade(hr_patch, config, seed=12)
    reference_lr = torch.from_numpy(np.stack([block_mean(x, 4) for x in hr_patch.numpy()]))
    results["synthetic_from_real_hr"] = {"hr_patch": list(hr_patch.shape), "lr": list(a.shape), "deterministic_same_seed": bool(torch.equal(a, b)),
                                         "differs_other_seed": not torch.equal(a, c), "recorded": rec_a.to_dict(),
                                         "mean_abs_diff_vs_plain_block_mean": round(float((a - reference_lr).abs().mean()), 5)}
    (OUT / "smoke_results.json").write_text(json.dumps(results, indent=2, sort_keys=True) + "\n")
    print(json.dumps({k: results[k] for k in ("downloaded_bytes_incl_metadata", "grid", "random_rgbn_batch")}, indent=2))
    return 0 if report.ok else 1


if __name__ == "__main__":
    sys.exit(main())
