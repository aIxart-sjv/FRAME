# Phase 3 data smoke run (real SEN2NEON sample)

A small, real, reproducible check of the paired-data layer (`frame/data/`, see `docs/DATA.md`). No training.

```bash
FRAME_DATA_ROOT=~/.cache/frame_data sen2sr_venv/bin/python experiments/data_smoke/run_smoke.py            # downloads ~40 MB if missing
FRAME_DATA_ROOT=~/.cache/frame_data sen2sr_venv/bin/python experiments/data_smoke/run_smoke.py --offline  # never downloads
```

* Source: `isp-uv-es/SEN2NEON` at revision `9f076b4f652aa0253127d382dcb6e611250b2e67`; three tiles
  (`2018_MLBS_3__0_2`, `2018_MLBS_3__1_1`, `2022_KONZ_7__5_3`), plus `metadata.csv` and the LR checksum list.
  8 files, 39,792,288 bytes, stored under `$FRAME_DATA_ROOT/sen2neon/` (outside the repository).
* What it does: verifies the published LR SHA-256 checksums, builds the manifest, runs deep QC (records, headers,
  pixels, split integrity), pulls aligned patches through `PairedPatchDataset`/`DataLoader`, measures LR/HR
  registration on real data, and makes one seeded synthetic pair from a real HR patch.
* Outputs (`metadata/`, ≈ 11 KB, committed): `sen2neon_sample.manifest.jsonl`, `qc_report.json`, `smoke_results.json`.
  `frame/tests/test_data_real_sample.py` asserts the committed manifest still matches what the code builds.
* Exit code is `0` when QC passes.
