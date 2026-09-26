# FRAME paired LR–HR data layer (Phase 3)

`frame/data/` is the reusable data layer that later training and evaluation code will sit on. It knows what
an LR–HR pair *is*, where it came from, which split it belongs to, how it was made, whether it is spatially
and spectrally valid, and how to hand small aligned patches to a model. It **does not train anything**, and it
is deliberately not a data platform: datasets stay outside the repository, manifests are small JSONL indexes,
and no pixel is read until something asks for it.

Primary requirements source: `docs/Requirements 142.txt` (Requirement 3, sections 3–36, and Requirement 9,
sections 24–27). Where a dataset's own published card differs from the requirements, the profile records the
dataset's facts and says so.

## 1. Architecture

```
 Dataset adapter        Manifest            Scene / Pair          Validation           Paired patch          LR / HR sample
 (per dataset)          (JSONL, digest)     (PairRecord)          (qc, geo, splits)    extractor (patches)   (PairedSample)
 ──────────────  ──▶  ───────────────  ──▶  ───────────────  ──▶  ─────────────────  ──▶ ─────────────────  ──▶ ──────────────────
 SEN2NEON              sorted, relative     one LR–HR pair,       record · file ·      HR window =           lr (C,h,w)
 OpenSR-Test           paths, no times-     its geography,        pixel · manifest     LR window × scale     hr (C',h·s,w·s)
 SEN2NAIPv2 (format)   tamps, content       split, pair type,     · split leakage      no padding, no        masks, band names,
 SEN2VENµS  (format)   digest               degradation,          · LR/HR geometry     independent crops     patch coordinates,
 India holdout (plan)                       licence, provenance                                              provenance dict
```

| Module | Purpose |
|---|---|
| `contract.py` | `PairRecord` (the index entry), `PairedSample` (pixels + masks), `RasterSpec`, `PatchCoords`, `DegradationRecord`; `PairType`, `Split`, `HRStatus`, `DatasetRole` |
| `roles.py` | one `DatasetProfile` per dataset: role, permitted splits, bands, scales, sizes, licence, **honest status** |
| `bands.py` | canonical band names (`B01`…`B12`, `B8A`) and explicit **by-name** band selection |
| `geo.py` | LR/HR geometry validation (CRS, resolution ratio, footprint origin, dimensions) |
| `patches.py` | aligned paired patch extraction |
| `degradation.py` | HR → Sentinel-2-like LR, seeded, configurable, versioned, recorded |
| `manifest.py` | the JSONL manifest format, byte-reproducible, with a content digest |
| `splits.py` | geographic (scene / region) split assignment and leakage checks |
| `qc.py`, `issues.py` | record / file / pixel / manifest validation and the machine-readable report |
| `adapters/` | `sen2neon`, `opensr_test`, `sen2naipv2`, `sen2venus`, `india_holdout`, `synthetic` (Phase 4 smoke set), and the generic GeoTIFF reader |
| `loader.py` | `PairedPatchDataset` (optional `return_masks=True`, added for Phase 4 training), `collate_pairs`, `dispatching_loader` |
| `cli.py` | `python -m frame.data {qc, split, sen2neon-manifest, synthetic}` |

The layer imports only what it needs. Importing `frame.data` does **not** import `opensr_test`, `skimage`,
`fastapi`, `sen2sr`, `mamba_ssm`, `huggingface_hub`, or any TACO reader (enforced by a test). It does not touch
`sen2sr/`, the Lite or Mamba inference paths, the tiling engine, or the API.

## 2. Dataset roles

| Dataset | Role (requirements) | Bands (LR → HR) | Resolution | Pair type | Phase 3 status |
|---|---|---|---|---|---|
| **SEN2NAIPv2** `unet` / `histmatch` | **Primary training**; may also be used for validation | RGBN → RGBN | 10 m → 2.5 m (×4); 130×130 → 520×520 | synthetic (LR shipped by the dataset, upstream degradation) | **format only** at Phase 3; **format verified on a 100-pair real sample in Phase 5** (§16); no whole part read |
| SEN2NAIPv2 `crosssensor` | Development **validation** | RGBN → RGBN | 10 m → 2.5 m (×4) | real cross-sensor (S2 within a day of NAIP) | **format only** at Phase 3; **format verified on a 30-pair real sample in Phase 5** (§16) |
| **SEN2VENµS** `rgbn_10m` / `rededge_20m` | **Supplementary training** | B2,B3,B4,B8 → same (×2, 10→5 m); B5,B6,B7,B8A → same (×4, 20→5 m) | 5 m HR | real cross-sensor | **format only** |
| **SEN2NEON** `2.5m` | **Independent benchmark — test only, never trained on** | 12 → 12 | 10 m → 2.5 m (×4); 256×256 → 1024×1024 | real cross-sensor (Sentinel-2 vs AVIRIS-NG-derived) | **implemented, real data read** (3-tile sample, 39.8 MB; in Phase 5 a seeded random 30-tile sample, §16) |
| **OpenSR-Test** `spot`, `spain_crops`, `spain_urban` | **Independent benchmark — test only** | 12 → RGBN | 10 m → 2.5 m (×4); 128×128 → 512×512 | real cross-sensor | **implemented, real data read** (`spot`; in Phase 5 also `spain_crops` and `spain_urban`, §16) |
| **Indian holdout** | **Domain holdout — test only** | 12 → (none yet) | 10 m → unknown | independent HR reference (may be absent) | **planned** — profile + record builder, no data |
| **synthetic_smoke** (Phase 4) | **Smoke test** — pipeline exercise only, never a benchmark | RGBN → RGBN | 10 m → 2.5 m (×4); 128×128 → 512×512 | synthetic (FRAME `frame_default_v1` degradation, recorded) | **generated** by `python -m frame.data synthetic`; fixtures, not evidence about real imagery |

"Format only" means exactly this: the reader and record builder follow the dataset's *published* format and
are tested against synthetic files written in that format. **At the end of Phase 3 no real SEN2NAIPv2 or SEN2VENµS file had been read
by FRAME** (Phase 5 has since read a small SEN2NAIPv2 sample, §16; SEN2VENµS is still format-only), and the profiles say so in a `status` field. Nothing here should be described as "SEN2NAIPv2
support" beyond that.

Roles are *enforced*, not just documented: QC rejects a SEN2NEON, OpenSR-Test or Indian record in `train`/`val`
(`role_violation`), a SEN2NAIPv2 `crosssensor` record in `train`, and a SEN2VENµS record in `test`.

### Dataset facts worth knowing (from the dataset cards, see `roles.py` caveats)

* **SEN2NEON**: the LR stack is a *convenience product* — all 12 bands on one 10 m grid; the 60 m and 20 m
  bands were **not measured at 10 m**. The HR is derived from AVIRIS-NG hyperspectral data convolved with
  Sentinel-2 response functions. LR nodata is `65535`, HR nodata is `0`, and many tiles have large NEON nodata
  fractions. North American NEON sites only: no evidence about Indian landscapes. A release correction replaced the
  LR files (20 Aug 2026), so every record stores the dataset revision and LR checksums can be verified.
* **SEN2VENµS**: 5 m HR and only 8 of 12 bands, so it can never by itself train the 12-band 2.5 m task; LR is Theia
  L2A, not the ESA L2A the deployed pipeline consumes; the VENµS half is **CC-BY-NC** (non-commercial).
* **SEN2NAIPv2**: distributed as 139.1 GB of TACO files (needs `tacoreader`, **not installed**). Its stored
  patches are 130×130 / 520×520; model crops are 128 / 512 (see §7). The value scale of the stored `uint16` is not
  stated on the card, so exports must state it (default 10000, unverified).
* **OpenSR-Test** and SEN2NEON are benchmarks: using them for training would contaminate every later claim
  (requirements lines 3066–3086).

### The synthetic smoke set (added in Phase 4)

`synthetic_smoke` (`frame/data/adapters/synthetic.py`) generates deterministic RGBN scenes with sharp edges and lines, degrades them with `frame_default_v1` (seeded, recorded, `parameters_verified=False`) and writes real
GeoTIFF LR/HR pairs so the whole pipeline (adapter → manifest → region split → QC → training → validation) runs on real files in seconds. `python -m frame.data synthetic --out M.jsonl --regions 5 --scenes-per-region 2 --seed 0` writes the files
under `$FRAME_DATA_ROOT/synthetic_smoke/` and an *unsplit* manifest (all `train`); splits come from `python -m frame.data split`. Its role, `smoke_test`, allows train/val/test; its LR can be regenerated exactly from the stored HR file and its
degradation record (tested). It exists for Phase 4 experiments (`docs/TRAINING.md`); any number measured on it says nothing about real imagery.

## 3. Where data lives

Datasets are never stored in the repository and never downloaded automatically.

```
$FRAME_DATA_ROOT/<dataset>/<relative path from the manifest record>
```

`FRAME_DATA_ROOT` defaults to `~/.cache/frame_data`. Manifests contain only relative paths, so they are portable
and contain no machine-specific location (a test scans `frame/data/` for absolute home paths).

## 4. The paired-sample contract

A `PairRecord` (one JSON line in a manifest) carries: `sample_id` (`<dataset>:<…>`), `dataset`, `variant`,
`dataset_revision`, `scene_id`, `region_id`, `split`, `source_split`, `pair_type`, `hr_status`, `scale_factor`,
`lr` / `hr` (`RasterSpec`: band names, size, pixel size, relative path, dtype, reflectance scale, nodata, CRS,
transform), `lon`/`lat`, `degradation`, `license`, `provenance`. A `PairedSample` adds the tensors, validity
masks, band names and (for patches) the `PatchCoords`.

Pair types (never conflated):

| `pair_type` | Meaning | `degradation` | HR |
|---|---|---|---|
| `synthetic` | LR = f(HR) | required (FRAME's own, or the dataset's upstream one) | required |
| `real_cross_sensor` | real Sentinel-2 + a different sensor's HR, co-registered by the dataset authors | forbidden | required |
| `independent_hr_reference` | real Sentinel-2 + an independent HR reference that may not exist | forbidden | `available` / `unavailable` / `unknown` |

Invariants enforced at construction (`ContractError` with a stable `code`): a paired record must have an HR
counterpart; an LR-only record must say `unavailable` or `unknown`, never `available`; absolute paths are refused;
unknown fields in a manifest are refused; `PatchCoords` derives the HR window from the LR window, so the two
cannot disagree.

**Spectral handling.** Bands are canonicalised (`B1`→`B01`, `b8a`→`B8A`) and always selected **by name**
(`adapter.load_pair(record, lr_bands=RGBN_BANDS)`); nothing relies on implicit channel order. FRAME's RGBN order
(`B04,B03,B02,B08`) differs from the ascending order stored in dataset files (`B02,B03,B04,B08`); the reorder is
explicit and tested. Nodata is carried as a boolean mask (a pixel is nodata only if **every** band equals the
nodata value) and zeroed in the tensor; **NaN is never repaired or replaced** — QC reports it.

## 5. Manifests

```
line 1    {"_header": {"kind": "frame-paired-manifest", "manifest_version": 1, "n_records": N, "datasets": {name: {revisions, variants}}, "digest": "<sha256>", ...}}
line 2…   one PairRecord per line
```

Records are sorted by `sample_id`, keys are sorted, there are no timestamps, and paths are relative, so the same
records always produce a byte-identical file and the same content digest. A training run can store the digest to
prove exactly which data it used. Reading is strict by default; `strict=False` turns bad lines into issues so QC can
report all of them at once (with line numbers).

A real manifest for the SEN2NEON sample is committed at
`experiments/data_smoke/metadata/sen2neon_sample.manifest.jsonl` (3 records, 5.9 KB); a test asserts it matches what
the code builds now.

## 6. Geographic splits and leakage prevention

Splits are decided per **geographic unit**, never per patch (requirements section 32–34; Requirement 9 section 24).

* **Scene** (one acquisition / ROI) — *hard rule*: a scene never contributes to more than one split.
* **Region** (e.g. a NEON site, a VENµS site) — *default rule* (`require_region_disjoint=True`): a region never
  spans splits either. Scene-level splitting (`--level scene`) is available for datasets with too few regions and
  then legitimately allows a region to span splits while still forbidding scene leakage.
* Units are **namespaced by dataset**, so `ABBY` in one dataset never collides with `ABBY` in another.
* `assign_splits` orders units by a seeded SHA-256 (reproducible and independent of input order) and allocates whole
  units by cumulative weight; a dataset restricted to some splits (SEN2NEON, OpenSR-Test: test only) is forced into
  them; explicit assignments override the hash (e.g. a named Indian region → test).
* `check_split_integrity` reports `scene_leakage`, `region_leakage`, `role_violation`, and — as a *warning* —
  `spatial_proximity_across_splits`: samples from different splits within the same 0.1° (~11 km) cell or an adjacent
  cell, which catches leakage between **different datasets covering the same ground** (SEN2NEON vs SEN2NAIPv2).
  Proximity is approximate (centroid based) and is a warning, not proof.

A unit test places two patches of one scene in different splits and confirms the validator rejects it, both in
`splits` and end-to-end through the `qc` CLI (exit code 1, `scene_leakage` in the JSON report).

## 7. Aligned patch extraction

`patches.py` is *training/evaluation* patch extraction, not inference tiling (`frame.tiling`). One rule:

    HR window = LR window × scale        (origin and size)

so an LR `128×128` patch at `(r, c)` pairs with the HR `512×512` patch at `(4r, 4c)` for a ×4 pair, and with a
`256×256` HR patch for a ×2 pair — whatever the pair's own scale. It never pads, never blends, and never crops LR
and HR independently.

* Dense grid (`dense_origins`) reuses `frame.tiling.plan.axis_starts` and clamps the last window inside the raster.
* Random origins draw from a seeded generator; `centred_origin` gives the centred crop.
* SEN2NAIPv2's stored `130→520` pairs are cropped **centred** to `128→512`, with the HR offset exactly 4× the LR
  offset (tested).
* `check_alignment` rejects a pair whose HR is not exactly `scale ×` the LR; `geo.validate_pair_geometry` checks CRS,
  resolution ratio, footprint origin (within 0.01 HR pixel) and dimension relation for georeferenced pairs.

## 8. Synthetic degradation

Grounded in requirements sections 6, 9, 30–31 and the SEN2NAIPv2 card, which give the **sequence**:

```
HR ─▶ Gaussian blur (σ in HR px) ─▶ bilinear downsample ×scale ─▶ reflectance harmonisation ─▶ signal-dependent noise ─▶ LR
```

* **Configurable**: `DegradationConfig(scale, blur_sigma, harmonisation, noise_a, noise_b, clip_min)`. Harmonisation
  is `none` or `histogram_match` (needs a reference LR image). The learned U-Net variant needs trained weights and is
  **not available**.
* **Versioned**: every result carries `DegradationRecord(name="frame-sen2naipv2-style", version="frame-degradation/1", …)`.
* **Deterministic**: blur, downsampling and harmonisation are deterministic; the only randomness is the noise, drawn
  from a CPU generator seeded by `seed` and recorded, so an LR can be regenerated from its record.
* **Aligned**: bilinear sampling with `align_corners=False` puts LR pixel *i* at HR position `i·scale + (scale−1)/2`,
  the centre of the HR block it covers (verified with an impulse test — no shift).
* **Honest about parameters**: the requirements say to reproduce the established SEN2NAIPv2 degradation and *not*
  invent a new one. The **numeric parameters of the published process are not in the requirements or on the card**,
  so `frame_default_v1` (σ = 0.5·scale, noise variance `2e-5·x + 1e-6`) is a clearly-labelled FRAME default and every
  record carries `parameters_verified = False`. SEN2NAIPv2's own ready-made LR images are described by
  `origin="upstream"` records instead. **Do not describe `frame_default_v1` as the SEN2NAIPv2 degradation.**

## 9. Validation and QC

`python -m frame.data qc MANIFEST [--data-root DIR] [--headers] [--pixels] [--report OUT.json] [--strict]`

| Level | Checked | Finding codes (errors unless noted) |
|---|---|---|
| record | profile facts (bands, sizes, scale, pixel size, pair type, role) and LR/HR geometry | `unknown_dataset`, `unknown_variant`, `role_violation`, `band_mismatch`, `shape_mismatch`, `scale_mismatch`, `resolution_mismatch`, `pair_type_mismatch`, `crs_mismatch`, `footprint_mismatch`, `dimension_mismatch`, `not_georeferenced` *(warning)* |
| file (`--data-root`) | file exists; with `--headers`, the file's band count / size / dtype / CRS / transform agree with the record | `missing_file`, `unreadable_file`, `band_count_mismatch`, `shape_mismatch`, `dtype_mismatch`, `crs_mismatch`, `transform_mismatch` |
| pixel (`--pixels`) | NaN/Inf, reflectance range (float32-rounded model bounds), all-nodata, nodata fraction, constant images | `non_finite`, `invalid_range`, `no_valid_pixels`, `load_failed`, `high_nodata` *(warning)*, `constant_image` *(warning)* |
| manifest | duplicate ids, record count, scene/region leakage, role violations, proximity | `duplicate_id`, `record_count_mismatch`, `invalid_json`, `scene_leakage`, `region_leakage`, `spatial_proximity_across_splits` *(warning)* |

Exit code: `0` when there is no error-level finding (`--strict` also fails on warnings), `1` otherwise, `2` for an
unreadable manifest or other `DataError`. The JSON report (`total_samples`, `valid_samples`, `invalid_samples`,
`issue_counts`, `per_split`, `per_dataset`, `manifest_digest`, `checks_performed`, `invalid_sample_ids`, `issues`)
is byte-deterministic. QC never modifies or repairs data.

## 10. DataLoader

```python
from frame.data import PairedPatchDataset, collate_pairs
from frame.data.adapters import Sen2NeonAdapter
from frame.preprocessing import RGBN_BANDS

adapter = Sen2NeonAdapter.from_manifest("experiments/data_smoke/metadata/sen2neon_sample.manifest.jsonl")
load = lambda r: adapter.load_pair(r, lr_bands=RGBN_BANDS, hr_bands=RGBN_BANDS)
ds = PairedPatchDataset(list(adapter.iter_records()), load, lr_patch=128, mode="random", patches_per_pair=4, seed=0, augment=True)
# item -> {"lr": (4,128,128), "hr": (4,512,512), "metadata": {sample_id, patch_id, split, scale_factor, patch: {...}, ...}}
```

`mode="grid"` enumerates a dense deterministic grid (evaluation, reports each patch's `valid_fraction`);
`mode="random"` draws origins from a generator seeded by `(seed, epoch, pair, k)` — the same index always gives the
same patch, and `set_epoch` changes the draws. Mostly-nodata random patches are redrawn (up to `max_attempts`); a pair
with no valid pixel is an error, not silent zeros. `augment=True` applies the *same* flip/rotation to LR and HR
(the dihedral set `frame.uncertainty` already uses), keeps them aligned, and records the choice. All records in one
dataset must share a scale factor so a batch can be stacked; mix datasets by building one dataset per scale.
It is a minimal `torch.utils.data.Dataset` — no sampler, no distributed logic, no training loop.

## 11. Getting data

**SEN2NEON sample (done in Phase 3, 39.8 MB).** Pinned to revision `9f076b4f652aa0253127d382dcb6e611250b2e67`:

```bash
FRAME_DATA_ROOT=~/.cache/frame_data sen2sr_venv/bin/python experiments/data_smoke/run_smoke.py
```

This downloads `metadata.csv`, the LR checksum list and three tiles (two acquisitions at two NEON sites), verifies
the LR files against the published SHA-256 list, builds the manifest, runs deep QC and writes
`experiments/data_smoke/metadata/smoke_results.json`. `--offline` refuses to download. A larger subset is built the
same way: `python -m frame.data sen2neon-manifest --metadata-csv … --ids … --out …`, then fetch exactly the files
`sen2neon.relative_files(records)` lists. The full LR + 2.5 m HR is ≈ 30 GB; the whole repository is 354.6 GB.

**OpenSR-Test.** Uses the existing `frame.validation` loader and its local cache; no new download.

**SEN2NAIPv2 (format only — recipe is untested).** The official files are `.taco`. FRAME does not read them and
`tacoreader` is not installed. To use the data: read it with `tacoreader` in a separate environment, write each pair
as an LR and an HR GeoTIFF under `$FRAME_DATA_ROOT/sen2naipv2/`, and describe them with rows for
`frame.data.adapters.sen2naipv2.record_from_row` (keys: `id`, `variant`, `scene_id`, `region_id`, `split`,
`lr_path`, `hr_path`; optional `crs`, `lr_transform`, `hr_transform`, `lon`, `lat`, `reflectance_scale`).
`scene_id`/`region_id` must be supplied by the exporter because the TACO metadata schema was not inspected. Full size
is 139.1 GB, which exceeds the free disk on the development machine (≈126 GB) — do not download it in full.

**SEN2VENµS (format only).** 139.5 GB in 29 per-site zips of nested per-date zips (Zenodo record 14603764). Unpack the
patches you want and describe them with `sen2venus.record_from_row` (keys: `site`, `date`, `mgrs`, `idx`, `variant`,
`split`, `lr_path`, `hr_path`). `sen2venus.file_name` / `expected_file_names` encode the documented naming convention.

**Indian holdout.** Nothing exists yet. `india_holdout.lr_only_record` builds an LR-only record whose `hr_status`
stays `unknown` until a real, co-registered reference is found; it refuses to label one `available`.

## 12. Storage (Phase 3)

| Item | Size | Location |
|---|---|---|
| SEN2NEON sample (`metadata.csv` 3.1 MB, checksum list 0.2 MB, 3 LR tiles 1.4 MB, 3 HR tiles 35.0 MB) | **39.8 MB** (8 files) | `$FRAME_DATA_ROOT/sen2neon/` (outside the repo) |
| OpenSR-Test `spot` (cache from earlier phases) | not re-downloaded | `~/.config/opensr_test/` |
| SEN2NAIPv2, SEN2VENµS, full SEN2NEON | **not downloaded** (139.1 GB / 139.5 GB / 354.6 GB) | — |
| In-repo outputs (`experiments/data_smoke/metadata/`) | ≈ 11 KB | repository |

## 13. Measured on the real SEN2NEON sample

From `experiments/data_smoke/metadata/smoke_results.json` (RTX 3050 laptop, CPU-only data code):

* 8 files, 39,792,288 bytes; **all 3 LR checksums match** the published SHA-256 list.
* Deep QC (records + headers + pixels + split integrity): **3/3 valid, 0 issues**, 0.29 s. Each 12×256×256 / 12×1024×1024 pair loads in ≈ 0.06–0.07 s.
* HR valid-pixel fraction: 0.533, 1.0, 1.0 (the first tile has large NEON nodata; the LR has none).
* Evaluation grid at LR 128: 12 patches, 8 fully valid, 4 with some nodata, none empty.
* A random RGBN batch through a `DataLoader`: LR `(4,4,128,128)`, HR `(4,4,512,512)`, augmentations `rot90, identity, hflip, rot270`.
* **Registration sanity check on real data** (B08, Pearson correlation between the LR and the 4×4 block-mean of the HR): 0.524 / 0.871 / 0.935 at zero shift, versus 0.319 / 0.776 / 0.912 with the HR shifted one LR pixel east and 0.132 / 0.667 / 0.794 with two. The correlation is highest without a shift on every tile, which is consistent with the grids being correctly registered. It is a coarse check, not a sub-pixel registration measurement, and the low absolute value on the first tile reflects real cross-sensor differences and heavy nodata.
* A synthetic pair made from a real 512×512 HR patch is bit-identical under the same seed and differs under another.

## 14. Limitations (what Phase 3 is not)

* **SEN2NAIPv2 and SEN2VENµS are format adapters only** at the end of Phase 3 — no real file had been read; they were tested on synthetic fixtures written in their documented format. (Phase 5 later read a 130-pair SEN2NAIPv2 sample, §16; SEN2NEON/OpenSR-Test samples grew as well.) SEN2NAIPv2's `.taco` container is still not readable by FRAME itself: export is done in a separate `tacoreader` environment.
* Only a **3-tile** SEN2NEON sample and the existing OpenSR-Test `spot` subset were read. No large-scale processing was performed and none is claimed.
* The **Indian holdout is a profile and a record builder only**; there is no Indian data, no HR reference, and no Indian generalisation result.
* `frame_default_v1` is **not** the published SEN2NAIPv2 degradation (`parameters_verified=False`); the U-Net harmonisation variant is not available.
* No training, no fine-tuning, no metric-based dataset comparison, no HR-reference experiment, no downstream analytics.
* Spatial-proximity leakage detection is centroid-based and approximate.
* The 12-band ×4 task itself (the 10-band cascade) is not implemented; the data layer only keeps all 12 LR bands available by name.
* No STAC, COG, Docker, async queues or new dependencies were added.

## 15. Tests

`frame/tests/test_data_*.py` (13 files), with shared fixtures in `frame/tests/data_fixtures.py` (a tiny coherent
synthetic dataset whose LR is the exact block mean of its HR) and `frame/tests/data_real_rows.py` (three real SEN2NEON
metadata rows, CC BY 4.0). Tests never download. Tests that need the real sample (`test_data_real_sample.py`) or the
cached OpenSR-Test `spot` subset skip themselves when it is absent.

```bash
export TMPDIR=~/.cache/frame_tmp                                     # keep pytest temp files off a small /tmp
sen2sr_venv/bin/python -m pytest frame/tests/test_data_*.py -q -p no:cacheprovider --basetemp=~/.cache/frame_pytest_tmp
```

## 16. Phase 5 update: what has now been read for real

Phase 5 (`docs/EVALUATION.md`) read more real data. Nothing here was used for training; SEN2NEON and OpenSR-Test remain test-only benchmarks.

| Dataset | What was read | Size | Location |
|---|---|---|---|
| SEN2NEON | a **seeded uniform random** draw of 30 of the 2,269 tiles (seed 0, no filtering by nodata, land cover or site; `experiments/evaluation/manifests/sen2neon_random30_seed0.jsonl`), LR checksums verified; 28 acquisitions | 423 MB with the Phase 3 tiles | `$FRAME_DATA_ROOT/sen2neon/` |
| OpenSR-Test | `spot` (9), `spain_crops` (28), `spain_urban` (20), MIT licence | 240 MB for the three | `~/.config/opensr_test/` |
| SEN2NAIPv2 | **130 real pairs** (100 `unet`, 30 `crosssensor`), a seeded draw over all 61,282 / 8,000 samples, exported unmodified from the TACO containers with ranged reads. Not a training set; **not** an evaluation set (SEN2SR was trained on this dataset) | 287 MB | `$FRAME_DATA_ROOT/sen2naipv2/` |

SEN2NAIPv2 facts now **verified on real files** (`experiments/evaluation/data_acquisition/naipv2_sample_report.json`): pairs are 130×130 (LR) and 520×520 (HR) `uint16`, band order B04, B03, B02, B08 (block-mean of the HR correlates 0.92-0.99 with the LR of
the same band); no nodata in the sample; LR/HR origins agree to 0.3 mm (the Phase 3 tolerance is 25 mm); a median 99th-percentile DN of 3.5-3.8k is consistent with DN / 10,000 reflectance but the scale is **inferred, not documented**. The dataset's own split labels
(train / validation / test) are geographically **mixed** inside a small draw, so they cannot serve as a geographic split; FRAME's seeded state-level split of the 100 `unet` pairs gave 79 / 21 pairs over 37 / 5 disjoint states with no leakage. The `.taco` container needs
`tacoreader` (0.4.5 used), which lives in a **separate export environment** and is not a FRAME or Mamba dependency; `tacoreader` returns rows in a different order on every load, so draws sort by `tortilla:id` first. Whole-part acquisition was assessed and deferred:
`experiments/evaluation/data_acquisition/SEN2NAIPV2_DECISION.md`. SEN2VENµS is still format-only, and the Indian holdout still has no data.
