# SEN2NAIPv2 acquisition decision (Phase 5)

**Decision: no whole part was downloaded.** A small, seeded, uniformly drawn sample (130 real pairs, 287 MB) was exported instead and verified; that answered every question a whole part
would have answered *for Phase 5*. Whole-part acquisition is deferred to the phase that actually trains on it, with the numbers below.

## What was checked before acquiring anything (remote metadata only, seconds, no pixel data)

| Variant | Pairs | Parts (size) | Dataset's own labels | Geography |
|---|---:|---|---|---|
| `unet` (synthetic LR, primary training) | 61,282 | 4 parts: 20.0 + 20.0 + 20.0 + **10.28 GB** | train 49,642 / validation 5,512 / test 6,128 | 56 states, 3,158 counties; **each part is a longitude band** (e.g. part 0003: 8,761 pairs, 19 states, lon −81.2…−67.1) |
| `histmatch` (synthetic LR) | 61,282 | 4 parts: 20.0 + 20.0 + 20.0 + 9.36 GB | same | same, but each part spans wider longitudes |
| `crosssensor` (real S2 ↔ NAIP within a day) | 8,000 | **one file, 9.72 GB** | all 8,000 labelled train | 53 states, 1,824 counties, lon −124.2…−67.8 |

* **Licence**: CC0-1.0 (from the dataset card, recorded in `frame/data/roles.py`).
* **Reader**: `tacoreader` (0.4.5 was used, the version of the dataset card's examples). It is **not** a FRAME dependency and was **not** installed into `sen2sr_venv` or `sen2sr_mamba_venv`; it lives in a
  separate export environment (`~/.venvs/frame_taco`, 516 MB, outside the repo), and everything under `data_acquisition/` runs there. The main environment only sees plain GeoTIFFs and a `rows.jsonl`.
* **Free disk**: 124 GB on the target volume (measured after the 287 MB sample), i.e. a 10 GB part would fit comfortably.
* **Purpose**: (1) verify the real file format FRAME's adapter was written against without ever having seen real data; (2) exercise the geographic-split machinery on real coordinates;
  (3) decide whether a whole part is worth its cost. The sample is **not** a training set and was **not** used to score any model (SEN2SR was trained on SEN2NAIPv2, so it is not an independent benchmark, and the
  evaluation config refuses it).

## What the sample verified (`naipv2_sample_report.json`; 100 `unet` pairs + 30 `crosssensor` pairs, 0 failures)

* All 130 pairs open, headers and pixels read, LR 130×130 / HR 520×520, four bands in the order B04, B03, B02, B08 (block-mean of the HR correlates with the LR on the matching band at 0.92–0.99 and
  weakly on the others, i.e. the band correspondence is diagonal), no nodata, and the LR/HR origins agree to 0.3 mm (FRAME's own tolerance is 0.01 HR pixel = 25 mm).
* The stored `uint16` values are used unscaled as DN with reflectance = DN / 10,000: the median 99th-percentile DN is 3,538-3,816 (a plausible reflectance of ~0.35-0.38); this supports the working assumption
  of the value scale, which the dataset card does not state, but it is an inference, not a documentation fact.
* The Phase 3 records validate (roles, checksums, geometry): 100/100 and 30/30 valid.
* The dataset's own split labels are geographically **mixed** inside every small draw (85 train / 8 validation / 7 test among 100 unet pairs), so its labels cannot be used as a geographic split. FRAME's
  seeded split at region = US state gave 79 train / 21 val pairs with 37 / 5 disjoint states and no leakage error (`geographic_split`), which is the workflow a later training phase should use.
* Geography of the 100-pair unet sample: 42 states, 95 counties, lat 26.8-48.2, lon −123.4…−71.0.

## Cost profile measured

Ranged reads of individual pairs cost about **6.5 pairs/min** (latency-bound, ~1.2 MB per pair; a transient network error mid-run is retried and reported); streaming a whole part is estimated (from the measured download throughput, not from a completed part) at about
**92 pairs/min**, i.e. roughly 1.5 h for the 8,761-pair part. Ranged access is therefore the right tool for hundreds of pairs and a whole part the right tool beyond ~1,500 pairs.

## Suitability for later geographically split training / validation

* **Suitable.** Real geography, an explicit CC0 licence, a documented format, verified alignment, and a demonstrated leakage-free state split.
* **Limits that a later phase must respect**: North America only (no Indian evidence; requirements §26 still needs an Indian holdout); the `unet`/`histmatch` LR is synthetic (upstream degradation
  whose parameters FRAME has not verified); `crosssensor` has only 8,000 pairs; a **single** `unet` part covers a longitude band, not the country, so a state-level split from one part means
  a few states, and a national split needs at least two parts of different longitude bands.

## If a part is acquired later (recommended options, none done now)

1. **`crosssensor` (one 9.72 GB file, 8,000 real pairs, 53 states)** - the only single download that covers the whole country; useful for *validation* under the Phase 3 role (real cross-sensor pairs
   are not a training source in FRAME's data roles).
2. **`unet` part 0003 (10.28 GB, 8,761 pairs, 19 states, east coast)** - the smallest training part; enough for a first, geographically narrow training run, not for national claims.

Either would be streamed to a directory **outside the repository** (`$FRAME_DATA_ROOT`), verified with the size and (if published) checksum, then inspected for coverage before use. Not the full 149 GB.
The next phase should decide this with the training compute plan (a GPU with enough memory for Mamba is not available locally, see `docs/TRAINING.md`), not now.
