# Operational GIS-ready delivery (Requirement 11, Step I)

What Step I actually built on top of the already-real pipeline (Steps B–H: real Sentinel-2
acquisition → 12-band harmonisation → RGBN adapter → real Lite/Mamba SR), and the conservative
Requirements 1–12 closure this project can currently support. Every number here was measured on
this machine during Step I; nothing is copied from an earlier phase's numbers unless the source
document is named.

## 1. The product model

Three kinds of product exist, and Step I keeps them explicitly distinct everywhere they surface
(the API, STAC properties, GeoTIFF tags, this document):

| Product | Bands | Grid | Where it lives |
|---|---|---|---|
| **A. Source / harmonised observation** | up to 12 real L2A bands (B01–B09, B8A, B11, B12; B10 correctly excluded) + SCL | the common computational grid `frame.acquisition.harmonise` resolves everything onto | `HarmonisedProduct` (in-process); the RGBN subset is what actually reaches SR |
| **B. Learned SR product** | exactly the 4 bands the registered model reconstructs: **B04, B03, B02, B08** | reconstructed 2.5 m pixel grid (10 m → ×4) | `/products/{job_id}/cog` (a real COG), `/sr/download/{job_id}` (plain GeoTIFF, unchanged from Phase 7) |
| **C. Evidence products** | TTA stability (5 bands: 4×`{band}_std` + `overall_std`); NDVI/NDWI self-consistency (scalar, in the job/analysis result) | same grids as B | `/products/{job_id}/stability`, `/products/{job_id}/analysis` |

**No 12-band learned SR exists or is claimed.** `select_rgbn_adapter` (Step F/G) is still the one
boundary between A and B; the other 8 bands are named as excluded in every STAC item's
`frame:source_scene_id`/asset list, never silently dropped or implied to be super-resolved.
LDSR-S2 remains capability-only (docs/REGISTRY.md §1a) and was not touched this step.

## 2. Cloud Optimized GeoTIFF (`frame.geospatial.write_cog` / `validate_cog`)

Real GDAL COG driver (bundled in this project's rasterio 1.4.4 / GDAL 3.10.3 — no separate
`osgeo.gdal` or system `gdalinfo` is installed). Creation options: `COMPRESS=DEFLATE`,
`PREDICTOR=3` (lossless, correct for float32 scientific values — never JPEG), `BLOCKSIZE=512`,
`OVERVIEWS=AUTO`.

**Validation** (`frame.geospatial.cog.validate_cog`) checks, through rasterio directly:
GDAL's own `IMAGE_STRUCTURE` `LAYOUT=COG` self-report (the validity gate), CRS present, a real
(non-identity) affine transform present; it also reports internal tiling, compression and
overview presence as diagnostics, *not* as additional validity gates.

**A real, measured edge case:** a 512×512 output (the common case — a single 128×128 tile at
scale 4) writes `blockxsize=blockysize=512` as one TIFF *strip*, not a *tile*, because the whole
image fits in exactly one block; `rasterio`'s `is_tiled` then reports `False` even though GDAL's
own `LAYOUT` tag still says `COG`. Verified at sizes 256/511/512/513/600/1024 — only 512 (and any
other exact multiple of the blocksize) shows this. `validate_cog` follows the `LAYOUT` tag, not
`is_tiled`, for exactly this reason (`frame/tests/test_geospatial_cog.py`).

**What was *not* checked:** the official GDAL `validate_cloud_optimized_geotiff.py` script and
`rio-cogeo`'s validator were not run (neither `osgeo.gdal` nor `rio-cogeo` is installed, and
adding either was judged an unnecessary new dependency for this stage — Step I's own instruction
to reuse existing primitives rather than add new infrastructure). `validate_cog`'s docstring
states this explicitly so a reader never mistakes it for the official validator.

## 3. STAC (`frame.product.stac`)

A real STAC 1.0.0 Item as a plain dict (no `pystac` dependency — Step I judged a single Item
sufficient for this project's scale, per its own instructions). FRAME-specific facts are
namespaced under `frame:` (the same extension-prefix convention STAC's own `proj:`/`eo:`/`sat:`
extensions use): `frame:model_id`, `frame:checkpoint_sha256`, `frame:scale_factor`,
`frame:radiometric_convention`, `frame:source_scene_id`, `frame:product_statement` (the fixed
"reconstructed 2.5 m grid, not native" sentence, carried in the file itself, not only in prose).
`bbox`/`geometry` are computed by real reprojection (`rasterio.warp.transform_bounds`) from the
product's own CRS/bounds — never fabricated, and `build_stac_item` raises rather than guessing a
geometry when a raster has none. `frame:source_scene_id`/`platform`/`source_provider` are `None`
(not invented) for a manually-uploaded GeoTIFF with no real acquisition provenance.

`validate_stac_item` checks the spec's required top-level fields, `bbox` shape, `geometry`
fields, `properties.datetime`, and that every asset has `href`/`type` — real, itemised structural
checks, not the official JSON-Schema STAC validator (not installed either, same reasoning as §2).

## 4. Checkpoint hashes

`frame.api.services.model.checkpoint_sha256(model_id)` computes the real SHA-256 of the cached
weight file on disk at call time (`hashlib.sha256` streamed over the file) — Lite:
`479aa796d5068d0b1206118ccbca27bd3223df0214db1a9b31a1e18349ed1c7e` (`model.safetensor`), Mamba:
`11e551b03663e873c5dcd4f02ec3fc8cdfa41ef46677540295ee4c6686ddc854` (`sr_model.safetensor`), both
independently recomputed this step and matching the values `docs/REGISTRY.md` already recorded.
`None` if the file is not cached locally — never a placeholder value.

## 5. Product registry (`frame.api.services.product`, new API routes)

`product_id == job_id`: no second identity space. New routes (all backed by the existing
`JobStore`, nothing duplicated):

| Route | What it does |
|---|---|
| `GET /products/{job_id}` | Status, COG availability + its real validation result, evidence asset paths, linked analyses, STAC URL |
| `GET /products/{job_id}/stac` | The real STAC Item (§3) |
| `GET /products/{job_id}/cog` | Downloads the real COG (written once, cached, from the exact tensor `/sr/run` already produced) |
| `GET /products/{job_id}/stability` | Same file as the existing `/uncertainty/download/{job_id}` — Step I's own instruction that "a single existing route can satisfy more than one conceptual requirement" applied directly |
| `GET /products/{job_id}/analysis` | Every NDVI analysis run against this job so far |
| `GET /products/{job_id}/tiles/{z}/{x}/{y}.png?layer=rgb\|stability` | §6 |

## 6. Tile serving (`frame.product.tiles`) — minimal, not TiTiler

**No `titiler`/`rio-tiler` dependency was added.** This environment has neither installed, and
adding a full ASGI tiling service was judged more infrastructure than this stage needs. Instead,
`frame.product.tiles` reads the requested Web Mercator XYZ tile window directly from the real COG
via `rasterio.warp.reproject` — the same reprojection/resampling primitive TiTiler itself is
built on — and encodes a 256×256 PNG (2–98th percentile stretch per request; no alpha channel, so
pixels outside the raster's real footprint render black, not transparent). Two layers: `rgb`
(bands 1–3 = B04/B03/B02, true colour) and `stability` (the `overall_std` band). This is real,
tested (`frame/tests/test_product_tiles.py`, `frame/tests/test_api_products.py`), and reachable
over HTTP — proven end-to-end in `experiments/final_demo/` (§9) — but it is a minimal in-process
renderer, not a general tile server: no caching, no vector tiles, no other tiling scheme.

## 7. A minimal background-job path (`frame.api.services.async_jobs`)

`POST /sr/run` is **unchanged and untouched** — every existing test and the entire frontend still
depend on it exactly as before. `POST /jobs/sr` / `GET /jobs/{async_job_id}` are new, additive:
a `ThreadPoolExecutor` and an in-memory status dict (`queued`/`running`/`completed`/`failed`),
not Celery/Redis or a durable queue — Step I's own instruction that a minimal local abstraction
is sufficient here. Model resolution (and any "unavailable model" 503) still happens
synchronously before a job is even queued, so a bad request never fails silently inside a
background thread (`frame/tests/test_api_async_jobs.py`, 5 tests).

## 8. Engineering experiments (`experiments/operational_delivery/run_experiment.py`)

Real measurements on this machine (RTX 3050 Laptop, 4 GB VRAM), synthetic reflectance-range
scenes (engineering timing/memory, not an accuracy claim — the separate §9 demonstration uses a
real live retrieval instead).

**A — tile size.** Only **128×128 succeeds**; 64 and 256 both raise the same real, reproducible
`RuntimeError` (`frame.tiling` itself accepts any `tile_size`, but the registered model's own
hard-constraint mask is baked to a fixed 128→512 shape, confirmed by calling the model directly
at each size before even involving the tiling engine). This is not a `frame.tiling` limitation —
it is the model contract already documented in `docs/REGISTRY.md`, now verified with a specific,
reproducible error message rather than only asserted.

**B / F — overlap and seam quality**, 300×400 scene, overlap ∈ {0, 8, 16, 32}: all four run
successfully (12 tiles each, 0.11–0.15 s). Seam RMSE (frame.tiling's own no-reference seam
diagnostic) measured 0.0268 (overlap 8) → 0.0191 (16) → 0.0134 (32) — lower with more blending,
as expected mathematically from averaging more overlapping predictions; this is a measurement of
*this specific run*, not a general claim that more overlap is "better" (it also costs more
tiles/compute at scale, per Experiment C).

**C — scaling**, synthetic scenes 128/256/512/768 px: near-linear per-tile cost, 0.0100 → 0.0103
→ 0.0117 → 0.0120 s/tile. 1024×1024 (the API's own `MAX_INPUT_PIXELS` cap) was not additionally
run; extrapolating from this trend is arithmetic, the same method `docs/REGISTRY.md` §3 already
uses and discloses as such for Mamba.

**D — CPU vs GPU**, Lite, 256×256 scene: CPU 0.369 s, GPU (CUDA) 0.100 s (≈3.7×), peak VRAM 109.5
MiB — consistent with `docs/REGISTRY.md`'s existing ≈100 MiB figure for Lite.

**E — output integrity**: 12/12 automated checks passed (CRS, transform, sane bounds, exact
scale-factor dimensions, band count/names, float32 dtype, no NaN/Inf, valid COG, pixel size
exactly 2.5 m in the output CRS, valid STAC item, real asset file existence).

Full record: `experiments/operational_delivery/record.json`.

## 9. GIS compatibility (`frame/tests/test_gis_compatibility.py`)

A permanent, re-runnable pytest suite (not only a one-off script) proving the generated product
is a real geospatial raster: rasterio can reopen it; CRS, transform and sane bounds are real;
pixel size is exactly 2.5 m in the output CRS; band descriptions are correct and in order;
dtype is float32 with no NaN/Inf; COG validation passes; a STAC item's asset reference points at
a file that actually exists on disk. `gdalinfo`/`rio-cogeo` are confirmed absent (asserted, not
just stated) so the limitation in §2 is enforced as an executable fact, not only prose.

## 10. Final real end-to-end demonstration (`experiments/final_demo/`)

One complete, real run, real network, real weights, through the actual FastAPI app (no
dependency overrides — the real `get_model`/`get_device`):

```
real AOI (80.204, 13.006, 80.2168, 13.0188)
  -> real CDSE discovery: 7 candidates, 3.97 s
  -> real scene selection: S2B_MSIL2A_20240330T045659_N0510_R119_T44PMV_20240330T072508 (2.4% cloud)
  -> real retrieval: 139x142 px, 12 bands, 7.24 s
  -> 12-band harmonisation: complete L2A, validity mask coverage 1.0
  -> RGBN adapter: 4 bands kept (B04,B03,B02,B08), 8 excluded (B05,B06,B07,B8A,B11,B12,B01,B09)
  -> real upload -> real Lite inference: 1.84 s, (4,142,139) -> (4,568,556), EPSG:32644, 2.5 m
  -> real NDVI analysis: COMPUTABLE
  -> API product registration: COG valid
  -> STAC generation: valid (0 problems)
  -> COG download: 4,789,861 bytes, re-validated valid
  -> stability evidence download: 6,324,455 bytes
  -> tile access: real PNG, 72,743 bytes, z=15/x=23684/y=15189
  -> analysis listing: 1 analysis
```

Full machine-readable record: `experiments/final_demo/record.json` (no secrets — checked by
grep as part of this step; only `"no_secrets_recorded": true` matches a secret-related search
term, itself not a secret).

**What this demonstration does *not* include:** an actual browser rendering the tile layer in
MapLibre. The tile endpoint is proven reachable and returns a real, correctly-shaped PNG; the
frontend was not modified to add a MapLibre layer consuming it (see §11) and no live browser
session was used to visually confirm map rendering this step. This is a disclosed gap, not an
implicit claim.

## 11. Frontend

**Not modified this step.** `frontend/src/components/workspace/InspectViewer.tsx` already
implements exactly the four-mode structure Requirement 11 asks for — **Original / Super-resolved
/ NDVI / Evidence** — via client-side GeoTIFF decoding and canvas rendering (pixel probe, region
statistics, a ruler, split-compare, a real geospatial metadata drawer), all against real backend
data; this was true before Step I and is unaffected by it. Adding a second, MapLibre-tile-based
view of the same product (§6's new capability) was judged out of scope for this step: the
existing viewer already satisfies the functional requirement, and modifying a working, tested,
already-reviewed component risked exactly the "redesign" Step I's own instructions warn against.
**This is the one clear, disclosed gap between what the backend now supports and what the
frontend currently shows.**

Verified this step: `npm run lint` (tsc --noEmit) clean; `npm run test` (vitest) 45/45 passed;
`npm run build` succeeds (one pre-existing chunk-size warning, unrelated to Step I).

## 12. Requirements 1–12 closure matrix

Conservative: a requirement is **CLOSED** only when the capability is both implemented and
supported by evidence sufficient for that specific claim, not merely because code exists.
Machine-readable copy: `experiments/final_demo/closure_matrix.json`.

| # | Requirement | Status | Evidence | Remaining gap | Safe claim |
|---|---|---|---|---|---|
| 1 | Sentinel-2 data acquisition | PARTIALLY CLOSED | Real CDSE discovery/retrieval, §10 live run | No async/queued acquisition; single-provider-at-a-time per band group | "Real Sentinel-2 scenes are discovered and retrieved through the Copernicus Data Space Ecosystem." |
| 2 | Multispectral preprocessing / harmonisation | PARTIALLY CLOSED (RGBN path) | 12-band harmonisation + RGBN adapter, real (Step F/G), exercised again in §10 | Non-RGBN bands are resampled/aligned only, never learned-SR'd | "A real 12-band L2A product is harmonised onto one grid; only RGBN reaches the SR model." |
| 3 | Paired LR–HR dataset creation | EVIDENCE-ONLY | `frame.data` roles/splits/leakage checks, real samples (SEN2NEON, OpenSR-Test, a SEN2NAIPv2 sample) | SEN2VENµS format-only; full geography-checked SEN2NAIPv2 not done; MuS2 absent; no Indian data | "Dataset roles, splits and leakage checks are implemented and tested on real samples; full-scale acquisition was not performed." |
| 4 | Super-resolution model | PARTIALLY CLOSED | Real Lite/Mamba behind one registry, LDSR-S2 verified-but-undeployable (docs/REGISTRY.md) | No 12-band/multi-resolution model; no ranking is or will be made | "Two real RGBN ×4 models are selectable, with a documented, verified compute-budget reason a third is not." |
| 5 | Spectral consistency | PARTIALLY CLOSED | `frame.consistency`/`frame.evaluate` metrics (RMSE/MAE/bias/correlation/SAM/ERGAS/NDVI/NDWI), Step H additions | RGBN self-consistency only; MNDWI/NDBI/BSI HR-reference-only | "Spectral-consistency metrics are implemented and measured; they are not accuracy against ground truth." |
| 6 | Geospatial consistency | CLOSED for the stated scope | CRS/transform/bounds preserved and derived, COG-valid output (§2, §9), tile-seam diagnostic | Co-registration of a reference *image* is a separate, measured problem (`docs/RELIABILITY.md` §3) | "The output grid's CRS, transform, pixel size and bounds are correct and automatically verified." |
| 7 | Uncertainty / model-stability | EVIDENCE-ONLY, negative-to-mixed | `frame.uncertainty`/`frame.reliability`, real Indian measurement (Step H) | Weakly associated with error, uncalibrated, texture-confounded (`docs/RELIABILITY.md`) | "TTA stability is a reconstruction-variation diagnostic, not calibrated uncertainty." |
| 8 | SR accuracy and validation | PARTIALLY CLOSED | `frame.evaluate`, real benchmark runs (SEN2NEON, OpenSR-Test), §8 engineering experiments | No SEN2VENµS/full-SEN2NAIPv2 evaluation, no ESA baseline, no perceptual metrics | "SR is measured against independent references with a registration gate; no ranking is made." |
| 9 | High-resolution reference validation | OPEN for India | Reference eligibility gate, public-reference results | **No Indian HR reference exists or was substituted** | "No Indian high-resolution reference has been obtained; none is claimed." |
| 10 | Downstream analytical utility | EVIDENCE-ONLY, mixed to null | `frame.downstream` NDVI/NDWI, real Indian self-consistency (Step H/this step's demo) | No labelled task (land cover, crop, flood, burn-scar); India has self-consistency only, not downstream validation | "NDVI/NDWI self-consistency is demonstrated; no task-level (e.g. land-cover) performance is claimed." |
| 11 | Operational / GIS-ready delivery | PARTIALLY CLOSED | COG, STAC, product registry, tiling, a minimal background-job path, a real end-to-end demonstration (this document) | No MapLibre tile-layer integration in the frontend (§11); no TiTiler; no auth/containers/deployment | "A real Sentinel-2 scene can be turned into a validated, downloadable COG with a real STAC item, over a real (demo-grade) API, end to end." |
| 12 | Geographic / Indian generalisation | EVIDENCE-ONLY, single scene | `frame.generalization` (Step H): two real Indian-scene runs (Lite, Mamba), no HR reference | One geographic point (Tamil Nadu/Chennai) ≠ India-wide generalisation; no urban/agriculture/coastal/forest/dry diversity was executed | "A reproducible domain-shift protocol exists and was run once on a real Indian scene; this is not evidence of India-wide generalisation." |

## 13. What remains a genuine gap (not fabricated around)

Indian HR reference; MuS2; full geography-checked SEN2NAIPv2 evaluation; SEN2VENµS real
exercise; an ESA baseline; perceptual metrics; full 12-band learned SR; a MapLibre tile-layer
frontend integration for the new `/products/{job_id}/tiles/...` endpoint; the official GDAL/STAC
validators (not installed); a true durable job queue (the local `ThreadPoolExecutor` is
disclosed as minimal). None of these are asserted as done anywhere in this document or its code.
