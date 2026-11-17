# Requirements traceability

Source of the requirements: `docs/Requirements 142.txt`, the list of twelve at its head (lines 2–195). It is not rewritten here. The table says, for each requirement, what FRAME built, what was measured, and what data exist, **as three
separate statuses**, because an implemented pipeline can have limited scientific validation and a validation can be limited by the data that could be obtained:

* **Engineering** — does the code exist, is it tested, and does a real end-to-end path work? `implemented` · `partially implemented` · `deferred`.
* **Scientific validation** — what does measured real-data evidence establish? `evidence-limited` (measured, small or narrow or mixed) · `not validated` (nothing measured) · `n/a`.
* **Data** — could the data the requirement asks for be obtained? `available (sample)` · `partial` · `none`.

No requirement is declared "complete". Numbers and caveats are in the linked documents; the claim wording is in [`CLAIMS.md`](CLAIMS.md), the datasets in [`REGISTRY.md`](REGISTRY.md) §2.

> *Numbering.* The memo's own index numbers the uncertainty requirement 7 and the SR-accuracy requirement 8 (lines 104, 126) and its body follows that (line 7766: "Requirement 7 asks the harder question"); a section near line 10650 of the memo is nevertheless headed "Requirement 8 final verdict" while discussing uncertainty. This page follows the index.
> The repository's other "Phase" numbers refer to two different sequences (the original nine-phase build and the later gap-closure roadmap): see [`README.md`](README.md) §3.

## Summary

| # | Requirement | Engineering | Scientific validation | Data |
|---|---|---|---|---|
| 1 | Sentinel-2 data acquisition | partially implemented | n/a | available (user-supplied GeoTIFFs) |
| 2 | Multispectral preprocessing and harmonisation | partially implemented (RGBN 10 m path) | n/a | available |
| 3 | Paired LR–HR dataset creation | implemented | evidence-limited (real samples only) | partial |
| 4 | Super-resolution model | partially implemented (two SEN2SR RGBN models) | evidence-limited | available (published weights) |
| 5 | Spectral consistency | implemented | evidence-limited | partial |
| 6 | Geospatial consistency | implemented | evidence-limited (the reference registration limits it) | available |
| 7 | Uncertainty / model-stability / error awareness | implemented (a TTA diagnostic) | **evidence-limited, negative-to-mixed**: weak, texture-like, uncalibrated | partial |
| 8 | SR accuracy and validation | implemented | evidence-limited | partial |
| 9 | High-resolution reference validation | partially implemented (public references + a registration gate) | evidence-limited | partial (public only; **no Indian reference**) |
| 10 | Downstream analytical utility | partially implemented (NDVI only) | evidence-limited, **mixed to null** | partial (**no land-cover labels**) |
| 11 | Operational / GIS-ready delivery | implemented as a demo-grade prototype; real COG/STAC/product API and a real end-to-end run added (Step I) | demonstrated once, end-to-end, real network | n/a (no throughput claim) |
| 12 | Geographic / Indian generalisation | implemented (a reproducible domain-shift protocol + runner) | **single-scene evidence only, not a generalisation claim** | 2 real Indian scenes, no HR reference |

## Detail

Test counts are from `pytest frame/tests --collect-only` at the end of Phase 8 (2,044 unit tests plus 33 marked `integration`; frontend 70).

### 1. Sentinel-2 data acquisition — engineering *partially implemented* · validation n/a · data *available (user-supplied, and now real Copernicus scenes)*

* **Built (requirements-recovery pass, see `docs/ACQUISITION.md`).** A real, tested `frame.acquisition` package: `AcquisitionQuery`/`SceneCandidate`/`RetrievedScene` contracts; real scene discovery against the Copernicus Data Space Ecosystem STAC API (`frame.acquisition.search`, collection `sentinel-2-l2a`, client-side cloud-cover filtering, a named `NoScenesFoundError`/`ProviderUnavailableError` rather than a silently-empty result); real, credentialed pixel retrieval via the CDSE Sentinel Hub Process API (`frame.acquisition.providers.cdse`, OAuth2 client-credentials, per-resolution-group evalscript requests, `REFLECTANCE` units server-side) as the primary provider, with Microsoft Planetary Computer (`frame.acquisition.providers.planetary_computer`, via the existing `cubo` dependency) as an honestly-recorded fallback when CDSE is unconfigured or fails; multi-resolution harmonisation onto one grid (`frame.acquisition.retrieve`, bilinear for reflectance bands, nearest-neighbour for the categorical SCL layer, native resolution recorded per band, never silently presented as uniform). The two providers' differing DN-to-reflectance conventions are reconciled and the residual discrepancy is disclosed as a provenance warning, not absorbed silently.
  The product's own `POST /upload` (user-supplied GeoTIFF) and `POST /aoi/preview` (shape-only preview, does not fetch imagery) are unchanged and remain the fallback path.
* **Evidence.** `test_acquisition_contract.py`, `test_acquisition_search.py`, `test_acquisition_cdse.py`, `test_acquisition_geometry.py`, `test_acquisition_retrieve.py`, `test_acquisition_planetary_computer.py` (fixture-backed, no live network required); `test_acquisition_live.py` (skipped without real `FRAME_CDSE_CLIENT_ID`/`SECRET`; run live in this pass — real STAC search plus real 10 m/20 m/60 m + SCL retrieval and harmonisation over a live AOI near Chennai, all passing). `test_api.py` (16), `test_api_pipeline.py`, `test_api_tiling.py` cover the pre-existing upload path unchanged.
* **Built (API layer, same pass).** `POST /acquisition/search` (bbox or GeoJSON AOI), `GET /acquisition/scenes/{search_id}/{scene_id}`, `POST /acquisition/retrieve` (writes real GeoTIFF artifacts), `GET /acquisition/download/{retrieval_id}/{bands,scl}` — real endpoints over the library above, with named, typed error codes (`invalid_acquisition_query`, `acquisition_provider_unavailable`, `acquisition_authentication_failed`, `acquisition_scene_not_found`, `acquisition_retrieval_mismatch`, `acquisition_retrieval_not_found`) rather than a generic passthrough, following the same schema/error/JobStore conventions as every other route.
* **Evidence (API layer).** `test_api_acquisition.py` (15, FastAPI TestClient, fixture-backed), `test_api_errors.py` (extended with the new status/code mapping). Verified directly against a real live CDSE search that the response schema carries a real GeoJSON footprint, MGRS tile, processing baseline, platform, and cloud cover -- the fields Step D's map needs (`docs/ACQUISITION.md` §5.1).
* **Built (frontend, same pass).** A real MapLibre GL JS map-first flow (`SelectArea` → `SelectScene` → `SceneConfirmed`): draw a rectangle/polygon AOI, choose date range/cloud threshold, real search, real scene footprints and metadata, selection, and an optional real retrieval action -- all against the real endpoints above. Manual GeoTIFF upload (`UploadScene`) kept intact as an explicit fallback. Live testing against the production Copernicus API found and fixed a real bug: retrieval used to request every band group (and SCL) at its native resolution regardless of the target grid, so a large AOI at a coarse target resolution still triggered an oversized real request; fixed to request at `max(native, target)` resolution per group.
* **Not done.** A retrieved scene's bands are not yet registered as an `UploadRecord`, so `POST /sr/run` cannot yet consume them (Step E). Bhoonidhi (the Indian-specific route) was not attempted: it requires separate Indian-government registration/credentials this environment does not have. A GeoJSON AOI is reduced to its bounding box, not filtered by its real polygon shape. 12-band ingestion validation and the multispectral preprocessing extension (Requirement 2) that would consume this package's output are also not yet built.

### 2. Multispectral preprocessing and harmonisation — *partially implemented* · n/a · *available*

* **Built (original RGBN path, unchanged).** `frame.preprocessing` for the four-band RGBN 10 m path: bands selected **by name** and reordered to B04, B03, B02, B08; explicit reflectance scaling (never guessed); nodata, non-finite and optional SCL masks kept apart from the tensor; opt-in content validation (no valid pixel, contradicted scale); provenance metadata.
* **Built (requirements-recovery, Step F — real 12-band harmonisation, see `docs/ACQUISITION.md` §8).** `frame.acquisition.harmonise`: `validate_full_l2a_bands` (the real 12-band L2A contract — B01–B09, B8A, B11, B12; B10 correctly excluded and never checked for); `HarmonisedProduct`, wrapping an already-retrieved scene with a real SCL-derived `ValidityMask` (never used to delete pixels) and per-band radiometric-conversion provenance naming which real provider (CDSE Process API vs. Planetary Computer) served each band; `select_rgbn_adapter`, the one explicit, tested boundary from a (potentially 12-band) harmonised product into the existing, unmodified RGBN SR pipeline. Real per-band native-resolution (10/20/60 m) and resampling-method provenance was already produced by `frame.acquisition.retrieve` (Step B) and is now surfaced through `POST /acquisition/retrieve`'s response (`band_radiometric_method`, `is_complete_l2a`, `validity_mask_coverage`).
* **Evidence.** Original path: `test_pipeline.py` (23), `test_masks.py` (11), `test_reflectance.py` (8), `test_metadata.py` (8), `test_integration_contract.py`. New: `test_acquisition_harmonise.py` (34 fixture-backed tests covering the band contract, native-resolution provenance, resampling discipline, radiometric-method tagging, masking, geospatial pass-through, and a real integration test proving a full 12-band harmonised product feeds the existing, unmodified `/sr/run`; plus 1 live test against the real Copernicus API), `test_api_acquisition.py`'s new harmonisation-metadata assertions.
* **Not done.** SCL upload in the manual `/upload` API path (only the acquisition path retrieves and masks with real SCL); the **BOA offset (−1000) for processing baseline ≥ 04.00 is still not applied for the manual-upload RGBN path** (documented in `MAMBA_INTEGRATION.md`) — the acquisition path's CDSE provider *does* get this right (Sentinel Hub applies it server-side; see `docs/ACQUISITION.md` §2), but the Planetary Computer fallback still does not, and this discrepancy is disclosed, not silently absorbed. No learned/controlled harmonisation of 20 m/60 m bands into the 10 m grid was built (Requirement 2 explicitly separates this "Option A" learned step from the ordinary bilinear/nearest-neighbour resampling actually implemented — see `Requirements 142.txt` §15); B01/B09's participation in a genuine 2.5 m reconstruction was explicitly flagged in the requirements as needing an experiment, not a default, and no such experiment was run in Step F (out of scope: "do not train a new model").

### 3. Paired LR–HR dataset creation — *implemented* · *evidence-limited* · data *partial*

* **Built.** `frame.data`: dataset roles, adapters (SEN2NEON, OpenSR-Test, SEN2NAIPv2, SEN2VENµS, India profile, synthetic), manifests with digests, geographic scene / region splits and leakage checks, aligned patch extraction, a seeded recorded degradation chain, QC and a DataLoader.
* **Evidence.** `test_data_*.py` (353 tests); `DATA.md`. Real files were read for SEN2NEON (30 of 2,269 tiles), OpenSR-Test (all three subsets) and SEN2NAIPv2 (130 pairs).
* **Limits.** SEN2VENµS is format-only; SEN2NAIPv2 was not downloaded whole (the training decision and its two open questions are in `TRAINING.md` §11); no Indian data. Which data FRAME should *train* on is decided on paper (one SEN2NAIPv2 part first) and not executed.

### 4. Super-resolution model — *partially implemented* · *evidence-limited* · data *available (published weights)*

* **Built.** The published SEN2SR-Lite and SEN2SR-Mamba RGBN ×4 models behind one selector (Mamba isolated in its own environment), the shared tile engine, measured cost and memory. No model was added, trained or modified. A real capability registry (`GET /models`) records verified input/output bands, resolution, scale factor and uncertainty method per model, distinct from what is actually selectable via `/sr/run`.
* **Evidence.** `test_models_*.py` (142, 16 real-GPU), `test_tiling_*.py` (217), `test_api_model_selection.py` (23), `test_api_model_registry.py` (9); `MAMBA_INTEGRATION.md`, `TILING.md`, [`REGISTRY.md`](REGISTRY.md) §1, §1a, §3.
* **Not done.** The comparison the requirement asks for across CNN / ResNet / GAN / Transformer / Swin / diffusion families, the 12-band cascade. **LDSR-S2 (Step G):** verified genuinely executable — real checkpoint (1.13 GB), real forward pass on real Sentinel-2 data producing a genuine 4× RGBN output, real (uncalibrated) diffusion-resampling uncertainty map ([`REGISTRY.md`](REGISTRY.md) §1a) — but not deployable on the project's declared 4 GB dev GPU (measured CUDA OOM at a 70×71 px input) and impractically slow on CPU; registered as `available: false`, not selectable via `/sr/run`. Real-data training (the infrastructure exists, §*training* below).

### 5. Spectral consistency — *implemented* · *evidence-limited* · data *partial*

* **Built.** The upstream Fourier hard constraint used unchanged (both models); `frame.consistency` self-consistency (downsample agreement -- MAE, RMSE, signed bias, Pearson correlation, added Step H; NDVI and NDWI index self-consistency, added Step H; B08/B04 ratio); reference-based SAM, ERGAS, band and index errors (NDVI/NDWI/MNDWI/NDBI/BSI, each with an explicit missing-band skip reason) in `frame.evaluate`.
* **Evidence.** `test_consistency_*.py` (66), `test_evaluate_metrics.py` (45); `EVALUATION.md` §8.4 (NIR carries the error; constrained systems reduce back to the input within 0.0012–0.0019 MAE).
* **Limits.** Self-consistency is not accuracy. The constraint ablation was run for Lite only. Index self-consistency is RGBN-only (NDVI, NDWI): MNDWI/NDBI/BSI need SWIR B11, which no registered SR model produces, so they remain HR-reference-only.

### 6. Geospatial consistency — *implemented* · *evidence-limited* · data *available*

* **Built.** `frame.geospatial` (CRS / transform / bounds preserved and derived, GeoTIFF write and read, a missing CRS is a hard error), the tile engine's geospatial rules, a tile-seam diagnostic, and (Phase 8) a read-back contract test and a smoke check on the real output.
* **Evidence.** `test_geospatial_*.py` (39), `test_tiling_geospatial.py` (26), `test_integration_contract.py` (27), `frame.smoke` records for toy / Lite / Mamba; the Phase 6 scene check on both real models (coverage, orientation, seams, georeferencing all pass).
* **Limits.** What is validated is the **output grid**. Co-registration of *reference* imagery is a measured problem: the bicubic baseline is displaced from SEN2NEON references by a median of 1.0 HR pixel (max 6.5), which is why a gate exists ([`RELIABILITY.md`](RELIABILITY.md) §3).

### 7. Uncertainty / model-stability / error awareness — *implemented* · **evidence-limited, negative-to-mixed** · data *partial*

* **Built.** `frame.uncertainty` (six-view TTA ensemble; per-pixel spread; scalar summary) exposed by the API as a **TTA stability — reconstruction-variation diagnostic**; `frame.reliability` (reference eligibility gate, error targets, within- and across-tile association with texture and added-detail baselines, partial correlation, unit-clustered bootstrap, risk-coverage, high-error detection, calibration).
* **Evidence.** `test_uncertainty_*.py` (69), `test_reliability_*.py` (250); `RELIABILITY.md`. The memo asked "whether the proposed 6-view TTA stability raster is actually scientifically defensible. If it isn't, we'll replace it". Measured: weakly associated with error and no stronger than image texture; near chance for high-error detection on SEN2NEON; **uncalibrated** (spread 16–32× smaller than error).
* **Not done.** Replacement or additional methods (deep ensembles, MC dropout, probabilistic / diffusion uncertainty), calibration, an out-of-distribution evaluation. The diagnostic was neither replaced nor promoted.

### 8. SR accuracy and validation — *implemented* · *evidence-limited* · data *partial*

* **Built.** `frame.validation` (opensr-test metrics, bicubic baseline) and `frame.evaluate` (strict mask, PSNR / SSIM / RMSE / SAM / ERGAS, detail metrics, registration shift sensitivity, aggregation over scene units, paired differences, provenance, role and overlap safety, `python -m frame.evaluate`).
* **Evidence.** `test_evaluate_*.py` (227), `test_validation_*.py` (68); `EVALUATION.md` §8 (30 SEN2NEON tiles; OpenSR-Test `spot`, `spain_crops`, `spain_urban`; bicubic, Lite, Mamba, a synthetic-trained tiny CNN). **Not a ranking.**
* **Not done.** MuS2, SEN2NAIPv2 / SEN2VENµS evaluation, an ESA baseline, perceptual metrics, the raw (non-harmonised) OpenSR `HR`.

### 9. High-resolution reference validation — *partially implemented* · *evidence-limited* · data *partial (public only)*

* **Built.** The reference eligibility gate; role-enforced benchmark adapters; a registration estimator with sub-pixel refinement.
* **Evidence.** `test_reliability_eligibility.py` (34), `test_evaluate_datasets.py`, `test_data_real_sample.py`; `RELIABILITY.md` §3 (12 / 30 SEN2NEON tiles, 6 / 9, 21 / 28, 13 / 20 eligible).
* **Not done.** **Indian references** (Bhoonidhi, Cartosat, Resourcesat, AVIRIS-NG) — none obtained; nothing synthetic was created in their place.

### 10. Downstream analytical utility — *partially implemented (NDVI)* · *evidence-limited, mixed to null* · data *partial*

* **Built.** `frame.downstream` (NDVI on reflectance, fixed regions, a predeclared vegetation threshold, decision agreement, stability association, risk-coverage; the same gate); an API and UI NDVI **demonstration** that says it is one.
* **Evidence.** `test_downstream_*.py` (138), `test_analysis_*.py` (45); `DOWNSTREAM.md`.
* **Not done.** Land-cover classification and every other task in the requirement (sub-pixel, crop, urban, flood, burn-scar, change detection): no region-level labels (`secondary_landcover_task_deferred_no_supported_reference`); NDBI needs SWIR; Indian validation unavailable (`india_downstream_validation_unavailable`).

### 11. Operational / GIS-ready delivery — *partially closed (real COG/STAC/product API, Step I)* · demonstrated once end-to-end · n/a

* **Built.** GeoTIFF output with CRS, transform and a model tag; provenance in every result; tiled inference; a FastAPI backend (synchronous, in-memory) and a React front end with model selection, comparison slider, stability view, NDVI demonstration, metadata and downloads; a documented runbook and a smoke path. **Step I adds:** real Cloud Optimized GeoTIFF writing and validation (`frame.geospatial.write_cog`/`validate_cog`); a real STAC 1.0.0 Item (`frame.product.stac`); a product registry (`GET /products/{job_id}[/stac|/cog|/stability|/analysis|/tiles/{z}/{x}/{y}.png]`); a minimal, real in-process COG→PNG XYZ tile renderer (not titiler/rio-tiler, neither installed); a minimal additive background-job path (`POST /jobs/sr`, `GET /jobs/{async_job_id}`) alongside the unchanged, still-synchronous `/sr/run`.
* **Evidence.** `test_api*.py` (120, 7 real-model), `test_integration_contract.py`, `test_smoke.py` (19), `test_final_readiness.py` (7), the front end's 70 tests; **Step I:** `test_geospatial_cog.py` (10), `test_product_stac.py` (9), `test_product_tiles.py` (6), `test_api_product.py` (6), `test_api_products.py` (10), `test_api_async_jobs.py` (5), `test_gis_compatibility.py` (5); a real, live end-to-end run (`experiments/final_demo/`: real CDSE discovery → retrieval → harmonisation → RGBN adapter → real Lite inference → COG/STAC generation and validation → tile access → download, no secrets in the record); real engineering experiments (`experiments/operational_delivery/`: tile-size, overlap/seam, scaling, CPU-vs-GPU, output-integrity). `frame/api/README.md`, [`DEMO_RUNBOOK.md`](DEMO_RUNBOOK.md), [`OPERATIONAL_DELIVERY.md`](OPERATIONAL_DELIVERY.md).
* **Not done.** Asynchronous jobs remain a minimal local thread pool, not a durable queue; authentication, containers, deployment, load testing, throughput scaling; the official GDAL/STAC validators (not installed); a MapLibre tile-layer frontend view of the new tile endpoint (the existing `InspectViewer` already covers Original/SR/NDVI/Evidence a different way — see `OPERATIONAL_DELIVERY.md` §11). **No production or real-time claim is made.**

### 12. Geographic / Indian generalisation — *protocol implemented, single-scene* · *not a generalisation claim* · data *2 real Indian scenes, no HR reference*

* **Built (Step H).** `frame.generalization`: a reproducible domain-shift protocol (`DomainShiftProtocol` -- training domain, test domain, geographic separation, season/date, land cover, model, metrics, HR-reference status) plus a real runner composing the existing real primitives (`frame.acquisition` search/retrieve/harmonise/RGBN-adapter, the registered model, `frame.uncertainty` TTA stability, `frame.consistency` self-consistency/NDVI/NDWI). Explicitly records that the deployed SEN2SR-Lite/Mamba weights' training domain is **not established** (third-party, undocumented geography) rather than fabricating "trained on SEN2NAIPv2". Still built: the India holdout profile, dataset roles, geographic split and leakage checks, the reference gate.
* **Evidence.** `test_generalization_*.py` (10, incl. one live real-CDSE test); two real executed records over one real Chennai-area Sentinel-2 scene (`S2B_MSIL2A_20240330T045659_...`), Lite and Mamba, at `experiments/generalization/india_chennai_{lite,mamba}/`: self-consistency MAE 0.0019 (Lite) / 0.0037 (Mamba), NDVI/NDWI self-consistency computable, TTA stability 0.00133 / 0.00195 -- **no reference accuracy** (no Indian HR reference exists). SEN2NEON is North America; OpenSR-Test is Spain and SPOT.
* **Not done.** A defensible generalisation claim: this is one scene through two models, not a geographically diverse Indian sample. No Indian HR reference, so no PSNR/SSIM/SAM/ERGAS/downstream-accuracy evidence exists or is claimed. No cross-sensor-domain or temporal generalisation experiment. **An Indian AOI is not evidence of Indian generalisation** (Requirements 142's own words) -- this record is deliberately reported as a single-scene protocol execution, never as validation.

### Cross-cutting: training

Training (`frame.train`, 249 tests) is infrastructure for Requirements 3–4 and is **not** a requirement of its own in the list. It was exercised on synthetic data only; the first real-data run (one SEN2NAIPv2 part) is decided on paper and not executed ([`TRAINING.md`](TRAINING.md) §9, §11, §12).
