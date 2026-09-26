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
| 11 | Operational / GIS-ready delivery | implemented as a demo-grade prototype | n/a (no throughput claim) | n/a |
| 12 | Geographic / Indian generalisation | deferred (controls exist, experiments not run) | not validated | **none for India** |

## Detail

Test counts are from `pytest frame/tests --collect-only` at the end of Phase 8 (2,044 unit tests plus 33 marked `integration`; frontend 70).

### 1. Sentinel-2 data acquisition — engineering *partially implemented* · validation n/a · data *available (user-supplied)*

* **Built.** The product takes a **user-supplied Sentinel-2 L2A GeoTIFF** (`POST /upload`): required bands, CRS and size are validated, larger-than-cap scenes are refused. `POST /aoi/preview` validates an AOI and date window and reports the shapes a run would produce; it **does not fetch imagery**.
  STAC acquisition (`pystac-client` + `cubo`) exists only in `experiments/baseline` and the experiments derived from it.
* **Evidence.** `test_api.py` (16), `test_api_pipeline.py`, `test_api_tiling.py`; `experiments/end_to_end/README.md`, which documents that the catalogue drifted (a re-query now returns 3 items for a window that returned 1), so the Baseline 0 scene is reused, never re-fetched.
* **Not done.** Programmatic acquisition inside the product; Copernicus / Bhoonidhi interoperability; cloud, date and quality filtering at acquisition.

### 2. Multispectral preprocessing and harmonisation — *partially implemented* · n/a · *available*

* **Built.** `frame.preprocessing` for the four-band RGBN 10 m path: bands selected **by name** and reordered to B04, B03, B02, B08; explicit reflectance scaling (never guessed); nodata, non-finite and optional SCL masks kept apart from the tensor; opt-in content validation (no valid pixel, contradicted scale); provenance metadata.
* **Evidence.** `test_pipeline.py` (23), `test_masks.py` (11), `test_reflectance.py` (8), `test_metadata.py` (8); `test_integration_contract.py`.
* **Not done.** The 12-band L2A configuration and B10 handling, 20 / 60 m harmonisation (DSen2-style), SCL upload in the API, the **BOA offset (−1000) for processing baseline ≥ 04.00 is not applied** (documented in `MAMBA_INTEGRATION.md`).

### 3. Paired LR–HR dataset creation — *implemented* · *evidence-limited* · data *partial*

* **Built.** `frame.data`: dataset roles, adapters (SEN2NEON, OpenSR-Test, SEN2NAIPv2, SEN2VENµS, India profile, synthetic), manifests with digests, geographic scene / region splits and leakage checks, aligned patch extraction, a seeded recorded degradation chain, QC and a DataLoader.
* **Evidence.** `test_data_*.py` (353 tests); `DATA.md`. Real files were read for SEN2NEON (30 of 2,269 tiles), OpenSR-Test (all three subsets) and SEN2NAIPv2 (130 pairs).
* **Limits.** SEN2VENµS is format-only; SEN2NAIPv2 was not downloaded whole (the training decision and its two open questions are in `TRAINING.md` §11); no Indian data. Which data FRAME should *train* on is decided on paper (one SEN2NAIPv2 part first) and not executed.

### 4. Super-resolution model — *partially implemented* · *evidence-limited* · data *available (published weights)*

* **Built.** The published SEN2SR-Lite and SEN2SR-Mamba RGBN ×4 models behind one selector (Mamba isolated in its own environment), the shared tile engine, measured cost and memory. No model was added, trained or modified.
* **Evidence.** `test_models_*.py` (142, 16 real-GPU), `test_tiling_*.py` (217), `test_api_model_selection.py` (23); `MAMBA_INTEGRATION.md`, `TILING.md`, [`REGISTRY.md`](REGISTRY.md) §1, §3.
* **Not done.** The comparison the requirement asks for across CNN / ResNet / GAN / Transformer / Swin / diffusion families, LDSR-S2, the 12-band cascade. Real-data training (the infrastructure exists, §*training* below).

### 5. Spectral consistency — *implemented* · *evidence-limited* · data *partial*

* **Built.** The upstream Fourier hard constraint used unchanged (both models); `frame.consistency` self-consistency (downsample agreement, NDVI, B08/B04 ratio); reference-based SAM, ERGAS, band and index errors in `frame.evaluate`.
* **Evidence.** `test_consistency_*.py` (57), `test_evaluate_metrics.py` (44); `EVALUATION.md` §8.4 (NIR carries the error; constrained systems reduce back to the input within 0.0012–0.0019 MAE).
* **Limits.** Self-consistency is not accuracy. The constraint ablation was run for Lite only. RGBN only (no SWIR indices).

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

### 11. Operational / GIS-ready delivery — *implemented (demo-grade)* · n/a · n/a

* **Built.** GeoTIFF output with CRS, transform and a model tag; provenance in every result; tiled inference; a FastAPI backend (synchronous, in-memory) and a React front end with model selection, comparison slider, stability view, NDVI demonstration, metadata and downloads; a documented runbook and a smoke path.
* **Evidence.** `test_api*.py` (120, 7 real-model), `test_integration_contract.py`, `test_smoke.py` (19), `test_final_readiness.py` (7), the front end's 70 tests; `frame/api/README.md`, [`DEMO_RUNBOOK.md`](DEMO_RUNBOOK.md).
* **Not done.** COG or STAC output, asynchronous jobs, authentication, containers, deployment, load testing, throughput scaling. **No production or real-time claim is made.**

### 12. Geographic / Indian generalisation — *deferred* · *not validated* · data *none for India*

* **Built (controls only).** The India holdout profile and record builder, dataset roles, geographic split and leakage checks, the reference gate.
* **Evidence.** None about generalisation: SEN2NEON is North America; OpenSR-Test is Spain and SPOT. The memo's own status line for this requirement is that it is strong "only if the geographic/domain-shift experiments are actually performed". They were not.
* **Not done.** Any geographic, cross-sensor-domain or temporal generalisation experiment, and anything Indian.

### Cross-cutting: training

Training (`frame.train`, 249 tests) is infrastructure for Requirements 3–4 and is **not** a requirement of its own in the list. It was exercised on synthetic data only; the first real-data run (one SEN2NAIPv2 part) is decided on paper and not executed ([`TRAINING.md`](TRAINING.md) §9, §11, §12).
