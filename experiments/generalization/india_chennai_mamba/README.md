# FRAME domain-shift record (Requirements 142 Requirement 12)

This is evidence about ONE real Indian Sentinel-2 scene, not a validated claim of Indian generalisation. Read every number here against the limitations below before repeating it.

* **Scene**: `S2B_MSIL2A_20240330T045659_N0510_R119_T44PMV_20240330T072508` (provider: copernicus-cdse-process-api, cloud cover 2.4%)
* **Model**: `mamba`
* **Test domain**: Real Sentinel-2 L2A over India, scene S2B_MSIL2A_20240330T045659_N0510_R119_T44PMV_20240330T072508, bbox (80.204, 13.006, 80.2168, 13.0188)
* **Training domain**: The deployed SEN2SR-Lite/Mamba weights are third-party (tacofoundation/sen2sr); their original training data and geography are not documented in the artifact manifest FRAME inspected (docs/REGISTRY.md Sec 1) and are not established here. FRAME's own training work (docs/TRAINING.md) never touched these weights -- it trained a separate, small synthetic-data smoke model only. Naming a training domain for the model actually serving /sr/run would be a fabricated claim.
* **Geographic separation**: True -- The test scene is Indian; the reference datasets this project has real evidence from (SEN2NEON: North America; OpenSR-Test: Spain/SPOT) are geographically disjoint from it. This establishes geographic separation of the EVIDENCE datasets, not of the deployed model's unknown training set (see training_domain_known).
* **Temporal**: acquisition_datetime='2024-03-30T04:56:59.024000Z'
* **Land cover**: not established -- no land-cover label exists for this scene (see frame.data.adapters.india_holdout)
* **HR reference status**: unavailable -- No Indian HR reference (Cartosat, Resourcesat, AVIRIS-NG or otherwise) has been obtained or validated by this project (docs/TRACEABILITY.md Sec 9). Reference-based accuracy (PSNR/SSIM/RMSE/SAM/ERGAS) is therefore not computed for this scene -- not omitted by oversight, structurally unavailable. Substituting a non-Indian HR reference here would misrepresent it as Indian-reference evidence, which Requirements 142 explicitly forbids.

## What was measured (no HR reference exists for this scene)

* Self-consistency (SR downsampled to 10 m vs. the original Sentinel-2 observation): MAE 0.0037, RMSE 0.0052, bias -0.0000, correlation 0.9979 (n=78952).
* NDVI self-consistency: COMPUTABLE, mean |Δ| 0.0184.
* NDWI self-consistency: COMPUTABLE, mean |Δ| 0.0149.
* TTA stability -- reconstruction-variation diagnostic (uncalibrated; see docs/RELIABILITY.md): scalar summary 0.001946 (Mean per-pixel standard deviation across the TTA ensemble, averaged over bands. A relative model-stability proxy, not a calibrated confidence value.).
* Valid-pixel fraction: 1.000.
* Bands excluded from the SR model (not RGBN, harmonised context only): ['B05', 'B06', 'B07', 'B8A', 'B11', 'B12', 'B01', 'B09'].

## What this record is NOT

* NOT reference accuracy: no PSNR/SSIM/RMSE/SAM/ERGAS against an independent HR reference -- none exists for this scene (Requirements 142 Requirement 9; docs/TRACEABILITY.md Sec 9).
* NOT a downstream-task result: no Indian land-cover, crop, urban, flood or burn-scar label exists here.
* NOT evidence the deployed model was trained for, or excludes, Indian imagery: its training domain is not established (see `training_domain` above).
* NOT a multi-scene generalisation claim: this is one scene. A defensible generalisation claim needs many geographically diverse Indian scenes, which this record does not attempt.

Inference: cuda, seed 0, 42.79 s.
