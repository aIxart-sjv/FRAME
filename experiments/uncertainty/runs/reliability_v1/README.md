# Reliability validation `reliability_v1`

Status: **completed**. Question: is the TTA model-stability signal informative about reconstruction error, judged **only on evidence whose reference is registered to the prediction grid**? The stability is a **relative model-stability proxy, not a calibrated uncertainty, a confidence or a probability of error**: whether it is informative is what is tested, and this README reports it faithfully whatever the answer. Every association is descriptive and non-causal; nothing is pooled across datasets; nothing is ranked.

Code: git `eb3769559525ea81e902422253f8950c36b075a5` (working tree dirty: True); metrics `frame-eval-metrics/1`; tiling {'tile_size': 128, 'overlap': 32, 'stride': 96, 'scale': 4, 'padding_mode': 'reflect', 'blend_mode': 'linear'}.

## What is measured

* **Stability** (the deployed quantity): frame.uncertainty.run_tta_ensemble over frame.tiling.TiledModel: the model is run on each geometric view of the whole scene and every prediction is de-transformed; stability = per-pixel population standard deviation over the members, averaged over the four bands; the product evaluated is the ensemble mean. Members: identity, hflip, vflip, rot90, rot180, rot270 (6); seed 42.
* **Error targets** (against the independent HR reference, strict valid mask only): per-pixel band-mean absolute reflectance error (E1); per-pixel spectral angle (E2); at cell level (10 m = 4 HR px, 40 m = 16 HR px) their means; per tile RMSE, MAE, SAM, ERGAS; per scene unit their means. The evaluated product is the TTA ensemble mean; the single pass (identity view) is recorded separately in `metrics.jsonl`.
* **Trivial predictors of error, reported next to the stability**: image texture (Sobel gradient of the bicubic input) and the amount the model moved away from bicubic. A stability that only re-encodes them adds nothing, so a partial correlation controlling for both is reported.
* **Units of analysis**: within-tile correlations are summarised over scene units (NEON acquisition / source orthophoto); intervals resample whole units; fewer than 5 units is descriptive only.

## The reference eligibility gate

The displacement of the bicubic baseline against the reference is estimated (`bicubic_baseline_cross_correlation`, system-neutral, before any model runs). A tile is pixel-level **eligible** only if, after at most one whole-pixel translation of 4 HR px (an integer crop of both grids, no resampling, recorded per tile), the residual is within **0.5 HR px** and the four quadrants agree within 1.0 HR px, with at least 50% strict-valid pixels. Between one and 2 tolerances the evidence is `uncertain`; beyond it `not_eligible`. Excluded tiles receive **no** error number and are listed below with their reason (`excluded_from_uncertainty_error_analysis`).

## Evidence

| dataset | class | records | invalid | unreadable | eligible tiles | uncertain | not eligible | exclusion reasons | eligible scene units | all scene units |
|---|---|---|---|---|---|---|---|---|---|---|
| `sen2neon_random30` | real_cross_sensor | 30 | 0 | 0 | 12 | 15 | 3 | reference_alignment_invalid: 3, reference_alignment_uncertain: 15 | 11 | 28 |
| `opensr_spot` | real_cross_sensor | 9 | 0 | 0 | 6 | 3 | 0 | reference_alignment_uncertain: 3 | 6 | 9 |
| `opensr_spain_crops` | real_cross_sensor | 28 | 0 | 0 | 21 | 7 | 0 | reference_alignment_uncertain: 7 | 5 | 5 |
| `opensr_spain_urban` | real_cross_sensor | 20 | 0 | 0 | 13 | 7 | 0 | reference_alignment_uncertain: 7 | 4 | 4 |


Registration of every tile (raw displacement of the bicubic baseline, the correction applied, the residual) is in `summary.json` (`datasets.<name>.tiles`) and in each row of `metrics.jsonl`.

## Dataset `sen2neon_random30`

### `sen2sr_lite`

12 eligible tiles from 11 scene units.

**Within-tile Spearman correlation between stability and error**, mean over scene units [95% interval over units]; the baselines are correlated with the same error; `partial` controls the stability for both baselines; `share > 0` is the share of tiles with a positive correlation.

| scale | stability vs abs error | stability vs SAM | texture vs abs error | added detail vs abs error | partial (stability given both) | share > 0 | tiles | units |
|---|---|---|---|---|---|---|---|---|
| pixel (2.5 m) | 0.094 [0.046, 0.143] | 0.123 [0.067, 0.176] | 0.050 [0.010, 0.086] | 0.091 [0.044, 0.139] | 0.063 [0.031, 0.098] | 0.83 | 12 | 11 |
| cell 10 m | 0.161 [0.089, 0.233] | 0.211 [0.135, 0.292] | 0.132 [0.069, 0.188] | 0.121 [0.060, 0.176] | 0.093 [0.047, 0.140] | 0.83 | 12 | 11 |
| cell 40 m | 0.260 [0.147, 0.381] | 0.317 [0.215, 0.433] | 0.269 [0.159, 0.382] | 0.213 [0.117, 0.316] | 0.110 [0.052, 0.169] | 0.92 | 12 | 11 |

**Pooled cell correlation** (a seeded subsample of each tile's cells; the interval resamples whole scene units):

| scale | stability vs abs error | stability vs SAM | texture vs abs error | added detail vs abs error | partial |
|---|---|---|---|---|---|
| cell 10 m | 0.106 [-0.041, 0.275] | 0.102 [0.003, 0.262] | 0.107 [-0.040, 0.260] | 0.120 [-0.059, 0.305] | 0.011 [-0.058, 0.123] |
| cell 40 m | 0.177 [0.003, 0.381] | 0.167 [0.050, 0.362] | 0.232 [0.046, 0.426] | 0.216 [-0.019, 0.444] | -0.057 [-0.196, 0.153] |

**Sensitivity to registration** (why misregistered evidence is excluded): the pixel-level correlation of the same eligible tiles with the reference displaced by known whole pixels:

| reference displaced by (HR px) | 0 | 1 | 2 | 4 |
|---|---|---|---|---|
| mean over units | 0.094 [0.046, 0.143] | 0.108 [0.060, 0.156] | 0.134 [0.083, 0.184] | 0.172 [0.115, 0.231] |

**Tile and scene reliability**: Spearman correlation across tiles (interval resamples scene units) and across scene units (tiles of a unit averaged first).

| error target | tile: stability | tiles | scene: stability | units | tile: texture | tile: added detail | tile: partial (stability given both) |
|---|---|---|---|---|---|---|---|
| rmse | 0.434 [-0.077, 0.831] | 12 | 0.373 [-0.336, 0.846] | 11 | 0.608 [0.197, 0.840] | 0.483 [-0.051, 0.826] | -0.082 [-0.588, 0.743] |
| mae | 0.476 [-0.103, 0.888] | 12 | 0.445 [-0.336, 0.907] | 11 | 0.490 [-0.103, 0.821] | 0.308 [-0.344, 0.811] | 0.157 [-0.322, 0.841] |
| sam_degrees | 0.441 [-0.022, 0.942] | 12 | 0.509 [-0.242, 0.963] | 11 | 0.175 [-0.402, 0.638] | -0.007 [-0.592, 0.544] | 0.491 [-0.076, 0.940] |
| ergas | 0.552 [0.237, 0.970] | 12 | 0.664 [0.088, 0.970] | 11 | 0.601 [0.239, 0.849] | 0.462 [0.044, 0.897] | 0.164 [-0.463, 0.967] |

**Risk-coverage** (most unstable removed first; risk = error of what remains; *not* calibrated selective prediction):

`tile_rmse` (12 items, 11 scene units). Selective efficiency (1 = oracle, 0 = random, < 0 worse than random): stability 0.425 [-0.333, 0.929], texture 0.495, added detail 0.084.

| coverage | risk (stability order) | oracle | texture order | added-detail order |
|---|---|---|---|---|
| 100% | 0.02430 | 0.02430 | 0.02430 | 0.02430 |
| 90% | 0.02513 | 0.01985 | 0.02424 | 0.02351 |
| 80% | 0.02049 | 0.01825 | 0.02337 | 0.02438 |
| 70% | 0.02029 | 0.01588 | 0.01793 | 0.02521 |
| 60% | 0.01996 | 0.01426 | 0.01826 | 0.02504 |
| 50% | 0.01725 | 0.01237 | 0.01653 | 0.01653 |

Relative risk reduction vs no removal: at 90% coverage -0.034 [-0.038, 0.210]; at 80% coverage 0.157 [-0.091, 0.299]; at 70% coverage 0.165 [-0.087, 0.386]; at 60% coverage 0.178 [-0.072, 0.426]; at 50% coverage 0.290 [-0.057, 0.510]

`tile_sam` (12 items, 11 scene units). Selective efficiency (1 = oracle, 0 = random, < 0 worse than random): stability 0.599 [-0.093, 1.000], texture 0.252, added detail -0.273.

| coverage | risk (stability order) | oracle | texture order | added-detail order |
|---|---|---|---|---|
| 100% | 2.62001 | 2.62001 | 2.62001 | 2.62001 |
| 90% | 2.75597 | 1.95244 | 2.59197 | 2.77112 |
| 80% | 2.03524 | 1.78327 | 2.75539 | 2.93579 |
| 70% | 1.72247 | 1.44153 | 2.05828 | 3.08822 |
| 60% | 1.56008 | 1.23901 | 2.10607 | 3.24557 |
| 50% | 1.66047 | 1.11438 | 2.12596 | 2.12596 |

Relative risk reduction vs no removal: at 90% coverage -0.052 [-0.063, 0.282]; at 80% coverage 0.223 [-0.129, 0.433]; at 70% coverage 0.343 [-0.024, 0.537]; at 60% coverage 0.405 [-0.029, 0.587]; at 50% coverage 0.366 [-0.063, 0.630]

`cell_4_abs_error` (98304 items, 11 scene units). Selective efficiency (1 = oracle, 0 = random, < 0 worse than random): stability 0.144 [0.056, 0.313], texture 0.123, added detail 0.110.

| coverage | risk (stability order) | oracle | texture order | added-detail order |
|---|---|---|---|---|
| 100% | 0.01335 | 0.01335 | 0.01335 | 0.01335 |
| 90% | 0.01267 | 0.00982 | 0.01263 | 0.01279 |
| 80% | 0.01246 | 0.00829 | 0.01252 | 0.01264 |
| 70% | 0.01239 | 0.00718 | 0.01254 | 0.01260 |
| 60% | 0.01246 | 0.00624 | 0.01269 | 0.01272 |
| 50% | 0.01274 | 0.00540 | 0.01298 | 0.01295 |

Relative risk reduction vs no removal: at 90% coverage 0.051 [0.036, 0.071]; at 80% coverage 0.067 [0.040, 0.105]; at 70% coverage 0.071 [0.031, 0.126]; at 60% coverage 0.066 [0.011, 0.147]; at 50% coverage 0.045 [-0.034, 0.164]

**High-error detection** (10 m cells):

Threshold 0.02796 = the 90% quantile of error over **same evidence (no held-out split): descriptive only**; prevalence of high error in the evaluated cells 0.100 (98304 cells; test units: 11). **Descriptive only**: 11 scene unit(s): fewer than the 12 needed for a development/test split.

| score | AUROC | AUPRC (chance = prevalence) | precision, top 10% flagged | lift | precision, top 20% | lift |
|---|---|---|---|---|---|---|
| stability (ensemble spread) | 0.482 | 0.153 | 0.189 | 1.89 | 0.145 | 1.45 |
| texture baseline | 0.472 | 0.156 | 0.192 | 1.92 | 0.136 | 1.36 |
| added-detail baseline | 0.454 | 0.137 | 0.155 | 1.55 | 0.121 | 1.21 |

**Calibration**: verdict `uncalibrated_stability_evidence`. The raw TTA spread is not a calibrated uncertainty: it is far smaller than the error and covers a small fraction of it. Whether a monotone map from stability to expected error, fitted on development units, is calibrated on held-out test units is reported below; even where it is, it is calibrated for this dataset and error scale only and is not a probability of error, a confidence or a conformal guarantee.

* Scale: the median error is 18.2 times the median spread (median over tiles).
* Coverage of `prediction +/- k * spread` against a Gaussian's nominal level: k = 1: 0.038 (nominal 0.683); k = 2: 0.075 (nominal 0.954).
* Recalibration: not_assessable (11 scene unit(s): fewer than the 12 needed for a development/test split; recalibration needs held-out units).

**Relationship with the Phase 5 detail categories** (Spearman of stability with each category; not semantic labels; instability and added detail both grow with texture): cell-level, mean over units [interval]:

| scale | unsupported detail | omission | supported synthesis |
|---|---|---|---|
| cell 10 m | 0.503 [0.426, 0.577] | -0.150 [-0.300, -0.007] | 0.367 [0.298, 0.443] |
| cell 40 m | 0.766 [0.717, 0.813] | -0.116 [-0.314, 0.082] | 0.650 [0.588, 0.714] |

Tile-level (tile mean stability vs tile share of each category):

| category | Spearman across tiles |
|---|---|
| unsupported_detail | 0.720 [-0.014, 0.947] |
| omission | 0.448 [-0.116, 1.000] |
| supported_synthesis | 0.420 [-0.390, 0.718] |

**TTA cost**: 6 members; single pass 0.096 s, TTA 0.828 s (x8.60) per scene, median over 11 tiles (the first tile, 0.92 s, carries warm-up and is excluded).

### `sen2sr_mamba`

12 eligible tiles from 11 scene units.

**Within-tile Spearman correlation between stability and error**, mean over scene units [95% interval over units]; the baselines are correlated with the same error; `partial` controls the stability for both baselines; `share > 0` is the share of tiles with a positive correlation.

| scale | stability vs abs error | stability vs SAM | texture vs abs error | added detail vs abs error | partial (stability given both) | share > 0 | tiles | units |
|---|---|---|---|---|---|---|---|---|
| pixel (2.5 m) | 0.078 [0.030, 0.124] | 0.148 [0.082, 0.212] | 0.051 [0.007, 0.090] | 0.091 [0.037, 0.146] | 0.037 [-0.002, 0.070] | 0.83 | 12 | 11 |
| cell 10 m | 0.133 [0.067, 0.199] | 0.225 [0.136, 0.317] | 0.126 [0.058, 0.188] | 0.128 [0.058, 0.196] | 0.052 [0.003, 0.096] | 0.83 | 12 | 11 |
| cell 40 m | 0.213 [0.105, 0.326] | 0.307 [0.190, 0.432] | 0.256 [0.139, 0.367] | 0.218 [0.110, 0.327] | 0.041 [-0.019, 0.101] | 0.83 | 12 | 11 |

**Pooled cell correlation** (a seeded subsample of each tile's cells; the interval resamples whole scene units):

| scale | stability vs abs error | stability vs SAM | texture vs abs error | added detail vs abs error | partial |
|---|---|---|---|---|---|
| cell 10 m | 0.060 [-0.071, 0.231] | 0.036 [-0.051, 0.214] | 0.103 [-0.045, 0.249] | 0.111 [-0.079, 0.315] | -0.049 [-0.129, 0.092] |
| cell 40 m | 0.119 [-0.048, 0.314] | 0.077 [-0.044, 0.312] | 0.225 [0.043, 0.412] | 0.216 [-0.022, 0.441] | -0.122 [-0.222, 0.056] |

**Sensitivity to registration** (why misregistered evidence is excluded): the pixel-level correlation of the same eligible tiles with the reference displaced by known whole pixels:

| reference displaced by (HR px) | 0 | 1 | 2 | 4 |
|---|---|---|---|---|
| mean over units | 0.078 [0.030, 0.124] | 0.092 [0.041, 0.138] | 0.119 [0.064, 0.173] | 0.160 [0.095, 0.228] |

**Tile and scene reliability**: Spearman correlation across tiles (interval resamples scene units) and across scene units (tiles of a unit averaged first).

| error target | tile: stability | tiles | scene: stability | units | tile: texture | tile: added detail | tile: partial (stability given both) |
|---|---|---|---|---|---|---|---|
| rmse | 0.343 [-0.243, 0.846] | 12 | 0.145 [-0.570, 0.846] | 11 | 0.545 [0.083, 0.834] | 0.434 [-0.109, 0.831] | -0.037 [-0.879, 0.885] |
| mae | 0.287 [-0.355, 0.778] | 12 | 0.173 [-0.644, 0.775] | 11 | 0.490 [-0.103, 0.821] | 0.406 [-0.200, 0.828] | -0.153 [-0.920, 0.842] |
| sam_degrees | 0.084 [-0.531, 0.641] | 12 | -0.091 [-0.791, 0.587] | 11 | 0.119 [-0.529, 0.586] | 0.049 [-0.598, 0.615] | 0.162 [-0.854, 0.897] |
| ergas | 0.385 [-0.064, 0.925] | 12 | 0.282 [-0.458, 0.925] | 11 | 0.573 [0.162, 0.848] | 0.476 [0.060, 0.926] | -0.037 [-0.817, 0.858] |

**Risk-coverage** (most unstable removed first; risk = error of what remains; *not* calibrated selective prediction):

`tile_rmse` (12 items, 11 scene units). Selective efficiency (1 = oracle, 0 = random, < 0 worse than random): stability 0.340 [-0.283, 0.986], texture 0.491, added detail 0.219.

| coverage | risk (stability order) | oracle | texture order | added-detail order |
|---|---|---|---|---|
| 100% | 0.02427 | 0.02427 | 0.02427 | 0.02427 |
| 90% | 0.02508 | 0.01979 | 0.02425 | 0.02508 |
| 80% | 0.02585 | 0.01819 | 0.02338 | 0.02433 |
| 70% | 0.01912 | 0.01587 | 0.01790 | 0.02520 |
| 60% | 0.01856 | 0.01423 | 0.01823 | 0.01823 |
| 50% | 0.01739 | 0.01228 | 0.01649 | 0.01739 |

Relative risk reduction vs no removal: at 90% coverage -0.034 [-0.038, 0.159]; at 80% coverage -0.065 [-0.103, 0.253]; at 70% coverage 0.212 [-0.137, 0.357]; at 60% coverage 0.235 [-0.086, 0.427]; at 50% coverage 0.283 [-0.023, 0.516]

`tile_sam` (12 items, 11 scene units). Selective efficiency (1 = oracle, 0 = random, < 0 worse than random): stability 0.215 [-0.592, 0.872], texture 0.262, added detail -0.001.

| coverage | risk (stability order) | oracle | texture order | added-detail order |
|---|---|---|---|---|
| 100% | 2.57821 | 2.57821 | 2.57821 | 2.57821 |
| 90% | 2.70227 | 1.90908 | 2.57169 | 2.70227 |
| 80% | 2.80030 | 1.76305 | 2.72159 | 2.86522 |
| 70% | 2.12395 | 1.43571 | 2.00796 | 3.03505 |
| 60% | 2.01627 | 1.26226 | 2.04882 | 2.04882 |
| 50% | 1.91067 | 1.14668 | 2.06433 | 1.91067 |

Relative risk reduction vs no removal: at 90% coverage -0.048 [-0.061, 0.242]; at 80% coverage -0.086 [-0.157, 0.341]; at 70% coverage 0.176 [-0.199, 0.470]; at 60% coverage 0.218 [-0.198, 0.524]; at 50% coverage 0.259 [-0.188, 0.574]

`cell_4_abs_error` (98304 items, 11 scene units). Selective efficiency (1 = oracle, 0 = random, < 0 worse than random): stability 0.091 [-0.018, 0.264], texture 0.117, added detail 0.105.

| coverage | risk (stability order) | oracle | texture order | added-detail order |
|---|---|---|---|---|
| 100% | 0.01321 | 0.01321 | 0.01321 | 0.01321 |
| 90% | 0.01277 | 0.00969 | 0.01251 | 0.01259 |
| 80% | 0.01262 | 0.00817 | 0.01241 | 0.01246 |
| 70% | 0.01255 | 0.00707 | 0.01244 | 0.01249 |
| 60% | 0.01267 | 0.00615 | 0.01260 | 0.01270 |
| 50% | 0.01297 | 0.00533 | 0.01289 | 0.01298 |

Relative risk reduction vs no removal: at 90% coverage 0.033 [0.015, 0.062]; at 80% coverage 0.044 [0.008, 0.089]; at 70% coverage 0.050 [-0.004, 0.106]; at 60% coverage 0.040 [-0.027, 0.121]; at 50% coverage 0.018 [-0.069, 0.138]

**High-error detection** (10 m cells):

Threshold 0.02759 = the 90% quantile of error over **same evidence (no held-out split): descriptive only**; prevalence of high error in the evaluated cells 0.100 (98304 cells; test units: 11). **Descriptive only**: 11 scene unit(s): fewer than the 12 needed for a development/test split.

| score | AUROC | AUPRC (chance = prevalence) | precision, top 10% flagged | lift | precision, top 20% | lift |
|---|---|---|---|---|---|---|
| stability (ensemble spread) | 0.456 | 0.126 | 0.154 | 1.54 | 0.123 | 1.23 |
| texture baseline | 0.468 | 0.156 | 0.190 | 1.90 | 0.133 | 1.33 |
| added-detail baseline | 0.444 | 0.147 | 0.170 | 1.70 | 0.125 | 1.25 |

**Calibration**: verdict `uncalibrated_stability_evidence`. The raw TTA spread is not a calibrated uncertainty: it is far smaller than the error and covers a small fraction of it. Whether a monotone map from stability to expected error, fitted on development units, is calibrated on held-out test units is reported below; even where it is, it is calibrated for this dataset and error scale only and is not a probability of error, a confidence or a conformal guarantee.

* Scale: the median error is 31.9 times the median spread (median over tiles).
* Coverage of `prediction +/- k * spread` against a Gaussian's nominal level: k = 1: 0.041 (nominal 0.683); k = 2: 0.080 (nominal 0.954).
* Recalibration: not_assessable (11 scene unit(s): fewer than the 12 needed for a development/test split; recalibration needs held-out units).

**Relationship with the Phase 5 detail categories** (Spearman of stability with each category; not semantic labels; instability and added detail both grow with texture): cell-level, mean over units [interval]:

| scale | unsupported detail | omission | supported synthesis |
|---|---|---|---|
| cell 10 m | 0.568 [0.495, 0.637] | -0.213 [-0.354, -0.087] | 0.463 [0.387, 0.543] |
| cell 40 m | 0.758 [0.680, 0.828] | -0.180 [-0.357, -0.013] | 0.674 [0.581, 0.762] |

Tile-level (tile mean stability vs tile share of each category):

| category | Spearman across tiles |
|---|---|
| unsupported_detail | 0.965 [0.735, 1.000] |
| omission | 0.007 [-0.625, 0.576] |
| supported_synthesis | 0.636 [0.263, 0.894] |

**TTA cost**: 6 members; single pass 14.788 s, TTA 88.858 s (x6.01) per scene, median over 11 tiles (the first tile, 98.24 s, carries warm-up and is excluded).

### Comparison of the systems on the same eligible tiles

Descriptive comparison of the same protocol on the tiles that are eligible for BOTH systems, paired over scene units (A - B). It is not a ranking: a different spread magnitude or a different association is a property of each model's stability signal, not a verdict on which model is better.

12 shared tiles, 11 scene units; tiles eligible for one system only: {'sen2sr_lite': 0, 'sen2sr_mamba': 0}.

**sen2sr_lite - sen2sr_mamba** (paired over scene units):

| metric | units | mean difference | 95% interval |
|---|---|---|---|
| `cells.4.spearman_abs_error.value` | 11 | 0.0281 | [0.004929, 0.0514] |
| `pixel.spearman_abs_error.value` | 11 | 0.0156 | [-0.002671, 0.03479] |
| `stability.mean` | 11 | -0.0002 | [-0.000252, -0.0001236] |
| `targets.product.rmse` | 11 | 0.0001 | [-9.748e-05, 0.0002498] |
| `targets.product.sam_degrees` | 11 | 0.0496 | [-0.02052, 0.1391] |
| `tta.total_seconds` | 11 | -88.9548 | [-91.03, -87.1] |
| `tta.single_pass_seconds` | 11 | -14.8003 | [-15.14, -14.49] |

## Dataset `opensr_spot`

### `sen2sr_lite`

6 eligible tiles from 6 scene units.

**Within-tile Spearman correlation between stability and error**, mean over scene units [95% interval over units]; the baselines are correlated with the same error; `partial` controls the stability for both baselines; `share > 0` is the share of tiles with a positive correlation.

| scale | stability vs abs error | stability vs SAM | texture vs abs error | added detail vs abs error | partial (stability given both) | share > 0 | tiles | units |
|---|---|---|---|---|---|---|---|---|
| pixel (2.5 m) | 0.167 [0.106, 0.235] | 0.170 [0.067, 0.281] | 0.129 [0.073, 0.194] | 0.144 [0.092, 0.197] | 0.087 [0.063, 0.116] | 1.00 | 6 | 6 |
| cell 10 m | 0.341 [0.256, 0.429] | 0.292 [0.159, 0.431] | 0.263 [0.171, 0.364] | 0.318 [0.226, 0.411] | 0.141 [0.108, 0.177] | 1.00 | 6 | 6 |
| cell 40 m | 0.510 [0.397, 0.619] | 0.415 [0.243, 0.599] | 0.422 [0.310, 0.542] | 0.507 [0.383, 0.625] | 0.131 [0.053, 0.195] | 1.00 | 6 | 6 |

**Pooled cell correlation** (a seeded subsample of each tile's cells; the interval resamples whole scene units):

| scale | stability vs abs error | stability vs SAM | texture vs abs error | added detail vs abs error | partial |
|---|---|---|---|---|---|
| cell 10 m | 0.733 [0.444, 0.803] | 0.721 [0.364, 0.798] | 0.674 [0.364, 0.771] | 0.697 [0.387, 0.788] | 0.294 [0.163, 0.327] |
| cell 40 m | 0.843 [0.583, 0.897] | 0.806 [0.434, 0.879] | 0.825 [0.559, 0.883] | 0.837 [0.556, 0.906] | 0.251 [0.060, 0.349] |

**Sensitivity to registration** (why misregistered evidence is excluded): the pixel-level correlation of the same eligible tiles with the reference displaced by known whole pixels:

| reference displaced by (HR px) | 0 | 1 | 2 | 4 |
|---|---|---|---|---|
| mean over units | 0.167 [0.106, 0.235] | 0.180 [0.116, 0.250] | 0.208 [0.140, 0.282] | 0.255 [0.183, 0.333] |

**Tile and scene reliability**: Spearman correlation across tiles (interval resamples scene units) and across scene units (tiles of a unit averaged first).

| error target | tile: stability | tiles | scene: stability | units | tile: texture | tile: added detail | tile: partial (stability given both) |
|---|---|---|---|---|---|---|---|
| rmse | 1.000 (descriptive only) | 6 | 1.000 (descriptive only) | 6 | 1.000 (descriptive only) | 1.000 (descriptive only) | 1.000 (descriptive only) |
| mae | 1.000 (descriptive only) | 6 | 1.000 (descriptive only) | 6 | 1.000 (descriptive only) | 1.000 (descriptive only) | 1.000 (descriptive only) |
| sam_degrees | 0.943 (descriptive only) | 6 | 0.943 (descriptive only) | 6 | 0.943 (descriptive only) | 0.943 (descriptive only) | -0.094 (descriptive only) |
| ergas | 0.943 (descriptive only) | 6 | 0.943 (descriptive only) | 6 | 0.943 (descriptive only) | 0.943 (descriptive only) | -0.094 (descriptive only) |

**Risk-coverage** (most unstable removed first; risk = error of what remains; *not* calibrated selective prediction):

`tile_rmse` (6 items, 6 scene units). Selective efficiency (1 = oracle, 0 = random, < 0 worse than random): stability 1.000 (descriptive only), texture 1.000, added detail 1.000.

| coverage | risk (stability order) | oracle | texture order | added-detail order |
|---|---|---|---|---|
| 100% | 0.02792 | 0.02792 | 0.02792 | 0.02792 |
| 90% | 0.02256 | 0.02256 | 0.02256 | 0.02256 |
| 80% | 0.02256 | 0.02256 | 0.02256 | 0.02256 |
| 70% | 0.01522 | 0.01522 | 0.01522 | 0.01522 |
| 60% | 0.01522 | 0.01522 | 0.01522 | 0.01522 |
| 50% | 0.01163 | 0.01163 | 0.01163 | 0.01163 |

Relative risk reduction vs no removal: at 90% coverage 0.192 (descriptive only); at 80% coverage 0.192 (descriptive only); at 70% coverage 0.455 (descriptive only); at 60% coverage 0.455 (descriptive only); at 50% coverage 0.584 (descriptive only)

`tile_sam` (6 items, 6 scene units). Selective efficiency (1 = oracle, 0 = random, < 0 worse than random): stability 0.989 (descriptive only), texture 0.989, added detail 0.989.

| coverage | risk (stability order) | oracle | texture order | added-detail order |
|---|---|---|---|---|
| 100% | 1.72719 | 1.72719 | 1.72719 | 1.72719 |
| 90% | 1.27155 | 1.27155 | 1.27155 | 1.27155 |
| 80% | 1.27155 | 1.27155 | 1.27155 | 1.27155 |
| 70% | 0.77380 | 0.77380 | 0.77380 | 0.77380 |
| 60% | 0.77380 | 0.77380 | 0.77380 | 0.77380 |
| 50% | 0.74235 | 0.67046 | 0.74235 | 0.74235 |

Relative risk reduction vs no removal: at 90% coverage 0.264 (descriptive only); at 80% coverage 0.264 (descriptive only); at 70% coverage 0.552 (descriptive only); at 60% coverage 0.552 (descriptive only); at 50% coverage 0.570 (descriptive only)

`cell_4_abs_error` (49152 items, 6 scene units). Selective efficiency (1 = oracle, 0 = random, < 0 worse than random): stability 0.772 [0.460, 0.835], texture 0.704, added detail 0.741.

| coverage | risk (stability order) | oracle | texture order | added-detail order |
|---|---|---|---|---|
| 100% | 0.01731 | 0.01731 | 0.01731 | 0.01731 |
| 90% | 0.01456 | 0.01338 | 0.01473 | 0.01459 |
| 80% | 0.01260 | 0.01115 | 0.01299 | 0.01278 |
| 70% | 0.01107 | 0.00942 | 0.01169 | 0.01138 |
| 60% | 0.01005 | 0.00809 | 0.01073 | 0.01036 |
| 50% | 0.00928 | 0.00704 | 0.00998 | 0.00960 |

Relative risk reduction vs no removal: at 90% coverage 0.158 [0.064, 0.221]; at 80% coverage 0.272 [0.110, 0.341]; at 70% coverage 0.361 [0.148, 0.441]; at 60% coverage 0.419 [0.184, 0.505]; at 50% coverage 0.464 [0.217, 0.567]

**High-error detection** (10 m cells):

Threshold 0.03657 = the 90% quantile of error over **same evidence (no held-out split): descriptive only**; prevalence of high error in the evaluated cells 0.100 (49152 cells; test units: 6). **Descriptive only**: 6 scene unit(s): fewer than the 12 needed for a development/test split.

| score | AUROC | AUPRC (chance = prevalence) | precision, top 10% flagged | lift | precision, top 20% | lift |
|---|---|---|---|---|---|---|
| stability (ensemble spread) | 0.892 | 0.506 | 0.485 | 4.85 | 0.370 | 3.70 |
| texture baseline | 0.862 | 0.471 | 0.455 | 4.55 | 0.346 | 3.46 |
| added-detail baseline | 0.880 | 0.496 | 0.487 | 4.87 | 0.358 | 3.58 |

**Calibration**: verdict `uncalibrated_stability_evidence`. The raw TTA spread is not a calibrated uncertainty: it is far smaller than the error and covers a small fraction of it. Whether a monotone map from stability to expected error, fitted on development units, is calibrated on held-out test units is reported below; even where it is, it is calibrated for this dataset and error scale only and is not a probability of error, a confidence or a conformal guarantee.

* Scale: the median error is 18.2 times the median spread (median over tiles).
* Coverage of `prediction +/- k * spread` against a Gaussian's nominal level: k = 1: 0.036 (nominal 0.683); k = 2: 0.071 (nominal 0.954).
* Recalibration: not_assessable (6 scene unit(s): fewer than the 12 needed for a development/test split; recalibration needs held-out units).

**Relationship with the Phase 5 detail categories** (Spearman of stability with each category; not semantic labels; instability and added detail both grow with texture): cell-level, mean over units [interval]:

| scale | unsupported detail | omission | supported synthesis |
|---|---|---|---|
| cell 10 m | 0.491 [0.389, 0.592] | -0.279 [-0.591, 0.004] | 0.540 [0.424, 0.641] |
| cell 40 m | 0.761 [0.696, 0.826] | -0.296 [-0.691, 0.102] | 0.783 [0.709, 0.851] |

Tile-level (tile mean stability vs tile share of each category):

| category | Spearman across tiles |
|---|---|
| unsupported_detail | 1.000 (descriptive only) |
| omission | 0.029 (descriptive only) |
| supported_synthesis | 1.000 (descriptive only) |

**TTA cost**: 6 members; single pass 0.013 s, TTA 0.189 s (x14.28) per scene, median over 5 tiles (the first tile, 0.18 s, carries warm-up and is excluded).

### `sen2sr_mamba`

6 eligible tiles from 6 scene units.

**Within-tile Spearman correlation between stability and error**, mean over scene units [95% interval over units]; the baselines are correlated with the same error; `partial` controls the stability for both baselines; `share > 0` is the share of tiles with a positive correlation.

| scale | stability vs abs error | stability vs SAM | texture vs abs error | added detail vs abs error | partial (stability given both) | share > 0 | tiles | units |
|---|---|---|---|---|---|---|---|---|
| pixel (2.5 m) | 0.198 [0.134, 0.267] | 0.244 [0.133, 0.365] | 0.133 [0.072, 0.203] | 0.150 [0.095, 0.205] | 0.121 [0.091, 0.151] | 1.00 | 6 | 6 |
| cell 10 m | 0.380 [0.289, 0.471] | 0.364 [0.231, 0.503] | 0.263 [0.165, 0.374] | 0.329 [0.239, 0.418] | 0.205 [0.166, 0.243] | 1.00 | 6 | 6 |
| cell 40 m | 0.527 [0.408, 0.632] | 0.486 [0.312, 0.659] | 0.418 [0.300, 0.545] | 0.509 [0.374, 0.633] | 0.192 [0.165, 0.221] | 1.00 | 6 | 6 |

**Pooled cell correlation** (a seeded subsample of each tile's cells; the interval resamples whole scene units):

| scale | stability vs abs error | stability vs SAM | texture vs abs error | added detail vs abs error | partial |
|---|---|---|---|---|---|
| cell 10 m | 0.718 [0.451, 0.804] | 0.751 [0.468, 0.809] | 0.675 [0.359, 0.775] | 0.706 [0.393, 0.797] | 0.246 [0.145, 0.346] |
| cell 40 m | 0.818 [0.552, 0.892] | 0.842 [0.596, 0.888] | 0.826 [0.554, 0.885] | 0.833 [0.543, 0.907] | 0.161 [0.051, 0.325] |

**Sensitivity to registration** (why misregistered evidence is excluded): the pixel-level correlation of the same eligible tiles with the reference displaced by known whole pixels:

| reference displaced by (HR px) | 0 | 1 | 2 | 4 |
|---|---|---|---|---|
| mean over units | 0.198 [0.134, 0.267] | 0.211 [0.144, 0.282] | 0.242 [0.174, 0.314] | 0.285 [0.212, 0.359] |

**Tile and scene reliability**: Spearman correlation across tiles (interval resamples scene units) and across scene units (tiles of a unit averaged first).

| error target | tile: stability | tiles | scene: stability | units | tile: texture | tile: added detail | tile: partial (stability given both) |
|---|---|---|---|---|---|---|---|
| rmse | 1.000 (descriptive only) | 6 | 1.000 (descriptive only) | 6 | 1.000 (descriptive only) | 1.000 (descriptive only) | 1.000 (descriptive only) |
| mae | 1.000 (descriptive only) | 6 | 1.000 (descriptive only) | 6 | 1.000 (descriptive only) | 1.000 (descriptive only) | 1.000 (descriptive only) |
| sam_degrees | 0.943 (descriptive only) | 6 | 0.943 (descriptive only) | 6 | 0.943 (descriptive only) | 0.943 (descriptive only) | -0.094 (descriptive only) |
| ergas | 0.943 (descriptive only) | 6 | 0.943 (descriptive only) | 6 | 0.943 (descriptive only) | 0.943 (descriptive only) | -0.094 (descriptive only) |

**Risk-coverage** (most unstable removed first; risk = error of what remains; *not* calibrated selective prediction):

`tile_rmse` (6 items, 6 scene units). Selective efficiency (1 = oracle, 0 = random, < 0 worse than random): stability 1.000 (descriptive only), texture 1.000, added detail 1.000.

| coverage | risk (stability order) | oracle | texture order | added-detail order |
|---|---|---|---|---|
| 100% | 0.02818 | 0.02818 | 0.02818 | 0.02818 |
| 90% | 0.02259 | 0.02259 | 0.02259 | 0.02259 |
| 80% | 0.02259 | 0.02259 | 0.02259 | 0.02259 |
| 70% | 0.01512 | 0.01512 | 0.01512 | 0.01512 |
| 60% | 0.01512 | 0.01512 | 0.01512 | 0.01512 |
| 50% | 0.01153 | 0.01153 | 0.01153 | 0.01153 |

Relative risk reduction vs no removal: at 90% coverage 0.198 (descriptive only); at 80% coverage 0.198 (descriptive only); at 70% coverage 0.463 (descriptive only); at 60% coverage 0.463 (descriptive only); at 50% coverage 0.591 (descriptive only)

`tile_sam` (6 items, 6 scene units). Selective efficiency (1 = oracle, 0 = random, < 0 worse than random): stability 0.989 (descriptive only), texture 0.989, added detail 0.989.

| coverage | risk (stability order) | oracle | texture order | added-detail order |
|---|---|---|---|---|
| 100% | 1.71348 | 1.71348 | 1.71348 | 1.71348 |
| 90% | 1.27689 | 1.27689 | 1.27689 | 1.27689 |
| 80% | 1.27689 | 1.27689 | 1.27689 | 1.27689 |
| 70% | 0.73682 | 0.73682 | 0.73682 | 0.73682 |
| 60% | 0.73682 | 0.73682 | 0.73682 | 0.73682 |
| 50% | 0.69887 | 0.62757 | 0.69887 | 0.69887 |

Relative risk reduction vs no removal: at 90% coverage 0.255 (descriptive only); at 80% coverage 0.255 (descriptive only); at 70% coverage 0.570 (descriptive only); at 60% coverage 0.570 (descriptive only); at 50% coverage 0.592 (descriptive only)

`cell_4_abs_error` (49152 items, 6 scene units). Selective efficiency (1 = oracle, 0 = random, < 0 worse than random): stability 0.767 [0.493, 0.831], texture 0.708, added detail 0.747.

| coverage | risk (stability order) | oracle | texture order | added-detail order |
|---|---|---|---|---|
| 100% | 0.01740 | 0.01740 | 0.01740 | 0.01740 |
| 90% | 0.01458 | 0.01343 | 0.01478 | 0.01467 |
| 80% | 0.01249 | 0.01112 | 0.01297 | 0.01276 |
| 70% | 0.01107 | 0.00935 | 0.01163 | 0.01126 |
| 60% | 0.01019 | 0.00801 | 0.01066 | 0.01026 |
| 50% | 0.00946 | 0.00696 | 0.00992 | 0.00949 |

Relative risk reduction vs no removal: at 90% coverage 0.162 [0.072, 0.230]; at 80% coverage 0.282 [0.118, 0.356]; at 70% coverage 0.364 [0.159, 0.453]; at 60% coverage 0.414 [0.191, 0.515]; at 50% coverage 0.456 [0.226, 0.563]

**High-error detection** (10 m cells):

Threshold 0.03746 = the 90% quantile of error over **same evidence (no held-out split): descriptive only**; prevalence of high error in the evaluated cells 0.100 (49152 cells; test units: 6). **Descriptive only**: 6 scene unit(s): fewer than the 12 needed for a development/test split.

| score | AUROC | AUPRC (chance = prevalence) | precision, top 10% flagged | lift | precision, top 20% | lift |
|---|---|---|---|---|---|---|
| stability (ensemble spread) | 0.893 | 0.512 | 0.495 | 4.95 | 0.386 | 3.86 |
| texture baseline | 0.868 | 0.477 | 0.464 | 4.64 | 0.350 | 3.50 |
| added-detail baseline | 0.885 | 0.507 | 0.485 | 4.85 | 0.365 | 3.65 |

**Calibration**: verdict `uncalibrated_stability_evidence`. The raw TTA spread is not a calibrated uncertainty: it is far smaller than the error and covers a small fraction of it. Whether a monotone map from stability to expected error, fitted on development units, is calibrated on held-out test units is reported below; even where it is, it is calibrated for this dataset and error scale only and is not a probability of error, a confidence or a conformal guarantee.

* Scale: the median error is 15.7 times the median spread (median over tiles).
* Coverage of `prediction +/- k * spread` against a Gaussian's nominal level: k = 1: 0.034 (nominal 0.683); k = 2: 0.069 (nominal 0.954).
* Recalibration: not_assessable (6 scene unit(s): fewer than the 12 needed for a development/test split; recalibration needs held-out units).

**Relationship with the Phase 5 detail categories** (Spearman of stability with each category; not semantic labels; instability and added detail both grow with texture): cell-level, mean over units [interval]:

| scale | unsupported detail | omission | supported synthesis |
|---|---|---|---|
| cell 10 m | 0.528 [0.410, 0.644] | -0.294 [-0.563, -0.028] | 0.591 [0.485, 0.679] |
| cell 40 m | 0.739 [0.625, 0.832] | -0.329 [-0.684, 0.049] | 0.812 [0.759, 0.861] |

Tile-level (tile mean stability vs tile share of each category):

| category | Spearman across tiles |
|---|---|
| unsupported_detail | 1.000 (descriptive only) |
| omission | -0.257 (descriptive only) |
| supported_synthesis | 1.000 (descriptive only) |

**TTA cost**: 6 members; single pass 1.709 s, TTA 10.159 s (x5.95) per scene, median over 5 tiles (the first tile, 9.69 s, carries warm-up and is excluded).

### Comparison of the systems on the same eligible tiles

Descriptive comparison of the same protocol on the tiles that are eligible for BOTH systems, paired over scene units (A - B). It is not a ranking: a different spread magnitude or a different association is a property of each model's stability signal, not a verdict on which model is better.

6 shared tiles, 6 scene units; tiles eligible for one system only: {'sen2sr_lite': 0, 'sen2sr_mamba': 0}.

**sen2sr_lite - sen2sr_mamba** (paired over scene units):

| metric | units | mean difference | 95% interval |
|---|---|---|---|
| `cells.4.spearman_abs_error.value` | 6 | -0.0389 | [-0.07277, -0.01194] |
| `pixel.spearman_abs_error.value` | 6 | -0.0309 | [-0.04935, -0.01407] |
| `stability.mean` | 6 | -0.0001 | [-0.0003592, 2.234e-05] |
| `targets.product.rmse` | 6 | -0.0001 | [-0.0004312, 0.0001065] |
| `targets.product.sam_degrees` | 6 | 0.0137 | [-0.06587, 0.07106] |
| `tta.total_seconds` | 6 | -9.9662 | [-10.22, -9.709] |
| `tta.single_pass_seconds` | 6 | -1.6784 | [-1.725, -1.632] |

## Dataset `opensr_spain_crops`

### `sen2sr_lite`

21 eligible tiles from 5 scene units.

**Within-tile Spearman correlation between stability and error**, mean over scene units [95% interval over units]; the baselines are correlated with the same error; `partial` controls the stability for both baselines; `share > 0` is the share of tiles with a positive correlation.

| scale | stability vs abs error | stability vs SAM | texture vs abs error | added detail vs abs error | partial (stability given both) | share > 0 | tiles | units |
|---|---|---|---|---|---|---|---|---|
| pixel (2.5 m) | 0.263 [0.206, 0.319] | 0.169 [0.097, 0.259] | 0.224 [0.179, 0.273] | 0.235 [0.192, 0.279] | 0.113 [0.086, 0.141] | 1.00 | 21 | 5 |
| cell 10 m | 0.439 [0.357, 0.517] | 0.282 [0.168, 0.386] | 0.402 [0.336, 0.470] | 0.449 [0.387, 0.512] | 0.109 [0.070, 0.153] | 1.00 | 21 | 5 |
| cell 40 m | 0.594 [0.499, 0.681] | 0.416 [0.261, 0.548] | 0.570 [0.492, 0.649] | 0.621 [0.545, 0.690] | 0.059 [-0.005, 0.139] | 1.00 | 21 | 5 |

**Pooled cell correlation** (a seeded subsample of each tile's cells; the interval resamples whole scene units):

| scale | stability vs abs error | stability vs SAM | texture vs abs error | added detail vs abs error | partial |
|---|---|---|---|---|---|
| cell 10 m | 0.475 [0.392, 0.556] | 0.350 [0.294, 0.492] | 0.428 [0.351, 0.501] | 0.474 [0.409, 0.552] | 0.139 [0.102, 0.177] |
| cell 40 m | 0.619 [0.520, 0.716] | 0.444 [0.392, 0.627] | 0.591 [0.490, 0.675] | 0.630 [0.547, 0.729] | 0.110 [0.060, 0.161] |

**Sensitivity to registration** (why misregistered evidence is excluded): the pixel-level correlation of the same eligible tiles with the reference displaced by known whole pixels:

| reference displaced by (HR px) | 0 | 1 | 2 | 4 |
|---|---|---|---|---|
| mean over units | 0.263 [0.206, 0.319] | 0.274 [0.215, 0.332] | 0.300 [0.238, 0.361] | 0.349 [0.281, 0.416] |

**Tile and scene reliability**: Spearman correlation across tiles (interval resamples scene units) and across scene units (tiles of a unit averaged first).

| error target | tile: stability | tiles | scene: stability | units | tile: texture | tile: added detail | tile: partial (stability given both) |
|---|---|---|---|---|---|---|---|
| rmse | 0.757 (descriptive only) | 21 | 0.900 (descriptive only) | 5 | 0.732 (descriptive only) | 0.764 (descriptive only) | 0.064 (descriptive only) |
| mae | 0.860 (descriptive only) | 21 | 1.000 (descriptive only) | 5 | 0.862 (descriptive only) | 0.832 (descriptive only) | 0.071 (descriptive only) |
| sam_degrees | 0.399 (descriptive only) | 21 | 0.600 (descriptive only) | 5 | 0.219 (descriptive only) | 0.518 (descriptive only) | 0.242 (descriptive only) |
| ergas | 0.099 (descriptive only) | 21 | -0.100 (descriptive only) | 5 | -0.117 (descriptive only) | 0.361 (descriptive only) | 0.108 (descriptive only) |

**Risk-coverage** (most unstable removed first; risk = error of what remains; *not* calibrated selective prediction):

`tile_rmse` (21 items, 5 scene units). Selective efficiency (1 = oracle, 0 = random, < 0 worse than random): stability 0.670 (descriptive only), texture 0.622, added detail 0.649.

| coverage | risk (stability order) | oracle | texture order | added-detail order |
|---|---|---|---|---|
| 100% | 0.02211 | 0.02211 | 0.02211 | 0.02211 |
| 90% | 0.02126 | 0.02071 | 0.02161 | 0.02127 |
| 80% | 0.02058 | 0.01993 | 0.02082 | 0.02058 |
| 70% | 0.02028 | 0.01909 | 0.02041 | 0.02028 |
| 60% | 0.01981 | 0.01841 | 0.01981 | 0.02007 |
| 50% | 0.01811 | 0.01732 | 0.01792 | 0.01811 |

Relative risk reduction vs no removal: at 90% coverage 0.039 (descriptive only); at 80% coverage 0.069 (descriptive only); at 70% coverage 0.083 (descriptive only); at 60% coverage 0.104 (descriptive only); at 50% coverage 0.181 (descriptive only)

`tile_sam` (21 items, 5 scene units). Selective efficiency (1 = oracle, 0 = random, < 0 worse than random): stability 0.516 (descriptive only), texture 0.353, added detail 0.578.

| coverage | risk (stability order) | oracle | texture order | added-detail order |
|---|---|---|---|---|
| 100% | 2.26875 | 2.26875 | 2.26875 | 2.26875 |
| 90% | 2.14122 | 2.07318 | 2.28761 | 2.07318 |
| 80% | 2.07125 | 1.92936 | 2.15535 | 2.07125 |
| 70% | 2.08191 | 1.82278 | 2.11817 | 2.08191 |
| 60% | 2.04309 | 1.72491 | 2.04309 | 1.99732 |
| 50% | 1.82806 | 1.60769 | 1.90223 | 1.82806 |

Relative risk reduction vs no removal: at 90% coverage 0.056 (descriptive only); at 80% coverage 0.087 (descriptive only); at 70% coverage 0.082 (descriptive only); at 60% coverage 0.099 (descriptive only); at 50% coverage 0.194 (descriptive only)

`cell_4_abs_error` (172032 items, 5 scene units). Selective efficiency (1 = oracle, 0 = random, < 0 worse than random): stability 0.566 [0.499, 0.622], texture 0.525, added detail 0.543.

| coverage | risk (stability order) | oracle | texture order | added-detail order |
|---|---|---|---|---|
| 100% | 0.01421 | 0.01421 | 0.01421 | 0.01421 |
| 90% | 0.01291 | 0.01189 | 0.01304 | 0.01293 |
| 80% | 0.01214 | 0.01061 | 0.01226 | 0.01218 |
| 70% | 0.01157 | 0.00963 | 0.01173 | 0.01169 |
| 60% | 0.01115 | 0.00881 | 0.01137 | 0.01132 |
| 50% | 0.01083 | 0.00807 | 0.01113 | 0.01104 |

Relative risk reduction vs no removal: at 90% coverage 0.091 [0.082, 0.097]; at 80% coverage 0.145 [0.126, 0.155]; at 70% coverage 0.185 [0.149, 0.198]; at 60% coverage 0.215 [0.166, 0.238]; at 50% coverage 0.237 [0.181, 0.272]

**High-error detection** (10 m cells):

Threshold 0.02551 = the 90% quantile of error over **same evidence (no held-out split): descriptive only**; prevalence of high error in the evaluated cells 0.100 (172032 cells; test units: 5). **Descriptive only**: 5 scene unit(s): fewer than the 12 needed for a development/test split.

| score | AUROC | AUPRC (chance = prevalence) | precision, top 10% flagged | lift | precision, top 20% | lift |
|---|---|---|---|---|---|---|
| stability (ensemble spread) | 0.819 | 0.398 | 0.426 | 4.26 | 0.319 | 3.19 |
| texture baseline | 0.791 | 0.343 | 0.386 | 3.86 | 0.301 | 3.01 |
| added-detail baseline | 0.793 | 0.388 | 0.419 | 4.19 | 0.309 | 3.09 |

**Calibration**: verdict `uncalibrated_stability_evidence`. The raw TTA spread is not a calibrated uncertainty: it is far smaller than the error and covers a small fraction of it. Whether a monotone map from stability to expected error, fitted on development units, is calibrated on held-out test units is reported below; even where it is, it is calibrated for this dataset and error scale only and is not a probability of error, a confidence or a conformal guarantee.

* Scale: the median error is 23.2 times the median spread (median over tiles).
* Coverage of `prediction +/- k * spread` against a Gaussian's nominal level: k = 1: 0.028 (nominal 0.683); k = 2: 0.057 (nominal 0.954).
* Recalibration: not_assessable (5 scene unit(s): fewer than the 12 needed for a development/test split; recalibration needs held-out units).

**Relationship with the Phase 5 detail categories** (Spearman of stability with each category; not semantic labels; instability and added detail both grow with texture): cell-level, mean over units [interval]:

| scale | unsupported detail | omission | supported synthesis |
|---|---|---|---|
| cell 10 m | 0.628 [0.525, 0.713] | -0.234 [-0.425, -0.092] | 0.674 [0.569, 0.763] |
| cell 40 m | 0.849 [0.776, 0.909] | -0.287 [-0.529, -0.098] | 0.865 [0.803, 0.918] |

Tile-level (tile mean stability vs tile share of each category):

| category | Spearman across tiles |
|---|---|
| unsupported_detail | 0.649 (descriptive only) |
| omission | 0.499 (descriptive only) |
| supported_synthesis | 0.890 (descriptive only) |

**TTA cost**: 6 members; single pass 0.012 s, TTA 0.179 s (x14.89) per scene, median over 20 tiles (the first tile, 0.17 s, carries warm-up and is excluded).

### `sen2sr_mamba`

21 eligible tiles from 5 scene units.

**Within-tile Spearman correlation between stability and error**, mean over scene units [95% interval over units]; the baselines are correlated with the same error; `partial` controls the stability for both baselines; `share > 0` is the share of tiles with a positive correlation.

| scale | stability vs abs error | stability vs SAM | texture vs abs error | added detail vs abs error | partial (stability given both) | share > 0 | tiles | units |
|---|---|---|---|---|---|---|---|---|
| pixel (2.5 m) | 0.289 [0.230, 0.349] | 0.206 [0.130, 0.305] | 0.224 [0.182, 0.272] | 0.256 [0.213, 0.299] | 0.136 [0.101, 0.173] | 1.00 | 21 | 5 |
| cell 10 m | 0.471 [0.386, 0.552] | 0.317 [0.201, 0.433] | 0.401 [0.336, 0.464] | 0.464 [0.393, 0.528] | 0.156 [0.106, 0.206] | 1.00 | 21 | 5 |
| cell 40 m | 0.598 [0.506, 0.682] | 0.429 [0.282, 0.560] | 0.567 [0.484, 0.639] | 0.615 [0.537, 0.681] | 0.100 [0.022, 0.182] | 1.00 | 21 | 5 |

**Pooled cell correlation** (a seeded subsample of each tile's cells; the interval resamples whole scene units):

| scale | stability vs abs error | stability vs SAM | texture vs abs error | added detail vs abs error | partial |
|---|---|---|---|---|---|
| cell 10 m | 0.476 [0.390, 0.576] | 0.346 [0.287, 0.516] | 0.424 [0.346, 0.494] | 0.475 [0.408, 0.558] | 0.158 [0.097, 0.199] |
| cell 40 m | 0.595 [0.486, 0.709] | 0.423 [0.357, 0.627] | 0.584 [0.482, 0.668] | 0.617 [0.533, 0.710] | 0.105 [0.001, 0.165] |

**Sensitivity to registration** (why misregistered evidence is excluded): the pixel-level correlation of the same eligible tiles with the reference displaced by known whole pixels:

| reference displaced by (HR px) | 0 | 1 | 2 | 4 |
|---|---|---|---|---|
| mean over units | 0.289 [0.230, 0.349] | 0.303 [0.241, 0.366] | 0.332 [0.267, 0.397] | 0.374 [0.310, 0.438] |

**Tile and scene reliability**: Spearman correlation across tiles (interval resamples scene units) and across scene units (tiles of a unit averaged first).

| error target | tile: stability | tiles | scene: stability | units | tile: texture | tile: added detail | tile: partial (stability given both) |
|---|---|---|---|---|---|---|---|
| rmse | 0.536 (descriptive only) | 21 | 0.900 (descriptive only) | 5 | 0.705 (descriptive only) | 0.765 (descriptive only) | -0.622 (descriptive only) |
| mae | 0.712 (descriptive only) | 21 | 1.000 (descriptive only) | 5 | 0.857 (descriptive only) | 0.817 (descriptive only) | -0.484 (descriptive only) |
| sam_degrees | 0.300 (descriptive only) | 21 | 0.600 (descriptive only) | 5 | 0.217 (descriptive only) | 0.573 (descriptive only) | -0.153 (descriptive only) |
| ergas | -0.095 (descriptive only) | 21 | -0.100 (descriptive only) | 5 | -0.117 (descriptive only) | 0.384 (descriptive only) | -0.683 (descriptive only) |

**Risk-coverage** (most unstable removed first; risk = error of what remains; *not* calibrated selective prediction):

`tile_rmse` (21 items, 5 scene units). Selective efficiency (1 = oracle, 0 = random, < 0 worse than random): stability 0.546 (descriptive only), texture 0.612, added detail 0.665.

| coverage | risk (stability order) | oracle | texture order | added-detail order |
|---|---|---|---|---|
| 100% | 0.02200 | 0.02200 | 0.02200 | 0.02200 |
| 90% | 0.02114 | 0.02059 | 0.02152 | 0.02112 |
| 80% | 0.02097 | 0.01981 | 0.02071 | 0.02044 |
| 70% | 0.02061 | 0.01897 | 0.02033 | 0.02000 |
| 60% | 0.02004 | 0.01828 | 0.01971 | 0.01995 |
| 50% | 0.01854 | 0.01723 | 0.01785 | 0.01802 |

Relative risk reduction vs no removal: at 90% coverage 0.039 (descriptive only); at 80% coverage 0.047 (descriptive only); at 70% coverage 0.063 (descriptive only); at 60% coverage 0.089 (descriptive only); at 50% coverage 0.157 (descriptive only)

`tile_sam` (21 items, 5 scene units). Selective efficiency (1 = oracle, 0 = random, < 0 worse than random): stability 0.439 (descriptive only), texture 0.347, added detail 0.629.

| coverage | risk (stability order) | oracle | texture order | added-detail order |
|---|---|---|---|---|
| 100% | 2.26199 | 2.26199 | 2.26199 | 2.26199 |
| 90% | 2.12990 | 2.05494 | 2.28187 | 2.05494 |
| 80% | 2.15668 | 1.91239 | 2.14431 | 2.05280 |
| 70% | 2.08632 | 1.79894 | 2.11037 | 1.96859 |
| 60% | 1.99979 | 1.69795 | 2.02753 | 1.98423 |
| 50% | 1.92413 | 1.58209 | 1.89410 | 1.81732 |

Relative risk reduction vs no removal: at 90% coverage 0.058 (descriptive only); at 80% coverage 0.047 (descriptive only); at 70% coverage 0.078 (descriptive only); at 60% coverage 0.116 (descriptive only); at 50% coverage 0.149 (descriptive only)

`cell_4_abs_error` (172032 items, 5 scene units). Selective efficiency (1 = oracle, 0 = random, < 0 worse than random): stability 0.584 [0.507, 0.647], texture 0.520, added detail 0.554.

| coverage | risk (stability order) | oracle | texture order | added-detail order |
|---|---|---|---|---|
| 100% | 0.01412 | 0.01412 | 0.01412 | 0.01412 |
| 90% | 0.01268 | 0.01182 | 0.01297 | 0.01279 |
| 80% | 0.01191 | 0.01054 | 0.01220 | 0.01202 |
| 70% | 0.01143 | 0.00956 | 0.01168 | 0.01155 |
| 60% | 0.01109 | 0.00874 | 0.01132 | 0.01124 |
| 50% | 0.01081 | 0.00800 | 0.01109 | 0.01100 |

Relative risk reduction vs no removal: at 90% coverage 0.102 [0.090, 0.111]; at 80% coverage 0.157 [0.130, 0.167]; at 70% coverage 0.190 [0.154, 0.210]; at 60% coverage 0.215 [0.174, 0.247]; at 50% coverage 0.235 [0.189, 0.280]

**High-error detection** (10 m cells):

Threshold 0.02536 = the 90% quantile of error over **same evidence (no held-out split): descriptive only**; prevalence of high error in the evaluated cells 0.100 (172032 cells; test units: 5). **Descriptive only**: 5 scene unit(s): fewer than the 12 needed for a development/test split.

| score | AUROC | AUPRC (chance = prevalence) | precision, top 10% flagged | lift | precision, top 20% | lift |
|---|---|---|---|---|---|---|
| stability (ensemble spread) | 0.826 | 0.464 | 0.476 | 4.76 | 0.336 | 3.36 |
| texture baseline | 0.787 | 0.337 | 0.383 | 3.83 | 0.299 | 2.99 |
| added-detail baseline | 0.792 | 0.409 | 0.440 | 4.40 | 0.319 | 3.19 |

**Calibration**: verdict `uncalibrated_stability_evidence`. The raw TTA spread is not a calibrated uncertainty: it is far smaller than the error and covers a small fraction of it. Whether a monotone map from stability to expected error, fitted on development units, is calibrated on held-out test units is reported below; even where it is, it is calibrated for this dataset and error scale only and is not a probability of error, a confidence or a conformal guarantee.

* Scale: the median error is 26.4 times the median spread (median over tiles).
* Coverage of `prediction +/- k * spread` against a Gaussian's nominal level: k = 1: 0.024 (nominal 0.683); k = 2: 0.048 (nominal 0.954).
* Recalibration: not_assessable (5 scene unit(s): fewer than the 12 needed for a development/test split; recalibration needs held-out units).

**Relationship with the Phase 5 detail categories** (Spearman of stability with each category; not semantic labels; instability and added detail both grow with texture): cell-level, mean over units [interval]:

| scale | unsupported detail | omission | supported synthesis |
|---|---|---|---|
| cell 10 m | 0.705 [0.639, 0.761] | -0.327 [-0.489, -0.184] | 0.740 [0.663, 0.805] |
| cell 40 m | 0.866 [0.818, 0.904] | -0.388 [-0.591, -0.208] | 0.885 [0.841, 0.919] |

Tile-level (tile mean stability vs tile share of each category):

| category | Spearman across tiles |
|---|---|
| unsupported_detail | 0.694 (descriptive only) |
| omission | 0.452 (descriptive only) |
| supported_synthesis | 0.855 (descriptive only) |

**TTA cost**: 6 members; single pass 1.719 s, TTA 10.423 s (x6.06) per scene, median over 20 tiles (the first tile, 10.46 s, carries warm-up and is excluded).

### Comparison of the systems on the same eligible tiles

Descriptive comparison of the same protocol on the tiles that are eligible for BOTH systems, paired over scene units (A - B). It is not a ranking: a different spread magnitude or a different association is a property of each model's stability signal, not a verdict on which model is better.

21 shared tiles, 5 scene units; tiles eligible for one system only: {'sen2sr_lite': 0, 'sen2sr_mamba': 0}.

**sen2sr_lite - sen2sr_mamba** (paired over scene units):

| metric | units | mean difference | 95% interval |
|---|---|---|---|
| `cells.4.spearman_abs_error.value` | 5 | -0.0319 | [-0.04225, -0.02237] |
| `pixel.spearman_abs_error.value` | 5 | -0.0267 | [-0.03441, -0.01906] |
| `stability.mean` | 5 | -0.0001 | [-0.0001547, -3.293e-05] |
| `targets.product.rmse` | 5 | 0.0001 | [9.862e-06, 0.0002138] |
| `targets.product.sam_degrees` | 5 | -0.0021 | [-0.04531, 0.03215] |
| `tta.total_seconds` | 5 | -10.1818 | [-10.42, -9.961] |
| `tta.single_pass_seconds` | 5 | -1.7023 | [-1.747, -1.657] |

## Dataset `opensr_spain_urban`

### `sen2sr_lite`

13 eligible tiles from 4 scene units. **Descriptive only (fewer than 5 scene units).**

**Within-tile Spearman correlation between stability and error**, mean over scene units [95% interval over units]; the baselines are correlated with the same error; `partial` controls the stability for both baselines; `share > 0` is the share of tiles with a positive correlation.

| scale | stability vs abs error | stability vs SAM | texture vs abs error | added detail vs abs error | partial (stability given both) | share > 0 | tiles | units |
|---|---|---|---|---|---|---|---|---|
| pixel (2.5 m) | 0.333 (descriptive only) | 0.299 (descriptive only) | 0.271 (descriptive only) | 0.295 (descriptive only) | 0.152 (descriptive only) | 1.00 | 13 | 4 |
| cell 10 m | 0.540 (descriptive only) | 0.455 (descriptive only) | 0.475 (descriptive only) | 0.541 (descriptive only) | 0.169 (descriptive only) | 1.00 | 13 | 4 |
| cell 40 m | 0.713 (descriptive only) | 0.608 (descriptive only) | 0.667 (descriptive only) | 0.731 (descriptive only) | 0.131 (descriptive only) | 1.00 | 13 | 4 |

**Pooled cell correlation** (a seeded subsample of each tile's cells; the interval resamples whole scene units):

| scale | stability vs abs error | stability vs SAM | texture vs abs error | added detail vs abs error | partial |
|---|---|---|---|---|---|
| cell 10 m | 0.608 (descriptive only) | 0.430 (descriptive only) | 0.548 (descriptive only) | 0.581 (descriptive only) | 0.212 (descriptive only) |
| cell 40 m | 0.760 (descriptive only) | 0.538 (descriptive only) | 0.731 (descriptive only) | 0.747 (descriptive only) | 0.167 (descriptive only) |

**Sensitivity to registration** (why misregistered evidence is excluded): the pixel-level correlation of the same eligible tiles with the reference displaced by known whole pixels:

| reference displaced by (HR px) | 0 | 1 | 2 | 4 |
|---|---|---|---|---|
| mean over units | 0.333 (descriptive only) | 0.346 (descriptive only) | 0.372 (descriptive only) | 0.418 (descriptive only) |

**Tile and scene reliability**: Spearman correlation across tiles (interval resamples scene units) and across scene units (tiles of a unit averaged first).

| error target | tile: stability | tiles | scene: stability | units | tile: texture | tile: added detail | tile: partial (stability given both) |
|---|---|---|---|---|---|---|---|
| rmse | 0.852 (descriptive only) | 13 | 0.800 (descriptive only) | 4 | 0.703 (descriptive only) | 0.830 (descriptive only) | 0.382 (descriptive only) |
| mae | 0.896 (descriptive only) | 13 | 0.800 (descriptive only) | 4 | 0.736 (descriptive only) | 0.901 (descriptive only) | 0.362 (descriptive only) |
| sam_degrees | 0.363 (descriptive only) | 13 | -0.400 (descriptive only) | 4 | 0.093 (descriptive only) | 0.522 (descriptive only) | -0.096 (descriptive only) |
| ergas | -0.297 (descriptive only) | 13 | -0.400 (descriptive only) | 4 | -0.560 (descriptive only) | -0.220 (descriptive only) | 0.207 (descriptive only) |

**Risk-coverage** (most unstable removed first; risk = error of what remains; *not* calibrated selective prediction):

`tile_rmse` (13 items, 4 scene units). Selective efficiency (1 = oracle, 0 = random, < 0 worse than random): stability 0.887 (descriptive only), texture 0.606, added detail 0.845.

| coverage | risk (stability order) | oracle | texture order | added-detail order |
|---|---|---|---|---|
| 100% | 0.03070 | 0.03070 | 0.03070 | 0.03070 |
| 90% | 0.02961 | 0.02961 | 0.02961 | 0.03041 |
| 80% | 0.02774 | 0.02762 | 0.02864 | 0.02774 |
| 70% | 0.02694 | 0.02679 | 0.02796 | 0.02756 |
| 60% | 0.02663 | 0.02573 | 0.02788 | 0.02662 |
| 50% | 0.02479 | 0.02339 | 0.02786 | 0.02339 |

Relative risk reduction vs no removal: at 90% coverage 0.035 (descriptive only); at 80% coverage 0.096 (descriptive only); at 70% coverage 0.122 (descriptive only); at 60% coverage 0.132 (descriptive only); at 50% coverage 0.192 (descriptive only)

`tile_sam` (13 items, 4 scene units). Selective efficiency (1 = oracle, 0 = random, < 0 worse than random): stability 0.323 (descriptive only), texture -0.279, added detail 0.482.

| coverage | risk (stability order) | oracle | texture order | added-detail order |
|---|---|---|---|---|
| 100% | 3.35249 | 3.35249 | 3.35249 | 3.35249 |
| 90% | 3.39898 | 3.16760 | 3.39898 | 3.35933 |
| 80% | 3.19461 | 2.94568 | 3.43561 | 3.19461 |
| 70% | 3.19832 | 2.89064 | 3.43500 | 3.13438 |
| 60% | 3.13102 | 2.84317 | 3.50154 | 3.09604 |
| 50% | 3.11740 | 2.73429 | 3.67565 | 2.79824 |

Relative risk reduction vs no removal: at 90% coverage -0.014 (descriptive only); at 80% coverage 0.047 (descriptive only); at 70% coverage 0.046 (descriptive only); at 60% coverage 0.066 (descriptive only); at 50% coverage 0.070 (descriptive only)

`cell_4_abs_error` (106496 items, 4 scene units). Selective efficiency (1 = oracle, 0 = random, < 0 worse than random): stability 0.644 (descriptive only), texture 0.552, added detail 0.600.

| coverage | risk (stability order) | oracle | texture order | added-detail order |
|---|---|---|---|---|
| 100% | 0.02034 | 0.02034 | 0.02034 | 0.02034 |
| 90% | 0.01829 | 0.01692 | 0.01864 | 0.01844 |
| 80% | 0.01697 | 0.01497 | 0.01755 | 0.01723 |
| 70% | 0.01583 | 0.01342 | 0.01654 | 0.01617 |
| 60% | 0.01490 | 0.01209 | 0.01559 | 0.01526 |
| 50% | 0.01412 | 0.01088 | 0.01470 | 0.01446 |

Relative risk reduction vs no removal: at 90% coverage 0.101 (descriptive only); at 80% coverage 0.166 (descriptive only); at 70% coverage 0.222 (descriptive only); at 60% coverage 0.267 (descriptive only); at 50% coverage 0.306 (descriptive only)

**High-error detection** (10 m cells):

Threshold 0.03748 = the 90% quantile of error over **same evidence (no held-out split): descriptive only**; prevalence of high error in the evaluated cells 0.100 (106496 cells; test units: 4). **Descriptive only**: 4 scene unit(s): fewer than the 12 needed for a development/test split.

| score | AUROC | AUPRC (chance = prevalence) | precision, top 10% flagged | lift | precision, top 20% | lift |
|---|---|---|---|---|---|---|
| stability (ensemble spread) | 0.837 | 0.430 | 0.435 | 4.35 | 0.327 | 3.27 |
| texture baseline | 0.794 | 0.359 | 0.358 | 3.58 | 0.278 | 2.78 |
| added-detail baseline | 0.813 | 0.392 | 0.406 | 4.06 | 0.302 | 3.02 |

**Calibration**: verdict `uncalibrated_stability_evidence`. The raw TTA spread is not a calibrated uncertainty: it is far smaller than the error and covers a small fraction of it. Whether a monotone map from stability to expected error, fitted on development units, is calibrated on held-out test units is reported below; even where it is, it is calibrated for this dataset and error scale only and is not a probability of error, a confidence or a conformal guarantee.

* Scale: the median error is 19.6 times the median spread (median over tiles).
* Coverage of `prediction +/- k * spread` against a Gaussian's nominal level: k = 1: 0.031 (nominal 0.683); k = 2: 0.061 (nominal 0.954).
* Recalibration: not_assessable (4 scene unit(s): fewer than the 12 needed for a development/test split; recalibration needs held-out units).

**Relationship with the Phase 5 detail categories** (Spearman of stability with each category; not semantic labels; instability and added detail both grow with texture): cell-level, mean over units [interval]:

| scale | unsupported detail | omission | supported synthesis |
|---|---|---|---|
| cell 10 m | 0.674 (descriptive only) | -0.519 (descriptive only) | 0.731 (descriptive only) |
| cell 40 m | 0.865 (descriptive only) | -0.658 (descriptive only) | 0.892 (descriptive only) |

Tile-level (tile mean stability vs tile share of each category):

| category | Spearman across tiles |
|---|---|
| unsupported_detail | 0.857 (descriptive only) |
| omission | 0.505 (descriptive only) |
| supported_synthesis | 0.830 (descriptive only) |

**TTA cost**: 6 members; single pass 0.015 s, TTA 0.183 s (x12.44) per scene, median over 12 tiles (the first tile, 0.19 s, carries warm-up and is excluded).

### `sen2sr_mamba`

13 eligible tiles from 4 scene units. **Descriptive only (fewer than 5 scene units).**

**Within-tile Spearman correlation between stability and error**, mean over scene units [95% interval over units]; the baselines are correlated with the same error; `partial` controls the stability for both baselines; `share > 0` is the share of tiles with a positive correlation.

| scale | stability vs abs error | stability vs SAM | texture vs abs error | added detail vs abs error | partial (stability given both) | share > 0 | tiles | units |
|---|---|---|---|---|---|---|---|---|
| pixel (2.5 m) | 0.377 (descriptive only) | 0.361 (descriptive only) | 0.276 (descriptive only) | 0.322 (descriptive only) | 0.200 (descriptive only) | 1.00 | 13 | 4 |
| cell 10 m | 0.589 (descriptive only) | 0.510 (descriptive only) | 0.479 (descriptive only) | 0.558 (descriptive only) | 0.261 (descriptive only) | 1.00 | 13 | 4 |
| cell 40 m | 0.725 (descriptive only) | 0.623 (descriptive only) | 0.667 (descriptive only) | 0.732 (descriptive only) | 0.223 (descriptive only) | 1.00 | 13 | 4 |

**Pooled cell correlation** (a seeded subsample of each tile's cells; the interval resamples whole scene units):

| scale | stability vs abs error | stability vs SAM | texture vs abs error | added detail vs abs error | partial |
|---|---|---|---|---|---|
| cell 10 m | 0.637 (descriptive only) | 0.446 (descriptive only) | 0.547 (descriptive only) | 0.591 (descriptive only) | 0.279 (descriptive only) |
| cell 40 m | 0.755 (descriptive only) | 0.514 (descriptive only) | 0.724 (descriptive only) | 0.746 (descriptive only) | 0.204 (descriptive only) |

**Sensitivity to registration** (why misregistered evidence is excluded): the pixel-level correlation of the same eligible tiles with the reference displaced by known whole pixels:

| reference displaced by (HR px) | 0 | 1 | 2 | 4 |
|---|---|---|---|---|
| mean over units | 0.377 (descriptive only) | 0.393 (descriptive only) | 0.418 (descriptive only) | 0.447 (descriptive only) |

**Tile and scene reliability**: Spearman correlation across tiles (interval resamples scene units) and across scene units (tiles of a unit averaged first).

| error target | tile: stability | tiles | scene: stability | units | tile: texture | tile: added detail | tile: partial (stability given both) |
|---|---|---|---|---|---|---|---|
| rmse | 0.775 (descriptive only) | 13 | 0.800 (descriptive only) | 4 | 0.665 (descriptive only) | 0.813 (descriptive only) | 0.116 (descriptive only) |
| mae | 0.780 (descriptive only) | 13 | 0.800 (descriptive only) | 4 | 0.698 (descriptive only) | 0.835 (descriptive only) | -0.031 (descriptive only) |
| sam_degrees | 0.346 (descriptive only) | 13 | -0.400 (descriptive only) | 4 | 0.044 (descriptive only) | 0.555 (descriptive only) | 0.230 (descriptive only) |
| ergas | -0.264 (descriptive only) | 13 | -0.400 (descriptive only) | 4 | -0.560 (descriptive only) | -0.132 (descriptive only) | 0.474 (descriptive only) |

**Risk-coverage** (most unstable removed first; risk = error of what remains; *not* calibrated selective prediction):

`tile_rmse` (13 items, 4 scene units). Selective efficiency (1 = oracle, 0 = random, < 0 worse than random): stability 0.654 (descriptive only), texture 0.557, added detail 0.666.

| coverage | risk (stability order) | oracle | texture order | added-detail order |
|---|---|---|---|---|
| 100% | 0.03082 | 0.03082 | 0.03082 | 0.03082 |
| 90% | 0.03055 | 0.02972 | 0.02977 | 0.03055 |
| 80% | 0.02887 | 0.02759 | 0.02887 | 0.02881 |
| 70% | 0.02823 | 0.02678 | 0.02823 | 0.02817 |
| 60% | 0.02811 | 0.02575 | 0.02816 | 0.02804 |
| 50% | 0.02346 | 0.02346 | 0.02827 | 0.02346 |

Relative risk reduction vs no removal: at 90% coverage 0.009 (descriptive only); at 80% coverage 0.063 (descriptive only); at 70% coverage 0.084 (descriptive only); at 60% coverage 0.088 (descriptive only); at 50% coverage 0.239 (descriptive only)

`tile_sam` (13 items, 4 scene units). Selective efficiency (1 = oracle, 0 = random, < 0 worse than random): stability -0.023 (descriptive only), texture -0.337, added detail 0.497.

| coverage | risk (stability order) | oracle | texture order | added-detail order |
|---|---|---|---|---|
| 100% | 3.42435 | 3.42435 | 3.42435 | 3.42435 |
| 90% | 3.43714 | 3.22159 | 3.47748 | 3.43714 |
| 80% | 3.52934 | 2.98339 | 3.52934 | 3.22227 |
| 70% | 3.54165 | 2.93504 | 3.54165 | 3.20047 |
| 60% | 3.51510 | 2.89306 | 3.61543 | 3.13126 |
| 50% | 2.86538 | 2.78912 | 3.82524 | 2.86538 |

Relative risk reduction vs no removal: at 90% coverage -0.004 (descriptive only); at 80% coverage -0.031 (descriptive only); at 70% coverage -0.034 (descriptive only); at 60% coverage -0.027 (descriptive only); at 50% coverage 0.163 (descriptive only)

`cell_4_abs_error` (106496 items, 4 scene units). Selective efficiency (1 = oracle, 0 = random, < 0 worse than random): stability 0.668 (descriptive only), texture 0.549, added detail 0.613.

| coverage | risk (stability order) | oracle | texture order | added-detail order |
|---|---|---|---|---|
| 100% | 0.02040 | 0.02040 | 0.02040 | 0.02040 |
| 90% | 0.01824 | 0.01696 | 0.01872 | 0.01848 |
| 80% | 0.01683 | 0.01499 | 0.01763 | 0.01720 |
| 70% | 0.01569 | 0.01343 | 0.01660 | 0.01608 |
| 60% | 0.01475 | 0.01208 | 0.01563 | 0.01514 |
| 50% | 0.01399 | 0.01087 | 0.01472 | 0.01435 |

Relative risk reduction vs no removal: at 90% coverage 0.106 (descriptive only); at 80% coverage 0.175 (descriptive only); at 70% coverage 0.231 (descriptive only); at 60% coverage 0.277 (descriptive only); at 50% coverage 0.314 (descriptive only)

**High-error detection** (10 m cells):

Threshold 0.03772 = the 90% quantile of error over **same evidence (no held-out split): descriptive only**; prevalence of high error in the evaluated cells 0.100 (106496 cells; test units: 4). **Descriptive only**: 4 scene unit(s): fewer than the 12 needed for a development/test split.

| score | AUROC | AUPRC (chance = prevalence) | precision, top 10% flagged | lift | precision, top 20% | lift |
|---|---|---|---|---|---|---|
| stability (ensemble spread) | 0.841 | 0.450 | 0.458 | 4.58 | 0.337 | 3.37 |
| texture baseline | 0.791 | 0.352 | 0.351 | 3.51 | 0.274 | 2.74 |
| added-detail baseline | 0.813 | 0.389 | 0.404 | 4.04 | 0.306 | 3.06 |

**Calibration**: verdict `uncalibrated_stability_evidence`. The raw TTA spread is not a calibrated uncertainty: it is far smaller than the error and covers a small fraction of it. Whether a monotone map from stability to expected error, fitted on development units, is calibrated on held-out test units is reported below; even where it is, it is calibrated for this dataset and error scale only and is not a probability of error, a confidence or a conformal guarantee.

* Scale: the median error is 19.2 times the median spread (median over tiles).
* Coverage of `prediction +/- k * spread` against a Gaussian's nominal level: k = 1: 0.029 (nominal 0.683); k = 2: 0.057 (nominal 0.954).
* Recalibration: not_assessable (4 scene unit(s): fewer than the 12 needed for a development/test split; recalibration needs held-out units).

**Relationship with the Phase 5 detail categories** (Spearman of stability with each category; not semantic labels; instability and added detail both grow with texture): cell-level, mean over units [interval]:

| scale | unsupported detail | omission | supported synthesis |
|---|---|---|---|
| cell 10 m | 0.738 (descriptive only) | -0.590 (descriptive only) | 0.770 (descriptive only) |
| cell 40 m | 0.871 (descriptive only) | -0.723 (descriptive only) | 0.896 (descriptive only) |

Tile-level (tile mean stability vs tile share of each category):

| category | Spearman across tiles |
|---|---|
| unsupported_detail | 0.747 (descriptive only) |
| omission | 0.434 (descriptive only) |
| supported_synthesis | 0.929 (descriptive only) |

**TTA cost**: 6 members; single pass 1.558 s, TTA 9.463 s (x6.07) per scene, median over 12 tiles (the first tile, 9.85 s, carries warm-up and is excluded).

### Comparison of the systems on the same eligible tiles

Descriptive comparison of the same protocol on the tiles that are eligible for BOTH systems, paired over scene units (A - B). It is not a ranking: a different spread magnitude or a different association is a property of each model's stability signal, not a verdict on which model is better.

13 shared tiles, 4 scene units; tiles eligible for one system only: {'sen2sr_lite': 0, 'sen2sr_mamba': 0}.

**sen2sr_lite - sen2sr_mamba** (paired over scene units):

| metric | units | mean difference | 95% interval |
|---|---|---|---|
| `cells.4.spearman_abs_error.value` | 4 | -0.0491 | (descriptive only) |
| `pixel.spearman_abs_error.value` | 4 | -0.0443 | (descriptive only) |
| `stability.mean` | 4 | -0.0002 | (descriptive only) |
| `targets.product.rmse` | 4 | -0.0000 | (descriptive only) |
| `targets.product.sam_degrees` | 4 | -0.0487 | (descriptive only) |
| `tta.total_seconds` | 4 | -9.5237 | (descriptive only) |
| `tta.single_pass_seconds` | 4 | -1.5902 | (descriptive only) |

## How to read this

* Correlations are between the **spread of the TTA ensemble** and **reconstruction error against the reference**. A positive value means more unstable places tend to have larger error; it does not mean instability causes error.
* Compare every stability figure with the **texture** and **added-detail** baselines: they need no ensemble. The partial correlation is what remains after controlling for both.
* `descriptive only` means too few scene units for an interval (or no held-out units for a threshold or a calibration). It is never turned into a claim.
* Risk-coverage describes an ordering. It is **not** calibrated selective prediction; the oracle and random removal bracket it.
* The raw spread is orders of magnitude smaller than the error (ensemble members share the model's bias): it is reported as **uncalibrated stability evidence**. A recalibration is judged only on held-out units.
* Registration limits everything: tiles whose reference is not registered were excluded, and the residual misregistration of the accepted tiles (<= the tolerance) still adds noise.

Files: `config.json`, `metrics.jsonl` (one row per tile and system; exclusions carry a reason and no numbers), `summary.json` (provenance, evidence, cost), `correlations.json`, `risk_coverage.json`, `detection.json`, `calibration.json`.
