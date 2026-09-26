# Downstream analytical utility `downstream_v1`

Status: **completed** (run `downstream_v1-20260925T164214Z`). Question: does super-resolution preserve or improve a downstream NDVI-derived analytical quantity and vegetation decision relative to the LR / bicubic baseline, and is the model's instability (the TTA spread) associated with downstream mistakes once image texture is controlled for? A null or mixed answer is a result. **No ranking of systems is made**; every difference is a paired value with its uncertainty where the number of scene units allows one.

Code: git `eb3769559525ea81e902422253f8950c36b075a5` (working tree dirty: True); config digest `cfc6c1a2c91b624f…`; metrics `frame-eval-metrics/1`; reference gate `frame-reliability-gate/1`.

## What was measured

* **Task**: NDVI-derived analysis, NDVI = (NIR - Red) / (NIR + Red) = (B08 - B04) / (B08 + B04), reflectance fractions (`reflectance fraction = stored DN / the dataset's reflectance_scale (10000); no BOA offset applied, no clipping by FRAME`), bands selected by name from the inputs in order ['B04', 'B03', 'B02', 'B08'].
* **Regions**: Fixed square cells on the aligned HR grid (the cells frame.reliability uses): 4 x 4 HR px = 10 m, the footprint of one Sentinel-2 pixel (the requirements' '10 m pixel = 16 cells at 2.5 m'), and 16 x 16 HR px = 40 m; every region starts on a multiple of its size in prediction coordinates; a region needs at least the declared share of valid pixels. Scales: [4, 16] HR px (headline: 4 HR px = 10 m). A region needs at least 75% valid pixels.
* **Decision rule** (fixed): vegetation iff NDVI >= threshold; threshold **0.3** (predeclared conventional value not selected or tuned on results: 0.3 is a conventional value; the requirements name no NDVI threshold. It was declared before any result and is neither selected nor tuned on results; the sensitivity thresholds are reported beside it.). Sensitivity thresholds [0.2, 0.4] are reported beside it. A region is called vegetated when at least 50% of its valid pixels are.
* **Valid pixels**: HR pixel valid iff it is not all-band nodata (the dataset's rule) AND none of the evaluated bands equals the HR nodata value AND all evaluated values are finite; then restricted to the aligned overlap and to pixels where the reference's Red + NIR reflectance exceeds 0.02 and NDVI is computable (|NIR + Red| > 1e-06) for the reference, the LR, bicubic and every model (identical pixels for every system).
* **Systems** (all on the same pixels): `lr_native` (the observed 10 m NDVI replicated to its cells), `bicubic`, `sen2sr_lite`, `sen2sr_mamba` (the ensemble mean of the deployed six-view TTA); reference = HR.
* **Reference gate** (frame-reliability-gate/1, the Phase 6 gate, unchanged): bicubic-baseline registration within 0.5 HR px after at most one recorded whole-pixel translation (<= 4 px), quadrant spread <= 1.0 px, >= 50% valid pixels. Ineligible tiles get no NDVI, no region and no error.
* **Inference**: over scene units (NEON acquisition / source orthophoto), never over pooled regions; an interval needs >= 5 units, a correlation across units >= 10; fewer is descriptive only. Bootstrap seed 0, 2000 resamples.

### Evidence

Regions at the headline scale (10 m):

| dataset | records | eligible tiles | uncertain | not eligible | candidate regions | in eligible tiles | analysed | excluded with their tile | excluded in eligible tiles | eligible scene units / all |
|---|---|---|---|---|---|---|---|---|---|---|
| `sen2neon_random30` | 30 | 12 | 15 | 3 | 1,966,080 | 786,432 | 711,979 | 1,179,648 | 74,453 | 11 / 28 |
| `opensr_spot` | 9 | 6 | 3 | 0 | 147,456 | 98,304 | 97,921 | 49,152 | 383 | 6 / 9 |
| `opensr_spain_crops` | 28 | 21 | 7 | 0 | 458,752 | 344,064 | 339,852 | 114,688 | 4,212 | 5 / 5 |
| `opensr_spain_urban` | 20 | 13 | 7 | 0 | 327,680 | 212,992 | 210,440 | 114,688 | 2,552 | 4 / 4 |

`sen2neon_random30`: tiles excluded (regions counted from the grid geometry alone) by reason {'reference_alignment_uncertain': 983040, 'reference_alignment_invalid': 196608}; regions excluded inside eligible tiles {'region_insufficient_valid_pixels': 71385, 'alignment_crop_edge_cells': 3068}; unreadable 0, invalid 0.
`opensr_spot`: tiles excluded (regions counted from the grid geometry alone) by reason {'reference_alignment_uncertain': 49152}; regions excluded inside eligible tiles {'region_insufficient_valid_pixels': 0, 'alignment_crop_edge_cells': 383}; unreadable 0, invalid 0.
`opensr_spain_crops`: tiles excluded (regions counted from the grid geometry alone) by reason {'reference_alignment_uncertain': 114688}; regions excluded inside eligible tiles {'region_insufficient_valid_pixels': 0, 'alignment_crop_edge_cells': 4212}; unreadable 0, invalid 0.
`opensr_spain_urban`: tiles excluded (regions counted from the grid geometry alone) by reason {'reference_alignment_uncertain': 114688}; regions excluded inside eligible tiles {'region_insufficient_valid_pixels': 0, 'alignment_crop_edge_cells': 2552}; unreadable 0, invalid 0.

Each tile's registration (raw displacement, applied translation, residual) is in `tiles.jsonl`; region tables are cached outside the repository (`cache_dir`) with their SHA-256 recorded per tile.

## What was found

### Dataset `sen2neon_random30`

12 eligible tiles from 11 scene units.

**NDVI fidelity and decision utility at 10 m, threshold 0.3** (mean over scene units [95% interval over units]; regions of a unit average first; decision metrics from each unit's pooled pixels):

| system | NDVI MAE | NDVI RMSE | NDVI bias | median |NDVI err| | vegetation-fraction MAE | region decision error | pixel decision disagreement | accuracy | balanced accuracy | false-positive rate | false-negative rate | valid coverage |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `lr_native` | 0.047 [0.025, 0.077] | 0.069 [0.040, 0.106] | -0.002 [-0.007, 0.001] | 0.039 [0.017, 0.072] | 0.055 [0.011, 0.119] | 0.054 [0.010, 0.118] | 0.055 [0.011, 0.118] | 0.945 [0.882, 0.989] | 0.900 [0.842, 0.944] | 0.092 [0.043, 0.142] | 0.079 [0.026, 0.151] | 0.999 [0.999, 1.000] |
| `bicubic` | 0.046 [0.024, 0.076] | 0.067 [0.038, 0.104] | -0.002 [-0.007, 0.001] | 0.037 [0.016, 0.070] | 0.053 [0.009, 0.117] | 0.052 [0.009, 0.116] | 0.053 [0.010, 0.117] | 0.947 [0.883, 0.990] | 0.911 [0.851, 0.956] | 0.079 [0.036, 0.126] | 0.074 [0.023, 0.144] | 0.999 [0.999, 1.000] |
| `sen2sr_lite` | 0.047 [0.025, 0.077] | 0.068 [0.039, 0.105] | -0.002 [-0.007, 0.001] | 0.038 [0.017, 0.071] | 0.055 [0.011, 0.119] | 0.054 [0.010, 0.118] | 0.055 [0.011, 0.119] | 0.945 [0.881, 0.989] | 0.913 [0.853, 0.959] | 0.076 [0.032, 0.128] | 0.072 [0.023, 0.143] | 0.999 [0.999, 1.000] |
| `sen2sr_mamba` | 0.046 [0.024, 0.076] | 0.067 [0.039, 0.103] | -0.003 [-0.007, 0.000] | 0.037 [0.016, 0.070] | 0.055 [0.010, 0.118] | 0.054 [0.010, 0.117] | 0.055 [0.011, 0.119] | 0.945 [0.881, 0.989] | 0.907 [0.848, 0.951] | 0.090 [0.041, 0.141] | 0.072 [0.023, 0.144] | 0.999 [0.999, 1.000] |

**Change relative to bicubic** (paired over the same scene units; a negative difference in an error is a smaller error; not a ranking):

| system − bicubic | units | NDVI MAE | vegetation-fraction MAE | region decision error | pixel decision disagreement | balanced accuracy |
|---|---|---|---|---|---|---|
| `lr_native - bicubic` | 11 | +0.0014 [0.0010, 0.0019] | +0.0022 [0.0007, 0.0041] | +0.0014 [0.0007, 0.0022] | +0.0020 [0.0006, 0.0038] | -0.0108 [-0.0155, -0.0061] |
| `sen2sr_lite - bicubic` | 11 | +0.0011 [0.0007, 0.0015] | +0.0018 [0.0008, 0.0029] | +0.0015 [0.0007, 0.0025] | +0.0019 [0.0009, 0.0030] | +0.0022 [-0.0024, 0.0068] |
| `sen2sr_mamba - bicubic` | 11 | +0.0002 [-0.0008, 0.0012] | +0.0016 [-0.0001, 0.0032] | +0.0011 [-0.0004, 0.0027] | +0.0017 [0.0000, 0.0035] | -0.0047 [-0.0118, 0.0019] |

Against the native LR result, and between models (descriptive):

| A − B | units | NDVI MAE | vegetation-fraction MAE | region decision error | pixel decision disagreement | balanced accuracy |
|---|---|---|---|---|---|---|
| `bicubic - lr_native` | 11 | -0.0014 [-0.0019, -0.0010] | -0.0022 [-0.0041, -0.0007] | -0.0014 [-0.0022, -0.0007] | -0.0020 [-0.0038, -0.0006] | +0.0108 [0.0061, 0.0155] |
| `sen2sr_lite - lr_native` | 11 | -0.0003 [-0.0006, -0.0000] | -0.0003 [-0.0022, 0.0009] | +0.0001 [-0.0002, 0.0007] | -0.0001 [-0.0018, 0.0011] | +0.0130 [0.0061, 0.0204] |
| `sen2sr_mamba - lr_native` | 11 | -0.0012 [-0.0026, 0.0000] | -0.0006 [-0.0023, 0.0010] | -0.0003 [-0.0016, 0.0010] | -0.0002 [-0.0018, 0.0014] | +0.0061 [-0.0004, 0.0127] |
| `sen2sr_lite - sen2sr_mamba` | 11 | +0.0009 [-0.0002, 0.0021] | +0.0003 [-0.0008, 0.0016] | +0.0004 [-0.0005, 0.0015] | +0.0002 [-0.0009, 0.0015] | +0.0069 [0.0009, 0.0150] |

**Threshold sensitivity** (pixel decision disagreement with the reference, mean over units; the primary threshold was fixed in advance and is not chosen among these):

| system | 0.3 (primary) | 0.2 | 0.4 |
|---|---|---|---|
| `lr_native` | 0.055 [0.011, 0.118] | 0.029 [0.009, 0.056] | 0.089 [0.023, 0.167] |
| `bicubic` | 0.053 [0.010, 0.117] | 0.026 [0.009, 0.050] | 0.086 [0.021, 0.164] |
| `sen2sr_lite` | 0.055 [0.011, 0.119] | 0.029 [0.009, 0.056] | 0.088 [0.022, 0.165] |
| `sen2sr_mamba` | 0.055 [0.011, 0.119] | 0.029 [0.009, 0.057] | 0.086 [0.021, 0.165] |

**At the 40 m scale (16 HR px)** (same metrics, threshold 0.3):

| system | NDVI MAE | vegetation-fraction MAE | region decision error | pixel decision disagreement |
|---|---|---|---|---|
| `lr_native` | 0.041 [0.020, 0.069] | 0.053 [0.010, 0.117] | 0.049 [0.007, 0.113] | 0.056 [0.011, 0.119] |
| `bicubic` | 0.040 [0.020, 0.069] | 0.052 [0.009, 0.117] | 0.050 [0.007, 0.113] | 0.054 [0.010, 0.118] |
| `sen2sr_lite` | 0.040 [0.020, 0.069] | 0.054 [0.010, 0.118] | 0.050 [0.007, 0.114] | 0.055 [0.011, 0.120] |
| `sen2sr_mamba` | 0.040 [0.020, 0.068] | 0.053 [0.010, 0.118] | 0.050 [0.007, 0.114] | 0.055 [0.010, 0.119] |

**Is instability associated with downstream error? (within tiles, then over scene units)**

Spearman correlation of a region's stability with its downstream error, mean over units [interval over units]; `texture` and `added detail` are trivial predictors correlated with the same error; `partial` = the stability controlling for texture (and, in the last column, for texture and added detail):

| region scale | model | downstream error | stability | texture | added detail | partial (given texture) | partial (given texture + added detail) | units |
|---|---|---|---|---|---|---|---|---|
| 10 m | `sen2sr_lite` | |NDVI error| | 0.095 [0.030, 0.157] | 0.073 [0.035, 0.111] | 0.077 [0.016, 0.128] | 0.063 [0.006, 0.115] | 0.054 [0.015, 0.095] | 11 |
| 10 m | `sen2sr_lite` | decision disagreement | 0.105 [0.019, 0.194] | 0.117 [0.030, 0.196] | 0.068 [-0.020, 0.153] | 0.029 [-0.037, 0.085] | 0.050 [0.016, 0.084] | 10 |
| 10 m | `sen2sr_mamba` | |NDVI error| | 0.108 [0.034, 0.180] | 0.090 [0.036, 0.152] | 0.105 [0.029, 0.185] | 0.066 [-0.000, 0.126] | 0.046 [0.006, 0.095] | 11 |
| 10 m | `sen2sr_mamba` | decision disagreement | 0.082 [-0.015, 0.180] | 0.113 [0.026, 0.196] | 0.073 [-0.012, 0.159] | 0.008 [-0.069, 0.079] | 0.028 [-0.018, 0.077] | 10 |
| 40 m | `sen2sr_lite` | |NDVI error| | 0.034 [-0.045, 0.116] | 0.032 [-0.025, 0.092] | 0.012 [-0.064, 0.088] | 0.015 [-0.053, 0.080] | 0.037 [-0.013, 0.087] | 11 |
| 40 m | `sen2sr_lite` | decision disagreement | 0.241 [0.122, 0.373] | 0.268 [0.142, 0.397] | 0.187 [0.058, 0.327] | 0.040 [-0.048, 0.113] | 0.116 [0.077, 0.157] | 10 |
| 40 m | `sen2sr_mamba` | |NDVI error| | 0.043 [-0.034, 0.119] | 0.043 [-0.016, 0.113] | 0.029 [-0.045, 0.112] | 0.014 [-0.054, 0.086] | 0.028 [-0.026, 0.090] | 11 |
| 40 m | `sen2sr_mamba` | decision disagreement | 0.182 [0.050, 0.319] | 0.255 [0.135, 0.376] | 0.185 [0.067, 0.309] | -0.014 [-0.111, 0.079] | 0.035 [-0.025, 0.092] | 10 |

Pooled regions (a seeded subsample of each tile; the interval resamples whole scene units) and across tiles (tile mean stability against tile mean error; an interval needs >= 10 units):

| model (10 m) | downstream error | pooled: stability | pooled: texture | pooled: partial | across tiles: stability |
|---|---|---|---|---|---|
| `sen2sr_lite` | |NDVI error| | 0.020 [-0.048, 0.132] | 0.001 [-0.065, 0.097] | 0.027 [-0.043, 0.130] | 0.413 [-0.050, 0.972] |
| `sen2sr_lite` | decision disagreement | 0.031 [-0.053, 0.123] | 0.068 [-0.008, 0.151] | -0.025 [-0.087, 0.054] | 0.035 [-0.672, 0.610] |
| `sen2sr_mamba` | |NDVI error| | -0.036 [-0.101, 0.085] | 0.019 [-0.040, 0.110] | -0.065 [-0.142, 0.051] | 0.105 [-0.443, 0.651] |
| `sen2sr_mamba` | decision disagreement | 0.032 [-0.062, 0.137] | 0.070 [-0.008, 0.157] | -0.019 [-0.103, 0.071] | 0.007 [-0.649, 0.564] |

**Risk-coverage on the downstream error** (within each scene unit, drop the most unstable fraction of regions, report the relative reduction of the remaining error; texture ranked the same way; the oracle is ranked by the true error; **not calibrated selective prediction**), mean over units [interval]:

| model (10 m) | downstream error | retained | reduction, stability order | reduction, texture order | stability − texture (paired over scene units) | reduction, oracle |
|---|---|---|---|---|---|---|
| `sen2sr_lite` | |NDVI error| | 80% kept | 0.057 [0.027, 0.090] | 0.040 [0.015, 0.066] | +0.017 [0.005, 0.030] | 0.401 [0.337, 0.489] |
| `sen2sr_lite` | |NDVI error| | 60% kept | 0.071 [0.012, 0.127] | 0.042 [-0.020, 0.089] | +0.029 [0.009, 0.051] | 0.579 [0.529, 0.641] |
| `sen2sr_lite` | |NDVI error| | 40% kept | 0.067 [-0.056, 0.170] | 0.023 [-0.121, 0.125] | +0.043 [0.011, 0.081] | 0.731 [0.702, 0.770] |
| `sen2sr_lite` | decision disagreement | 80% kept | 0.352 [0.131, 0.605] | 0.267 [0.100, 0.448] | +0.085 [-0.040, 0.271] | 0.948 [0.844, 1.000] |
| `sen2sr_lite` | decision disagreement | 60% kept | 0.412 [0.115, 0.695] | 0.310 [0.047, 0.536] | +0.102 [-0.022, 0.281] | 1.000 [0.999, 1.000] |
| `sen2sr_lite` | decision disagreement | 40% kept | 0.416 [-0.014, 0.772] | 0.289 [-0.107, 0.583] | +0.127 [-0.015, 0.287] | 1.000 [1.000, 1.000] |
| `sen2sr_mamba` | |NDVI error| | 80% kept | 0.049 [0.008, 0.088] | 0.047 [0.016, 0.079] | +0.001 [-0.020, 0.021] | 0.406 [0.341, 0.495] |
| `sen2sr_mamba` | |NDVI error| | 60% kept | 0.063 [-0.025, 0.137] | 0.054 [-0.014, 0.116] | +0.009 [-0.022, 0.041] | 0.585 [0.533, 0.648] |
| `sen2sr_mamba` | |NDVI error| | 40% kept | 0.048 [-0.122, 0.181] | 0.037 [-0.113, 0.154] | +0.011 [-0.027, 0.054] | 0.735 [0.706, 0.774] |
| `sen2sr_mamba` | decision disagreement | 80% kept | 0.314 [0.088, 0.571] | 0.262 [0.100, 0.427] | +0.053 [-0.074, 0.229] | 0.948 [0.844, 1.000] |
| `sen2sr_mamba` | decision disagreement | 60% kept | 0.352 [0.024, 0.666] | 0.284 [0.027, 0.508] | +0.068 [-0.111, 0.326] | 1.000 [0.999, 1.000] |
| `sen2sr_mamba` | decision disagreement | 40% kept | 0.317 [-0.155, 0.713] | 0.348 [-0.076, 0.666] | -0.031 [-0.142, 0.071] | 1.000 [1.000, 1.000] |

### Dataset `opensr_spot`

6 eligible tiles from 6 scene units.

**NDVI fidelity and decision utility at 10 m, threshold 0.3** (mean over scene units [95% interval over units]; regions of a unit average first; decision metrics from each unit's pooled pixels):

| system | NDVI MAE | NDVI RMSE | NDVI bias | median |NDVI err| | vegetation-fraction MAE | region decision error | pixel decision disagreement | accuracy | balanced accuracy | false-positive rate | false-negative rate | valid coverage |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `lr_native` | 0.020 [0.007, 0.036] | 0.028 [0.010, 0.049] | -0.001 [-0.002, -0.000] | 0.015 [0.006, 0.028] | 0.033 [0.001, 0.084] | 0.024 [0.001, 0.059] | 0.033 [0.001, 0.084] | 0.967 [0.916, 0.999] | 0.878 (descriptive only) | 0.026 [0.001, 0.067] | 0.204 (descriptive only) | 1.000 [1.000, 1.000] |
| `bicubic` | 0.018 [0.006, 0.031] | 0.025 [0.009, 0.044] | -0.001 [-0.003, -0.000] | 0.013 [0.005, 0.024] | 0.025 [0.000, 0.061] | 0.023 [0.000, 0.058] | 0.027 [0.000, 0.068] | 0.973 [0.932, 1.000] | 0.893 (descriptive only) | 0.021 [0.000, 0.051] | 0.182 (descriptive only) | 1.000 [1.000, 1.000] |
| `sen2sr_lite` | 0.020 [0.007, 0.036] | 0.028 [0.010, 0.049] | -0.000 [-0.001, 0.000] | 0.015 [0.006, 0.028] | 0.026 [0.001, 0.063] | 0.026 [0.001, 0.064] | 0.029 [0.001, 0.071] | 0.971 [0.929, 0.999] | 0.898 (descriptive only) | 0.025 [0.001, 0.061] | 0.166 (descriptive only) | 1.000 [1.000, 1.000] |
| `sen2sr_mamba` | 0.019 [0.007, 0.033] | 0.027 [0.009, 0.046] | -0.001 [-0.003, 0.000] | 0.014 [0.005, 0.025] | 0.032 [0.001, 0.078] | 0.032 [0.001, 0.081] | 0.036 [0.001, 0.088] | 0.964 [0.912, 0.999] | 0.879 (descriptive only) | 0.031 [0.001, 0.077] | 0.195 (descriptive only) | 1.000 [1.000, 1.000] |

**Change relative to bicubic** (paired over the same scene units; a negative difference in an error is a smaller error; not a ranking):

| system − bicubic | units | NDVI MAE | vegetation-fraction MAE | region decision error | pixel decision disagreement | balanced accuracy |
|---|---|---|---|---|---|---|
| `lr_native - bicubic` | 6 | +0.0023 [0.0008, 0.0041] | +0.0081 [0.0001, 0.0224] | +0.0008 [0.0001, 0.0017] | +0.0056 [0.0001, 0.0156] | -0.0152 (descriptive only) |
| `sen2sr_lite - bicubic` | 6 | +0.0024 [0.0010, 0.0042] | +0.0015 [-0.0001, 0.0043] | +0.0029 [0.0002, 0.0061] | +0.0020 [0.0002, 0.0049] | +0.0045 (descriptive only) |
| `sen2sr_mamba - bicubic` | 6 | +0.0014 [0.0004, 0.0029] | +0.0071 [0.0003, 0.0169] | +0.0091 [0.0002, 0.0233] | +0.0083 [0.0003, 0.0202] | -0.0142 (descriptive only) |

Against the native LR result, and between models (descriptive):

| A − B | units | NDVI MAE | vegetation-fraction MAE | region decision error | pixel decision disagreement | balanced accuracy |
|---|---|---|---|---|---|---|
| `bicubic - lr_native` | 6 | -0.0023 [-0.0041, -0.0008] | -0.0081 [-0.0224, -0.0001] | -0.0008 [-0.0017, -0.0001] | -0.0056 [-0.0156, -0.0001] | +0.0152 (descriptive only) |
| `sen2sr_lite - lr_native` | 6 | +0.0002 [0.0000, 0.0003] | -0.0066 [-0.0216, 0.0017] | +0.0021 [0.0001, 0.0047] | -0.0036 [-0.0139, 0.0028] | +0.0197 (descriptive only) |
| `sen2sr_mamba - lr_native` | 6 | -0.0008 [-0.0027, 0.0004] | -0.0010 [-0.0065, 0.0032] | +0.0082 [0.0001, 0.0220] | +0.0026 [0.0002, 0.0055] | +0.0010 (descriptive only) |
| `sen2sr_lite - sen2sr_mamba` | 6 | +0.0010 [-0.0001, 0.0027] | -0.0056 [-0.0156, -0.0001] | -0.0062 [-0.0173, -0.0001] | -0.0063 [-0.0175, -0.0001] | +0.0186 (descriptive only) |

**Threshold sensitivity** (pixel decision disagreement with the reference, mean over units; the primary threshold was fixed in advance and is not chosen among these):

| system | 0.3 (primary) | 0.2 | 0.4 |
|---|---|---|---|
| `lr_native` | 0.033 [0.001, 0.084] | 0.061 [0.016, 0.110] | 0.026 [0.000, 0.071] |
| `bicubic` | 0.027 [0.000, 0.068] | 0.053 [0.014, 0.096] | 0.023 [0.000, 0.061] |
| `sen2sr_lite` | 0.029 [0.001, 0.071] | 0.058 [0.016, 0.104] | 0.026 [0.000, 0.070] |
| `sen2sr_mamba` | 0.036 [0.001, 0.088] | 0.059 [0.017, 0.105] | 0.026 [0.000, 0.069] |

**At the 40 m scale (16 HR px)** (same metrics, threshold 0.3):

| system | NDVI MAE | vegetation-fraction MAE | region decision error | pixel decision disagreement |
|---|---|---|---|---|
| `lr_native` | 0.013 [0.004, 0.022] | 0.019 [0.001, 0.045] | 0.020 [0.000, 0.056] | 0.033 [0.001, 0.084] |
| `bicubic` | 0.012 [0.004, 0.022] | 0.018 [0.000, 0.042] | 0.017 [0.000, 0.048] | 0.027 [0.000, 0.068] |
| `sen2sr_lite` | 0.012 [0.004, 0.022] | 0.018 [0.001, 0.038] | 0.017 [0.000, 0.049] | 0.029 [0.001, 0.071] |
| `sen2sr_mamba` | 0.012 [0.004, 0.021] | 0.021 [0.001, 0.048] | 0.020 [0.000, 0.058] | 0.036 [0.001, 0.088] |

**Is instability associated with downstream error? (within tiles, then over scene units)**

Spearman correlation of a region's stability with its downstream error, mean over units [interval over units]; `texture` and `added detail` are trivial predictors correlated with the same error; `partial` = the stability controlling for texture (and, in the last column, for texture and added detail):

| region scale | model | downstream error | stability | texture | added detail | partial (given texture) | partial (given texture + added detail) | units |
|---|---|---|---|---|---|---|---|---|
| 10 m | `sen2sr_lite` | |NDVI error| | 0.143 [0.037, 0.263] | 0.103 [0.013, 0.201] | 0.158 [0.061, 0.260] | 0.109 [0.033, 0.183] | 0.054 [-0.000, 0.108] | 6 |
| 10 m | `sen2sr_lite` | decision disagreement | 0.145 (descriptive only) | 0.126 (descriptive only) | 0.144 (descriptive only) | 0.074 (descriptive only) | 0.042 (descriptive only) | 4 |
| 10 m | `sen2sr_mamba` | |NDVI error| | 0.183 [0.057, 0.312] | 0.123 [0.033, 0.221] | 0.173 [0.068, 0.288] | 0.142 [0.047, 0.238] | 0.087 [0.018, 0.154] | 6 |
| 10 m | `sen2sr_mamba` | decision disagreement | 0.154 (descriptive only) | 0.141 (descriptive only) | 0.174 (descriptive only) | 0.075 (descriptive only) | 0.012 (descriptive only) | 4 |
| 40 m | `sen2sr_lite` | |NDVI error| | 0.111 [-0.063, 0.277] | 0.067 [-0.088, 0.211] | 0.105 [-0.083, 0.275] | 0.110 [0.003, 0.209] | 0.061 [-0.012, 0.116] | 6 |
| 40 m | `sen2sr_lite` | decision disagreement | 0.277 (descriptive only) | 0.247 (descriptive only) | 0.282 (descriptive only) | 0.148 (descriptive only) | 0.067 (descriptive only) | 4 |
| 40 m | `sen2sr_mamba` | |NDVI error| | 0.123 [-0.066, 0.296] | 0.075 [-0.080, 0.210] | 0.110 [-0.074, 0.278] | 0.113 [-0.028, 0.243] | 0.066 [-0.036, 0.162] | 6 |
| 40 m | `sen2sr_mamba` | decision disagreement | 0.304 (descriptive only) | 0.268 (descriptive only) | 0.310 (descriptive only) | 0.158 (descriptive only) | 0.065 (descriptive only) | 4 |

Pooled regions (a seeded subsample of each tile; the interval resamples whole scene units) and across tiles (tile mean stability against tile mean error; an interval needs >= 10 units):

| model (10 m) | downstream error | pooled: stability | pooled: texture | pooled: partial | across tiles: stability |
|---|---|---|---|---|---|
| `sen2sr_lite` | |NDVI error| | 0.527 [0.181, 0.634] | 0.492 [0.174, 0.602] | 0.243 [0.079, 0.337] | 0.943 (descriptive only) |
| `sen2sr_lite` | decision disagreement | 0.417 [0.063, 0.504] | 0.384 [0.054, 0.476] | 0.191 [0.032, 0.231] | 0.928 (descriptive only) |
| `sen2sr_mamba` | |NDVI error| | 0.558 [0.259, 0.633] | 0.504 [0.194, 0.616] | 0.295 [0.162, 0.344] | 0.943 (descriptive only) |
| `sen2sr_mamba` | decision disagreement | 0.392 [0.075, 0.478] | 0.404 [0.060, 0.507] | 0.120 [0.014, 0.182] | 0.928 (descriptive only) |

**Risk-coverage on the downstream error** (within each scene unit, drop the most unstable fraction of regions, report the relative reduction of the remaining error; texture ranked the same way; the oracle is ranked by the true error; **not calibrated selective prediction**), mean over units [interval]:

| model (10 m) | downstream error | retained | reduction, stability order | reduction, texture order | stability − texture (paired over scene units) | reduction, oracle |
|---|---|---|---|---|---|---|
| `sen2sr_lite` | |NDVI error| | 80% kept | 0.058 [0.023, 0.093] | 0.042 [0.014, 0.069] | +0.016 [0.001, 0.030] | 0.353 [0.318, 0.398] |
| `sen2sr_lite` | |NDVI error| | 60% kept | 0.111 [0.042, 0.184] | 0.075 [0.022, 0.128] | +0.036 [0.014, 0.057] | 0.548 [0.514, 0.591] |
| `sen2sr_lite` | |NDVI error| | 40% kept | 0.161 [0.056, 0.281] | 0.108 [0.022, 0.201] | +0.054 [0.019, 0.086] | 0.709 [0.684, 0.741] |
| `sen2sr_lite` | decision disagreement | 80% kept | 0.543 (descriptive only) | 0.352 (descriptive only) | +0.191 (descriptive only) | 0.903 (descriptive only) |
| `sen2sr_lite` | decision disagreement | 60% kept | 0.636 (descriptive only) | 0.520 (descriptive only) | +0.116 (descriptive only) | 0.980 (descriptive only) |
| `sen2sr_lite` | decision disagreement | 40% kept | 0.711 (descriptive only) | 0.565 (descriptive only) | +0.146 (descriptive only) | 1.000 (descriptive only) |
| `sen2sr_mamba` | |NDVI error| | 80% kept | 0.077 [0.036, 0.121] | 0.052 [0.025, 0.077] | +0.025 [0.002, 0.053] | 0.361 [0.325, 0.405] |
| `sen2sr_mamba` | |NDVI error| | 60% kept | 0.135 [0.059, 0.209] | 0.091 [0.038, 0.144] | +0.044 [0.017, 0.073] | 0.556 [0.521, 0.599] |
| `sen2sr_mamba` | |NDVI error| | 40% kept | 0.192 [0.065, 0.329] | 0.129 [0.044, 0.222] | +0.064 [0.007, 0.113] | 0.716 [0.690, 0.748] |
| `sen2sr_mamba` | decision disagreement | 80% kept | 0.531 (descriptive only) | 0.346 (descriptive only) | +0.184 (descriptive only) | 0.893 (descriptive only) |
| `sen2sr_mamba` | decision disagreement | 60% kept | 0.600 (descriptive only) | 0.524 (descriptive only) | +0.075 (descriptive only) | 0.971 (descriptive only) |
| `sen2sr_mamba` | decision disagreement | 40% kept | 0.692 (descriptive only) | 0.588 (descriptive only) | +0.103 (descriptive only) | 1.000 (descriptive only) |

### Dataset `opensr_spain_crops`

21 eligible tiles from 5 scene units.

**NDVI fidelity and decision utility at 10 m, threshold 0.3** (mean over scene units [95% interval over units]; regions of a unit average first; decision metrics from each unit's pooled pixels):

| system | NDVI MAE | NDVI RMSE | NDVI bias | median |NDVI err| | vegetation-fraction MAE | region decision error | pixel decision disagreement | accuracy | balanced accuracy | false-positive rate | false-negative rate | valid coverage |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `lr_native` | 0.038 [0.025, 0.051] | 0.051 [0.037, 0.065] | -0.004 [-0.007, -0.001] | 0.030 [0.017, 0.044] | 0.086 [0.043, 0.130] | 0.064 [0.030, 0.101] | 0.086 [0.043, 0.130] | 0.914 [0.870, 0.957] | 0.832 [0.773, 0.868] | 0.124 [0.052, 0.220] | 0.211 [0.076, 0.365] | 1.000 [1.000, 1.000] |
| `bicubic` | 0.038 [0.025, 0.052] | 0.051 [0.036, 0.066] | -0.004 [-0.007, -0.001] | 0.030 [0.017, 0.045] | 0.077 [0.037, 0.119] | 0.066 [0.031, 0.104] | 0.083 [0.040, 0.126] | 0.917 [0.874, 0.960] | 0.838 [0.778, 0.874] | 0.119 [0.050, 0.211] | 0.204 [0.071, 0.358] | 1.000 [1.000, 1.000] |
| `sen2sr_lite` | 0.037 [0.025, 0.050] | 0.050 [0.036, 0.064] | -0.003 [-0.007, -0.001] | 0.029 [0.017, 0.044] | 0.074 [0.034, 0.115] | 0.064 [0.029, 0.100] | 0.080 [0.038, 0.123] | 0.920 [0.877, 0.962] | 0.851 [0.787, 0.886] | 0.104 [0.051, 0.171] | 0.195 [0.067, 0.345] | 1.000 [1.000, 1.000] |
| `sen2sr_mamba` | 0.038 [0.025, 0.051] | 0.051 [0.036, 0.066] | -0.004 [-0.007, -0.001] | 0.029 [0.017, 0.044] | 0.073 [0.034, 0.115] | 0.065 [0.030, 0.103] | 0.080 [0.038, 0.122] | 0.920 [0.878, 0.962] | 0.851 [0.789, 0.887] | 0.107 [0.051, 0.181] | 0.190 [0.064, 0.339] | 1.000 [1.000, 1.000] |

**Change relative to bicubic** (paired over the same scene units; a negative difference in an error is a smaller error; not a ranking):

| system − bicubic | units | NDVI MAE | vegetation-fraction MAE | region decision error | pixel decision disagreement | balanced accuracy |
|---|---|---|---|---|---|---|
| `lr_native - bicubic` | 5 | -0.0001 [-0.0005, 0.0003] | +0.0092 [0.0045, 0.0149] | -0.0019 [-0.0037, -0.0002] | +0.0036 [0.0018, 0.0062] | -0.0059 [-0.0073, -0.0044] |
| `sen2sr_lite - bicubic` | 5 | -0.0007 [-0.0016, -0.0000] | -0.0036 [-0.0077, -0.0007] | -0.0025 [-0.0061, -0.0002] | -0.0027 [-0.0064, -0.0003] | +0.0125 [0.0070, 0.0209] |
| `sen2sr_mamba - bicubic` | 5 | -0.0002 [-0.0005, 0.0001] | -0.0043 [-0.0079, -0.0015] | -0.0015 [-0.0039, 0.0002] | -0.0033 [-0.0065, -0.0009] | +0.0131 [0.0088, 0.0185] |

Against the native LR result, and between models (descriptive):

| A − B | units | NDVI MAE | vegetation-fraction MAE | region decision error | pixel decision disagreement | balanced accuracy |
|---|---|---|---|---|---|---|
| `bicubic - lr_native` | 5 | +0.0001 [-0.0003, 0.0005] | -0.0092 [-0.0149, -0.0045] | +0.0019 [0.0002, 0.0037] | -0.0036 [-0.0062, -0.0018] | +0.0059 [0.0044, 0.0073] |
| `sen2sr_lite - lr_native` | 5 | -0.0006 [-0.0011, -0.0001] | -0.0128 [-0.0226, -0.0051] | -0.0006 [-0.0026, 0.0007] | -0.0064 [-0.0126, -0.0021] | +0.0184 [0.0114, 0.0276] |
| `sen2sr_mamba - lr_native` | 5 | -0.0001 [-0.0003, 0.0001] | -0.0135 [-0.0229, -0.0058] | +0.0004 [-0.0006, 0.0020] | -0.0069 [-0.0127, -0.0027] | +0.0190 [0.0135, 0.0253] |
| `sen2sr_lite - sen2sr_mamba` | 5 | -0.0004 [-0.0012, 0.0001] | +0.0007 [0.0002, 0.0012] | -0.0011 [-0.0024, 0.0002] | +0.0005 [0.0001, 0.0009] | -0.0006 [-0.0039, 0.0032] |

**Threshold sensitivity** (pixel decision disagreement with the reference, mean over units; the primary threshold was fixed in advance and is not chosen among these):

| system | 0.3 (primary) | 0.2 | 0.4 |
|---|---|---|---|
| `lr_native` | 0.086 [0.043, 0.130] | 0.117 [0.051, 0.184] | 0.055 [0.026, 0.090] |
| `bicubic` | 0.083 [0.040, 0.126] | 0.114 [0.050, 0.181] | 0.052 [0.024, 0.083] |
| `sen2sr_lite` | 0.080 [0.038, 0.123] | 0.112 [0.049, 0.176] | 0.050 [0.024, 0.078] |
| `sen2sr_mamba` | 0.080 [0.038, 0.122] | 0.111 [0.049, 0.171] | 0.050 [0.024, 0.080] |

**At the 40 m scale (16 HR px)** (same metrics, threshold 0.3):

| system | NDVI MAE | vegetation-fraction MAE | region decision error | pixel decision disagreement |
|---|---|---|---|---|
| `lr_native` | 0.031 [0.020, 0.042] | 0.064 [0.028, 0.104] | 0.054 [0.021, 0.089] | 0.087 [0.043, 0.131] |
| `bicubic` | 0.031 [0.020, 0.042] | 0.064 [0.028, 0.104] | 0.049 [0.019, 0.080] | 0.083 [0.040, 0.127] |
| `sen2sr_lite` | 0.031 [0.020, 0.041] | 0.061 [0.026, 0.099] | 0.050 [0.019, 0.082] | 0.081 [0.038, 0.123] |
| `sen2sr_mamba` | 0.031 [0.020, 0.041] | 0.061 [0.026, 0.100] | 0.050 [0.019, 0.081] | 0.080 [0.038, 0.123] |

**Is instability associated with downstream error? (within tiles, then over scene units)**

Spearman correlation of a region's stability with its downstream error, mean over units [interval over units]; `texture` and `added detail` are trivial predictors correlated with the same error; `partial` = the stability controlling for texture (and, in the last column, for texture and added detail):

| region scale | model | downstream error | stability | texture | added detail | partial (given texture) | partial (given texture + added detail) | units |
|---|---|---|---|---|---|---|---|---|
| 10 m | `sen2sr_lite` | |NDVI error| | 0.039 [-0.087, 0.165] | 0.035 [-0.075, 0.156] | 0.052 [-0.064, 0.168] | 0.028 [-0.027, 0.090] | 0.004 [-0.035, 0.053] | 5 |
| 10 m | `sen2sr_lite` | decision disagreement | 0.151 [-0.009, 0.287] | 0.144 [-0.003, 0.269] | 0.193 [0.050, 0.308] | 0.066 [-0.004, 0.125] | -0.015 [-0.077, 0.048] | 5 |
| 10 m | `sen2sr_mamba` | |NDVI error| | 0.058 [-0.066, 0.182] | 0.049 [-0.065, 0.166] | 0.063 [-0.040, 0.172] | 0.040 [-0.021, 0.115] | 0.013 [-0.044, 0.077] | 5 |
| 10 m | `sen2sr_mamba` | decision disagreement | 0.216 [0.039, 0.368] | 0.145 [-0.002, 0.270] | 0.203 [0.066, 0.320] | 0.170 [0.067, 0.273] | 0.095 [-0.019, 0.197] | 5 |
| 40 m | `sen2sr_lite` | |NDVI error| | -0.047 [-0.223, 0.134] | -0.033 [-0.187, 0.127] | -0.045 [-0.228, 0.138] | -0.027 [-0.125, 0.060] | 0.003 [-0.018, 0.024] | 5 |
| 40 m | `sen2sr_lite` | decision disagreement | 0.278 [0.030, 0.470] | 0.253 [0.012, 0.435] | 0.327 [0.085, 0.501] | 0.137 [0.067, 0.200] | -0.052 [-0.115, 0.044] | 5 |
| 40 m | `sen2sr_mamba` | |NDVI error| | -0.040 [-0.194, 0.144] | -0.032 [-0.182, 0.125] | -0.045 [-0.210, 0.131] | -0.016 [-0.098, 0.071] | 0.009 [-0.087, 0.089] | 5 |
| 40 m | `sen2sr_mamba` | decision disagreement | 0.338 [0.085, 0.534] | 0.255 [0.017, 0.440] | 0.337 [0.111, 0.506] | 0.246 [0.121, 0.371] | 0.104 [-0.017, 0.217] | 5 |

Pooled regions (a seeded subsample of each tile; the interval resamples whole scene units) and across tiles (tile mean stability against tile mean error; an interval needs >= 10 units):

| model (10 m) | downstream error | pooled: stability | pooled: texture | pooled: partial | across tiles: stability |
|---|---|---|---|---|---|
| `sen2sr_lite` | |NDVI error| | 0.148 [0.111, 0.282] | 0.115 [0.084, 0.244] | 0.094 [0.059, 0.186] | 0.239 (descriptive only) |
| `sen2sr_lite` | decision disagreement | 0.238 [0.094, 0.373] | 0.223 [0.098, 0.334] | 0.110 [0.026, 0.214] | 0.552 (descriptive only) |
| `sen2sr_mamba` | |NDVI error| | 0.141 [0.111, 0.265] | 0.122 [0.090, 0.264] | 0.076 [0.040, 0.193] | 0.106 (descriptive only) |
| `sen2sr_mamba` | decision disagreement | 0.320 [0.217, 0.422] | 0.223 [0.100, 0.333] | 0.236 [0.177, 0.310] | 0.695 (descriptive only) |

**Risk-coverage on the downstream error** (within each scene unit, drop the most unstable fraction of regions, report the relative reduction of the remaining error; texture ranked the same way; the oracle is ranked by the true error; **not calibrated selective prediction**), mean over units [interval]:

| model (10 m) | downstream error | retained | reduction, stability order | reduction, texture order | stability − texture (paired over scene units) | reduction, oracle |
|---|---|---|---|---|---|---|
| `sen2sr_lite` | |NDVI error| | 80% kept | 0.067 [0.003, 0.129] | 0.056 [0.004, 0.112] | +0.011 [-0.002, 0.030] | 0.370 [0.313, 0.421] |
| `sen2sr_lite` | |NDVI error| | 60% kept | 0.087 [-0.017, 0.191] | 0.072 [-0.015, 0.159] | +0.015 [-0.004, 0.042] | 0.559 [0.494, 0.609] |
| `sen2sr_lite` | |NDVI error| | 40% kept | 0.096 [-0.031, 0.235] | 0.064 [-0.058, 0.191] | +0.032 [0.013, 0.052] | 0.715 [0.659, 0.756] |
| `sen2sr_lite` | decision disagreement | 80% kept | 0.220 [-0.013, 0.552] | 0.186 [-0.011, 0.489] | +0.035 [-0.007, 0.076] | 0.885 [0.777, 0.992] |
| `sen2sr_lite` | decision disagreement | 60% kept | 0.318 [-0.014, 0.713] | 0.257 [-0.017, 0.593] | +0.061 [-0.009, 0.159] | 0.999 [0.996, 1.000] |
| `sen2sr_lite` | decision disagreement | 40% kept | 0.402 [0.014, 0.820] | 0.352 [-0.010, 0.719] | +0.050 [-0.010, 0.128] | 1.000 [1.000, 1.000] |
| `sen2sr_mamba` | |NDVI error| | 80% kept | 0.073 [0.008, 0.138] | 0.060 [0.010, 0.119] | +0.013 [-0.017, 0.054] | 0.373 [0.317, 0.423] |
| `sen2sr_mamba` | |NDVI error| | 60% kept | 0.093 [-0.012, 0.203] | 0.080 [-0.006, 0.166] | +0.013 [-0.017, 0.058] | 0.562 [0.500, 0.611] |
| `sen2sr_mamba` | |NDVI error| | 40% kept | 0.099 [-0.035, 0.232] | 0.076 [-0.047, 0.199] | +0.022 [-0.002, 0.055] | 0.717 [0.664, 0.758] |
| `sen2sr_mamba` | decision disagreement | 80% kept | 0.302 [0.062, 0.591] | 0.187 [-0.010, 0.489] | +0.116 [0.035, 0.226] | 0.885 [0.778, 0.992] |
| `sen2sr_mamba` | decision disagreement | 60% kept | 0.408 [0.066, 0.780] | 0.258 [-0.013, 0.591] | +0.151 [0.047, 0.276] | 0.998 [0.995, 1.000] |
| `sen2sr_mamba` | decision disagreement | 40% kept | 0.462 [0.064, 0.851] | 0.353 [-0.009, 0.720] | +0.108 [0.034, 0.191] | 1.000 [1.000, 1.000] |

### Dataset `opensr_spain_urban`

13 eligible tiles from 4 scene units. **Descriptive only (fewer than 5 scene units).**

**NDVI fidelity and decision utility at 10 m, threshold 0.3** (mean over scene units [95% interval over units]; regions of a unit average first; decision metrics from each unit's pooled pixels):

| system | NDVI MAE | NDVI RMSE | NDVI bias | median |NDVI err| | vegetation-fraction MAE | region decision error | pixel decision disagreement | accuracy | balanced accuracy | false-positive rate | false-negative rate | valid coverage |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `lr_native` | 0.041 (descriptive only) | 0.056 (descriptive only) | -0.002 (descriptive only) | 0.031 (descriptive only) | 0.086 (descriptive only) | 0.060 (descriptive only) | 0.086 (descriptive only) | 0.914 (descriptive only) | 0.864 (descriptive only) | 0.104 (descriptive only) | 0.169 (descriptive only) | 1.000 (descriptive only) |
| `bicubic` | 0.041 (descriptive only) | 0.056 (descriptive only) | -0.002 (descriptive only) | 0.030 (descriptive only) | 0.074 (descriptive only) | 0.063 (descriptive only) | 0.081 (descriptive only) | 0.919 (descriptive only) | 0.872 (descriptive only) | 0.099 (descriptive only) | 0.158 (descriptive only) | 1.000 (descriptive only) |
| `sen2sr_lite` | 0.040 (descriptive only) | 0.055 (descriptive only) | -0.002 (descriptive only) | 0.030 (descriptive only) | 0.068 (descriptive only) | 0.060 (descriptive only) | 0.077 (descriptive only) | 0.923 (descriptive only) | 0.884 (descriptive only) | 0.093 (descriptive only) | 0.139 (descriptive only) | 1.000 (descriptive only) |
| `sen2sr_mamba` | 0.041 (descriptive only) | 0.056 (descriptive only) | -0.002 (descriptive only) | 0.031 (descriptive only) | 0.069 (descriptive only) | 0.063 (descriptive only) | 0.078 (descriptive only) | 0.922 (descriptive only) | 0.883 (descriptive only) | 0.099 (descriptive only) | 0.135 (descriptive only) | 1.000 (descriptive only) |

**Change relative to bicubic** (paired over the same scene units; a negative difference in an error is a smaller error; not a ranking):

| system − bicubic | units | NDVI MAE | vegetation-fraction MAE | region decision error | pixel decision disagreement | balanced accuracy |
|---|---|---|---|---|---|---|
| `lr_native - bicubic` | 4 | +0.0001 (descriptive only) | +0.0126 (descriptive only) | -0.0022 (descriptive only) | +0.0053 (descriptive only) | -0.0080 (descriptive only) |
| `sen2sr_lite - bicubic` | 4 | -0.0006 (descriptive only) | -0.0051 (descriptive only) | -0.0024 (descriptive only) | -0.0042 (descriptive only) | +0.0124 (descriptive only) |
| `sen2sr_mamba - bicubic` | 4 | +0.0003 (descriptive only) | -0.0045 (descriptive only) | +0.0000 (descriptive only) | -0.0028 (descriptive only) | +0.0110 (descriptive only) |

Against the native LR result, and between models (descriptive):

| A − B | units | NDVI MAE | vegetation-fraction MAE | region decision error | pixel decision disagreement | balanced accuracy |
|---|---|---|---|---|---|---|
| `bicubic - lr_native` | 4 | -0.0001 (descriptive only) | -0.0126 (descriptive only) | +0.0022 (descriptive only) | -0.0053 (descriptive only) | +0.0080 (descriptive only) |
| `sen2sr_lite - lr_native` | 4 | -0.0007 (descriptive only) | -0.0177 (descriptive only) | -0.0003 (descriptive only) | -0.0095 (descriptive only) | +0.0203 (descriptive only) |
| `sen2sr_mamba - lr_native` | 4 | +0.0001 (descriptive only) | -0.0171 (descriptive only) | +0.0022 (descriptive only) | -0.0081 (descriptive only) | +0.0190 (descriptive only) |
| `sen2sr_lite - sen2sr_mamba` | 4 | -0.0009 (descriptive only) | -0.0006 (descriptive only) | -0.0025 (descriptive only) | -0.0014 (descriptive only) | +0.0014 (descriptive only) |

**Threshold sensitivity** (pixel decision disagreement with the reference, mean over units; the primary threshold was fixed in advance and is not chosen among these):

| system | 0.3 (primary) | 0.2 | 0.4 |
|---|---|---|---|
| `lr_native` | 0.086 (descriptive only) | 0.144 (descriptive only) | 0.065 (descriptive only) |
| `bicubic` | 0.081 (descriptive only) | 0.140 (descriptive only) | 0.061 (descriptive only) |
| `sen2sr_lite` | 0.077 (descriptive only) | 0.135 (descriptive only) | 0.059 (descriptive only) |
| `sen2sr_mamba` | 0.078 (descriptive only) | 0.134 (descriptive only) | 0.061 (descriptive only) |

**At the 40 m scale (16 HR px)** (same metrics, threshold 0.3):

| system | NDVI MAE | vegetation-fraction MAE | region decision error | pixel decision disagreement |
|---|---|---|---|---|
| `lr_native` | 0.033 (descriptive only) | 0.058 (descriptive only) | 0.057 (descriptive only) | 0.087 (descriptive only) |
| `bicubic` | 0.033 (descriptive only) | 0.057 (descriptive only) | 0.049 (descriptive only) | 0.081 (descriptive only) |
| `sen2sr_lite` | 0.032 (descriptive only) | 0.052 (descriptive only) | 0.047 (descriptive only) | 0.077 (descriptive only) |
| `sen2sr_mamba` | 0.032 (descriptive only) | 0.054 (descriptive only) | 0.049 (descriptive only) | 0.079 (descriptive only) |

**Is instability associated with downstream error? (within tiles, then over scene units)**

Spearman correlation of a region's stability with its downstream error, mean over units [interval over units]; `texture` and `added detail` are trivial predictors correlated with the same error; `partial` = the stability controlling for texture (and, in the last column, for texture and added detail):

| region scale | model | downstream error | stability | texture | added detail | partial (given texture) | partial (given texture + added detail) | units |
|---|---|---|---|---|---|---|---|---|
| 10 m | `sen2sr_lite` | |NDVI error| | 0.165 (descriptive only) | 0.125 (descriptive only) | 0.174 (descriptive only) | 0.110 (descriptive only) | 0.044 (descriptive only) | 4 |
| 10 m | `sen2sr_lite` | decision disagreement | 0.235 (descriptive only) | 0.212 (descriptive only) | 0.270 (descriptive only) | 0.113 (descriptive only) | 0.011 (descriptive only) | 4 |
| 10 m | `sen2sr_mamba` | |NDVI error| | 0.185 (descriptive only) | 0.144 (descriptive only) | 0.198 (descriptive only) | 0.118 (descriptive only) | 0.038 (descriptive only) | 4 |
| 10 m | `sen2sr_mamba` | decision disagreement | 0.304 (descriptive only) | 0.213 (descriptive only) | 0.296 (descriptive only) | 0.223 (descriptive only) | 0.106 (descriptive only) | 4 |
| 40 m | `sen2sr_lite` | |NDVI error| | 0.112 (descriptive only) | 0.076 (descriptive only) | 0.113 (descriptive only) | 0.093 (descriptive only) | 0.041 (descriptive only) | 4 |
| 40 m | `sen2sr_lite` | decision disagreement | 0.398 (descriptive only) | 0.344 (descriptive only) | 0.446 (descriptive only) | 0.218 (descriptive only) | -0.021 (descriptive only) | 4 |
| 40 m | `sen2sr_mamba` | |NDVI error| | 0.108 (descriptive only) | 0.078 (descriptive only) | 0.112 (descriptive only) | 0.077 (descriptive only) | 0.019 (descriptive only) | 4 |
| 40 m | `sen2sr_mamba` | decision disagreement | 0.440 (descriptive only) | 0.343 (descriptive only) | 0.449 (descriptive only) | 0.295 (descriptive only) | 0.084 (descriptive only) | 4 |

Pooled regions (a seeded subsample of each tile; the interval resamples whole scene units) and across tiles (tile mean stability against tile mean error; an interval needs >= 10 units):

| model (10 m) | downstream error | pooled: stability | pooled: texture | pooled: partial | across tiles: stability |
|---|---|---|---|---|---|
| `sen2sr_lite` | |NDVI error| | 0.133 (descriptive only) | 0.100 (descriptive only) | 0.090 (descriptive only) | 0.038 (descriptive only) |
| `sen2sr_lite` | decision disagreement | 0.331 (descriptive only) | 0.315 (descriptive only) | 0.133 (descriptive only) | 0.462 (descriptive only) |
| `sen2sr_mamba` | |NDVI error| | 0.135 (descriptive only) | 0.114 (descriptive only) | 0.074 (descriptive only) | 0.011 (descriptive only) |
| `sen2sr_mamba` | decision disagreement | 0.400 (descriptive only) | 0.319 (descriptive only) | 0.255 (descriptive only) | 0.390 (descriptive only) |

**Risk-coverage on the downstream error** (within each scene unit, drop the most unstable fraction of regions, report the relative reduction of the remaining error; texture ranked the same way; the oracle is ranked by the true error; **not calibrated selective prediction**), mean over units [interval]:

| model (10 m) | downstream error | retained | reduction, stability order | reduction, texture order | stability − texture (paired over scene units) | reduction, oracle |
|---|---|---|---|---|---|---|
| `sen2sr_lite` | |NDVI error| | 80% kept | 0.075 (descriptive only) | 0.050 (descriptive only) | +0.025 (descriptive only) | 0.368 (descriptive only) |
| `sen2sr_lite` | |NDVI error| | 60% kept | 0.129 (descriptive only) | 0.091 (descriptive only) | +0.038 (descriptive only) | 0.556 (descriptive only) |
| `sen2sr_lite` | |NDVI error| | 40% kept | 0.170 (descriptive only) | 0.139 (descriptive only) | +0.031 (descriptive only) | 0.711 (descriptive only) |
| `sen2sr_lite` | decision disagreement | 80% kept | 0.161 (descriptive only) | 0.129 (descriptive only) | +0.031 (descriptive only) | 0.904 (descriptive only) |
| `sen2sr_lite` | decision disagreement | 60% kept | 0.301 (descriptive only) | 0.256 (descriptive only) | +0.044 (descriptive only) | 1.000 (descriptive only) |
| `sen2sr_lite` | decision disagreement | 40% kept | 0.407 (descriptive only) | 0.407 (descriptive only) | +0.001 (descriptive only) | 1.000 (descriptive only) |
| `sen2sr_mamba` | |NDVI error| | 80% kept | 0.083 (descriptive only) | 0.056 (descriptive only) | +0.027 (descriptive only) | 0.371 (descriptive only) |
| `sen2sr_mamba` | |NDVI error| | 60% kept | 0.139 (descriptive only) | 0.104 (descriptive only) | +0.035 (descriptive only) | 0.560 (descriptive only) |
| `sen2sr_mamba` | |NDVI error| | 40% kept | 0.185 (descriptive only) | 0.155 (descriptive only) | +0.030 (descriptive only) | 0.714 (descriptive only) |
| `sen2sr_mamba` | decision disagreement | 80% kept | 0.230 (descriptive only) | 0.133 (descriptive only) | +0.097 (descriptive only) | 0.903 (descriptive only) |
| `sen2sr_mamba` | decision disagreement | 60% kept | 0.367 (descriptive only) | 0.264 (descriptive only) | +0.103 (descriptive only) | 1.000 (descriptive only) |
| `sen2sr_mamba` | decision disagreement | 40% kept | 0.503 (descriptive only) | 0.418 (descriptive only) | +0.085 (descriptive only) | 1.000 (descriptive only) |

## What was not shown

* **Calibrated uncertainty.** Phase 6 found the TTA spread uncalibrated (16-32x smaller than the error); nothing here changes that. Any association above is an ordering, not a probability, confidence or interval.
* **Causal relationships.** Texture, brightness and the amount of detail added are confounded with both instability and error; a partial correlation reduces that, it does not remove it.
* **Indian generalisation.** `india_downstream_validation_unavailable`: no Indian HR reference and no Indian downstream labels exist in this repository; nothing Indian was evaluated or claimed.
* **Land-cover performance.** `secondary_landcover_task_deferred_no_supported_reference`: no dataset used here carries region-level land-cover labels or a defensible reference segmentation; the only label (NEON land-cover superclass) describes a whole tile. The thresholded NDVI decision is a vegetation proxy, not land cover, and its agreement is not a classification accuracy against labels.
* **A downstream advantage of super-resolution, or of any system over another.** Values and paired differences are reported with their uncertainty; where an interval includes zero, or there are too few scene units for one, no advantage is claimed. No ranking of systems is made.
* **Behaviour on misregistered evidence.** Tiles that failed the reference gate were excluded and are not used for any conclusion.

## Limitations

* **Few eligible scene units.** The registration gate leaves few units per dataset (see the evidence table); several tables are descriptive only, and dev/test protocols were not run.
* **Registration residual.** Eligible tiles carry a residual misregistration of up to the gate tolerance, and OpenSR-Test tiles needed a recorded whole-pixel translation that assumes one global displacement; the reference is a different sensor with its own radiometry.
* **Geographic scope.** North America (SEN2NEON) and Spain (OpenSR-Test); no Indian reference exists here.
* **Texture, brightness and detail confounding.** The trivial baselines and the partial correlations are controls, not a proof of independence.
* **A threshold is a choice.** 0.3 was fixed in advance from convention; the sensitivity thresholds show how much the decision depends on it. Decision metrics inherit the reference's own classification noise (it is not ground truth).
* **One index, one decision rule, one perturbation set, two models.** No other index, task or ensemble was tested.

Files: `config.json`, `tiles.jsonl` (one row per tile: registration, region counts, exclusion reason), `summary.json` (provenance, evidence, cost), `downstream_metrics.json` (per unit and across units, paired differences), `association.json`, `risk_coverage.json`.
