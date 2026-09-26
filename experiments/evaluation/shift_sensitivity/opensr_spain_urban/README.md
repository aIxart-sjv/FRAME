# Spatial-shift sensitivity: `opensr_spain_urban`

Status: **completed**. Requirements 142 §27: the SR of each system is displaced against the reference by known amounts and the same metrics are recomputed on the overlap. **This is an interpretation aid, not a ranking**: it shows how large a difference a systematic misregistration alone produces, so that differences between systems on this dataset can be read against it. Every number is a mean ± standard deviation over scene units and is **descriptive** unless the paired table carries an interval.

Dataset `opensr_spain_urban` (real_cross_sensor, role independent_benchmark, split test): 20 records, **20 evaluated**, 0 skipped, 0 invalid, 0 unreadable; manifest digest `1596f37c40c631a6…`. Code git `eb3769559525ea81e902422253f8950c36b075a5` (working tree dirty: True); metrics `frame-eval-metrics/1`.

**Method.** Shifts are whole HR pixels (no resampling): 0 LR px = 0 HR px, 0.25 LR px = 1 HR px, 0.5 LR px = 2 HR px, 1 LR px = 4 HR px, 2 LR px = 8 HR px (scale x4), along the x (columns). The SR content is displaced against the reference and both are cropped to the overlap. `aligned_to_bicubic` removes the displacement that phase correlation finds between the **bicubic baseline** and the reference, rounded to a whole HR pixel: a system-neutral estimate, identical for every system on a sample, so no system aligns itself. Not computed: classification accuracy and area error (they need a downstream task, which is outside Phase 5).

Estimated displacement of the bicubic baseline against the reference (the reference's own misregistration, HR px): median 1.63, mean 1.67, max 3.44 over 20 tiles.

## Systems
| system | model | hard constraint |
|---|---|---|
| bicubic | bicubic | None |
| sen2sr_lite | SEN2SRLite/NonReference_RGBN_x4 | True |
| sen2sr_mamba | SEN2SR/MambaSR_RGBN_x4 | True |
| tiny_cnn_seed0 | tiny_cnn | False |

## PSNR (dB) ↑

| condition | bicubic | sen2sr_lite | sen2sr_mamba | tiny_cnn_seed0 | n units |
|---|---|---|---|---|---|
| `shift_0_lr_px` (0 LR px) | 29.286 ± 1.729 | 29.279 ± 1.897 | 29.251 ± 1.908 | 28.867 ± 1.976 | 4 |
| `shift_0.25_lr_px` (0.25 LR px) | 28.945 ± 1.644 | 28.780 ± 1.747 | 28.746 ± 1.763 | 28.326 ± 1.811 | 4 |
| `shift_0.5_lr_px` (0.5 LR px) | 28.452 ± 1.555 | 28.125 ± 1.593 | 28.104 ± 1.622 | 27.629 ± 1.661 | 4 |
| `shift_1_lr_px` (1 LR px) | 27.433 ± 1.490 | 26.987 ± 1.482 | 27.019 ± 1.543 | 26.459 ± 1.592 | 4 |
| `shift_2_lr_px` (2 LR px) | 26.207 ± 1.632 | 25.926 ± 1.633 | 25.965 ± 1.685 | 25.418 ± 1.738 | 4 |
| `aligned_to_bicubic` | 29.720 ± 1.374 | 29.917 ± 1.411 | 29.904 ± 1.375 | 29.556 ± 1.455 | 4 |

Paired difference (system − bicubic) over the same scene units, mean and 95% interval where the unit count allows one:

| condition | sen2sr_lite − bicubic | sen2sr_mamba − bicubic | tiny_cnn_seed0 − bicubic |
|---|---|---|---|
| `shift_0_lr_px` (0 LR px) | -0.006 (descriptive only) | -0.034 (descriptive only) | -0.418 (descriptive only) |
| `shift_0.25_lr_px` (0.25 LR px) | -0.166 (descriptive only) | -0.199 (descriptive only) | -0.620 (descriptive only) |
| `shift_0.5_lr_px` (0.5 LR px) | -0.327 (descriptive only) | -0.348 (descriptive only) | -0.823 (descriptive only) |
| `shift_1_lr_px` (1 LR px) | -0.446 (descriptive only) | -0.414 (descriptive only) | -0.974 (descriptive only) |
| `shift_2_lr_px` (2 LR px) | -0.281 (descriptive only) | -0.243 (descriptive only) | -0.790 (descriptive only) |
| `aligned_to_bicubic` | +0.196 (descriptive only) | +0.184 (descriptive only) | -0.164 (descriptive only) |

## SSIM ↑

| condition | bicubic | sen2sr_lite | sen2sr_mamba | tiny_cnn_seed0 | n units |
|---|---|---|---|---|---|
| `shift_0_lr_px` (0 LR px) | 0.700 ± 0.059 | 0.710 ± 0.074 | 0.709 ± 0.075 | 0.699 ± 0.079 | 4 |
| `shift_0.25_lr_px` (0.25 LR px) | 0.684 ± 0.055 | 0.685 ± 0.067 | 0.683 ± 0.068 | 0.673 ± 0.072 | 4 |
| `shift_0.5_lr_px` (0.5 LR px) | 0.660 ± 0.050 | 0.649 ± 0.058 | 0.648 ± 0.059 | 0.634 ± 0.061 | 4 |
| `shift_1_lr_px` (1 LR px) | 0.608 ± 0.046 | 0.582 ± 0.048 | 0.586 ± 0.052 | 0.563 ± 0.053 | 4 |
| `shift_2_lr_px` (2 LR px) | 0.564 ± 0.054 | 0.550 ± 0.057 | 0.553 ± 0.059 | 0.525 ± 0.063 | 4 |
| `aligned_to_bicubic` | 0.722 ± 0.036 | 0.745 ± 0.038 | 0.744 ± 0.035 | 0.736 ± 0.041 | 4 |

## SAM (°) ↓

| condition | bicubic | sen2sr_lite | sen2sr_mamba | tiny_cnn_seed0 | n units |
|---|---|---|---|---|---|
| `shift_0_lr_px` (0 LR px) | 3.919 ± 0.987 | 3.952 ± 1.036 | 3.998 ± 1.063 | 4.240 ± 1.044 | 4 |
| `shift_0.25_lr_px` (0.25 LR px) | 4.102 ± 1.036 | 4.198 ± 1.086 | 4.227 ± 1.091 | 4.478 ± 1.090 | 4 |
| `shift_0.5_lr_px` (0.5 LR px) | 4.364 ± 1.085 | 4.528 ± 1.129 | 4.527 ± 1.108 | 4.808 ± 1.128 | 4 |
| `shift_1_lr_px` (1 LR px) | 4.936 ± 1.171 | 5.168 ± 1.195 | 5.107 ± 1.145 | 5.473 ± 1.194 | 4 |
| `shift_2_lr_px` (2 LR px) | 5.718 ± 1.306 | 5.889 ± 1.311 | 5.817 ± 1.267 | 6.243 ± 1.320 | 4 |
| `aligned_to_bicubic` | 3.699 ± 0.804 | 3.651 ± 0.779 | 3.704 ± 0.827 | 3.950 ± 0.782 | 4 |

Paired difference (system − bicubic) over the same scene units, mean and 95% interval where the unit count allows one:

| condition | sen2sr_lite − bicubic | sen2sr_mamba − bicubic | tiny_cnn_seed0 − bicubic |
|---|---|---|---|
| `shift_0_lr_px` (0 LR px) | +0.033 (descriptive only) | +0.079 (descriptive only) | +0.321 (descriptive only) |
| `shift_0.25_lr_px` (0.25 LR px) | +0.096 (descriptive only) | +0.125 (descriptive only) | +0.376 (descriptive only) |
| `shift_0.5_lr_px` (0.5 LR px) | +0.163 (descriptive only) | +0.162 (descriptive only) | +0.444 (descriptive only) |
| `shift_1_lr_px` (1 LR px) | +0.233 (descriptive only) | +0.172 (descriptive only) | +0.538 (descriptive only) |
| `shift_2_lr_px` (2 LR px) | +0.171 (descriptive only) | +0.099 (descriptive only) | +0.525 (descriptive only) |
| `aligned_to_bicubic` | -0.048 (descriptive only) | +0.006 (descriptive only) | +0.251 (descriptive only) |

## ERGAS ↓

| condition | bicubic | sen2sr_lite | sen2sr_mamba | tiny_cnn_seed0 | n units |
|---|---|---|---|---|---|
| `shift_0_lr_px` (0 LR px) | 6.218 ± 2.050 | 6.269 ± 2.210 | 6.265 ± 2.183 | 6.719 ± 2.401 | 4 |
| `shift_0.25_lr_px` (0.25 LR px) | 6.500 ± 2.157 | 6.693 ± 2.362 | 6.652 ± 2.249 | 7.217 ± 2.577 | 4 |
| `shift_0.5_lr_px` (0.5 LR px) | 6.908 ± 2.244 | 7.244 ± 2.453 | 7.157 ± 2.254 | 7.849 ± 2.667 | 4 |
| `shift_1_lr_px` (1 LR px) | 7.798 ± 2.343 | 8.257 ± 2.492 | 8.094 ± 2.223 | 8.961 ± 2.684 | 4 |
| `shift_2_lr_px` (2 LR px) | 8.975 ± 2.377 | 9.298 ± 2.463 | 9.146 ± 2.247 | 10.016 ± 2.634 | 4 |
| `aligned_to_bicubic` | 5.833 ± 1.801 | 5.672 ± 1.772 | 5.714 ± 1.854 | 6.023 ± 1.885 | 4 |

## NDVI MAE ↓

| condition | bicubic | sen2sr_lite | sen2sr_mamba | tiny_cnn_seed0 | n units |
|---|---|---|---|---|---|
| `shift_0_lr_px` (0 LR px) | 0.062 ± 0.020 | 0.063 ± 0.021 | 0.063 ± 0.021 | 0.067 ± 0.020 | 4 |
| `shift_0.25_lr_px` (0.25 LR px) | 0.065 ± 0.021 | 0.067 ± 0.022 | 0.067 ± 0.022 | 0.071 ± 0.022 | 4 |
| `shift_0.5_lr_px` (0.5 LR px) | 0.069 ± 0.023 | 0.072 ± 0.023 | 0.072 ± 0.023 | 0.077 ± 0.023 | 4 |
| `shift_1_lr_px` (1 LR px) | 0.078 ± 0.025 | 0.082 ± 0.025 | 0.081 ± 0.024 | 0.087 ± 0.025 | 4 |
| `shift_2_lr_px` (2 LR px) | 0.091 ± 0.027 | 0.093 ± 0.027 | 0.092 ± 0.026 | 0.099 ± 0.027 | 4 |
| `aligned_to_bicubic` | 0.059 ± 0.018 | 0.058 ± 0.017 | 0.059 ± 0.019 | 0.063 ± 0.017 | 4 |

## detail rel. error (1 = no detail added) ↓

| condition | bicubic | sen2sr_lite | sen2sr_mamba | tiny_cnn_seed0 | n units |
|---|---|---|---|---|---|
| `shift_0_lr_px` (0 LR px) | 0.973 ± 0.025 | 0.978 ± 0.053 | 0.985 ± 0.061 | 1.024 ± 0.062 | 4 |
| `shift_0.25_lr_px` (0.25 LR px) | 0.986 ± 0.023 | 1.007 ± 0.047 | 1.018 ± 0.051 | 1.059 ± 0.050 | 4 |
| `shift_0.5_lr_px` (0.5 LR px) | 1.003 ± 0.018 | 1.042 ± 0.033 | 1.055 ± 0.034 | 1.101 ± 0.032 | 4 |
| `shift_1_lr_px` (1 LR px) | 1.033 ± 0.005 | 1.086 ± 0.008 | 1.093 ± 0.012 | 1.150 ± 0.021 | 4 |
| `shift_2_lr_px` (2 LR px) | 1.033 ± 0.008 | 1.067 ± 0.008 | 1.075 ± 0.012 | 1.132 ± 0.025 | 4 |
| `aligned_to_bicubic` | 0.957 ± 0.011 | 0.938 ± 0.020 | 0.939 ± 0.024 | 0.977 ± 0.021 | 4 |

Paired difference (system − bicubic) over the same scene units, mean and 95% interval where the unit count allows one:

| condition | sen2sr_lite − bicubic | sen2sr_mamba − bicubic | tiny_cnn_seed0 − bicubic |
|---|---|---|---|
| `shift_0_lr_px` (0 LR px) | +0.004 (descriptive only) | +0.011 (descriptive only) | +0.050 (descriptive only) |
| `shift_0.25_lr_px` (0.25 LR px) | +0.021 (descriptive only) | +0.032 (descriptive only) | +0.073 (descriptive only) |
| `shift_0.5_lr_px` (0.5 LR px) | +0.039 (descriptive only) | +0.051 (descriptive only) | +0.098 (descriptive only) |
| `shift_1_lr_px` (1 LR px) | +0.053 (descriptive only) | +0.060 (descriptive only) | +0.117 (descriptive only) |
| `shift_2_lr_px` (2 LR px) | +0.034 (descriptive only) | +0.042 (descriptive only) | +0.099 (descriptive only) |
| `aligned_to_bicubic` | -0.019 (descriptive only) | -0.018 (descriptive only) | +0.020 (descriptive only) |

## detail correlation ↑

| condition | bicubic | sen2sr_lite | sen2sr_mamba | tiny_cnn_seed0 | n units |
|---|---|---|---|---|---|
| `shift_0_lr_px` (0 LR px) | 0.234 ± 0.093 | 0.252 ± 0.129 | 0.236 ± 0.141 | 0.219 ± 0.114 | 4 |
| `shift_0.25_lr_px` (0.25 LR px) | 0.181 ± 0.087 | 0.169 ± 0.118 | 0.151 ± 0.123 | 0.145 ± 0.102 | 4 |
| `shift_0.5_lr_px` (0.5 LR px) | 0.107 ± 0.072 | 0.070 ± 0.088 | 0.056 ± 0.081 | 0.058 ± 0.072 | 4 |
| `shift_1_lr_px` (1 LR px) | -0.020 ± 0.028 | -0.056 ± 0.019 | -0.047 ± 0.017 | -0.044 ± 0.015 | 4 |
| `shift_2_lr_px` (2 LR px) | -0.019 ± 0.024 | 0.001 ± 0.023 | -0.001 ± 0.018 | -0.004 ± 0.023 | 4 |
| `aligned_to_bicubic` | 0.304 ± 0.032 | 0.365 ± 0.039 | 0.358 ± 0.047 | 0.323 ± 0.032 | 4 |

## How to read this

* The zero row is the ordinary evaluation of this dataset (restricted to the metrics shown). Each following row scores exactly the same SR against a reference that is further displaced.
* If a displacement of a fraction of an LR pixel changes a metric by more than the differences between systems, then that difference cannot be attributed to the systems on this dataset alone.
* `aligned_to_bicubic` removes an estimated whole-pixel misregistration; a residual sub-pixel misregistration and any radiometric difference between sensors remain.
* Nothing here is combined across datasets and nothing is a ranking.
