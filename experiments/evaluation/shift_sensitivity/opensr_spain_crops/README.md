# Spatial-shift sensitivity: `opensr_spain_crops`

Status: **completed**. Requirements 142 §27: the SR of each system is displaced against the reference by known amounts and the same metrics are recomputed on the overlap. **This is an interpretation aid, not a ranking**: it shows how large a difference a systematic misregistration alone produces, so that differences between systems on this dataset can be read against it. Every number is a mean ± standard deviation over scene units and is **descriptive** unless the paired table carries an interval.

Dataset `opensr_spain_crops` (real_cross_sensor, role independent_benchmark, split test): 28 records, **28 evaluated**, 0 skipped, 0 invalid, 0 unreadable; manifest digest `8fda3d88b850d574…`. Code git `eb3769559525ea81e902422253f8950c36b075a5` (working tree dirty: True); metrics `frame-eval-metrics/1`.

**Method.** Shifts are whole HR pixels (no resampling): 0 LR px = 0 HR px, 0.25 LR px = 1 HR px, 0.5 LR px = 2 HR px, 1 LR px = 4 HR px, 2 LR px = 8 HR px (scale x4), along the x (columns). The SR content is displaced against the reference and both are cropped to the overlap. `aligned_to_bicubic` removes the displacement that phase correlation finds between the **bicubic baseline** and the reference, rounded to a whole HR pixel: a system-neutral estimate, identical for every system on a sample, so no system aligns itself. Not computed: classification accuracy and area error (they need a downstream task, which is outside Phase 5).

Estimated displacement of the bicubic baseline against the reference (the reference's own misregistration, HR px): median 1.54, mean 1.63, max 3.26 over 28 tiles.

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
| `shift_0_lr_px` (0 LR px) | 32.531 ± 0.813 | 32.549 ± 0.741 | 32.551 ± 0.769 | 32.334 ± 0.825 | 5 |
| `shift_0.25_lr_px` (0.25 LR px) | 32.216 ± 0.894 | 32.098 ± 0.827 | 32.079 ± 0.869 | 31.831 ± 0.946 | 5 |
| `shift_0.5_lr_px` (0.5 LR px) | 31.760 ± 1.047 | 31.510 ± 1.013 | 31.487 ± 1.061 | 31.201 ± 1.175 | 5 |
| `shift_1_lr_px` (1 LR px) | 30.802 ± 1.400 | 30.459 ± 1.411 | 30.462 ± 1.454 | 30.117 ± 1.600 | 5 |
| `shift_2_lr_px` (2 LR px) | 29.534 ± 1.860 | 29.318 ± 1.845 | 29.323 ± 1.879 | 28.969 ± 2.003 | 5 |
| `aligned_to_bicubic` | 32.918 ± 0.887 | 33.107 ± 0.777 | 33.153 ± 0.760 | 32.938 ± 0.790 | 5 |

Paired difference (system − bicubic) over the same scene units, mean and 95% interval where the unit count allows one:

| condition | sen2sr_lite − bicubic | sen2sr_mamba − bicubic | tiny_cnn_seed0 − bicubic |
|---|---|---|---|
| `shift_0_lr_px` (0 LR px) | +0.019 [-0.168, 0.229] | +0.020 [-0.186, 0.251] | -0.196 [-0.390, 0.004] |
| `shift_0.25_lr_px` (0.25 LR px) | -0.118 [-0.261, 0.034] | -0.137 [-0.276, 0.015] | -0.385 [-0.519, -0.252] |
| `shift_0.5_lr_px` (0.5 LR px) | -0.251 [-0.338, -0.167] | -0.274 [-0.339, -0.197] | -0.559 [-0.668, -0.436] |
| `shift_1_lr_px` (1 LR px) | -0.344 [-0.402, -0.284] | -0.340 [-0.392, -0.289] | -0.685 [-0.846, -0.535] |
| `shift_2_lr_px` (2 LR px) | -0.216 [-0.256, -0.176] | -0.211 [-0.236, -0.187] | -0.565 [-0.683, -0.454] |
| `aligned_to_bicubic` | +0.189 [0.069, 0.330] | +0.234 [0.092, 0.379] | +0.020 [-0.075, 0.130] |

## SSIM ↑

| condition | bicubic | sen2sr_lite | sen2sr_mamba | tiny_cnn_seed0 | n units |
|---|---|---|---|---|---|
| `shift_0_lr_px` (0 LR px) | 0.814 ± 0.026 | 0.819 ± 0.023 | 0.820 ± 0.024 | 0.815 ± 0.025 | 5 |
| `shift_0.25_lr_px` (0.25 LR px) | 0.803 ± 0.029 | 0.802 ± 0.026 | 0.802 ± 0.028 | 0.796 ± 0.029 | 5 |
| `shift_0.5_lr_px` (0.5 LR px) | 0.787 ± 0.036 | 0.779 ± 0.035 | 0.779 ± 0.037 | 0.772 ± 0.040 | 5 |
| `shift_1_lr_px` (1 LR px) | 0.755 ± 0.054 | 0.739 ± 0.058 | 0.740 ± 0.060 | 0.729 ± 0.067 | 5 |
| `shift_2_lr_px` (2 LR px) | 0.724 ± 0.076 | 0.715 ± 0.078 | 0.715 ± 0.080 | 0.701 ± 0.088 | 5 |
| `aligned_to_bicubic` | 0.825 ± 0.026 | 0.838 ± 0.019 | 0.841 ± 0.018 | 0.835 ± 0.018 | 5 |

## SAM (°) ↓

| condition | bicubic | sen2sr_lite | sen2sr_mamba | tiny_cnn_seed0 | n units |
|---|---|---|---|---|---|
| `shift_0_lr_px` (0 LR px) | 2.417 ± 0.529 | 2.442 ± 0.518 | 2.433 ± 0.520 | 2.539 ± 0.508 | 5 |
| `shift_0.25_lr_px` (0.25 LR px) | 2.507 ± 0.552 | 2.563 ± 0.549 | 2.548 ± 0.550 | 2.662 ± 0.544 | 5 |
| `shift_0.5_lr_px` (0.5 LR px) | 2.629 ± 0.572 | 2.713 ± 0.575 | 2.689 ± 0.572 | 2.814 ± 0.573 | 5 |
| `shift_1_lr_px` (1 LR px) | 2.893 ± 0.603 | 3.001 ± 0.609 | 2.961 ± 0.602 | 3.109 ± 0.615 | 5 |
| `shift_2_lr_px` (2 LR px) | 3.283 ± 0.637 | 3.364 ± 0.640 | 3.329 ± 0.637 | 3.489 ± 0.660 | 5 |
| `aligned_to_bicubic` | 2.326 ± 0.548 | 2.312 ± 0.534 | 2.302 ± 0.535 | 2.404 ± 0.517 | 5 |

Paired difference (system − bicubic) over the same scene units, mean and 95% interval where the unit count allows one:

| condition | sen2sr_lite − bicubic | sen2sr_mamba − bicubic | tiny_cnn_seed0 − bicubic |
|---|---|---|---|
| `shift_0_lr_px` (0 LR px) | +0.025 [0.008, 0.052] | +0.016 [-0.003, 0.039] | +0.122 [0.080, 0.167] |
| `shift_0.25_lr_px` (0.25 LR px) | +0.056 [0.038, 0.073] | +0.042 [0.018, 0.065] | +0.155 [0.114, 0.205] |
| `shift_0.5_lr_px` (0.5 LR px) | +0.084 [0.062, 0.101] | +0.060 [0.018, 0.093] | +0.185 [0.137, 0.246] |
| `shift_1_lr_px` (1 LR px) | +0.108 [0.084, 0.137] | +0.068 [0.008, 0.115] | +0.216 [0.161, 0.299] |
| `shift_2_lr_px` (2 LR px) | +0.082 [0.061, 0.110] | +0.046 [-0.014, 0.090] | +0.206 [0.147, 0.293] |
| `aligned_to_bicubic` | -0.015 [-0.030, -0.001] | -0.024 [-0.045, -0.006] | +0.078 [0.030, 0.133] |

## ERGAS ↓

| condition | bicubic | sen2sr_lite | sen2sr_mamba | tiny_cnn_seed0 | n units |
|---|---|---|---|---|---|
| `shift_0_lr_px` (0 LR px) | 3.938 ± 1.480 | 3.967 ± 1.596 | 3.944 ± 1.531 | 4.208 ± 1.822 | 5 |
| `shift_0.25_lr_px` (0.25 LR px) | 4.130 ± 1.600 | 4.251 ± 1.777 | 4.210 ± 1.641 | 4.546 ± 2.026 | 5 |
| `shift_0.5_lr_px` (0.5 LR px) | 4.394 ± 1.707 | 4.594 ± 1.903 | 4.529 ± 1.711 | 4.928 ± 2.148 | 5 |
| `shift_1_lr_px` (1 LR px) | 4.954 ± 1.845 | 5.208 ± 2.011 | 5.111 ± 1.791 | 5.582 ± 2.237 | 5 |
| `shift_2_lr_px` (2 LR px) | 5.736 ± 1.900 | 5.915 ± 2.018 | 5.834 ± 1.842 | 6.304 ± 2.234 | 5 |
| `aligned_to_bicubic` | 3.719 ± 1.368 | 3.613 ± 1.341 | 3.606 ± 1.368 | 3.778 ± 1.492 | 5 |

## NDVI MAE ↓

| condition | bicubic | sen2sr_lite | sen2sr_mamba | tiny_cnn_seed0 | n units |
|---|---|---|---|---|---|
| `shift_0_lr_px` (0 LR px) | 0.039 ± 0.012 | 0.039 ± 0.011 | 0.039 ± 0.011 | 0.041 ± 0.011 | 5 |
| `shift_0.25_lr_px` (0.25 LR px) | 0.041 ± 0.012 | 0.041 ± 0.012 | 0.041 ± 0.012 | 0.043 ± 0.012 | 5 |
| `shift_0.5_lr_px` (0.5 LR px) | 0.043 ± 0.013 | 0.044 ± 0.013 | 0.043 ± 0.012 | 0.045 ± 0.013 | 5 |
| `shift_1_lr_px` (1 LR px) | 0.047 ± 0.013 | 0.049 ± 0.014 | 0.048 ± 0.013 | 0.050 ± 0.013 | 5 |
| `shift_2_lr_px` (2 LR px) | 0.054 ± 0.014 | 0.055 ± 0.014 | 0.054 ± 0.014 | 0.057 ± 0.014 | 5 |
| `aligned_to_bicubic` | 0.038 ± 0.012 | 0.037 ± 0.011 | 0.037 ± 0.011 | 0.038 ± 0.011 | 5 |

## detail rel. error (1 = no detail added) ↓

| condition | bicubic | sen2sr_lite | sen2sr_mamba | tiny_cnn_seed0 | n units |
|---|---|---|---|---|---|
| `shift_0_lr_px` (0 LR px) | 0.974 ± 0.022 | 0.976 ± 0.046 | 0.979 ± 0.052 | 1.002 ± 0.049 | 5 |
| `shift_0.25_lr_px` (0.25 LR px) | 0.986 ± 0.020 | 1.001 ± 0.040 | 1.009 ± 0.041 | 1.034 ± 0.037 | 5 |
| `shift_0.5_lr_px` (0.5 LR px) | 1.001 ± 0.016 | 1.030 ± 0.028 | 1.040 ± 0.026 | 1.067 ± 0.023 | 5 |
| `shift_1_lr_px` (1 LR px) | 1.025 ± 0.007 | 1.064 ± 0.015 | 1.070 ± 0.015 | 1.102 ± 0.029 | 5 |
| `shift_2_lr_px` (2 LR px) | 1.026 ± 0.009 | 1.051 ± 0.013 | 1.058 ± 0.012 | 1.094 ± 0.029 | 5 |
| `aligned_to_bicubic` | 0.961 ± 0.015 | 0.942 ± 0.026 | 0.936 ± 0.030 | 0.962 ± 0.026 | 5 |

Paired difference (system − bicubic) over the same scene units, mean and 95% interval where the unit count allows one:

| condition | sen2sr_lite − bicubic | sen2sr_mamba − bicubic | tiny_cnn_seed0 − bicubic |
|---|---|---|---|
| `shift_0_lr_px` (0 LR px) | +0.001 [-0.017, 0.019] | +0.005 [-0.020, 0.028] | +0.028 [0.007, 0.048] |
| `shift_0.25_lr_px` (0.25 LR px) | +0.015 [-0.000, 0.030] | +0.023 [0.005, 0.040] | +0.048 [0.032, 0.064] |
| `shift_0.5_lr_px` (0.5 LR px) | +0.029 [0.020, 0.039] | +0.039 [0.029, 0.048] | +0.066 [0.052, 0.079] |
| `shift_1_lr_px` (1 LR px) | +0.040 [0.031, 0.047] | +0.045 [0.037, 0.054] | +0.078 [0.058, 0.098] |
| `shift_2_lr_px` (2 LR px) | +0.025 [0.019, 0.030] | +0.032 [0.027, 0.036] | +0.068 [0.052, 0.084] |
| `aligned_to_bicubic` | -0.018 [-0.028, -0.010] | -0.025 [-0.036, -0.013] | +0.001 [-0.008, 0.010] |

## detail correlation ↑

| condition | bicubic | sen2sr_lite | sen2sr_mamba | tiny_cnn_seed0 | n units |
|---|---|---|---|---|---|
| `shift_0_lr_px` (0 LR px) | 0.225 ± 0.085 | 0.240 ± 0.122 | 0.233 ± 0.136 | 0.211 ± 0.118 | 5 |
| `shift_0.25_lr_px` (0.25 LR px) | 0.172 ± 0.084 | 0.159 ± 0.113 | 0.146 ± 0.113 | 0.134 ± 0.105 | 5 |
| `shift_0.5_lr_px` (0.5 LR px) | 0.103 ± 0.073 | 0.069 ± 0.082 | 0.061 ± 0.069 | 0.057 ± 0.070 | 5 |
| `shift_1_lr_px` (1 LR px) | -0.007 ± 0.029 | -0.035 ± 0.015 | -0.026 ± 0.016 | -0.022 ± 0.016 | 5 |
| `shift_2_lr_px` (2 LR px) | -0.009 ± 0.025 | 0.008 ± 0.024 | 0.006 ± 0.023 | 0.001 ± 0.022 | 5 |
| `aligned_to_bicubic` | 0.290 ± 0.047 | 0.350 ± 0.054 | 0.361 ± 0.063 | 0.317 ± 0.060 | 5 |

## How to read this

* The zero row is the ordinary evaluation of this dataset (restricted to the metrics shown). Each following row scores exactly the same SR against a reference that is further displaced.
* If a displacement of a fraction of an LR pixel changes a metric by more than the differences between systems, then that difference cannot be attributed to the systems on this dataset alone.
* `aligned_to_bicubic` removes an estimated whole-pixel misregistration; a residual sub-pixel misregistration and any radiometric difference between sensors remain.
* Nothing here is combined across datasets and nothing is a ranking.
