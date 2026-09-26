# Spatial-shift sensitivity: `opensr_spot`

Status: **completed**. Requirements 142 §27: the SR of each system is displaced against the reference by known amounts and the same metrics are recomputed on the overlap. **This is an interpretation aid, not a ranking**: it shows how large a difference a systematic misregistration alone produces, so that differences between systems on this dataset can be read against it. Every number is a mean ± standard deviation over scene units and is **descriptive** unless the paired table carries an interval.

Dataset `opensr_spot` (real_cross_sensor, role independent_benchmark, split test): 9 records, **9 evaluated**, 0 skipped, 0 invalid, 0 unreadable; manifest digest `32495be456ae9ba3…`. Code git `eb3769559525ea81e902422253f8950c36b075a5` (working tree dirty: True); metrics `frame-eval-metrics/1`.

**Method.** Shifts are whole HR pixels (no resampling): 0 LR px = 0 HR px, 0.25 LR px = 1 HR px, 0.5 LR px = 2 HR px, 1 LR px = 4 HR px, 2 LR px = 8 HR px (scale x4), along the x (columns). The SR content is displaced against the reference and both are cropped to the overlap. `aligned_to_bicubic` removes the displacement that phase correlation finds between the **bicubic baseline** and the reference, rounded to a whole HR pixel: a system-neutral estimate, identical for every system on a sample, so no system aligns itself. Not computed: classification accuracy and area error (they need a downstream task, which is outside Phase 5).

Estimated displacement of the bicubic baseline against the reference (the reference's own misregistration, HR px): median 0.40, mean 0.47, max 0.85 over 9 tiles.

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
| `shift_0_lr_px` (0 LR px) | 33.117 ± 5.387 | 33.223 ± 5.412 | 33.197 ± 5.514 | 32.852 ± 5.654 | 9 |
| `shift_0.25_lr_px` (0.25 LR px) | 33.026 ± 5.351 | 33.070 ± 5.385 | 33.039 ± 5.474 | 32.697 ± 5.641 | 9 |
| `shift_0.5_lr_px` (0.5 LR px) | 32.607 ± 5.400 | 32.451 ± 5.483 | 32.410 ± 5.538 | 32.079 ± 5.771 | 9 |
| `shift_1_lr_px` (1 LR px) | 31.386 ± 5.599 | 30.925 ± 5.713 | 30.902 ± 5.693 | 30.545 ± 6.038 | 9 |
| `shift_2_lr_px` (2 LR px) | 29.722 ± 5.893 | 29.362 ± 5.960 | 29.371 ± 5.921 | 28.947 ± 6.228 | 9 |
| `aligned_to_bicubic` | 33.138 ± 5.412 | 33.257 ± 5.447 | 33.233 ± 5.551 | 32.877 ± 5.681 | 9 |

Paired difference (system − bicubic) over the same scene units, mean and 95% interval where the unit count allows one:

| condition | sen2sr_lite − bicubic | sen2sr_mamba − bicubic | tiny_cnn_seed0 − bicubic |
|---|---|---|---|
| `shift_0_lr_px` (0 LR px) | +0.106 [0.026, 0.181] | +0.079 [-0.033, 0.194] | -0.265 [-0.507, -0.052] |
| `shift_0.25_lr_px` (0.25 LR px) | +0.044 [-0.027, 0.114] | +0.013 [-0.093, 0.117] | -0.328 [-0.574, -0.112] |
| `shift_0.5_lr_px` (0.5 LR px) | -0.156 [-0.232, -0.083] | -0.198 [-0.305, -0.098] | -0.528 [-0.798, -0.280] |
| `shift_1_lr_px` (1 LR px) | -0.461 [-0.557, -0.363] | -0.484 [-0.574, -0.390] | -0.841 [-1.138, -0.557] |
| `shift_2_lr_px` (2 LR px) | -0.360 [-0.430, -0.287] | -0.351 [-0.433, -0.274] | -0.775 [-1.014, -0.540] |
| `aligned_to_bicubic` | +0.118 [0.044, 0.186] | +0.095 [-0.017, 0.204] | -0.261 [-0.505, -0.047] |

## SSIM ↑

| condition | bicubic | sen2sr_lite | sen2sr_mamba | tiny_cnn_seed0 | n units |
|---|---|---|---|---|---|
| `shift_0_lr_px` (0 LR px) | 0.818 ± 0.114 | 0.831 ± 0.104 | 0.825 ± 0.115 | 0.817 ± 0.117 | 9 |
| `shift_0.25_lr_px` (0.25 LR px) | 0.813 ± 0.117 | 0.823 ± 0.111 | 0.818 ± 0.120 | 0.809 ± 0.123 | 9 |
| `shift_0.5_lr_px` (0.5 LR px) | 0.795 ± 0.130 | 0.795 ± 0.131 | 0.792 ± 0.136 | 0.781 ± 0.144 | 9 |
| `shift_1_lr_px` (1 LR px) | 0.740 ± 0.170 | 0.718 ± 0.183 | 0.722 ± 0.175 | 0.703 ± 0.198 | 9 |
| `shift_2_lr_px` (2 LR px) | 0.674 ± 0.217 | 0.656 ± 0.224 | 0.661 ± 0.214 | 0.636 ± 0.239 | 9 |
| `aligned_to_bicubic` | 0.818 ± 0.114 | 0.831 ± 0.105 | 0.826 ± 0.116 | 0.817 ± 0.117 | 9 |

## SAM (°) ↓

| condition | bicubic | sen2sr_lite | sen2sr_mamba | tiny_cnn_seed0 | n units |
|---|---|---|---|---|---|
| `shift_0_lr_px` (0 LR px) | 1.846 ± 1.516 | 2.071 ± 1.727 | 2.015 ± 1.650 | 2.461 ± 2.230 | 9 |
| `shift_0.25_lr_px` (0.25 LR px) | 1.878 ± 1.556 | 2.111 ± 1.778 | 2.050 ± 1.691 | 2.488 ± 2.263 | 9 |
| `shift_0.5_lr_px` (0.5 LR px) | 2.021 ± 1.731 | 2.274 ± 1.976 | 2.193 ± 1.854 | 2.631 ± 2.430 | 9 |
| `shift_1_lr_px` (1 LR px) | 2.444 ± 2.232 | 2.729 ± 2.517 | 2.603 ± 2.306 | 3.073 ± 2.937 | 9 |
| `shift_2_lr_px` (2 LR px) | 3.127 ± 3.016 | 3.378 ± 3.254 | 3.222 ± 2.984 | 3.743 ± 3.696 | 9 |
| `aligned_to_bicubic` | 1.843 ± 1.518 | 2.068 ± 1.730 | 2.012 ± 1.653 | 2.458 ± 2.233 | 9 |

Paired difference (system − bicubic) over the same scene units, mean and 95% interval where the unit count allows one:

| condition | sen2sr_lite − bicubic | sen2sr_mamba − bicubic | tiny_cnn_seed0 − bicubic |
|---|---|---|---|
| `shift_0_lr_px` (0 LR px) | +0.225 [0.106, 0.365] | +0.170 [0.070, 0.280] | +0.615 [0.211, 1.076] |
| `shift_0.25_lr_px` (0.25 LR px) | +0.233 [0.107, 0.379] | +0.172 [0.072, 0.283] | +0.610 [0.211, 1.068] |
| `shift_0.5_lr_px` (0.5 LR px) | +0.252 [0.113, 0.413] | +0.172 [0.076, 0.280] | +0.609 [0.214, 1.061] |
| `shift_1_lr_px` (1 LR px) | +0.285 [0.125, 0.472] | +0.160 [0.075, 0.268] | +0.629 [0.228, 1.087] |
| `shift_2_lr_px` (2 LR px) | +0.251 [0.116, 0.408] | +0.096 [-0.001, 0.210] | +0.617 [0.225, 1.066] |
| `aligned_to_bicubic` | +0.225 [0.105, 0.364] | +0.169 [0.070, 0.280] | +0.614 [0.210, 1.076] |

## ERGAS ↓

| condition | bicubic | sen2sr_lite | sen2sr_mamba | tiny_cnn_seed0 | n units |
|---|---|---|---|---|---|
| `shift_0_lr_px` (0 LR px) | 3.767 ± 2.694 | 3.685 ± 2.608 | 3.728 ± 2.689 | 3.998 ± 2.938 | 9 |
| `shift_0.25_lr_px` (0.25 LR px) | 3.802 ± 2.713 | 3.758 ± 2.669 | 3.797 ± 2.738 | 4.079 ± 3.009 | 9 |
| `shift_0.5_lr_px` (0.5 LR px) | 4.054 ± 2.958 | 4.156 ± 3.071 | 4.186 ± 3.113 | 4.521 ± 3.466 | 9 |
| `shift_1_lr_px` (1 LR px) | 4.873 ± 3.780 | 5.220 ± 4.134 | 5.199 ± 4.068 | 5.693 ± 4.652 | 9 |
| `shift_2_lr_px` (2 LR px) | 6.177 ± 5.126 | 6.481 ± 5.410 | 6.422 ± 5.287 | 7.025 ± 5.969 | 9 |
| `aligned_to_bicubic` | 3.763 ± 2.699 | 3.678 ± 2.615 | 3.720 ± 2.697 | 3.993 ± 2.943 | 9 |

## NDVI MAE ↓

| condition | bicubic | sen2sr_lite | sen2sr_mamba | tiny_cnn_seed0 | n units |
|---|---|---|---|---|---|
| `shift_0_lr_px` (0 LR px) | 0.025 ± 0.024 | 0.029 ± 0.028 | 0.028 ± 0.025 | 0.036 ± 0.036 | 9 |
| `shift_0.25_lr_px` (0.25 LR px) | 0.026 ± 0.024 | 0.030 ± 0.028 | 0.029 ± 0.026 | 0.036 ± 0.037 | 9 |
| `shift_0.5_lr_px` (0.5 LR px) | 0.028 ± 0.028 | 0.033 ± 0.032 | 0.031 ± 0.029 | 0.039 ± 0.040 | 9 |
| `shift_1_lr_px` (1 LR px) | 0.035 ± 0.036 | 0.040 ± 0.041 | 0.038 ± 0.037 | 0.046 ± 0.049 | 9 |
| `shift_2_lr_px` (2 LR px) | 0.046 ± 0.050 | 0.050 ± 0.054 | 0.047 ± 0.049 | 0.056 ± 0.062 | 9 |
| `aligned_to_bicubic` | 0.025 ± 0.024 | 0.029 ± 0.028 | 0.028 ± 0.025 | 0.036 ± 0.036 | 9 |

## detail rel. error (1 = no detail added) ↓

| condition | bicubic | sen2sr_lite | sen2sr_mamba | tiny_cnn_seed0 | n units |
|---|---|---|---|---|---|
| `shift_0_lr_px` (0 LR px) | 0.946 ± 0.020 | 0.932 ± 0.032 | 0.936 ± 0.033 | 0.980 ± 0.046 | 9 |
| `shift_0.25_lr_px` (0.25 LR px) | 0.950 ± 0.020 | 0.943 ± 0.029 | 0.950 ± 0.031 | 0.991 ± 0.045 | 9 |
| `shift_0.5_lr_px` (0.5 LR px) | 0.970 ± 0.016 | 0.987 ± 0.021 | 1.000 ± 0.030 | 1.036 ± 0.050 | 9 |
| `shift_1_lr_px` (1 LR px) | 1.024 ± 0.009 | 1.087 ± 0.037 | 1.102 ± 0.042 | 1.138 ± 0.082 | 9 |
| `shift_2_lr_px` (2 LR px) | 1.054 ± 0.021 | 1.105 ± 0.044 | 1.115 ± 0.049 | 1.163 ± 0.086 | 9 |
| `aligned_to_bicubic` | 0.945 ± 0.020 | 0.930 ± 0.030 | 0.933 ± 0.030 | 0.979 ± 0.046 | 9 |

Paired difference (system − bicubic) over the same scene units, mean and 95% interval where the unit count allows one:

| condition | sen2sr_lite − bicubic | sen2sr_mamba − bicubic | tiny_cnn_seed0 − bicubic |
|---|---|---|---|
| `shift_0_lr_px` (0 LR px) | -0.014 [-0.022, -0.006] | -0.010 [-0.022, 0.002] | +0.034 [0.010, 0.061] |
| `shift_0.25_lr_px` (0.25 LR px) | -0.007 [-0.014, 0.000] | -0.001 [-0.013, 0.012] | +0.041 [0.016, 0.068] |
| `shift_0.5_lr_px` (0.5 LR px) | +0.018 [0.009, 0.027] | +0.030 [0.014, 0.049] | +0.066 [0.036, 0.099] |
| `shift_1_lr_px` (1 LR px) | +0.063 [0.047, 0.082] | +0.078 [0.057, 0.100] | +0.114 [0.071, 0.162] |
| `shift_2_lr_px` (2 LR px) | +0.050 [0.036, 0.064] | +0.060 [0.042, 0.078] | +0.108 [0.069, 0.150] |
| `aligned_to_bicubic` | -0.015 [-0.023, -0.007] | -0.012 [-0.023, -0.000] | +0.034 [0.010, 0.061] |

## detail correlation ↑

| condition | bicubic | sen2sr_lite | sen2sr_mamba | tiny_cnn_seed0 | n units |
|---|---|---|---|---|---|
| `shift_0_lr_px` (0 LR px) | 0.333 ± 0.061 | 0.378 ± 0.078 | 0.381 ± 0.072 | 0.324 ± 0.079 | 9 |
| `shift_0.25_lr_px` (0.25 LR px) | 0.315 ± 0.068 | 0.349 ± 0.083 | 0.349 ± 0.078 | 0.301 ± 0.079 | 9 |
| `shift_0.5_lr_px` (0.5 LR px) | 0.248 ± 0.066 | 0.248 ± 0.070 | 0.237 ± 0.070 | 0.217 ± 0.061 | 9 |
| `shift_1_lr_px` (1 LR px) | 0.053 ± 0.036 | 0.001 ± 0.033 | -0.007 ± 0.030 | 0.009 ± 0.026 | 9 |
| `shift_2_lr_px` (2 LR px) | -0.055 ± 0.021 | -0.038 ± 0.022 | -0.031 ± 0.021 | -0.039 ± 0.020 | 9 |
| `aligned_to_bicubic` | 0.337 ± 0.056 | 0.383 ± 0.071 | 0.388 ± 0.062 | 0.327 ± 0.076 | 9 |

## How to read this

* The zero row is the ordinary evaluation of this dataset (restricted to the metrics shown). Each following row scores exactly the same SR against a reference that is further displaced.
* If a displacement of a fraction of an LR pixel changes a metric by more than the differences between systems, then that difference cannot be attributed to the systems on this dataset alone.
* `aligned_to_bicubic` removes an estimated whole-pixel misregistration; a residual sub-pixel misregistration and any radiometric difference between sensors remain.
* Nothing here is combined across datasets and nothing is a ranking.
