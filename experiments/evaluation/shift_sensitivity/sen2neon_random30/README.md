# Spatial-shift sensitivity: `sen2neon_random30`

Status: **completed**. Requirements 142 §27: the SR of each system is displaced against the reference by known amounts and the same metrics are recomputed on the overlap. **This is an interpretation aid, not a ranking**: it shows how large a difference a systematic misregistration alone produces, so that differences between systems on this dataset can be read against it. Every number is a mean ± standard deviation over scene units and is **descriptive** unless the paired table carries an interval.

Dataset `sen2neon_random30` (real_cross_sensor, role independent_benchmark, split test): 30 records, **30 evaluated**, 0 skipped, 0 invalid, 0 unreadable; manifest digest `ab433be9ceada8a1…`. Code git `eb3769559525ea81e902422253f8950c36b075a5` (working tree dirty: True); metrics `frame-eval-metrics/1`.

**Method.** Shifts are whole HR pixels (no resampling): 0 LR px = 0 HR px, 0.25 LR px = 1 HR px, 0.5 LR px = 2 HR px, 1 LR px = 4 HR px, 2 LR px = 8 HR px (scale x4), along the x (columns). The SR content is displaced against the reference and both are cropped to the overlap. `aligned_to_bicubic` removes the displacement that phase correlation finds between the **bicubic baseline** and the reference, rounded to a whole HR pixel: a system-neutral estimate, identical for every system on a sample, so no system aligns itself. Not computed: classification accuracy and area error (they need a downstream task, which is outside Phase 5).

Estimated displacement of the bicubic baseline against the reference (the reference's own misregistration, HR px): median 1.00, mean 1.54, max 6.50 over 30 tiles.

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
| `shift_0_lr_px` (0 LR px) | 33.687 ± 6.025 | 33.231 ± 5.839 | 33.223 ± 5.795 | 32.875 ± 5.718 | 28 |
| `shift_0.25_lr_px` (0.25 LR px) | 33.430 ± 5.867 | 32.957 ± 5.689 | 32.966 ± 5.659 | 32.631 ± 5.593 | 28 |
| `shift_0.5_lr_px` (0.5 LR px) | 33.011 ± 5.611 | 32.514 ± 5.444 | 32.546 ± 5.432 | 32.230 ± 5.389 | 28 |
| `shift_1_lr_px` (1 LR px) | 32.133 ± 5.177 | 31.625 ± 5.043 | 31.687 ± 5.055 | 31.393 ± 5.040 | 28 |
| `shift_2_lr_px` (2 LR px) | 31.003 ± 4.815 | 30.623 ± 4.745 | 30.687 ± 4.769 | 30.383 ± 4.765 | 28 |
| `aligned_to_bicubic` | 33.835 ± 6.041 | 33.397 ± 5.851 | 33.384 ± 5.807 | 33.032 ± 5.735 | 28 |

Paired difference (system − bicubic) over the same scene units, mean and 95% interval where the unit count allows one:

| condition | sen2sr_lite − bicubic | sen2sr_mamba − bicubic | tiny_cnn_seed0 − bicubic |
|---|---|---|---|
| `shift_0_lr_px` (0 LR px) | -0.456 [-0.593, -0.335] | -0.463 [-0.638, -0.314] | -0.811 [-1.128, -0.548] |
| `shift_0.25_lr_px` (0.25 LR px) | -0.473 [-0.606, -0.354] | -0.464 [-0.627, -0.323] | -0.800 [-1.092, -0.551] |
| `shift_0.5_lr_px` (0.5 LR px) | -0.496 [-0.623, -0.381] | -0.465 [-0.614, -0.333] | -0.780 [-1.037, -0.557] |
| `shift_1_lr_px` (1 LR px) | -0.509 [-0.616, -0.407] | -0.447 [-0.573, -0.330] | -0.740 [-0.947, -0.555] |
| `shift_2_lr_px` (2 LR px) | -0.380 [-0.449, -0.314] | -0.316 [-0.398, -0.238] | -0.619 [-0.765, -0.487] |
| `aligned_to_bicubic` | -0.438 [-0.576, -0.318] | -0.451 [-0.625, -0.301] | -0.803 [-1.118, -0.541] |

## SSIM ↑

| condition | bicubic | sen2sr_lite | sen2sr_mamba | tiny_cnn_seed0 | n units |
|---|---|---|---|---|---|
| `shift_0_lr_px` (0 LR px) | 0.919 ± 0.123 | 0.903 ± 0.135 | 0.908 ± 0.126 | 0.897 ± 0.145 | 28 |
| `shift_0.25_lr_px` (0.25 LR px) | 0.913 ± 0.123 | 0.897 ± 0.136 | 0.902 ± 0.127 | 0.891 ± 0.145 | 28 |
| `shift_0.5_lr_px` (0.5 LR px) | 0.904 ± 0.123 | 0.885 ± 0.135 | 0.891 ± 0.126 | 0.880 ± 0.145 | 28 |
| `shift_1_lr_px` (1 LR px) | 0.882 ± 0.122 | 0.859 ± 0.134 | 0.868 ± 0.125 | 0.856 ± 0.144 | 28 |
| `shift_2_lr_px` (2 LR px) | 0.860 ± 0.123 | 0.840 ± 0.134 | 0.849 ± 0.126 | 0.834 ± 0.145 | 28 |
| `aligned_to_bicubic` | 0.924 ± 0.108 | 0.910 ± 0.122 | 0.913 ± 0.114 | 0.903 ± 0.134 | 28 |

## SAM (°) ↓

| condition | bicubic | sen2sr_lite | sen2sr_mamba | tiny_cnn_seed0 | n units |
|---|---|---|---|---|---|
| `shift_0_lr_px` (0 LR px) | 3.189 ± 2.518 | 3.348 ± 2.617 | 3.262 ± 2.519 | 3.516 ± 2.806 | 28 |
| `shift_0.25_lr_px` (0.25 LR px) | 3.274 ± 2.511 | 3.434 ± 2.616 | 3.339 ± 2.518 | 3.593 ± 2.810 | 28 |
| `shift_0.5_lr_px` (0.5 LR px) | 3.409 ± 2.506 | 3.575 ± 2.616 | 3.463 ± 2.518 | 3.726 ± 2.817 | 28 |
| `shift_1_lr_px` (1 LR px) | 3.717 ± 2.526 | 3.894 ± 2.644 | 3.754 ± 2.542 | 4.043 ± 2.854 | 28 |
| `shift_2_lr_px` (2 LR px) | 4.209 ± 2.630 | 4.364 ± 2.748 | 4.217 ± 2.639 | 4.520 ± 2.961 | 28 |
| `aligned_to_bicubic` | 3.130 ± 2.493 | 3.286 ± 2.587 | 3.205 ± 2.497 | 3.456 ± 2.771 | 28 |

Paired difference (system − bicubic) over the same scene units, mean and 95% interval where the unit count allows one:

| condition | sen2sr_lite − bicubic | sen2sr_mamba − bicubic | tiny_cnn_seed0 − bicubic |
|---|---|---|---|
| `shift_0_lr_px` (0 LR px) | +0.159 [0.085, 0.263] | +0.073 [0.028, 0.121] | +0.327 [0.163, 0.559] |
| `shift_0.25_lr_px` (0.25 LR px) | +0.160 [0.088, 0.263] | +0.065 [0.021, 0.111] | +0.319 [0.160, 0.545] |
| `shift_0.5_lr_px` (0.5 LR px) | +0.166 [0.095, 0.267] | +0.054 [0.009, 0.098] | +0.316 [0.162, 0.538] |
| `shift_1_lr_px` (1 LR px) | +0.177 [0.107, 0.277] | +0.037 [-0.013, 0.084] | +0.326 [0.174, 0.546] |
| `shift_2_lr_px` (2 LR px) | +0.155 [0.092, 0.250] | +0.008 [-0.039, 0.054] | +0.311 [0.165, 0.525] |
| `aligned_to_bicubic` | +0.156 [0.082, 0.259] | +0.074 [0.031, 0.120] | +0.326 [0.159, 0.559] |

## ERGAS ↓

| condition | bicubic | sen2sr_lite | sen2sr_mamba | tiny_cnn_seed0 | n units |
|---|---|---|---|---|---|
| `shift_0_lr_px` (0 LR px) | 5.515 ± 3.644 | 5.788 ± 3.725 | 5.629 ± 3.718 | 6.198 ± 3.839 | 28 |
| `shift_0.25_lr_px` (0.25 LR px) | 5.641 ± 3.633 | 5.934 ± 3.718 | 5.749 ± 3.710 | 6.341 ± 3.845 | 28 |
| `shift_0.5_lr_px` (0.5 LR px) | 5.857 ± 3.617 | 6.177 ± 3.707 | 5.958 ± 3.698 | 6.583 ± 3.854 | 28 |
| `shift_1_lr_px` (1 LR px) | 6.377 ± 3.638 | 6.736 ± 3.739 | 6.463 ± 3.722 | 7.148 ± 3.923 | 28 |
| `shift_2_lr_px` (2 LR px) | 7.218 ± 3.880 | 7.526 ± 3.987 | 7.258 ± 3.973 | 7.939 ± 4.200 | 28 |
| `aligned_to_bicubic` | 5.411 ± 3.605 | 5.664 ± 3.678 | 5.528 ± 3.678 | 6.075 ± 3.786 | 28 |

## NDVI MAE ↓

| condition | bicubic | sen2sr_lite | sen2sr_mamba | tiny_cnn_seed0 | n units |
|---|---|---|---|---|---|
| `shift_0_lr_px` (0 LR px) | 0.054 ± 0.044 | 0.057 ± 0.045 | 0.055 ± 0.043 | 0.059 ± 0.048 | 28 |
| `shift_0.25_lr_px` (0.25 LR px) | 0.056 ± 0.044 | 0.058 ± 0.044 | 0.056 ± 0.043 | 0.061 ± 0.048 | 28 |
| `shift_0.5_lr_px` (0.5 LR px) | 0.058 ± 0.043 | 0.061 ± 0.044 | 0.059 ± 0.043 | 0.063 ± 0.048 | 28 |
| `shift_1_lr_px` (1 LR px) | 0.064 ± 0.044 | 0.067 ± 0.045 | 0.064 ± 0.043 | 0.069 ± 0.049 | 28 |
| `shift_2_lr_px` (2 LR px) | 0.073 ± 0.046 | 0.075 ± 0.047 | 0.073 ± 0.045 | 0.078 ± 0.051 | 28 |
| `aligned_to_bicubic` | 0.053 ± 0.043 | 0.055 ± 0.044 | 0.054 ± 0.043 | 0.058 ± 0.047 | 28 |

## detail rel. error (1 = no detail added) ↓

| condition | bicubic | sen2sr_lite | sen2sr_mamba | tiny_cnn_seed0 | n units |
|---|---|---|---|---|---|
| `shift_0_lr_px` (0 LR px) | 1.138 ± 0.586 | 1.748 ± 1.035 | 1.833 ± 1.032 | 2.005 ± 1.237 | 28 |
| `shift_0.25_lr_px` (0.25 LR px) | 1.222 ± 0.565 | 1.834 ± 1.020 | 1.908 ± 1.020 | 2.079 ± 1.229 | 28 |
| `shift_0.5_lr_px` (0.5 LR px) | 1.362 ± 0.532 | 1.979 ± 0.988 | 2.032 ± 0.993 | 2.199 ± 1.206 | 28 |
| `shift_1_lr_px` (1 LR px) | 1.582 ± 0.500 | 2.214 ± 0.944 | 2.241 ± 0.953 | 2.402 ± 1.170 | 28 |
| `shift_2_lr_px` (2 LR px) | 1.643 ± 0.481 | 2.217 ± 0.933 | 2.251 ± 0.947 | 2.434 ± 1.150 | 28 |
| `aligned_to_bicubic` | 1.125 ± 0.532 | 1.713 ± 0.985 | 1.796 ± 0.987 | 1.971 ± 1.193 | 28 |

Paired difference (system − bicubic) over the same scene units, mean and 95% interval where the unit count allows one:

| condition | sen2sr_lite − bicubic | sen2sr_mamba − bicubic | tiny_cnn_seed0 − bicubic |
|---|---|---|---|
| `shift_0_lr_px` (0 LR px) | +0.610 [0.453, 0.797] | +0.695 [0.531, 0.888] | +0.867 [0.607, 1.179] |
| `shift_0.25_lr_px` (0.25 LR px) | +0.612 [0.457, 0.799] | +0.686 [0.524, 0.879] | +0.857 [0.598, 1.168] |
| `shift_0.5_lr_px` (0.5 LR px) | +0.617 [0.465, 0.801] | +0.670 [0.511, 0.861] | +0.837 [0.583, 1.145] |
| `shift_1_lr_px` (1 LR px) | +0.632 [0.485, 0.808] | +0.659 [0.506, 0.847] | +0.820 [0.571, 1.125] |
| `shift_2_lr_px` (2 LR px) | +0.574 [0.428, 0.756] | +0.608 [0.455, 0.800] | +0.791 [0.547, 1.095] |
| `aligned_to_bicubic` | +0.589 [0.433, 0.778] | +0.672 [0.509, 0.868] | +0.846 [0.589, 1.156] |

## detail correlation ↑

| condition | bicubic | sen2sr_lite | sen2sr_mamba | tiny_cnn_seed0 | n units |
|---|---|---|---|---|---|
| `shift_0_lr_px` (0 LR px) | 0.529 ± 0.231 | 0.463 ± 0.202 | 0.415 ± 0.176 | 0.403 ± 0.164 | 28 |
| `shift_0.25_lr_px` (0.25 LR px) | 0.456 ± 0.206 | 0.387 ± 0.180 | 0.347 ± 0.156 | 0.334 ± 0.146 | 28 |
| `shift_0.5_lr_px` (0.5 LR px) | 0.313 ± 0.154 | 0.248 ± 0.134 | 0.229 ± 0.117 | 0.214 ± 0.113 | 28 |
| `shift_1_lr_px` (1 LR px) | 0.042 ± 0.100 | -0.011 ± 0.088 | 0.007 ± 0.077 | -0.002 ± 0.077 | 28 |
| `shift_2_lr_px` (2 LR px) | -0.038 ± 0.041 | -0.008 ± 0.038 | -0.015 ± 0.031 | -0.020 ± 0.034 | 28 |
| `aligned_to_bicubic` | 0.547 ± 0.186 | 0.503 ± 0.163 | 0.452 ± 0.150 | 0.442 ± 0.131 | 28 |

## How to read this

* The zero row is the ordinary evaluation of this dataset (restricted to the metrics shown). Each following row scores exactly the same SR against a reference that is further displaced.
* If a displacement of a fraction of an LR pixel changes a metric by more than the differences between systems, then that difference cannot be attributed to the systems on this dataset alone.
* `aligned_to_bicubic` removes an estimated whole-pixel misregistration; a residual sub-pixel misregistration and any radiometric difference between sensors remain.
* Nothing here is combined across datasets and nothing is a ranking.
