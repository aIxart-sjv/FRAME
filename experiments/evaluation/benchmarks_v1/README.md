# Evaluation `benchmarks_v1`

Status: **completed**. Measured by `python -m frame.evaluate run`; every number below is a mean ± standard deviation over **scene units** (tiles of one scene averaged first), with `n units` beside it. **There is no ranking here and none should be read into it**: systems are listed in configuration order, differences are paired and labelled, and results of different datasets are never pooled (no combined score). Where the number of units is too small for an interval or a test, results are **descriptive only**.

Code: git `eb3769559525ea81e902422253f8950c36b075a5` (working tree dirty: True); metrics `frame-eval-metrics/1`; tiling 128 px, overlap 32, reflect padding, linear blend.

Preprocessing (all systems): reflectance fraction = stored DN / the dataset's reflectance_scale (10000); no BOA offset applied, no clipping by FRAME; band order B04, B03, B02, B08 (FRAME's RGBN order, selected by name from the dataset files, not the files' own order).

## Systems evaluated
| system | model | hard constraint | weights / checkpoint |
|---|---|---|---|
| bicubic | bicubic | None | none |
| sen2sr_lite | SEN2SRLite/NonReference_RGBN_x4 | True | model.safetensor:479aa796d506, hard_constraint.safetensor:fbad98151906 |
| sen2sr_lite_no_constraint | SEN2SRLite/NonReference_RGBN_x4 | False | model.safetensor:479aa796d506, hard_constraint.safetensor:fbad98151906 |
| sen2sr_mamba | SEN2SR/MambaSR_RGBN_x4 | True | weights_sha256:11e551b03663, hard_constraint_sha256:fbad98151906 |
| tiny_cnn_seed0 | tiny_cnn | False | checkpoint_sha256:58fb6bb7c1b2 |
| tiny_cnn_seed1 | tiny_cnn | False | checkpoint_sha256:21d17803dff6 |
| tiny_cnn_seed2 | tiny_cnn | False | checkpoint_sha256:ef5a9daa3fbf |
| tiny_cnn_seed3 | tiny_cnn | False | checkpoint_sha256:aa11b41babee |
| tiny_cnn_seed4 | tiny_cnn | False | checkpoint_sha256:5a9eee65e423 |

## Evaluation matrix

Each cell is samples evaluated / records in the dataset. Common to every cell of a row: model and weights as in the table above; hard constraint and tiling as stated there.

| system | sen2neon_random30 | sen2neon_phase3_sample | opensr_spot | opensr_spain_crops | opensr_spain_urban | synthetic_smoke_test |
|---|---|---|---|---|---|---|
| bicubic | 30/30 | 3/3 | 9/9 | 28/28 | 20/20 | 2/2 |
| sen2sr_lite | 30/30 | 3/3 | 9/9 | 28/28 | 20/20 | 2/2 |
| sen2sr_lite_no_constraint | 30/30 | 3/3 | 9/9 | 28/28 | 20/20 | 2/2 |
| sen2sr_mamba | 30/30 | 3/3 | 9/9 | 28/28 | 20/20 | 2/2 |
| tiny_cnn_seed0 | 30/30 | 3/3 | 9/9 | 28/28 | 20/20 | 2/2 |
| tiny_cnn_seed1 | 30/30 | 3/3 | 9/9 | 28/28 | 20/20 | 2/2 |
| tiny_cnn_seed2 | 30/30 | 3/3 | 9/9 | 28/28 | 20/20 | 2/2 |
| tiny_cnn_seed3 | 30/30 | 3/3 | 9/9 | 28/28 | 20/20 | 2/2 |
| tiny_cnn_seed4 | 30/30 | 3/3 | 9/9 | 28/28 | 20/20 | 2/2 |

| dataset | evidence class | split | scenes | LR | HR | bands |
|---|---|---|---|---|---|---|
| sen2neon_random30 | real_cross_sensor | test | 28 | 10 m | 2.5 m | B04, B03, B02, B08 |
| sen2neon_phase3_sample | real_cross_sensor | test | 2 | 10 m | 2.5 m | B04, B03, B02, B08 |
| opensr_spot | real_cross_sensor | test | 9 | 10 m | 2.5 m | B04, B03, B02, B08 |
| opensr_spain_crops | real_cross_sensor | test | 5 | 10 m | 2.5 m | B04, B03, B02, B08 |
| opensr_spain_urban | real_cross_sensor | test | 4 | 10 m | 2.5 m | B04, B03, B02, B08 |
| synthetic_smoke_test | synthetic | test | 1 | 10 m | 2.5 m | B04, B03, B02, B08 |

Hard constraint: `True` = the intended system (low-frequency Fourier constraint inside each 512×512 tile output); `False` = a system without it (the ablation of Lite, or a trained checkpoint); `None` = not applicable (bicubic).

## Evidence class: `real_cross_sensor`

### Dataset `sen2neon_random30` (sen2neon, role independent_benchmark, split test)

Records 30: **evaluated 30**, skipped 0, invalid 0, unreadable 0; 28 scene units; manifest digest `ab433be9ceada8a1…`.
Dataset-defined categories (descriptive only): {'Developed': 1, 'Forest': 10, 'Rural': 16, 'Water': 1, 'unlabelled': 2}

**Reference accuracy (SR vs the independent HR reference)**

| system | n units | PSNR (dB) ↑ | SSIM ↑ | RMSE ↓ | MAE ↓ | SAM (°) ↓ | ERGAS ↓ |
|---|---|---|---|---|---|---|---|
| bicubic | 28 | 33.6867 ± 6.0254 | 0.9188 ± 0.1229 | 0.0271 ± 0.0272 | 0.0176 ± 0.0222 | 3.1892 ± 2.5176 | 5.5151 ± 3.6443 |
| sen2sr_lite | 28 | 33.2307 ± 5.8387 | 0.9034 ± 0.1354 | 0.0282 ± 0.0285 | 0.0184 ± 0.0233 | 3.3482 ± 2.6174 | 5.7883 ± 3.7250 |
| sen2sr_lite_no_constraint | 28 | 32.1263 ± 5.4652 | 0.8799 ± 0.1418 | 0.0314 ± 0.0311 | 0.0212 ± 0.0252 | 4.0892 ± 2.8330 | 6.7321 ± 4.0438 |
| sen2sr_mamba | 28 | 33.2234 ± 5.7951 | 0.9077 ± 0.1262 | 0.0282 ± 0.0284 | 0.0183 ± 0.0232 | 3.2623 ± 2.5188 | 5.6290 ± 3.7178 |
| tiny_cnn_seed0 | 28 | 32.8754 ± 5.7183 | 0.8969 ± 0.1448 | 0.0294 ± 0.0303 | 0.0191 ± 0.0249 | 3.5164 ± 2.8058 | 6.1979 ± 3.8387 |
| tiny_cnn_seed1 | 28 | 32.8063 ± 5.7485 | 0.8940 ± 0.1461 | 0.0296 ± 0.0303 | 0.0193 ± 0.0249 | 3.5509 ± 2.8093 | 6.2519 ± 3.8463 |
| tiny_cnn_seed2 | 28 | 32.8073 ± 5.6824 | 0.8946 ± 0.1439 | 0.0294 ± 0.0297 | 0.0192 ± 0.0244 | 3.5354 ± 2.7568 | 6.2170 ± 3.7941 |
| tiny_cnn_seed3 | 28 | 32.9389 ± 5.7467 | 0.8991 ± 0.1411 | 0.0291 ± 0.0294 | 0.0190 ± 0.0242 | 3.5247 ± 2.8087 | 6.1596 ± 3.7881 |
| tiny_cnn_seed4 | 28 | 32.9244 ± 5.7580 | 0.8981 ± 0.1450 | 0.0292 ± 0.0300 | 0.0191 ± 0.0248 | 3.4957 ± 2.8031 | 6.1493 ± 3.8272 |

**Derived indices and band ratios (vs the HR reference)**

| system | n units | NDVI MAE ↓ | NDVI bias | NDVI r ↑ | NDWI MAE ↓ | log(B08/B04) bias |
|---|---|---|---|---|---|---|
| bicubic | 28 | 0.0541 ± 0.0437 | 0.0016 ± 0.0152 | 0.7489 ± 0.2362 | 0.0419 ± 0.0341 | 0.0217 ± 0.0516 |
| sen2sr_lite | 28 | 0.0567 ± 0.0445 | 0.0021 ± 0.0164 | 0.7409 ± 0.2338 | 0.0442 ± 0.0355 | 0.0245 ± 0.0537 |
| sen2sr_lite_no_constraint | 28 | 0.0715 ± 0.0470 | 0.0128 ± 0.0216 | 0.7169 ± 0.2531 | 0.0538 ± 0.0380 | 0.1275 ± 0.1408 |
| sen2sr_mamba | 28 | 0.0551 ± 0.0430 | 0.0009 ± 0.0150 | 0.7424 ± 0.2178 | 0.0433 ± 0.0335 | 0.0177 ± 0.0513 |
| tiny_cnn_seed0 | 28 | 0.0594 ± 0.0479 | 0.0030 ± 0.0192 | 0.7340 ± 0.2318 | 0.0465 ± 0.0388 | 0.0290 ± 0.0609 |
| tiny_cnn_seed1 | 28 | 0.0600 ± 0.0480 | 0.0034 ± 0.0191 | 0.7324 ± 0.2340 | 0.0469 ± 0.0389 | 0.0320 ± 0.0600 |
| tiny_cnn_seed2 | 28 | 0.0597 ± 0.0472 | 0.0026 ± 0.0185 | 0.7292 ± 0.2343 | 0.0468 ± 0.0380 | 0.0280 ± 0.0592 |
| tiny_cnn_seed3 | 28 | 0.0595 ± 0.0479 | 0.0022 ± 0.0184 | 0.7350 ± 0.2335 | 0.0467 ± 0.0390 | 0.0273 ± 0.0589 |
| tiny_cnn_seed4 | 28 | 0.0590 ± 0.0477 | 0.0029 ± 0.0194 | 0.7363 ± 0.2331 | 0.0463 ± 0.0388 | 0.0284 ± 0.0609 |

**Spatial / detail correctness (vs the HR reference)**

| system | n units | detail rel. error (1 = no detail added) | detail correlation ↑ | detail energy ratio (1 = same amount) | edge correlation ↑ | shift (HR px) | seam / interior error |
|---|---|---|---|---|---|---|---|
| bicubic | 28 | 1.1380 ± 0.5855 | 0.5293 ± 0.2310 | 1.8280 ± 1.9909 | 0.5932 ± 0.1983 | 1.5034 ± 1.3062 | n/a |
| sen2sr_lite | 28 | 1.7479 ± 1.0346 | 0.4634 ± 0.2019 | 4.7122 ± 5.8707 | 0.5491 ± 0.1800 | 1.4208 ± 1.3195 | 0.9881 ± 0.0902 |
| sen2sr_lite_no_constraint | 28 | 2.0065 ± 1.2021 | 0.4677 ± 0.2036 | 6.3149 ± 7.8034 | 0.5489 ± 0.1800 | 1.4213 ± 1.3267 | 0.9938 ± 0.0830 |
| sen2sr_mamba | 28 | 1.8327 ± 1.0318 | 0.4146 ± 0.1755 | 4.8766 ± 6.2884 | 0.5128 ± 0.1636 | 1.4387 ± 1.3659 | 0.9881 ± 0.0912 |
| tiny_cnn_seed0 | 28 | 2.0054 ± 1.2372 | 0.4032 ± 0.1644 | 6.1001 ± 8.9457 | 0.5107 ± 0.1539 | 1.4347 ± 1.3188 | 0.9900 ± 0.0908 |
| tiny_cnn_seed1 | 28 | 2.0551 ± 1.2541 | 0.4106 ± 0.1721 | 6.4483 ± 9.0658 | 0.5205 ± 0.1579 | 1.4202 ± 1.3247 | 0.9895 ± 0.0905 |
| tiny_cnn_seed2 | 28 | 2.0270 ± 1.1781 | 0.4143 ± 0.1725 | 6.1431 ± 7.8681 | 0.5194 ± 0.1574 | 1.4494 ± 1.3001 | 0.9894 ± 0.0903 |
| tiny_cnn_seed3 | 28 | 1.8627 ± 1.0980 | 0.4347 ± 0.1842 | 5.2619 ± 6.7968 | 0.5324 ± 0.1653 | 1.3646 ± 1.3524 | 0.9890 ± 0.0896 |
| tiny_cnn_seed4 | 28 | 1.9409 ± 1.1967 | 0.4168 ± 0.1733 | 5.7700 ± 8.2158 | 0.5202 ± 0.1586 | 1.4599 ± 1.2973 | 0.9892 ± 0.0909 |

**Per-band reflectance error** (RMSE ↓ / bias = SR − reference; bands are the RGBN bands every system produces)

| system | B02 RMSE | B03 RMSE | B04 RMSE | B08 RMSE | B02 bias | B03 bias | B04 bias | B08 bias |
|---|---|---|---|---|---|---|---|---|
| bicubic | 0.0168 ± 0.0305 | 0.0193 ± 0.0297 | 0.0226 ± 0.0301 | 0.0380 ± 0.0240 | 0.0001 ± 0.0002 | 0.0001 ± 0.0002 | 0.0001 ± 0.0001 | 0.0001 ± 0.0003 |
| sen2sr_lite | 0.0178 ± 0.0322 | 0.0203 ± 0.0312 | 0.0238 ± 0.0316 | 0.0393 ± 0.0246 | 0.0001 ± 0.0002 | 0.0001 ± 0.0002 | 0.0001 ± 0.0001 | 0.0001 ± 0.0003 |
| sen2sr_lite_no_constraint | 0.0201 ± 0.0355 | 0.0227 ± 0.0344 | 0.0270 ± 0.0342 | 0.0434 ± 0.0257 | -0.0017 ± 0.0023 | -0.0018 ± 0.0027 | -0.0031 ± 0.0037 | -0.0075 ± 0.0026 |
| sen2sr_mamba | 0.0176 ± 0.0323 | 0.0200 ± 0.0312 | 0.0234 ± 0.0313 | 0.0396 ± 0.0247 | 0.0001 ± 0.0002 | 0.0001 ± 0.0002 | 0.0001 ± 0.0001 | 0.0001 ± 0.0003 |
| tiny_cnn_seed0 | 0.0195 ± 0.0335 | 0.0218 ± 0.0331 | 0.0256 ± 0.0342 | 0.0397 ± 0.0257 | -0.0000 ± 0.0004 | 0.0000 ± 0.0002 | 0.0001 ± 0.0002 | 0.0000 ± 0.0004 |
| tiny_cnn_seed1 | 0.0195 ± 0.0335 | 0.0218 ± 0.0330 | 0.0257 ± 0.0342 | 0.0403 ± 0.0258 | -0.0000 ± 0.0004 | 0.0000 ± 0.0002 | 0.0000 ± 0.0001 | 0.0001 ± 0.0004 |
| tiny_cnn_seed2 | 0.0194 ± 0.0330 | 0.0217 ± 0.0324 | 0.0255 ± 0.0335 | 0.0401 ± 0.0253 | 0.0000 ± 0.0003 | 0.0001 ± 0.0002 | 0.0001 ± 0.0002 | -0.0000 ± 0.0004 |
| tiny_cnn_seed3 | 0.0192 ± 0.0329 | 0.0214 ± 0.0322 | 0.0252 ± 0.0332 | 0.0394 ± 0.0247 | 0.0001 ± 0.0002 | -0.0000 ± 0.0002 | 0.0001 ± 0.0002 | -0.0007 ± 0.0007 |
| tiny_cnn_seed4 | 0.0193 ± 0.0334 | 0.0216 ± 0.0328 | 0.0254 ± 0.0340 | 0.0396 ± 0.0253 | -0.0000 ± 0.0003 | 0.0001 ± 0.0002 | 0.0001 ± 0.0001 | 0.0000 ± 0.0004 |

**What was added relative to bicubic, judged against the reference** (element-wise; τ = 0.005 reflectance; fractions of valid elements; the reference is a different sensor, so "unsupported" means "not confirmed by this reference", not "false")

| system | n units | supported synthesis | unsupported detail | omission | MSE skill vs bicubic |
|---|---|---|---|---|---|
| sen2sr_lite | 28 | 0.0481 ± 0.0544 | 0.0937 ± 0.0912 | 0.4773 ± 0.1779 | -0.1145 ± 0.0969 |
| sen2sr_lite_no_constraint | 28 | 0.0952 ± 0.0378 | 0.3638 ± 0.1008 | 0.2818 ± 0.1287 | -0.4578 ± 0.2926 |
| sen2sr_mamba | 28 | 0.0581 ± 0.0622 | 0.0949 ± 0.0843 | 0.4694 ± 0.1770 | -0.1188 ± 0.1276 |
| tiny_cnn_seed0 | 28 | 0.0396 ± 0.0513 | 0.1015 ± 0.1183 | 0.4846 ± 0.1670 | -0.2287 ± 0.2762 |
| tiny_cnn_seed1 | 28 | 0.0464 ± 0.0513 | 0.1185 ± 0.1169 | 0.4646 ± 0.1670 | -0.2477 ± 0.2775 |
| tiny_cnn_seed2 | 28 | 0.0449 ± 0.0521 | 0.1150 ± 0.1140 | 0.4691 ± 0.1649 | -0.2481 ± 0.2791 |
| tiny_cnn_seed3 | 28 | 0.0365 ± 0.0496 | 0.1001 ± 0.1109 | 0.4867 ± 0.1652 | -0.2031 ± 0.2139 |
| tiny_cnn_seed4 | 28 | 0.0363 ± 0.0501 | 0.0940 ± 0.1174 | 0.4914 ± 0.1689 | -0.2127 ± 0.2605 |

**Self-consistency (SR → area-average → compare with the LR)**. This is NOT accuracy against an HR reference: a system can average back to the LR perfectly and still be far from the reference.

| system | n units | MAE vs LR | RMSE vs LR | NDVI MAE vs LR |
|---|---|---|---|---|
| bicubic | 28 | 0.0016 ± 0.0016 | 0.0026 ± 0.0022 | 0.0052 ± 0.0031 |
| sen2sr_lite | 28 | 0.0012 ± 0.0013 | 0.0019 ± 0.0018 | 0.0035 ± 0.0027 |
| sen2sr_lite_no_constraint | 28 | 0.0054 ± 0.0022 | 0.0067 ± 0.0031 | 0.0244 ± 0.0105 |
| sen2sr_mamba | 28 | 0.0019 ± 0.0021 | 0.0029 ± 0.0028 | 0.0099 ± 0.0077 |
| tiny_cnn_seed0 | 28 | 0.0019 ± 0.0029 | 0.0035 ± 0.0040 | 0.0064 ± 0.0074 |
| tiny_cnn_seed1 | 28 | 0.0020 ± 0.0030 | 0.0037 ± 0.0041 | 0.0070 ± 0.0072 |
| tiny_cnn_seed2 | 28 | 0.0018 ± 0.0025 | 0.0034 ± 0.0034 | 0.0067 ± 0.0069 |
| tiny_cnn_seed3 | 28 | 0.0019 ± 0.0023 | 0.0034 ± 0.0032 | 0.0072 ± 0.0083 |
| tiny_cnn_seed4 | 28 | 0.0018 ± 0.0028 | 0.0034 ± 0.0038 | 0.0063 ± 0.0076 |

**By dataset-defined category** (the dataset's own labels, never inferred; descriptive only: few samples per category, no interval or test)

| category | n samples | n units | system | PSNR (dB) | SAM (°) | detail rel. error |
|---|---|---|---|---|---|---|
| Developed | 1 | 1 | bicubic | 33.2851 | 2.7854 | 0.6595 |
| Developed | 1 | 1 | sen2sr_lite | 32.1124 | 3.4713 | 1.2290 |
| Developed | 1 | 1 | sen2sr_lite_no_constraint | 30.7428 | 4.5326 | 1.4659 |
| Developed | 1 | 1 | sen2sr_mamba | 31.9710 | 3.2270 | 1.4617 |
| Developed | 1 | 1 | tiny_cnn_seed0 | 30.3109 | 4.5629 | 2.0637 |
| Developed | 1 | 1 | tiny_cnn_seed1 | 29.9735 | 4.8727 | 2.1457 |
| Developed | 1 | 1 | tiny_cnn_seed2 | 30.1553 | 4.7123 | 2.0645 |
| Developed | 1 | 1 | tiny_cnn_seed3 | 30.6154 | 4.4476 | 1.8991 |
| Developed | 1 | 1 | tiny_cnn_seed4 | 30.3941 | 4.4843 | 1.9942 |
| Forest | 10 | 10 | bicubic | 32.6104 ± 6.2358 | 3.0898 ± 2.4681 | 1.4204 ± 0.7733 |
| Forest | 10 | 10 | sen2sr_lite | 32.1041 ± 6.1725 | 3.3421 ± 2.8054 | 2.2115 ± 1.3973 |
| Forest | 10 | 10 | sen2sr_lite_no_constraint | 30.9885 ± 5.9617 | 4.4129 ± 3.1690 | 2.5319 ± 1.6243 |
| Forest | 10 | 10 | sen2sr_mamba | 32.2510 ± 6.2572 | 3.1246 ± 2.5380 | 2.1790 ± 1.4537 |
| Forest | 10 | 10 | tiny_cnn_seed0 | 31.8990 ± 6.2262 | 3.5245 ± 3.2223 | 2.4548 ± 1.8344 |
| Forest | 10 | 10 | tiny_cnn_seed1 | 31.7495 ± 6.1768 | 3.5848 ± 3.1983 | 2.5542 ± 1.8292 |
| Forest | 10 | 10 | tiny_cnn_seed2 | 31.7817 ± 6.1190 | 3.5524 ± 3.0937 | 2.4980 ± 1.7054 |
| Forest | 10 | 10 | tiny_cnn_seed3 | 31.8797 ± 6.1145 | 3.5556 ± 3.2259 | 2.3129 ± 1.5840 |
| Forest | 10 | 10 | tiny_cnn_seed4 | 31.9141 ± 6.2034 | 3.5170 ± 3.2150 | 2.3938 ± 1.7500 |
| Rural | 16 | 14 | bicubic | 34.9635 ± 6.3923 | 3.3570 ± 2.7817 | 0.9706 ± 0.3093 |
| Rural | 16 | 14 | sen2sr_lite | 34.5267 ± 6.0783 | 3.4433 ± 2.7520 | 1.4077 ± 0.4760 |
| Rural | 16 | 14 | sen2sr_lite_no_constraint | 33.3884 ± 5.5637 | 3.9577 ± 2.9455 | 1.6074 ± 0.5613 |
| Rural | 16 | 14 | sen2sr_mamba | 34.4151 ± 5.9755 | 3.4509 ± 2.7310 | 1.5705 ± 0.5127 |
| Rural | 16 | 14 | tiny_cnn_seed0 | 34.0807 ± 5.8366 | 3.5667 ± 2.7857 | 1.7997 ± 0.6353 |
| Rural | 16 | 14 | tiny_cnn_seed1 | 34.1008 ± 5.9017 | 3.5606 ± 2.8074 | 1.7734 ± 0.6600 |
| Rural | 16 | 14 | tiny_cnn_seed2 | 34.0483 ± 5.8298 | 3.5731 ± 2.7842 | 1.7879 ± 0.6267 |
| Rural | 16 | 14 | tiny_cnn_seed3 | 34.2118 ± 5.9492 | 3.5668 ± 2.7941 | 1.6370 ± 0.5923 |
| Rural | 16 | 14 | tiny_cnn_seed4 | 34.1646 ± 5.9155 | 3.5355 ± 2.7917 | 1.7129 ± 0.6406 |
| Water | 1 | 1 | bicubic | 29.3964 | 2.7525 | 0.7432 |
| Water | 1 | 1 | sen2sr_lite | 29.1925 | 2.8082 | 1.2625 |
| Water | 1 | 1 | sen2sr_lite_no_constraint | 28.5533 | 3.9635 | 1.5016 |
| Water | 1 | 1 | sen2sr_mamba | 29.1765 | 2.7794 | 1.3849 |
| Water | 1 | 1 | tiny_cnn_seed0 | 29.2630 | 2.8518 | 0.9557 |
| Water | 1 | 1 | tiny_cnn_seed1 | 29.1397 | 2.9280 | 1.1413 |
| Water | 1 | 1 | tiny_cnn_seed2 | 29.1946 | 2.8720 | 1.0531 |
| Water | 1 | 1 | tiny_cnn_seed3 | 29.2509 | 2.8547 | 0.9023 |
| Water | 1 | 1 | tiny_cnn_seed4 | 29.2391 | 2.8615 | 0.9876 |
| unlabelled | 2 | 2 | bicubic | 35.3026 ± 0.5506 | 1.5751 ± 0.4096 | 1.4232 ± 0.9477 |
| unlabelled | 2 | 2 | sen2sr_lite | 35.0248 ± 0.3907 | 1.6008 ± 0.3963 | 2.4900 ± 1.6771 |
| unlabelled | 2 | 2 | sen2sr_lite_no_constraint | 34.0565 ± 0.2401 | 2.1782 ± 0.4314 | 2.9070 ± 1.9093 |
| unlabelled | 2 | 2 | sen2sr_mamba | 35.0419 ± 0.3857 | 1.5607 ± 0.3797 | 2.4934 ± 1.6308 |
| unlabelled | 2 | 2 | tiny_cnn_seed0 | 35.1771 ± 0.4973 | 1.6024 ± 0.4093 | 1.6705 ± 1.1427 |
| unlabelled | 2 | 2 | tiny_cnn_seed1 | 35.0697 ± 0.4153 | 1.6182 ± 0.4212 | 1.8910 ± 1.2392 |
| unlabelled | 2 | 2 | tiny_cnn_seed2 | 35.1155 ± 0.4658 | 1.6133 ± 0.4117 | 1.8171 ± 1.2272 |
| unlabelled | 2 | 2 | tiny_cnn_seed3 | 35.0728 ± 0.4624 | 1.6198 ± 0.4169 | 1.6505 ± 1.1197 |
| unlabelled | 2 | 2 | tiny_cnn_seed4 | 35.1737 ± 0.5083 | 1.5996 ± 0.4112 | 1.6777 ± 1.1623 |

**Paired differences A − B over scene units** (sign convention: A minus B; for error metrics lower is smaller error; a difference is not a ranking)

| A − B | metric | n units | mean difference | median | 95% CI | Wilcoxon p | label |
|---|---|---|---|---|---|---|---|
| sen2sr_lite - bicubic | PSNR (dB) ↑ | 28 | -0.4560 | -0.3962 | [-0.5931, -0.3346] | 0.0000 | inferential |
| sen2sr_lite - bicubic | SSIM ↑ | 28 | -0.0154 | -0.0120 | [-0.0225, -0.0101] | 0.0000 | inferential |
| sen2sr_lite - bicubic | RMSE ↓ | 28 | 0.0012 | 0.0008 | [0.0007, 0.0018] | 0.0000 | inferential |
| sen2sr_lite - bicubic | SAM (°) ↓ | 28 | 0.1590 | 0.0891 | [0.0850, 0.2631] | 0.0000 | inferential |
| sen2sr_lite - bicubic | ERGAS ↓ | 28 | 0.2732 | 0.2004 | [0.1965, 0.3599] | 0.0000 | inferential |
| sen2sr_lite - bicubic | NDVI MAE ↓ | 28 | 0.0026 | 0.0015 | [0.0015, 0.0043] | 0.0000 | inferential |
| sen2sr_lite - bicubic | detail rel. error (1 = no detail added) | 28 | 0.6099 | 0.5461 | [0.4530, 0.7975] | 0.0000 | inferential |
| sen2sr_lite - bicubic | MAE vs LR | 28 | -0.0004 | -0.0004 | [-0.0006, -0.0003] | 0.0000 | inferential |
| sen2sr_lite_no_constraint - bicubic | PSNR (dB) ↑ | 28 | -1.5604 | -1.4459 | [-1.8610, -1.2807] | 0.0000 | inferential |
| sen2sr_lite_no_constraint - bicubic | SSIM ↑ | 28 | -0.0389 | -0.0354 | [-0.0490, -0.0305] | 0.0000 | inferential |
| sen2sr_lite_no_constraint - bicubic | RMSE ↓ | 28 | 0.0043 | 0.0033 | [0.0032, 0.0061] | 0.0000 | inferential |
| sen2sr_lite_no_constraint - bicubic | SAM (°) ↓ | 26 | 0.8383 | 0.7544 | [0.6766, 1.0208] | 0.0000 | inferential |
| sen2sr_lite_no_constraint - bicubic | ERGAS ↓ | 28 | 1.2170 | 1.1395 | [0.9670, 1.4854] | 0.0000 | inferential |
| sen2sr_lite_no_constraint - bicubic | NDVI MAE ↓ | 28 | 0.0174 | 0.0175 | [0.0144, 0.0205] | 0.0000 | inferential |
| sen2sr_lite_no_constraint - bicubic | detail rel. error (1 = no detail added) | 28 | 0.8685 | 0.7811 | [0.6538, 1.1229] | 0.0000 | inferential |
| sen2sr_lite_no_constraint - bicubic | MAE vs LR | 28 | 0.0038 | 0.0038 | [0.0034, 0.0042] | 0.0000 | inferential |
| sen2sr_mamba - bicubic | PSNR (dB) ↑ | 28 | -0.4633 | -0.2970 | [-0.6384, -0.3141] | 0.0000 | inferential |
| sen2sr_mamba - bicubic | SSIM ↑ | 28 | -0.0112 | -0.0091 | [-0.0150, -0.0076] | 0.0000 | inferential |
| sen2sr_mamba - bicubic | RMSE ↓ | 28 | 0.0011 | 0.0008 | [0.0007, 0.0018] | 0.0000 | inferential |
| sen2sr_mamba - bicubic | SAM (°) ↓ | 28 | 0.0731 | 0.0582 | [0.0281, 0.1209] | 0.0034 | inferential |
| sen2sr_mamba - bicubic | ERGAS ↓ | 28 | 0.1138 | 0.0963 | [0.0170, 0.2144] | 0.0337 | inferential |
| sen2sr_mamba - bicubic | NDVI MAE ↓ | 28 | 0.0010 | 0.0008 | [0.0003, 0.0016] | 0.0118 | inferential |
| sen2sr_mamba - bicubic | detail rel. error (1 = no detail added) | 28 | 0.6947 | 0.6165 | [0.5312, 0.8883] | 0.0000 | inferential |
| sen2sr_mamba - bicubic | MAE vs LR | 28 | 0.0003 | 0.0002 | [0.0001, 0.0006] | 0.0095 | inferential |
| tiny_cnn_seed0 - bicubic | PSNR (dB) ↑ | 28 | -0.8114 | -0.6010 | [-1.1284, -0.5482] | 0.0000 | inferential |
| tiny_cnn_seed0 - bicubic | SSIM ↑ | 28 | -0.0220 | -0.0133 | [-0.0342, -0.0128] | 0.0000 | inferential |
| tiny_cnn_seed0 - bicubic | RMSE ↓ | 28 | 0.0023 | 0.0013 | [0.0012, 0.0039] | 0.0000 | inferential |
| tiny_cnn_seed0 - bicubic | SAM (°) ↓ | 28 | 0.3273 | 0.1759 | [0.1626, 0.5590] | 0.0000 | inferential |
| tiny_cnn_seed0 - bicubic | ERGAS ↓ | 28 | 0.6828 | 0.5173 | [0.4750, 0.9233] | 0.0000 | inferential |
| tiny_cnn_seed0 - bicubic | NDVI MAE ↓ | 28 | 0.0053 | 0.0024 | [0.0023, 0.0096] | 0.0000 | inferential |
| tiny_cnn_seed0 - bicubic | detail rel. error (1 = no detail added) | 28 | 0.8673 | 0.7065 | [0.6066, 1.1791] | 0.0000 | inferential |
| tiny_cnn_seed0 - bicubic | MAE vs LR | 28 | 0.0003 | 0.0000 | [-0.0000, 0.0008] | 0.6456 | inferential |
| tiny_cnn_seed1 - bicubic | PSNR (dB) ↑ | 28 | -0.8804 | -0.6708 | [-1.1949, -0.6187] | 0.0000 | inferential |
| tiny_cnn_seed1 - bicubic | SSIM ↑ | 28 | -0.0248 | -0.0176 | [-0.0379, -0.0151] | 0.0000 | inferential |
| tiny_cnn_seed1 - bicubic | RMSE ↓ | 28 | 0.0025 | 0.0017 | [0.0014, 0.0041] | 0.0000 | inferential |
| tiny_cnn_seed1 - bicubic | SAM (°) ↓ | 28 | 0.3618 | 0.2021 | [0.1876, 0.6048] | 0.0000 | inferential |
| tiny_cnn_seed1 - bicubic | ERGAS ↓ | 28 | 0.7367 | 0.5833 | [0.5319, 0.9756] | 0.0000 | inferential |
| tiny_cnn_seed1 - bicubic | NDVI MAE ↓ | 28 | 0.0059 | 0.0028 | [0.0028, 0.0104] | 0.0000 | inferential |
| tiny_cnn_seed1 - bicubic | detail rel. error (1 = no detail added) | 28 | 0.9170 | 0.7478 | [0.6630, 1.2244] | 0.0000 | inferential |
| tiny_cnn_seed1 - bicubic | MAE vs LR | 28 | 0.0004 | 0.0001 | [0.0000, 0.0010] | 0.0900 | inferential |
| tiny_cnn_seed2 - bicubic | PSNR (dB) ↑ | 28 | -0.8795 | -0.6612 | [-1.1933, -0.6123] | 0.0000 | inferential |
| tiny_cnn_seed2 - bicubic | SSIM ↑ | 28 | -0.0242 | -0.0168 | [-0.0363, -0.0149] | 0.0000 | inferential |
| tiny_cnn_seed2 - bicubic | RMSE ↓ | 28 | 0.0023 | 0.0017 | [0.0014, 0.0037] | 0.0000 | inferential |
| tiny_cnn_seed2 - bicubic | SAM (°) ↓ | 28 | 0.3462 | 0.2110 | [0.1887, 0.5593] | 0.0000 | inferential |
| tiny_cnn_seed2 - bicubic | ERGAS ↓ | 28 | 0.7019 | 0.5528 | [0.5081, 0.9204] | 0.0000 | inferential |
| tiny_cnn_seed2 - bicubic | NDVI MAE ↓ | 28 | 0.0056 | 0.0032 | [0.0028, 0.0095] | 0.0000 | inferential |
| tiny_cnn_seed2 - bicubic | detail rel. error (1 = no detail added) | 28 | 0.8890 | 0.7457 | [0.6477, 1.1661] | 0.0000 | inferential |
| tiny_cnn_seed2 - bicubic | MAE vs LR | 28 | 0.0002 | 0.0001 | [-0.0000, 0.0006] | 0.3620 | inferential |
| tiny_cnn_seed3 - bicubic | PSNR (dB) ↑ | 28 | -0.7478 | -0.5741 | [-1.0070, -0.5271] | 0.0000 | inferential |
| tiny_cnn_seed3 - bicubic | SSIM ↑ | 28 | -0.0197 | -0.0143 | [-0.0302, -0.0117] | 0.0000 | inferential |
| tiny_cnn_seed3 - bicubic | RMSE ↓ | 28 | 0.0020 | 0.0015 | [0.0012, 0.0031] | 0.0000 | inferential |
| tiny_cnn_seed3 - bicubic | SAM (°) ↓ | 28 | 0.3356 | 0.1994 | [0.1738, 0.5635] | 0.0000 | inferential |
| tiny_cnn_seed3 - bicubic | ERGAS ↓ | 28 | 0.6445 | 0.5130 | [0.4596, 0.8556] | 0.0000 | inferential |
| tiny_cnn_seed3 - bicubic | NDVI MAE ↓ | 28 | 0.0054 | 0.0029 | [0.0025, 0.0096] | 0.0000 | inferential |
| tiny_cnn_seed3 - bicubic | detail rel. error (1 = no detail added) | 28 | 0.7247 | 0.5918 | [0.5077, 0.9723] | 0.0000 | inferential |
| tiny_cnn_seed3 - bicubic | MAE vs LR | 28 | 0.0003 | 0.0002 | [0.0001, 0.0005] | 0.0118 | inferential |
| tiny_cnn_seed4 - bicubic | PSNR (dB) ↑ | 28 | -0.7623 | -0.5732 | [-1.0635, -0.5146] | 0.0000 | inferential |
| tiny_cnn_seed4 - bicubic | SSIM ↑ | 28 | -0.0207 | -0.0134 | [-0.0327, -0.0118] | 0.0000 | inferential |
| tiny_cnn_seed4 - bicubic | RMSE ↓ | 28 | 0.0021 | 0.0013 | [0.0012, 0.0036] | 0.0000 | inferential |
| tiny_cnn_seed4 - bicubic | SAM (°) ↓ | 28 | 0.3065 | 0.1540 | [0.1452, 0.5338] | 0.0000 | inferential |
| tiny_cnn_seed4 - bicubic | ERGAS ↓ | 28 | 0.6342 | 0.5089 | [0.4404, 0.8583] | 0.0000 | inferential |
| tiny_cnn_seed4 - bicubic | NDVI MAE ↓ | 28 | 0.0049 | 0.0024 | [0.0021, 0.0092] | 0.0000 | inferential |
| tiny_cnn_seed4 - bicubic | detail rel. error (1 = no detail added) | 28 | 0.8029 | 0.6200 | [0.5579, 1.0937] | 0.0000 | inferential |
| tiny_cnn_seed4 - bicubic | MAE vs LR | 28 | 0.0002 | -0.0000 | [-0.0001, 0.0007] | 0.4932 | inferential |
| sen2sr_mamba - sen2sr_lite | PSNR (dB) ↑ | 28 | -0.0073 | -0.0124 | [-0.0746, 0.0636] | 0.3741 | inferential |
| sen2sr_mamba - sen2sr_lite | SSIM ↑ | 28 | 0.0042 | 0.0013 | [0.0009, 0.0089] | 0.0595 | inferential |
| sen2sr_mamba - sen2sr_lite | RMSE ↓ | 28 | -0.0000 | 0.0000 | [-0.0002, 0.0001] | 0.4515 | inferential |
| sen2sr_mamba - sen2sr_lite | SAM (°) ↓ | 28 | -0.0859 | -0.0230 | [-0.1684, -0.0229] | 0.0451 | inferential |
| sen2sr_mamba - sen2sr_lite | ERGAS ↓ | 28 | -0.1593 | -0.0598 | [-0.2650, -0.0678] | 0.0095 | inferential |
| sen2sr_mamba - sen2sr_lite | NDVI MAE ↓ | 28 | -0.0017 | -0.0004 | [-0.0033, -0.0005] | 0.0247 | inferential |
| sen2sr_mamba - sen2sr_lite | detail rel. error (1 = no detail added) | 28 | 0.0848 | 0.1005 | [0.0182, 0.1438] | 0.0024 | inferential |
| sen2sr_mamba - sen2sr_lite | MAE vs LR | 28 | 0.0007 | 0.0005 | [0.0005, 0.0011] | 0.0000 | inferential |
| sen2sr_lite - sen2sr_lite_no_constraint | PSNR (dB) ↑ | 28 | 1.1044 | 1.0189 | [0.9309, 1.2900] | 0.0000 | inferential |
| sen2sr_lite - sen2sr_lite_no_constraint | SSIM ↑ | 28 | 0.0235 | 0.0234 | [0.0190, 0.0280] | 0.0000 | inferential |
| sen2sr_lite - sen2sr_lite_no_constraint | RMSE ↓ | 28 | -0.0031 | -0.0024 | [-0.0043, -0.0024] | 0.0000 | inferential |
| sen2sr_lite - sen2sr_lite_no_constraint | SAM (°) ↓ | 26 | -0.6729 | -0.6121 | [-0.7827, -0.5606] | 0.0000 | inferential |
| sen2sr_lite - sen2sr_lite_no_constraint | ERGAS ↓ | 28 | -0.9438 | -0.8562 | [-1.1779, -0.7326] | 0.0000 | inferential |
| sen2sr_lite - sen2sr_lite_no_constraint | NDVI MAE ↓ | 28 | -0.0147 | -0.0151 | [-0.0173, -0.0122] | 0.0000 | inferential |
| sen2sr_lite - sen2sr_lite_no_constraint | detail rel. error (1 = no detail added) | 28 | -0.2586 | -0.2331 | [-0.3259, -0.1998] | 0.0000 | inferential |
| sen2sr_lite - sen2sr_lite_no_constraint | MAE vs LR | 28 | -0.0042 | -0.0042 | [-0.0046, -0.0038] | 0.0000 | inferential |

### Dataset `sen2neon_phase3_sample` (sen2neon, role independent_benchmark, split test)

Records 3: **evaluated 3**, skipped 0, invalid 0, unreadable 0; 2 scene units; manifest digest `16db74fc35b0eee6…`.
Dataset-defined categories (descriptive only): {'Forest': 2, 'Rural': 1}

**Reference accuracy (SR vs the independent HR reference)**

| system | n units | PSNR (dB) ↑ | SSIM ↑ | RMSE ↓ | MAE ↓ | SAM (°) ↓ | ERGAS ↓ |
|---|---|---|---|---|---|---|---|
| bicubic | 2 | 36.3391 ± 1.8459 | 0.9739 ± 0.0018 | 0.0156 ± 0.0035 | 0.0080 ± 0.0005 | 0.9316 ± 0.4198 | 2.3066 ± 0.2932 |
| sen2sr_lite | 2 | 35.7513 ± 1.5734 | 0.9608 ± 0.0006 | 0.0166 ± 0.0032 | 0.0085 ± 0.0004 | 1.0016 ± 0.4917 | 2.5636 ± 0.1158 |
| sen2sr_lite_no_constraint | 2 | 34.5531 ± 1.3043 | 0.9423 ± 0.0065 | 0.0190 ± 0.0030 | 0.0110 ± 0.0006 | 1.5409 ± 0.5405 | 3.8746 ± 0.9678 |
| sen2sr_mamba | 2 | 35.6728 ± 1.4766 | 0.9609 ± 0.0024 | 0.0167 ± 0.0030 | 0.0084 ± 0.0003 | 0.9968 ± 0.5116 | 2.4994 ± 0.0776 |
| tiny_cnn_seed0 | 2 | 35.5965 ± 1.0067 | 0.9651 ± 0.0101 | 0.0168 ± 0.0021 | 0.0086 ± 0.0001 | 1.0651 ± 0.5748 | 3.0155 ± 0.5056 |
| tiny_cnn_seed1 | 2 | 35.3857 ± 0.9999 | 0.9609 ± 0.0092 | 0.0172 ± 0.0022 | 0.0088 ± 0.0001 | 1.1107 ± 0.5989 | 3.1057 ± 0.4340 |
| tiny_cnn_seed2 | 2 | 35.4403 ± 0.9665 | 0.9624 ± 0.0097 | 0.0171 ± 0.0021 | 0.0087 ± 0.0001 | 1.0885 ± 0.5832 | 3.0308 ± 0.4327 |
| tiny_cnn_seed3 | 2 | 35.6582 ± 1.0651 | 0.9663 ± 0.0095 | 0.0167 ± 0.0022 | 0.0085 ± 0.0000 | 1.0597 ± 0.5725 | 3.0239 ± 0.4712 |
| tiny_cnn_seed4 | 2 | 35.5677 ± 1.0095 | 0.9650 ± 0.0095 | 0.0169 ± 0.0022 | 0.0086 ± 0.0000 | 1.0601 ± 0.5672 | 2.9869 ± 0.4655 |

**Derived indices and band ratios (vs the HR reference)**

| system | n units | NDVI MAE ↓ | NDVI bias | NDVI r ↑ | NDWI MAE ↓ | log(B08/B04) bias |
|---|---|---|---|---|---|---|
| bicubic | 2 | 0.0153 ± 0.0086 | -0.0030 ± 0.0022 | 0.9171 ± 0.1064 | 0.0149 ± 0.0030 | 0.0015 ± 0.0028 |
| sen2sr_lite | 2 | 0.0165 ± 0.0098 | -0.0031 ± 0.0022 | 0.9124 ± 0.1096 | 0.0160 ± 0.0040 | 0.0025 ± 0.0037 |
| sen2sr_lite_no_constraint | 2 | 0.0310 ± 0.0080 | 0.0174 ± 0.0081 | 0.9118 ± 0.1084 | 0.0220 ± 0.0052 | 0.2143 ± 0.1389 |
| sen2sr_mamba | 2 | 0.0163 ± 0.0100 | -0.0032 ± 0.0022 | 0.9096 ± 0.1121 | 0.0160 ± 0.0043 | -0.0005 ± 0.0026 |
| tiny_cnn_seed0 | 2 | 0.0174 ± 0.0110 | -0.0029 ± 0.0022 | 0.9106 ± 0.1042 | 0.0166 ± 0.0050 | 0.0051 ± 0.0044 |
| tiny_cnn_seed1 | 2 | 0.0180 ± 0.0114 | -0.0026 ± 0.0022 | 0.9057 ± 0.1101 | 0.0170 ± 0.0053 | 0.0086 ± 0.0039 |
| tiny_cnn_seed2 | 2 | 0.0177 ± 0.0111 | -0.0030 ± 0.0024 | 0.9099 ± 0.1056 | 0.0168 ± 0.0051 | 0.0052 ± 0.0040 |
| tiny_cnn_seed3 | 2 | 0.0173 ± 0.0110 | -0.0027 ± 0.0024 | 0.9109 ± 0.1046 | 0.0165 ± 0.0050 | 0.0074 ± 0.0028 |
| tiny_cnn_seed4 | 2 | 0.0173 ± 0.0109 | -0.0029 ± 0.0023 | 0.9109 ± 0.1050 | 0.0165 ± 0.0049 | 0.0053 ± 0.0035 |

**Spatial / detail correctness (vs the HR reference)**

| system | n units | detail rel. error (1 = no detail added) | detail correlation ↑ | detail energy ratio (1 = same amount) | edge correlation ↑ | shift (HR px) | seam / interior error |
|---|---|---|---|---|---|---|---|
| bicubic | 2 | 1.0083 ± 0.2903 | 0.6890 ± 0.0932 | 1.7911 ± 1.3801 | 0.6388 ± 0.1885 | 1.5477 ± 0.8849 | n/a |
| sen2sr_lite | 2 | 1.6352 ± 0.6439 | 0.5912 ± 0.0385 | 4.7099 ± 3.9579 | 0.5813 ± 0.1573 | 1.3355 ± 0.4887 | 0.9656 ± 0.0854 |
| sen2sr_lite_no_constraint | 2 | 1.9119 ± 0.7956 | 0.5982 ± 0.0361 | 6.4287 ± 5.4238 | 0.5816 ± 0.1553 | 1.3355 ± 0.4887 | 0.9706 ± 0.0783 |
| sen2sr_mamba | 2 | 1.7375 ± 0.5712 | 0.5187 ± 0.0203 | 4.8975 ± 3.7744 | 0.5424 ± 0.1378 | 1.4011 ± 0.4782 | 0.9694 ± 0.0893 |
| tiny_cnn_seed0 | 2 | 1.4237 ± 0.0733 | 0.5328 ± 0.0381 | 3.0842 ± 0.5354 | 0.5484 ± 0.0764 | 1.5758 ± 0.8284 | 0.9682 ± 0.0889 |
| tiny_cnn_seed1 | 2 | 1.5873 ± 0.1009 | 0.5205 ± 0.0028 | 3.9746 ± 1.4059 | 0.5526 ± 0.0872 | 1.3355 ± 0.4887 | 0.9734 ± 0.0854 |
| tiny_cnn_seed2 | 2 | 1.5217 ± 0.0452 | 0.5405 ± 0.0232 | 3.6298 ± 1.0993 | 0.5549 ± 0.0829 | 1.3355 ± 0.4887 | 0.9693 ± 0.0850 |
| tiny_cnn_seed3 | 2 | 1.3620 ± 0.0673 | 0.5468 ± 0.0187 | 2.8672 ± 0.4825 | 0.5544 ± 0.0821 | 1.5758 ± 0.8284 | 0.9695 ± 0.0856 |
| tiny_cnn_seed4 | 2 | 1.4376 ± 0.0248 | 0.5369 ± 0.0278 | 3.2276 ± 0.7632 | 0.5502 ± 0.0735 | 1.5758 ± 0.8284 | 0.9704 ± 0.0845 |

**Per-band reflectance error** (RMSE ↓ / bias = SR − reference; bands are the RGBN bands every system produces)

| system | B02 RMSE | B03 RMSE | B04 RMSE | B08 RMSE | B02 bias | B03 bias | B04 bias | B08 bias |
|---|---|---|---|---|---|---|---|---|
| bicubic | 0.0037 ± 0.0016 | 0.0052 ± 0.0007 | 0.0048 ± 0.0026 | 0.0299 ± 0.0079 | 0.0001 ± 0.0000 | 0.0001 ± 0.0000 | 0.0001 ± 0.0000 | 0.0003 ± 0.0003 |
| sen2sr_lite | 0.0043 ± 0.0023 | 0.0059 ± 0.0013 | 0.0055 ± 0.0033 | 0.0316 ± 0.0078 | 0.0001 ± 0.0000 | 0.0001 ± 0.0000 | 0.0001 ± 0.0000 | 0.0003 ± 0.0003 |
| sen2sr_lite_no_constraint | 0.0059 ± 0.0024 | 0.0073 ± 0.0016 | 0.0087 ± 0.0027 | 0.0354 ± 0.0078 | -0.0029 ± 0.0003 | -0.0031 ± 0.0004 | -0.0054 ± 0.0008 | -0.0014 ± 0.0046 |
| sen2sr_mamba | 0.0043 ± 0.0026 | 0.0058 ± 0.0017 | 0.0054 ± 0.0037 | 0.0318 ± 0.0076 | 0.0001 ± 0.0000 | 0.0001 ± 0.0000 | 0.0001 ± 0.0000 | 0.0003 ± 0.0003 |
| tiny_cnn_seed0 | 0.0058 ± 0.0043 | 0.0070 ± 0.0029 | 0.0070 ± 0.0053 | 0.0309 ± 0.0073 | 0.0001 ± 0.0000 | 0.0001 ± 0.0000 | 0.0001 ± 0.0000 | 0.0003 ± 0.0003 |
| tiny_cnn_seed1 | 0.0060 ± 0.0042 | 0.0071 ± 0.0028 | 0.0071 ± 0.0053 | 0.0317 ± 0.0073 | 0.0001 ± 0.0000 | 0.0001 ± 0.0000 | -0.0000 ± 0.0000 | 0.0002 ± 0.0002 |
| tiny_cnn_seed2 | 0.0058 ± 0.0041 | 0.0070 ± 0.0027 | 0.0069 ± 0.0052 | 0.0316 ± 0.0070 | 0.0001 ± 0.0000 | 0.0001 ± 0.0000 | 0.0001 ± 0.0000 | 0.0003 ± 0.0004 |
| tiny_cnn_seed3 | 0.0058 ± 0.0043 | 0.0070 ± 0.0028 | 0.0070 ± 0.0053 | 0.0306 ± 0.0075 | 0.0001 ± 0.0000 | 0.0001 ± 0.0000 | 0.0000 ± 0.0001 | 0.0003 ± 0.0004 |
| tiny_cnn_seed4 | 0.0057 ± 0.0042 | 0.0069 ± 0.0028 | 0.0069 ± 0.0052 | 0.0311 ± 0.0072 | 0.0000 ± 0.0000 | 0.0001 ± 0.0000 | 0.0001 ± 0.0000 | 0.0002 ± 0.0003 |

**What was added relative to bicubic, judged against the reference** (element-wise; τ = 0.005 reflectance; fractions of valid elements; the reference is a different sensor, so "unsupported" means "not confirmed by this reference", not "false")

| system | n units | supported synthesis | unsupported detail | omission | MSE skill vs bicubic |
|---|---|---|---|---|---|
| sen2sr_lite | 2 | 0.0387 ± 0.0043 | 0.0786 ± 0.0036 | 0.2531 ± 0.0842 | -0.1461 ± 0.0718 |
| sen2sr_lite_no_constraint | 2 | 0.0562 ± 0.0054 | 0.3548 ± 0.0073 | 0.1654 ± 0.0355 | -0.5156 ± 0.1869 |
| sen2sr_mamba | 2 | 0.0395 ± 0.0006 | 0.0784 ± 0.0084 | 0.2532 ± 0.0781 | -0.1680 ± 0.0991 |
| tiny_cnn_seed0 | 2 | 0.0132 ± 0.0121 | 0.0388 ± 0.0404 | 0.3106 ± 0.0407 | -0.1976 ± 0.2300 |
| tiny_cnn_seed1 | 2 | 0.0247 ± 0.0094 | 0.0681 ± 0.0403 | 0.2793 ± 0.0476 | -0.2574 ± 0.2433 |
| tiny_cnn_seed2 | 2 | 0.0204 ± 0.0107 | 0.0579 ± 0.0424 | 0.2898 ± 0.0430 | -0.2426 ± 0.2499 |
| tiny_cnn_seed3 | 2 | 0.0101 ± 0.0118 | 0.0331 ± 0.0384 | 0.3189 ± 0.0431 | -0.1792 ± 0.2109 |
| tiny_cnn_seed4 | 2 | 0.0126 ± 0.0109 | 0.0381 ± 0.0373 | 0.3112 ± 0.0438 | -0.2055 ± 0.2307 |

**Self-consistency (SR → area-average → compare with the LR)**. This is NOT accuracy against an HR reference: a system can average back to the LR perfectly and still be far from the reference.

| system | n units | MAE vs LR | RMSE vs LR | NDVI MAE vs LR |
|---|---|---|---|---|
| bicubic | 2 | 0.0011 ± 0.0001 | 0.0021 ± 0.0001 | 0.0028 ± 0.0022 |
| sen2sr_lite | 2 | 0.0009 ± 0.0000 | 0.0016 ± 0.0001 | 0.0018 ± 0.0013 |
| sen2sr_lite_no_constraint | 2 | 0.0045 ± 0.0000 | 0.0053 ± 0.0001 | 0.0228 ± 0.0020 |
| sen2sr_mamba | 2 | 0.0012 ± 0.0000 | 0.0018 ± 0.0000 | 0.0043 ± 0.0021 |
| tiny_cnn_seed0 | 2 | 0.0010 ± 0.0003 | 0.0023 ± 0.0011 | 0.0027 ± 0.0024 |
| tiny_cnn_seed1 | 2 | 0.0011 ± 0.0004 | 0.0025 ± 0.0013 | 0.0034 ± 0.0025 |
| tiny_cnn_seed2 | 2 | 0.0010 ± 0.0004 | 0.0022 ± 0.0013 | 0.0030 ± 0.0023 |
| tiny_cnn_seed3 | 2 | 0.0011 ± 0.0003 | 0.0025 ± 0.0011 | 0.0030 ± 0.0025 |
| tiny_cnn_seed4 | 2 | 0.0009 ± 0.0003 | 0.0022 ± 0.0012 | 0.0027 ± 0.0022 |

**By dataset-defined category** (the dataset's own labels, never inferred; descriptive only: few samples per category, no interval or test)

| category | n samples | n units | system | PSNR (dB) | SAM (°) | detail rel. error |
|---|---|---|---|---|---|---|
| Forest | 2 | 1 | bicubic | 35.0338 ± 2.2847 | 0.6347 ± 0.1573 | 1.2136 ± 0.8281 |
| Forest | 2 | 1 | sen2sr_lite | 34.6388 ± 2.2123 | 0.6540 ± 0.1492 | 2.0905 ± 1.5623 |
| Forest | 2 | 1 | sen2sr_lite_no_constraint | 33.6309 ± 1.9394 | 1.1587 ± 0.1003 | 2.4745 ± 1.8495 |
| Forest | 2 | 1 | sen2sr_mamba | 34.6287 ± 2.1779 | 0.6350 ± 0.1399 | 2.1413 ± 1.5274 |
| Forest | 2 | 1 | tiny_cnn_seed0 | 34.8847 ± 2.2581 | 0.6586 ± 0.1493 | 1.3719 ± 0.9458 |
| Forest | 2 | 1 | tiny_cnn_seed1 | 34.6787 ± 2.2018 | 0.6872 ± 0.1350 | 1.6587 ± 1.1864 |
| Forest | 2 | 1 | tiny_cnn_seed2 | 34.7569 ± 2.2371 | 0.6761 ± 0.1424 | 1.5537 ± 1.1138 |
| Forest | 2 | 1 | tiny_cnn_seed3 | 34.9051 ± 2.2582 | 0.6549 ± 0.1493 | 1.3144 ± 0.9036 |
| Forest | 2 | 1 | tiny_cnn_seed4 | 34.8539 ± 2.2617 | 0.6591 ± 0.1492 | 1.4200 ± 1.0112 |
| Rural | 1 | 1 | bicubic | 37.6443 | 1.2284 | 0.8030 |
| Rural | 1 | 1 | sen2sr_lite | 36.8639 | 1.3493 | 1.1800 |
| Rural | 1 | 1 | sen2sr_lite_no_constraint | 35.4754 | 1.9231 | 1.3493 |
| Rural | 1 | 1 | sen2sr_mamba | 36.7169 | 1.3586 | 1.3336 |
| Rural | 1 | 1 | tiny_cnn_seed0 | 36.3084 | 1.4716 | 1.4755 |
| Rural | 1 | 1 | tiny_cnn_seed1 | 36.0927 | 1.5342 | 1.5160 |
| Rural | 1 | 1 | tiny_cnn_seed2 | 36.1237 | 1.5008 | 1.4898 |
| Rural | 1 | 1 | tiny_cnn_seed3 | 36.4114 | 1.4645 | 1.4096 |
| Rural | 1 | 1 | tiny_cnn_seed4 | 36.2816 | 1.4612 | 1.4551 |

**Paired differences A − B over scene units** (sign convention: A minus B; for error metrics lower is smaller error; a difference is not a ranking)

| A − B | metric | n units | mean difference | median | 95% CI | Wilcoxon p | label |
|---|---|---|---|---|---|---|---|
| sen2sr_lite - bicubic | PSNR (dB) ↑ | 2 | -0.5877 | -0.5877 | n/a | n/a | descriptive only |
| sen2sr_lite - bicubic | SSIM ↑ | 2 | -0.0130 | -0.0130 | n/a | n/a | descriptive only |
| sen2sr_lite - bicubic | RMSE ↓ | 2 | 0.0010 | 0.0010 | n/a | n/a | descriptive only |
| sen2sr_lite - bicubic | SAM (°) ↓ | 2 | 0.0701 | 0.0701 | n/a | n/a | descriptive only |
| sen2sr_lite - bicubic | ERGAS ↓ | 2 | 0.2570 | 0.2570 | n/a | n/a | descriptive only |
| sen2sr_lite - bicubic | NDVI MAE ↓ | 2 | 0.0012 | 0.0012 | n/a | n/a | descriptive only |
| sen2sr_lite - bicubic | detail rel. error (1 = no detail added) | 2 | 0.6269 | 0.6269 | n/a | n/a | descriptive only |
| sen2sr_lite - bicubic | MAE vs LR | 2 | -0.0002 | -0.0002 | n/a | n/a | descriptive only |
| sen2sr_lite_no_constraint - bicubic | PSNR (dB) ↑ | 2 | -1.7859 | -1.7859 | n/a | n/a | descriptive only |
| sen2sr_lite_no_constraint - bicubic | SSIM ↑ | 2 | -0.0316 | -0.0316 | n/a | n/a | descriptive only |
| sen2sr_lite_no_constraint - bicubic | RMSE ↓ | 2 | 0.0034 | 0.0034 | n/a | n/a | descriptive only |
| sen2sr_lite_no_constraint - bicubic | SAM (°) ↓ | 2 | 0.6093 | 0.6093 | n/a | n/a | descriptive only |
| sen2sr_lite_no_constraint - bicubic | ERGAS ↓ | 2 | 1.5680 | 1.5680 | n/a | n/a | descriptive only |
| sen2sr_lite_no_constraint - bicubic | NDVI MAE ↓ | 2 | 0.0157 | 0.0157 | n/a | n/a | descriptive only |
| sen2sr_lite_no_constraint - bicubic | detail rel. error (1 = no detail added) | 2 | 0.9036 | 0.9036 | n/a | n/a | descriptive only |
| sen2sr_lite_no_constraint - bicubic | MAE vs LR | 2 | 0.0034 | 0.0034 | n/a | n/a | descriptive only |
| sen2sr_mamba - bicubic | PSNR (dB) ↑ | 2 | -0.6663 | -0.6663 | n/a | n/a | descriptive only |
| sen2sr_mamba - bicubic | SSIM ↑ | 2 | -0.0130 | -0.0130 | n/a | n/a | descriptive only |
| sen2sr_mamba - bicubic | RMSE ↓ | 2 | 0.0012 | 0.0012 | n/a | n/a | descriptive only |
| sen2sr_mamba - bicubic | SAM (°) ↓ | 2 | 0.0652 | 0.0652 | n/a | n/a | descriptive only |
| sen2sr_mamba - bicubic | ERGAS ↓ | 2 | 0.1927 | 0.1927 | n/a | n/a | descriptive only |
| sen2sr_mamba - bicubic | NDVI MAE ↓ | 2 | 0.0010 | 0.0010 | n/a | n/a | descriptive only |
| sen2sr_mamba - bicubic | detail rel. error (1 = no detail added) | 2 | 0.7291 | 0.7291 | n/a | n/a | descriptive only |
| sen2sr_mamba - bicubic | MAE vs LR | 2 | 0.0001 | 0.0001 | n/a | n/a | descriptive only |
| tiny_cnn_seed0 - bicubic | PSNR (dB) ↑ | 2 | -0.7425 | -0.7425 | n/a | n/a | descriptive only |
| tiny_cnn_seed0 - bicubic | SSIM ↑ | 2 | -0.0088 | -0.0088 | n/a | n/a | descriptive only |
| tiny_cnn_seed0 - bicubic | RMSE ↓ | 2 | 0.0012 | 0.0012 | n/a | n/a | descriptive only |
| tiny_cnn_seed0 - bicubic | SAM (°) ↓ | 2 | 0.1335 | 0.1335 | n/a | n/a | descriptive only |
| tiny_cnn_seed0 - bicubic | ERGAS ↓ | 2 | 0.7089 | 0.7089 | n/a | n/a | descriptive only |
| tiny_cnn_seed0 - bicubic | NDVI MAE ↓ | 2 | 0.0021 | 0.0021 | n/a | n/a | descriptive only |
| tiny_cnn_seed0 - bicubic | detail rel. error (1 = no detail added) | 2 | 0.4154 | 0.4154 | n/a | n/a | descriptive only |
| tiny_cnn_seed0 - bicubic | MAE vs LR | 2 | -0.0001 | -0.0001 | n/a | n/a | descriptive only |
| tiny_cnn_seed1 - bicubic | PSNR (dB) ↑ | 2 | -0.9534 | -0.9534 | n/a | n/a | descriptive only |
| tiny_cnn_seed1 - bicubic | SSIM ↑ | 2 | -0.0130 | -0.0130 | n/a | n/a | descriptive only |
| tiny_cnn_seed1 - bicubic | RMSE ↓ | 2 | 0.0016 | 0.0016 | n/a | n/a | descriptive only |
| tiny_cnn_seed1 - bicubic | SAM (°) ↓ | 2 | 0.1792 | 0.1792 | n/a | n/a | descriptive only |
| tiny_cnn_seed1 - bicubic | ERGAS ↓ | 2 | 0.7991 | 0.7991 | n/a | n/a | descriptive only |
| tiny_cnn_seed1 - bicubic | NDVI MAE ↓ | 2 | 0.0028 | 0.0028 | n/a | n/a | descriptive only |
| tiny_cnn_seed1 - bicubic | detail rel. error (1 = no detail added) | 2 | 0.5790 | 0.5790 | n/a | n/a | descriptive only |
| tiny_cnn_seed1 - bicubic | MAE vs LR | 2 | 0.0000 | 0.0000 | n/a | n/a | descriptive only |
| tiny_cnn_seed2 - bicubic | PSNR (dB) ↑ | 2 | -0.8987 | -0.8987 | n/a | n/a | descriptive only |
| tiny_cnn_seed2 - bicubic | SSIM ↑ | 2 | -0.0115 | -0.0115 | n/a | n/a | descriptive only |
| tiny_cnn_seed2 - bicubic | RMSE ↓ | 2 | 0.0015 | 0.0015 | n/a | n/a | descriptive only |
| tiny_cnn_seed2 - bicubic | SAM (°) ↓ | 2 | 0.1569 | 0.1569 | n/a | n/a | descriptive only |
| tiny_cnn_seed2 - bicubic | ERGAS ↓ | 2 | 0.7242 | 0.7242 | n/a | n/a | descriptive only |
| tiny_cnn_seed2 - bicubic | NDVI MAE ↓ | 2 | 0.0024 | 0.0024 | n/a | n/a | descriptive only |
| tiny_cnn_seed2 - bicubic | detail rel. error (1 = no detail added) | 2 | 0.5134 | 0.5134 | n/a | n/a | descriptive only |
| tiny_cnn_seed2 - bicubic | MAE vs LR | 2 | -0.0001 | -0.0001 | n/a | n/a | descriptive only |
| tiny_cnn_seed3 - bicubic | PSNR (dB) ↑ | 2 | -0.6808 | -0.6808 | n/a | n/a | descriptive only |
| tiny_cnn_seed3 - bicubic | SSIM ↑ | 2 | -0.0076 | -0.0076 | n/a | n/a | descriptive only |
| tiny_cnn_seed3 - bicubic | RMSE ↓ | 2 | 0.0011 | 0.0011 | n/a | n/a | descriptive only |
| tiny_cnn_seed3 - bicubic | SAM (°) ↓ | 2 | 0.1281 | 0.1281 | n/a | n/a | descriptive only |
| tiny_cnn_seed3 - bicubic | ERGAS ↓ | 2 | 0.7173 | 0.7173 | n/a | n/a | descriptive only |
| tiny_cnn_seed3 - bicubic | NDVI MAE ↓ | 2 | 0.0021 | 0.0021 | n/a | n/a | descriptive only |
| tiny_cnn_seed3 - bicubic | detail rel. error (1 = no detail added) | 2 | 0.3537 | 0.3537 | n/a | n/a | descriptive only |
| tiny_cnn_seed3 - bicubic | MAE vs LR | 2 | -0.0000 | -0.0000 | n/a | n/a | descriptive only |
| tiny_cnn_seed4 - bicubic | PSNR (dB) ↑ | 2 | -0.7713 | -0.7713 | n/a | n/a | descriptive only |
| tiny_cnn_seed4 - bicubic | SSIM ↑ | 2 | -0.0089 | -0.0089 | n/a | n/a | descriptive only |
| tiny_cnn_seed4 - bicubic | RMSE ↓ | 2 | 0.0013 | 0.0013 | n/a | n/a | descriptive only |
| tiny_cnn_seed4 - bicubic | SAM (°) ↓ | 2 | 0.1285 | 0.1285 | n/a | n/a | descriptive only |
| tiny_cnn_seed4 - bicubic | ERGAS ↓ | 2 | 0.6803 | 0.6803 | n/a | n/a | descriptive only |
| tiny_cnn_seed4 - bicubic | NDVI MAE ↓ | 2 | 0.0020 | 0.0020 | n/a | n/a | descriptive only |
| tiny_cnn_seed4 - bicubic | detail rel. error (1 = no detail added) | 2 | 0.4292 | 0.4292 | n/a | n/a | descriptive only |
| tiny_cnn_seed4 - bicubic | MAE vs LR | 2 | -0.0002 | -0.0002 | n/a | n/a | descriptive only |
| sen2sr_mamba - sen2sr_lite | PSNR (dB) ↑ | 2 | -0.0785 | -0.0785 | n/a | n/a | descriptive only |
| sen2sr_mamba - sen2sr_lite | SSIM ↑ | 2 | 0.0000 | 0.0000 | n/a | n/a | descriptive only |
| sen2sr_mamba - sen2sr_lite | RMSE ↓ | 2 | 0.0001 | 0.0001 | n/a | n/a | descriptive only |
| sen2sr_mamba - sen2sr_lite | SAM (°) ↓ | 2 | -0.0048 | -0.0048 | n/a | n/a | descriptive only |
| sen2sr_mamba - sen2sr_lite | ERGAS ↓ | 2 | -0.0642 | -0.0642 | n/a | n/a | descriptive only |
| sen2sr_mamba - sen2sr_lite | NDVI MAE ↓ | 2 | -0.0002 | -0.0002 | n/a | n/a | descriptive only |
| sen2sr_mamba - sen2sr_lite | detail rel. error (1 = no detail added) | 2 | 0.1022 | 0.1022 | n/a | n/a | descriptive only |
| sen2sr_mamba - sen2sr_lite | MAE vs LR | 2 | 0.0003 | 0.0003 | n/a | n/a | descriptive only |
| sen2sr_lite - sen2sr_lite_no_constraint | PSNR (dB) ↑ | 2 | 1.1982 | 1.1982 | n/a | n/a | descriptive only |
| sen2sr_lite - sen2sr_lite_no_constraint | SSIM ↑ | 2 | 0.0186 | 0.0186 | n/a | n/a | descriptive only |
| sen2sr_lite - sen2sr_lite_no_constraint | RMSE ↓ | 2 | -0.0024 | -0.0024 | n/a | n/a | descriptive only |
| sen2sr_lite - sen2sr_lite_no_constraint | SAM (°) ↓ | 2 | -0.5392 | -0.5392 | n/a | n/a | descriptive only |
| sen2sr_lite - sen2sr_lite_no_constraint | ERGAS ↓ | 2 | -1.3110 | -1.3110 | n/a | n/a | descriptive only |
| sen2sr_lite - sen2sr_lite_no_constraint | NDVI MAE ↓ | 2 | -0.0144 | -0.0144 | n/a | n/a | descriptive only |
| sen2sr_lite - sen2sr_lite_no_constraint | detail rel. error (1 = no detail added) | 2 | -0.2767 | -0.2767 | n/a | n/a | descriptive only |
| sen2sr_lite - sen2sr_lite_no_constraint | MAE vs LR | 2 | -0.0037 | -0.0037 | n/a | n/a | descriptive only |

### Dataset `opensr_spot` (opensr_test, role independent_benchmark, split test)

Records 9: **evaluated 9**, skipped 0, invalid 0, unreadable 0; 9 scene units; manifest digest `32495be456ae9ba3…`.
Dataset-defined categories (descriptive only): {'spot_mixed': 9}

**Reference accuracy (SR vs the independent HR reference)**

| system | n units | PSNR (dB) ↑ | SSIM ↑ | RMSE ↓ | MAE ↓ | SAM (°) ↓ | ERGAS ↓ |
|---|---|---|---|---|---|---|---|
| bicubic | 9 | 33.1171 ± 5.3870 | 0.8176 ± 0.1136 | 0.0261 ± 0.0157 | 0.0183 ± 0.0106 | 1.8458 ± 1.5160 | 3.7674 ± 2.6943 |
| sen2sr_lite | 9 | 33.2229 ± 5.4117 | 0.8307 ± 0.1044 | 0.0258 ± 0.0155 | 0.0180 ± 0.0104 | 2.0710 ± 1.7269 | 3.6846 ± 2.6080 |
| sen2sr_lite_no_constraint | 9 | 32.4468 ± 5.1699 | 0.8295 ± 0.1058 | 0.0279 ± 0.0162 | 0.0197 ± 0.0109 | 2.6792 ± 1.9437 | 3.9759 ± 2.7353 |
| sen2sr_mamba | 9 | 33.1966 ± 5.5140 | 0.8248 ± 0.1151 | 0.0261 ± 0.0160 | 0.0182 ± 0.0108 | 2.0153 ± 1.6499 | 3.7279 ± 2.6891 |
| tiny_cnn_seed0 | 9 | 32.8519 ± 5.6537 | 0.8169 ± 0.1166 | 0.0274 ± 0.0172 | 0.0192 ± 0.0118 | 2.4609 ± 2.2300 | 3.9982 ± 2.9378 |
| tiny_cnn_seed1 | 9 | 32.8774 ± 5.6924 | 0.8183 ± 0.1173 | 0.0274 ± 0.0174 | 0.0192 ± 0.0119 | 2.4564 ± 2.2812 | 3.9806 ± 2.9463 |
| tiny_cnn_seed2 | 9 | 32.8746 ± 5.6624 | 0.8178 ± 0.1167 | 0.0273 ± 0.0172 | 0.0192 ± 0.0118 | 2.4595 ± 2.2447 | 3.9724 ± 2.9194 |
| tiny_cnn_seed3 | 9 | 32.9654 ± 5.6008 | 0.8200 ± 0.1142 | 0.0269 ± 0.0168 | 0.0189 ± 0.0115 | 2.4279 ± 2.2171 | 3.9174 ± 2.8698 |
| tiny_cnn_seed4 | 9 | 32.9124 ± 5.6587 | 0.8193 ± 0.1158 | 0.0272 ± 0.0171 | 0.0191 ± 0.0117 | 2.3966 ± 2.2017 | 3.9599 ± 2.9214 |

**Derived indices and band ratios (vs the HR reference)**

| system | n units | NDVI MAE ↓ | NDVI bias | NDVI r ↑ | NDWI MAE ↓ | log(B08/B04) bias |
|---|---|---|---|---|---|---|
| bicubic | 9 | 0.0253 ± 0.0237 | -0.0027 ± 0.0054 | 0.8655 ± 0.0656 | 0.0253 ± 0.0215 | -0.0008 ± 0.0015 |
| sen2sr_lite | 9 | 0.0295 ± 0.0276 | -0.0021 ± 0.0053 | 0.8411 ± 0.0741 | 0.0282 ± 0.0244 | 0.0017 ± 0.0024 |
| sen2sr_lite_no_constraint | 9 | 0.0415 ± 0.0313 | -0.0156 ± 0.0041 | 0.8225 ± 0.0744 | 0.0374 ± 0.0272 | -0.0222 ± 0.0232 |
| sen2sr_mamba | 9 | 0.0282 ± 0.0255 | -0.0032 ± 0.0069 | 0.8377 ± 0.0775 | 0.0275 ± 0.0233 | -0.0027 ± 0.0071 |
| tiny_cnn_seed0 | 9 | 0.0360 ± 0.0365 | 0.0002 ± 0.0049 | 0.8254 ± 0.0743 | 0.0341 ± 0.0314 | 0.0098 ± 0.0151 |
| tiny_cnn_seed1 | 9 | 0.0360 ± 0.0374 | -0.0000 ± 0.0053 | 0.8280 ± 0.0776 | 0.0338 ± 0.0319 | 0.0096 ± 0.0143 |
| tiny_cnn_seed2 | 9 | 0.0361 ± 0.0368 | -0.0003 ± 0.0054 | 0.8245 ± 0.0782 | 0.0339 ± 0.0314 | 0.0088 ± 0.0131 |
| tiny_cnn_seed3 | 9 | 0.0356 ± 0.0364 | -0.0006 ± 0.0048 | 0.8369 ± 0.0756 | 0.0336 ± 0.0313 | 0.0079 ± 0.0128 |
| tiny_cnn_seed4 | 9 | 0.0348 ± 0.0359 | 0.0003 ± 0.0054 | 0.8359 ± 0.0743 | 0.0331 ± 0.0310 | 0.0096 ± 0.0133 |

**Spatial / detail correctness (vs the HR reference)**

| system | n units | detail rel. error (1 = no detail added) | detail correlation ↑ | detail energy ratio (1 = same amount) | edge correlation ↑ | shift (HR px) | seam / interior error |
|---|---|---|---|---|---|---|---|
| bicubic | 9 | 0.9458 ± 0.0204 | 0.3328 ± 0.0611 | 0.0805 ± 0.0303 | 0.3847 ± 0.1071 | 0.4716 ± 0.2203 | n/a |
| sen2sr_lite | 9 | 0.9315 ± 0.0323 | 0.3776 ± 0.0784 | 0.1875 ± 0.0812 | 0.4448 ± 0.1170 | 0.4720 ± 0.2586 | n/a |
| sen2sr_lite_no_constraint | 9 | 0.9363 ± 0.0337 | 0.3827 ± 0.0781 | 0.2511 ± 0.1105 | 0.4499 ± 0.1160 | 0.4720 ± 0.2586 | n/a |
| sen2sr_mamba | 9 | 0.9359 ± 0.0329 | 0.3813 ± 0.0718 | 0.2135 ± 0.0970 | 0.4626 ± 0.1046 | 0.5025 ± 0.2338 | n/a |
| tiny_cnn_seed0 | 9 | 0.9796 ± 0.0462 | 0.3239 ± 0.0792 | 0.3134 ± 0.1857 | 0.4026 ± 0.1467 | 0.4702 ± 0.2372 | n/a |
| tiny_cnn_seed1 | 9 | 0.9741 ± 0.0465 | 0.3314 ± 0.0758 | 0.3016 ± 0.1902 | 0.4125 ± 0.1379 | 0.4867 ± 0.2313 | n/a |
| tiny_cnn_seed2 | 9 | 0.9752 ± 0.0442 | 0.3260 ± 0.0763 | 0.2962 ± 0.1769 | 0.4056 ± 0.1408 | 0.4684 ± 0.2386 | n/a |
| tiny_cnn_seed3 | 9 | 0.9625 ± 0.0349 | 0.3346 ± 0.0706 | 0.2463 ± 0.1566 | 0.4172 ± 0.1312 | 0.4675 ± 0.2582 | n/a |
| tiny_cnn_seed4 | 9 | 0.9691 ± 0.0450 | 0.3348 ± 0.0742 | 0.2833 ± 0.1851 | 0.4106 ± 0.1372 | 0.4611 ± 0.2492 | n/a |

**Per-band reflectance error** (RMSE ↓ / bias = SR − reference; bands are the RGBN bands every system produces)

| system | B02 RMSE | B03 RMSE | B04 RMSE | B08 RMSE | B02 bias | B03 bias | B04 bias | B08 bias |
|---|---|---|---|---|---|---|---|---|
| bicubic | 0.0185 ± 0.0138 | 0.0217 ± 0.0139 | 0.0260 ± 0.0136 | 0.0345 ± 0.0215 | -0.0000 ± 0.0000 | -0.0000 ± 0.0000 | -0.0000 ± 0.0001 | -0.0001 ± 0.0002 |
| sen2sr_lite | 0.0181 ± 0.0133 | 0.0210 ± 0.0133 | 0.0253 ± 0.0131 | 0.0347 ± 0.0222 | -0.0000 ± 0.0000 | -0.0000 ± 0.0000 | -0.0000 ± 0.0001 | -0.0001 ± 0.0002 |
| sen2sr_lite_no_constraint | 0.0196 ± 0.0141 | 0.0230 ± 0.0138 | 0.0278 ± 0.0135 | 0.0368 ± 0.0235 | 0.0018 ± 0.0024 | 0.0030 ± 0.0038 | 0.0040 ± 0.0051 | -0.0041 ± 0.0038 |
| sen2sr_mamba | 0.0183 ± 0.0137 | 0.0212 ± 0.0137 | 0.0255 ± 0.0136 | 0.0351 ± 0.0228 | -0.0000 ± 0.0000 | -0.0000 ± 0.0000 | -0.0000 ± 0.0001 | -0.0001 ± 0.0002 |
| tiny_cnn_seed0 | 0.0199 ± 0.0150 | 0.0227 ± 0.0150 | 0.0275 ± 0.0156 | 0.0358 ± 0.0233 | -0.0003 ± 0.0002 | -0.0003 ± 0.0002 | -0.0002 ± 0.0002 | -0.0003 ± 0.0002 |
| tiny_cnn_seed1 | 0.0197 ± 0.0150 | 0.0226 ± 0.0151 | 0.0273 ± 0.0156 | 0.0359 ± 0.0238 | -0.0003 ± 0.0002 | -0.0002 ± 0.0001 | -0.0001 ± 0.0001 | -0.0001 ± 0.0002 |
| tiny_cnn_seed2 | 0.0197 ± 0.0149 | 0.0226 ± 0.0149 | 0.0273 ± 0.0154 | 0.0359 ± 0.0237 | -0.0002 ± 0.0002 | -0.0002 ± 0.0001 | -0.0002 ± 0.0001 | -0.0002 ± 0.0004 |
| tiny_cnn_seed3 | 0.0195 ± 0.0148 | 0.0222 ± 0.0146 | 0.0269 ± 0.0151 | 0.0354 ± 0.0229 | -0.0002 ± 0.0001 | -0.0002 ± 0.0001 | -0.0001 ± 0.0001 | -0.0006 ± 0.0006 |
| tiny_cnn_seed4 | 0.0196 ± 0.0149 | 0.0225 ± 0.0150 | 0.0272 ± 0.0155 | 0.0357 ± 0.0233 | -0.0003 ± 0.0002 | -0.0002 ± 0.0001 | -0.0002 ± 0.0001 | -0.0001 ± 0.0003 |

**What was added relative to bicubic, judged against the reference** (element-wise; τ = 0.005 reflectance; fractions of valid elements; the reference is a different sensor, so "unsupported" means "not confirmed by this reference", not "false")

| system | n units | supported synthesis | unsupported detail | omission | MSE skill vs bicubic |
|---|---|---|---|---|---|
| sen2sr_lite | 9 | 0.1170 ± 0.1127 | 0.1079 ± 0.1063 | 0.5260 ± 0.1090 | 0.0237 ± 0.0288 |
| sen2sr_lite_no_constraint | 9 | 0.1990 ± 0.0816 | 0.3284 ± 0.1044 | 0.3148 ± 0.0897 | -0.1719 ± 0.1197 |
| sen2sr_mamba | 9 | 0.1235 ± 0.1071 | 0.1293 ± 0.1308 | 0.5030 ± 0.1176 | 0.0173 ± 0.0420 |
| tiny_cnn_seed0 | 9 | 0.1358 ± 0.1085 | 0.1832 ± 0.1521 | 0.4529 ± 0.1292 | -0.0666 ± 0.0959 |
| tiny_cnn_seed1 | 9 | 0.1335 ± 0.1129 | 0.1776 ± 0.1624 | 0.4576 ± 0.1424 | -0.0607 ± 0.0998 |
| tiny_cnn_seed2 | 9 | 0.1341 ± 0.1127 | 0.1792 ± 0.1591 | 0.4564 ± 0.1404 | -0.0608 ± 0.0923 |
| tiny_cnn_seed3 | 9 | 0.1148 ± 0.1111 | 0.1507 ± 0.1531 | 0.4953 ± 0.1408 | -0.0375 ± 0.0677 |
| tiny_cnn_seed4 | 9 | 0.1242 ± 0.1139 | 0.1626 ± 0.1593 | 0.4769 ± 0.1454 | -0.0518 ± 0.0941 |

**Self-consistency (SR → area-average → compare with the LR)**. This is NOT accuracy against an HR reference: a system can average back to the LR perfectly and still be far from the reference.

| system | n units | MAE vs LR | RMSE vs LR | NDVI MAE vs LR |
|---|---|---|---|---|
| bicubic | 9 | 0.0022 ± 0.0016 | 0.0031 ± 0.0023 | 0.0055 ± 0.0053 |
| sen2sr_lite | 9 | 0.0015 ± 0.0011 | 0.0022 ± 0.0017 | 0.0038 ± 0.0039 |
| sen2sr_lite_no_constraint | 9 | 0.0053 ± 0.0018 | 0.0069 ± 0.0023 | 0.0184 ± 0.0040 |
| sen2sr_mamba | 9 | 0.0022 ± 0.0023 | 0.0031 ± 0.0031 | 0.0076 ± 0.0106 |
| tiny_cnn_seed0 | 9 | 0.0032 ± 0.0030 | 0.0050 ± 0.0046 | 0.0092 ± 0.0099 |
| tiny_cnn_seed1 | 9 | 0.0033 ± 0.0032 | 0.0050 ± 0.0047 | 0.0090 ± 0.0102 |
| tiny_cnn_seed2 | 9 | 0.0031 ± 0.0030 | 0.0048 ± 0.0044 | 0.0091 ± 0.0101 |
| tiny_cnn_seed3 | 9 | 0.0029 ± 0.0028 | 0.0044 ± 0.0043 | 0.0107 ± 0.0115 |
| tiny_cnn_seed4 | 9 | 0.0031 ± 0.0030 | 0.0048 ± 0.0045 | 0.0089 ± 0.0099 |

**By dataset-defined category** (the dataset's own labels, never inferred; descriptive only: few samples per category, no interval or test)

| category | n samples | n units | system | PSNR (dB) | SAM (°) | detail rel. error |
|---|---|---|---|---|---|---|
| spot_mixed | 9 | 9 | bicubic | 33.1171 ± 5.3870 | 1.8458 ± 1.5160 | 0.9458 ± 0.0204 |
| spot_mixed | 9 | 9 | sen2sr_lite | 33.2229 ± 5.4117 | 2.0710 ± 1.7269 | 0.9315 ± 0.0323 |
| spot_mixed | 9 | 9 | sen2sr_lite_no_constraint | 32.4468 ± 5.1699 | 2.6792 ± 1.9437 | 0.9363 ± 0.0337 |
| spot_mixed | 9 | 9 | sen2sr_mamba | 33.1966 ± 5.5140 | 2.0153 ± 1.6499 | 0.9359 ± 0.0329 |
| spot_mixed | 9 | 9 | tiny_cnn_seed0 | 32.8519 ± 5.6537 | 2.4609 ± 2.2300 | 0.9796 ± 0.0462 |
| spot_mixed | 9 | 9 | tiny_cnn_seed1 | 32.8774 ± 5.6924 | 2.4564 ± 2.2812 | 0.9741 ± 0.0465 |
| spot_mixed | 9 | 9 | tiny_cnn_seed2 | 32.8746 ± 5.6624 | 2.4595 ± 2.2447 | 0.9752 ± 0.0442 |
| spot_mixed | 9 | 9 | tiny_cnn_seed3 | 32.9654 ± 5.6008 | 2.4279 ± 2.2171 | 0.9625 ± 0.0349 |
| spot_mixed | 9 | 9 | tiny_cnn_seed4 | 32.9124 ± 5.6587 | 2.3966 ± 2.2017 | 0.9691 ± 0.0450 |

**Paired differences A − B over scene units** (sign convention: A minus B; for error metrics lower is smaller error; a difference is not a ranking)

| A − B | metric | n units | mean difference | median | 95% CI | Wilcoxon p | label |
|---|---|---|---|---|---|---|---|
| sen2sr_lite - bicubic | PSNR (dB) ↑ | 9 | 0.1057 | 0.1165 | [0.0257, 0.1806] | 0.0391 | inferential |
| sen2sr_lite - bicubic | SSIM ↑ | 9 | 0.0131 | 0.0103 | [0.0067, 0.0203] | 0.0039 | inferential |
| sen2sr_lite - bicubic | RMSE ↓ | 9 | -0.0003 | -0.0003 | [-0.0005, 0.0000] | 0.0977 | inferential |
| sen2sr_lite - bicubic | SAM (°) ↓ | 9 | 0.2252 | 0.0958 | [0.1057, 0.3645] | 0.0039 | inferential |
| sen2sr_lite - bicubic | ERGAS ↓ | 9 | -0.0828 | -0.0542 | [-0.1828, -0.0026] | 0.1289 | inferential |
| sen2sr_lite - bicubic | NDVI MAE ↓ | 9 | 0.0042 | 0.0017 | [0.0020, 0.0067] | 0.0039 | inferential |
| sen2sr_lite - bicubic | detail rel. error (1 = no detail added) | 9 | -0.0143 | -0.0181 | [-0.0223, -0.0057] | 0.0273 | inferential |
| sen2sr_lite - bicubic | MAE vs LR | 9 | -0.0007 | -0.0004 | [-0.0010, -0.0004] | 0.0039 | inferential |
| sen2sr_lite_no_constraint - bicubic | PSNR (dB) ↑ | 9 | -0.6703 | -0.5134 | [-0.9492, -0.4387] | 0.0039 | inferential |
| sen2sr_lite_no_constraint - bicubic | SSIM ↑ | 9 | 0.0119 | 0.0074 | [0.0055, 0.0192] | 0.0039 | inferential |
| sen2sr_lite_no_constraint - bicubic | RMSE ↓ | 9 | 0.0018 | 0.0016 | [0.0012, 0.0023] | 0.0039 | inferential |
| sen2sr_lite_no_constraint - bicubic | SAM (°) ↓ | 9 | 0.8334 | 0.6556 | [0.5871, 1.1386] | 0.0039 | inferential |
| sen2sr_lite_no_constraint - bicubic | ERGAS ↓ | 9 | 0.2085 | 0.1785 | [0.1356, 0.2967] | 0.0039 | inferential |
| sen2sr_lite_no_constraint - bicubic | NDVI MAE ↓ | 9 | 0.0162 | 0.0120 | [0.0117, 0.0216] | 0.0039 | inferential |
| sen2sr_lite_no_constraint - bicubic | detail rel. error (1 = no detail added) | 9 | -0.0095 | -0.0156 | [-0.0190, 0.0015] | 0.1641 | inferential |
| sen2sr_lite_no_constraint - bicubic | MAE vs LR | 9 | 0.0031 | 0.0023 | [0.0022, 0.0042] | 0.0039 | inferential |
| sen2sr_mamba - bicubic | PSNR (dB) ↑ | 9 | 0.0795 | 0.0487 | [-0.0331, 0.1936] | 0.2500 | inferential |
| sen2sr_mamba - bicubic | SSIM ↑ | 9 | 0.0072 | 0.0074 | [0.0025, 0.0115] | 0.0391 | inferential |
| sen2sr_mamba - bicubic | RMSE ↓ | 9 | -0.0000 | -0.0001 | [-0.0003, 0.0003] | 0.8203 | inferential |
| sen2sr_mamba - bicubic | SAM (°) ↓ | 9 | 0.1695 | 0.0816 | [0.0703, 0.2805] | 0.0078 | inferential |
| sen2sr_mamba - bicubic | ERGAS ↓ | 9 | -0.0395 | -0.0580 | [-0.0869, 0.0147] | 0.1641 | inferential |
| sen2sr_mamba - bicubic | NDVI MAE ↓ | 9 | 0.0029 | 0.0016 | [0.0014, 0.0045] | 0.0039 | inferential |
| sen2sr_mamba - bicubic | detail rel. error (1 = no detail added) | 9 | -0.0099 | -0.0094 | [-0.0223, 0.0023] | 0.2031 | inferential |
| sen2sr_mamba - bicubic | MAE vs LR | 9 | 0.0001 | -0.0003 | [-0.0004, 0.0006] | 0.5703 | inferential |
| tiny_cnn_seed0 - bicubic | PSNR (dB) ↑ | 9 | -0.2652 | -0.2389 | [-0.5070, -0.0517] | 0.0547 | inferential |
| tiny_cnn_seed0 - bicubic | SSIM ↑ | 9 | -0.0006 | -0.0013 | [-0.0087, 0.0062] | 0.9102 | inferential |
| tiny_cnn_seed0 - bicubic | RMSE ↓ | 9 | 0.0013 | 0.0005 | [0.0002, 0.0025] | 0.0742 | inferential |
| tiny_cnn_seed0 - bicubic | SAM (°) ↓ | 9 | 0.6152 | 0.1568 | [0.2114, 1.0760] | 0.0039 | inferential |
| tiny_cnn_seed0 - bicubic | ERGAS ↓ | 9 | 0.2308 | 0.0714 | [0.0586, 0.4381] | 0.0391 | inferential |
| tiny_cnn_seed0 - bicubic | NDVI MAE ↓ | 9 | 0.0107 | 0.0021 | [0.0034, 0.0191] | 0.0039 | inferential |
| tiny_cnn_seed0 - bicubic | detail rel. error (1 = no detail added) | 9 | 0.0338 | 0.0351 | [0.0102, 0.0614] | 0.0391 | inferential |
| tiny_cnn_seed0 - bicubic | MAE vs LR | 9 | 0.0011 | 0.0002 | [0.0002, 0.0020] | 0.0273 | inferential |
| tiny_cnn_seed1 - bicubic | PSNR (dB) ↑ | 9 | -0.2398 | -0.1758 | [-0.4916, -0.0160] | 0.0977 | inferential |
| tiny_cnn_seed1 - bicubic | SSIM ↑ | 9 | 0.0007 | 0.0008 | [-0.0068, 0.0069] | 0.4961 | inferential |
| tiny_cnn_seed1 - bicubic | RMSE ↓ | 9 | 0.0013 | 0.0003 | [0.0002, 0.0025] | 0.1289 | inferential |
| tiny_cnn_seed1 - bicubic | SAM (°) ↓ | 9 | 0.6106 | 0.1135 | [0.1810, 1.1152] | 0.0039 | inferential |
| tiny_cnn_seed1 - bicubic | ERGAS ↓ | 9 | 0.2132 | 0.0457 | [0.0435, 0.4188] | 0.0742 | inferential |
| tiny_cnn_seed1 - bicubic | NDVI MAE ↓ | 9 | 0.0108 | 0.0017 | [0.0030, 0.0199] | 0.0039 | inferential |
| tiny_cnn_seed1 - bicubic | detail rel. error (1 = no detail added) | 9 | 0.0283 | 0.0186 | [0.0045, 0.0561] | 0.0977 | inferential |
| tiny_cnn_seed1 - bicubic | MAE vs LR | 9 | 0.0011 | 0.0001 | [0.0001, 0.0021] | 0.2500 | inferential |
| tiny_cnn_seed2 - bicubic | PSNR (dB) ↑ | 9 | -0.2425 | -0.2047 | [-0.4756, -0.0330] | 0.0977 | inferential |
| tiny_cnn_seed2 - bicubic | SSIM ↑ | 9 | 0.0002 | 0.0006 | [-0.0066, 0.0058] | 0.5703 | inferential |
| tiny_cnn_seed2 - bicubic | RMSE ↓ | 9 | 0.0012 | 0.0004 | [0.0002, 0.0024] | 0.0977 | inferential |
| tiny_cnn_seed2 - bicubic | SAM (°) ↓ | 9 | 0.6137 | 0.1461 | [0.1986, 1.0893] | 0.0039 | inferential |
| tiny_cnn_seed2 - bicubic | ERGAS ↓ | 9 | 0.2050 | 0.0542 | [0.0467, 0.3973] | 0.0391 | inferential |
| tiny_cnn_seed2 - bicubic | NDVI MAE ↓ | 9 | 0.0108 | 0.0021 | [0.0033, 0.0194] | 0.0039 | inferential |
| tiny_cnn_seed2 - bicubic | detail rel. error (1 = no detail added) | 9 | 0.0294 | 0.0276 | [0.0068, 0.0554] | 0.0547 | inferential |
| tiny_cnn_seed2 - bicubic | MAE vs LR | 9 | 0.0009 | 0.0001 | [0.0001, 0.0019] | 0.2031 | inferential |
| tiny_cnn_seed3 - bicubic | PSNR (dB) ↑ | 9 | -0.1518 | -0.1085 | [-0.3279, 0.0082] | 0.2500 | inferential |
| tiny_cnn_seed3 - bicubic | SSIM ↑ | 9 | 0.0025 | 0.0020 | [-0.0021, 0.0068] | 0.2500 | inferential |
| tiny_cnn_seed3 - bicubic | RMSE ↓ | 9 | 0.0008 | 0.0002 | [0.0001, 0.0017] | 0.2500 | inferential |
| tiny_cnn_seed3 - bicubic | SAM (°) ↓ | 9 | 0.5822 | 0.1053 | [0.1798, 1.0381] | 0.0039 | inferential |
| tiny_cnn_seed3 - bicubic | ERGAS ↓ | 9 | 0.1500 | 0.0331 | [0.0239, 0.3026] | 0.1289 | inferential |
| tiny_cnn_seed3 - bicubic | NDVI MAE ↓ | 9 | 0.0103 | 0.0015 | [0.0030, 0.0187] | 0.0039 | inferential |
| tiny_cnn_seed3 - bicubic | detail rel. error (1 = no detail added) | 9 | 0.0167 | 0.0100 | [0.0004, 0.0349] | 0.1289 | inferential |
| tiny_cnn_seed3 - bicubic | MAE vs LR | 9 | 0.0007 | -0.0000 | [-0.0000, 0.0015] | 0.4258 | inferential |
| tiny_cnn_seed4 - bicubic | PSNR (dB) ↑ | 9 | -0.2048 | -0.0860 | [-0.4489, 0.0023] | 0.1641 | inferential |
| tiny_cnn_seed4 - bicubic | SSIM ↑ | 9 | 0.0017 | 0.0021 | [-0.0056, 0.0076] | 0.2500 | inferential |
| tiny_cnn_seed4 - bicubic | RMSE ↓ | 9 | 0.0011 | 0.0002 | [0.0001, 0.0022] | 0.1641 | inferential |
| tiny_cnn_seed4 - bicubic | SAM (°) ↓ | 9 | 0.5508 | 0.1089 | [0.1631, 1.0009] | 0.0039 | inferential |
| tiny_cnn_seed4 - bicubic | ERGAS ↓ | 9 | 0.1925 | 0.0371 | [0.0323, 0.3942] | 0.0977 | inferential |
| tiny_cnn_seed4 - bicubic | NDVI MAE ↓ | 9 | 0.0096 | 0.0014 | [0.0026, 0.0177] | 0.0039 | inferential |
| tiny_cnn_seed4 - bicubic | detail rel. error (1 = no detail added) | 9 | 0.0232 | 0.0204 | [0.0013, 0.0504] | 0.1289 | inferential |
| tiny_cnn_seed4 - bicubic | MAE vs LR | 9 | 0.0009 | -0.0000 | [-0.0000, 0.0019] | 0.4961 | inferential |
| sen2sr_mamba - sen2sr_lite | PSNR (dB) ↑ | 9 | -0.0263 | -0.0039 | [-0.1023, 0.0390] | 0.8203 | inferential |
| sen2sr_mamba - sen2sr_lite | SSIM ↑ | 9 | -0.0059 | 0.0010 | [-0.0160, 0.0017] | 0.7344 | inferential |
| sen2sr_mamba - sen2sr_lite | RMSE ↓ | 9 | 0.0003 | 0.0000 | [-0.0000, 0.0006] | 0.6523 | inferential |
| sen2sr_mamba - sen2sr_lite | SAM (°) ↓ | 9 | -0.0557 | -0.0189 | [-0.1647, 0.0307] | 0.2500 | inferential |
| sen2sr_mamba - sen2sr_lite | ERGAS ↓ | 9 | 0.0433 | -0.0121 | [-0.0079, 0.1085] | 0.8203 | inferential |
| sen2sr_mamba - sen2sr_lite | NDVI MAE ↓ | 9 | -0.0013 | -0.0003 | [-0.0033, 0.0001] | 0.2500 | inferential |
| sen2sr_mamba - sen2sr_lite | detail rel. error (1 = no detail added) | 9 | 0.0044 | -0.0004 | [-0.0054, 0.0154] | 0.6523 | inferential |
| sen2sr_mamba - sen2sr_lite | MAE vs LR | 9 | 0.0007 | 0.0001 | [0.0001, 0.0016] | 0.0039 | inferential |
| sen2sr_lite - sen2sr_lite_no_constraint | PSNR (dB) ↑ | 9 | 0.7760 | 0.6447 | [0.5719, 1.0227] | 0.0039 | inferential |
| sen2sr_lite - sen2sr_lite_no_constraint | SSIM ↑ | 9 | 0.0012 | 0.0001 | [-0.0006, 0.0029] | 0.3594 | inferential |
| sen2sr_lite - sen2sr_lite_no_constraint | RMSE ↓ | 9 | -0.0020 | -0.0021 | [-0.0026, -0.0015] | 0.0039 | inferential |
| sen2sr_lite - sen2sr_lite_no_constraint | SAM (°) ↓ | 9 | -0.6082 | -0.5604 | [-0.7819, -0.4654] | 0.0039 | inferential |
| sen2sr_lite - sen2sr_lite_no_constraint | ERGAS ↓ | 9 | -0.2913 | -0.2404 | [-0.3848, -0.2066] | 0.0039 | inferential |
| sen2sr_lite - sen2sr_lite_no_constraint | NDVI MAE ↓ | 9 | -0.0120 | -0.0109 | [-0.0152, -0.0093] | 0.0039 | inferential |
| sen2sr_lite - sen2sr_lite_no_constraint | detail rel. error (1 = no detail added) | 9 | -0.0048 | -0.0040 | [-0.0089, -0.0008] | 0.0742 | inferential |
| sen2sr_lite - sen2sr_lite_no_constraint | MAE vs LR | 9 | -0.0038 | -0.0036 | [-0.0048, -0.0030] | 0.0039 | inferential |

### Dataset `opensr_spain_crops` (opensr_test, role independent_benchmark, split test)

Records 28: **evaluated 28**, skipped 0, invalid 0, unreadable 0; 5 scene units; manifest digest `8fda3d88b850d574…`.
Dataset-defined categories (descriptive only): {'crops': 28}

**Reference accuracy (SR vs the independent HR reference)**

| system | n units | PSNR (dB) ↑ | SSIM ↑ | RMSE ↓ | MAE ↓ | SAM (°) ↓ | ERGAS ↓ |
|---|---|---|---|---|---|---|---|
| bicubic | 5 | 32.5305 ± 0.8132 | 0.8137 ± 0.0260 | 0.0242 ± 0.0021 | 0.0163 ± 0.0023 | 2.4172 ± 0.5293 | 3.9377 ± 1.4798 |
| sen2sr_lite | 5 | 32.5494 ± 0.7408 | 0.8188 ± 0.0231 | 0.0242 ± 0.0019 | 0.0162 ± 0.0021 | 2.4420 ± 0.5175 | 3.9670 ± 1.5963 |
| sen2sr_lite_no_constraint | 5 | 32.0136 ± 0.6705 | 0.8149 ± 0.0215 | 0.0256 ± 0.0018 | 0.0177 ± 0.0019 | 2.8380 ± 0.5178 | 4.3230 ± 1.9429 |
| sen2sr_mamba | 5 | 32.5509 ± 0.7690 | 0.8197 ± 0.0239 | 0.0242 ± 0.0019 | 0.0162 ± 0.0022 | 2.4332 ± 0.5200 | 3.9444 ± 1.5313 |
| tiny_cnn_seed0 | 5 | 32.3343 ± 0.8254 | 0.8148 ± 0.0246 | 0.0248 ± 0.0021 | 0.0166 ± 0.0023 | 2.5394 ± 0.5084 | 4.2082 ± 1.8216 |
| tiny_cnn_seed1 | 5 | 32.3392 ± 0.8111 | 0.8160 ± 0.0241 | 0.0248 ± 0.0021 | 0.0166 ± 0.0023 | 2.5307 ± 0.5105 | 4.1966 ± 1.8168 |
| tiny_cnn_seed2 | 5 | 32.3325 ± 0.8034 | 0.8150 ± 0.0244 | 0.0248 ± 0.0021 | 0.0166 ± 0.0023 | 2.5400 ± 0.5068 | 4.2125 ± 1.8381 |
| tiny_cnn_seed3 | 5 | 32.3941 ± 0.7492 | 0.8158 ± 0.0236 | 0.0246 ± 0.0019 | 0.0165 ± 0.0022 | 2.5202 ± 0.5073 | 4.1643 ± 1.8193 |
| tiny_cnn_seed4 | 5 | 32.3700 ± 0.8157 | 0.8163 ± 0.0242 | 0.0247 ± 0.0021 | 0.0165 ± 0.0023 | 2.5159 ± 0.5113 | 4.1763 ± 1.7863 |

**Derived indices and band ratios (vs the HR reference)**

| system | n units | NDVI MAE ↓ | NDVI bias | NDVI r ↑ | NDWI MAE ↓ | log(B08/B04) bias |
|---|---|---|---|---|---|---|
| bicubic | 5 | 0.0391 ± 0.0115 | -0.0021 ± 0.0016 | 0.7816 ± 0.0808 | 0.0315 ± 0.0067 | -0.0009 ± 0.0009 |
| sen2sr_lite | 5 | 0.0393 ± 0.0113 | -0.0018 ± 0.0016 | 0.7832 ± 0.0821 | 0.0318 ± 0.0065 | 0.0006 ± 0.0004 |
| sen2sr_lite_no_constraint | 5 | 0.0476 ± 0.0136 | -0.0053 ± 0.0191 | 0.7808 ± 0.0813 | 0.0374 ± 0.0058 | 0.0182 ± 0.0907 |
| sen2sr_mamba | 5 | 0.0393 ± 0.0114 | -0.0020 ± 0.0019 | 0.7801 ± 0.0872 | 0.0316 ± 0.0065 | -0.0015 ± 0.0033 |
| tiny_cnn_seed0 | 5 | 0.0407 ± 0.0111 | -0.0010 ± 0.0016 | 0.7746 ± 0.0822 | 0.0333 ± 0.0063 | 0.0035 ± 0.0019 |
| tiny_cnn_seed1 | 5 | 0.0406 ± 0.0111 | -0.0007 ± 0.0016 | 0.7761 ± 0.0806 | 0.0332 ± 0.0063 | 0.0046 ± 0.0022 |
| tiny_cnn_seed2 | 5 | 0.0407 ± 0.0111 | -0.0008 ± 0.0019 | 0.7749 ± 0.0816 | 0.0334 ± 0.0062 | 0.0039 ± 0.0017 |
| tiny_cnn_seed3 | 5 | 0.0403 ± 0.0112 | -0.0016 ± 0.0017 | 0.7815 ± 0.0835 | 0.0331 ± 0.0062 | 0.0022 ± 0.0015 |
| tiny_cnn_seed4 | 5 | 0.0403 ± 0.0112 | -0.0007 ± 0.0018 | 0.7785 ± 0.0830 | 0.0330 ± 0.0063 | 0.0040 ± 0.0017 |

**Spatial / detail correctness (vs the HR reference)**

| system | n units | detail rel. error (1 = no detail added) | detail correlation ↑ | detail energy ratio (1 = same amount) | edge correlation ↑ | shift (HR px) | seam / interior error |
|---|---|---|---|---|---|---|---|
| bicubic | 5 | 0.9744 ± 0.0222 | 0.2251 ± 0.0852 | 0.0486 ± 0.0121 | 0.4040 ± 0.0654 | 1.4935 ± 0.8951 | n/a |
| sen2sr_lite | 5 | 0.9755 ± 0.0457 | 0.2402 ± 0.1220 | 0.1096 ± 0.0236 | 0.4312 ± 0.0808 | 1.5436 ± 0.8717 | n/a |
| sen2sr_lite_no_constraint | 5 | 0.9804 ± 0.0524 | 0.2440 ± 0.1216 | 0.1469 ± 0.0321 | 0.4340 ± 0.0813 | 1.5596 ± 0.8732 | n/a |
| sen2sr_mamba | 5 | 0.9792 ± 0.0518 | 0.2328 ± 0.1356 | 0.1258 ± 0.0265 | 0.4353 ± 0.0958 | 1.5752 ± 0.8692 | n/a |
| tiny_cnn_seed0 | 5 | 1.0020 ± 0.0487 | 0.2113 ± 0.1181 | 0.1972 ± 0.0584 | 0.4224 ± 0.0822 | 1.5724 ± 0.8550 | n/a |
| tiny_cnn_seed1 | 5 | 0.9989 ± 0.0490 | 0.2193 ± 0.1165 | 0.1971 ± 0.0583 | 0.4281 ± 0.0790 | 1.5615 ± 0.8336 | n/a |
| tiny_cnn_seed2 | 5 | 1.0014 ± 0.0483 | 0.2115 ± 0.1165 | 0.1951 ± 0.0550 | 0.4227 ± 0.0797 | 1.5759 ± 0.8704 | n/a |
| tiny_cnn_seed3 | 5 | 0.9917 ± 0.0424 | 0.2167 ± 0.1082 | 0.1496 ± 0.0293 | 0.4275 ± 0.0746 | 1.5421 ± 0.8745 | n/a |
| tiny_cnn_seed4 | 5 | 0.9969 ± 0.0486 | 0.2170 ± 0.1180 | 0.1843 ± 0.0537 | 0.4259 ± 0.0828 | 1.5639 ± 0.8483 | n/a |

**Per-band reflectance error** (RMSE ↓ / bias = SR − reference; bands are the RGBN bands every system produces)

| system | B02 RMSE | B03 RMSE | B04 RMSE | B08 RMSE | B02 bias | B03 bias | B04 bias | B08 bias |
|---|---|---|---|---|---|---|---|---|
| bicubic | 0.0159 ± 0.0049 | 0.0192 ± 0.0050 | 0.0254 ± 0.0062 | 0.0315 ± 0.0032 | 0.0001 ± 0.0000 | 0.0001 ± 0.0000 | 0.0001 ± 0.0000 | 0.0001 ± 0.0000 |
| sen2sr_lite | 0.0159 ± 0.0047 | 0.0191 ± 0.0047 | 0.0251 ± 0.0060 | 0.0316 ± 0.0036 | 0.0001 ± 0.0000 | 0.0001 ± 0.0000 | 0.0001 ± 0.0000 | 0.0001 ± 0.0000 |
| sen2sr_lite_no_constraint | 0.0171 ± 0.0047 | 0.0205 ± 0.0048 | 0.0268 ± 0.0058 | 0.0332 ± 0.0042 | 0.0011 ± 0.0027 | 0.0018 ± 0.0033 | 0.0017 ± 0.0045 | -0.0048 ± 0.0028 |
| sen2sr_mamba | 0.0159 ± 0.0048 | 0.0191 ± 0.0049 | 0.0251 ± 0.0061 | 0.0316 ± 0.0036 | 0.0001 ± 0.0000 | 0.0001 ± 0.0000 | 0.0001 ± 0.0000 | 0.0001 ± 0.0000 |
| tiny_cnn_seed0 | 0.0170 ± 0.0046 | 0.0200 ± 0.0048 | 0.0260 ± 0.0060 | 0.0318 ± 0.0032 | -0.0001 ± 0.0001 | -0.0001 ± 0.0001 | -0.0001 ± 0.0001 | -0.0001 ± 0.0001 |
| tiny_cnn_seed1 | 0.0170 ± 0.0046 | 0.0200 ± 0.0048 | 0.0259 ± 0.0060 | 0.0319 ± 0.0034 | -0.0001 ± 0.0001 | -0.0001 ± 0.0001 | -0.0001 ± 0.0001 | 0.0000 ± 0.0000 |
| tiny_cnn_seed2 | 0.0170 ± 0.0046 | 0.0200 ± 0.0048 | 0.0260 ± 0.0060 | 0.0318 ± 0.0034 | -0.0001 ± 0.0001 | -0.0001 ± 0.0001 | -0.0001 ± 0.0001 | 0.0000 ± 0.0001 |
| tiny_cnn_seed3 | 0.0168 ± 0.0045 | 0.0197 ± 0.0046 | 0.0257 ± 0.0058 | 0.0317 ± 0.0035 | -0.0000 ± 0.0001 | -0.0001 ± 0.0001 | -0.0000 ± 0.0001 | -0.0005 ± 0.0004 |
| tiny_cnn_seed4 | 0.0169 ± 0.0046 | 0.0199 ± 0.0048 | 0.0259 ± 0.0060 | 0.0317 ± 0.0033 | -0.0001 ± 0.0001 | -0.0000 ± 0.0001 | -0.0001 ± 0.0001 | 0.0001 ± 0.0001 |

**What was added relative to bicubic, judged against the reference** (element-wise; τ = 0.005 reflectance; fractions of valid elements; the reference is a different sensor, so "unsupported" means "not confirmed by this reference", not "false")

| system | n units | supported synthesis | unsupported detail | omission | MSE skill vs bicubic |
|---|---|---|---|---|---|
| sen2sr_lite | 5 | 0.0718 ± 0.0387 | 0.0665 ± 0.0200 | 0.5886 ± 0.0694 | 0.0026 ± 0.0582 |
| sen2sr_lite_no_constraint | 5 | 0.1580 ± 0.0327 | 0.3048 ± 0.0602 | 0.3593 ± 0.0794 | -0.1302 ± 0.0899 |
| sen2sr_mamba | 5 | 0.0844 ± 0.0426 | 0.0770 ± 0.0253 | 0.5680 ± 0.0597 | 0.0025 ± 0.0637 |
| tiny_cnn_seed0 | 5 | 0.1028 ± 0.0682 | 0.1195 ± 0.0655 | 0.5210 ± 0.0546 | -0.0488 ± 0.0645 |
| tiny_cnn_seed1 | 5 | 0.1036 ± 0.0665 | 0.1200 ± 0.0607 | 0.5195 ± 0.0581 | -0.0478 ± 0.0682 |
| tiny_cnn_seed2 | 5 | 0.1022 ± 0.0665 | 0.1196 ± 0.0610 | 0.5211 ± 0.0554 | -0.0494 ± 0.0678 |
| tiny_cnn_seed3 | 5 | 0.0814 ± 0.0495 | 0.0915 ± 0.0395 | 0.5621 ± 0.0473 | -0.0339 ± 0.0590 |
| tiny_cnn_seed4 | 5 | 0.0956 ± 0.0659 | 0.1074 ± 0.0599 | 0.5363 ± 0.0538 | -0.0404 ± 0.0674 |

**Self-consistency (SR → area-average → compare with the LR)**. This is NOT accuracy against an HR reference: a system can average back to the LR perfectly and still be far from the reference.

| system | n units | MAE vs LR | RMSE vs LR | NDVI MAE vs LR |
|---|---|---|---|---|
| bicubic | 5 | 0.0014 ± 0.0004 | 0.0021 ± 0.0005 | 0.0036 ± 0.0011 |
| sen2sr_lite | 5 | 0.0010 ± 0.0002 | 0.0014 ± 0.0003 | 0.0024 ± 0.0007 |
| sen2sr_lite_no_constraint | 5 | 0.0045 ± 0.0009 | 0.0056 ± 0.0008 | 0.0187 ± 0.0087 |
| sen2sr_mamba | 5 | 0.0012 ± 0.0003 | 0.0018 ± 0.0005 | 0.0045 ± 0.0037 |
| tiny_cnn_seed0 | 5 | 0.0020 ± 0.0008 | 0.0035 ± 0.0011 | 0.0050 ± 0.0015 |
| tiny_cnn_seed1 | 5 | 0.0020 ± 0.0009 | 0.0036 ± 0.0012 | 0.0050 ± 0.0015 |
| tiny_cnn_seed2 | 5 | 0.0020 ± 0.0009 | 0.0035 ± 0.0011 | 0.0051 ± 0.0015 |
| tiny_cnn_seed3 | 5 | 0.0017 ± 0.0005 | 0.0030 ± 0.0005 | 0.0056 ± 0.0015 |
| tiny_cnn_seed4 | 5 | 0.0019 ± 0.0008 | 0.0034 ± 0.0011 | 0.0049 ± 0.0014 |

**By dataset-defined category** (the dataset's own labels, never inferred; descriptive only: few samples per category, no interval or test)

| category | n samples | n units | system | PSNR (dB) | SAM (°) | detail rel. error |
|---|---|---|---|---|---|---|
| crops | 28 | 5 | bicubic | 32.8399 ± 2.1546 | 2.2481 ± 0.8991 | 0.9794 ± 0.0221 |
| crops | 28 | 5 | sen2sr_lite | 32.8180 ± 2.1571 | 2.2768 ± 0.8739 | 0.9849 ± 0.0464 |
| crops | 28 | 5 | sen2sr_lite_no_constraint | 32.2460 ± 2.0479 | 2.7159 ± 0.9446 | 0.9907 ± 0.0538 |
| crops | 28 | 5 | sen2sr_mamba | 32.8251 ± 2.1973 | 2.2642 ± 0.8947 | 0.9882 ± 0.0533 |
| crops | 28 | 5 | tiny_cnn_seed0 | 32.6263 ± 2.2256 | 2.3674 ± 0.9378 | 1.0094 ± 0.0515 |
| crops | 28 | 5 | tiny_cnn_seed1 | 32.6259 ± 2.2171 | 2.3586 ± 0.9349 | 1.0065 ± 0.0517 |
| crops | 28 | 5 | tiny_cnn_seed2 | 32.6147 ± 2.2109 | 2.3698 ± 0.9358 | 1.0089 ± 0.0512 |
| crops | 28 | 5 | tiny_cnn_seed3 | 32.6610 ± 2.1954 | 2.3549 ± 0.9452 | 1.0006 ± 0.0450 |
| crops | 28 | 5 | tiny_cnn_seed4 | 32.6539 ± 2.2288 | 2.3449 ± 0.9344 | 1.0046 ± 0.0512 |

**Paired differences A − B over scene units** (sign convention: A minus B; for error metrics lower is smaller error; a difference is not a ranking)

| A − B | metric | n units | mean difference | median | 95% CI | Wilcoxon p | label |
|---|---|---|---|---|---|---|---|
| sen2sr_lite - bicubic | PSNR (dB) ↑ | 5 | 0.0189 | -0.0112 | [-0.1676, 0.2291] | n/a | descriptive only |
| sen2sr_lite - bicubic | SSIM ↑ | 5 | 0.0051 | 0.0043 | [-0.0028, 0.0144] | n/a | descriptive only |
| sen2sr_lite - bicubic | RMSE ↓ | 5 | -0.0001 | 0.0000 | [-0.0006, 0.0004] | n/a | descriptive only |
| sen2sr_lite - bicubic | SAM (°) ↓ | 5 | 0.0249 | 0.0103 | [0.0081, 0.0515] | n/a | descriptive only |
| sen2sr_lite - bicubic | ERGAS ↓ | 5 | 0.0293 | 0.0003 | [-0.0836, 0.1498] | n/a | descriptive only |
| sen2sr_lite - bicubic | NDVI MAE ↓ | 5 | 0.0003 | 0.0001 | [-0.0001, 0.0008] | n/a | descriptive only |
| sen2sr_lite - bicubic | detail rel. error (1 = no detail added) | 5 | 0.0011 | -0.0006 | [-0.0168, 0.0189] | n/a | descriptive only |
| sen2sr_lite - bicubic | MAE vs LR | 5 | -0.0005 | -0.0004 | [-0.0006, -0.0003] | n/a | descriptive only |
| sen2sr_lite_no_constraint - bicubic | PSNR (dB) ↑ | 5 | -0.5169 | -0.4315 | [-0.7943, -0.2671] | n/a | descriptive only |
| sen2sr_lite_no_constraint - bicubic | SSIM ↑ | 5 | 0.0012 | 0.0028 | [-0.0122, 0.0152] | n/a | descriptive only |
| sen2sr_lite_no_constraint - bicubic | RMSE ↓ | 5 | 0.0014 | 0.0013 | [0.0007, 0.0021] | n/a | descriptive only |
| sen2sr_lite_no_constraint - bicubic | SAM (°) ↓ | 5 | 0.4209 | 0.3884 | [0.3139, 0.5525] | n/a | descriptive only |
| sen2sr_lite_no_constraint - bicubic | ERGAS ↓ | 5 | 0.3853 | 0.1928 | [0.1088, 0.8147] | n/a | descriptive only |
| sen2sr_lite_no_constraint - bicubic | NDVI MAE ↓ | 5 | 0.0085 | 0.0072 | [0.0056, 0.0129] | n/a | descriptive only |
| sen2sr_lite_no_constraint - bicubic | detail rel. error (1 = no detail added) | 5 | 0.0060 | 0.0040 | [-0.0173, 0.0292] | n/a | descriptive only |
| sen2sr_lite_no_constraint - bicubic | MAE vs LR | 5 | 0.0031 | 0.0032 | [0.0023, 0.0041] | n/a | descriptive only |
| sen2sr_mamba - bicubic | PSNR (dB) ↑ | 5 | 0.0204 | -0.0467 | [-0.1860, 0.2510] | n/a | descriptive only |
| sen2sr_mamba - bicubic | SSIM ↑ | 5 | 0.0059 | 0.0049 | [-0.0031, 0.0161] | n/a | descriptive only |
| sen2sr_mamba - bicubic | RMSE ↓ | 5 | -0.0000 | 0.0001 | [-0.0007, 0.0005] | n/a | descriptive only |
| sen2sr_mamba - bicubic | SAM (°) ↓ | 5 | 0.0160 | 0.0052 | [-0.0027, 0.0387] | n/a | descriptive only |
| sen2sr_mamba - bicubic | ERGAS ↓ | 5 | 0.0067 | 0.0181 | [-0.0825, 0.0910] | n/a | descriptive only |
| sen2sr_mamba - bicubic | NDVI MAE ↓ | 5 | 0.0002 | 0.0000 | [-0.0001, 0.0006] | n/a | descriptive only |
| sen2sr_mamba - bicubic | detail rel. error (1 = no detail added) | 5 | 0.0047 | 0.0076 | [-0.0195, 0.0276] | n/a | descriptive only |
| sen2sr_mamba - bicubic | MAE vs LR | 5 | -0.0002 | -0.0003 | [-0.0005, 0.0001] | n/a | descriptive only |
| tiny_cnn_seed0 - bicubic | PSNR (dB) ↑ | 5 | -0.1962 | -0.3026 | [-0.3902, 0.0037] | n/a | descriptive only |
| tiny_cnn_seed0 - bicubic | SSIM ↑ | 5 | 0.0011 | -0.0020 | [-0.0075, 0.0113] | n/a | descriptive only |
| tiny_cnn_seed0 - bicubic | RMSE ↓ | 5 | 0.0006 | 0.0008 | [0.0000, 0.0012] | n/a | descriptive only |
| tiny_cnn_seed0 - bicubic | SAM (°) ↓ | 5 | 0.1223 | 0.0903 | [0.0796, 0.1671] | n/a | descriptive only |
| tiny_cnn_seed0 - bicubic | ERGAS ↓ | 5 | 0.2705 | 0.1574 | [0.0492, 0.5870] | n/a | descriptive only |
| tiny_cnn_seed0 - bicubic | NDVI MAE ↓ | 5 | 0.0016 | 0.0013 | [0.0009, 0.0024] | n/a | descriptive only |
| tiny_cnn_seed0 - bicubic | detail rel. error (1 = no detail added) | 5 | 0.0276 | 0.0346 | [0.0075, 0.0484] | n/a | descriptive only |
| tiny_cnn_seed0 - bicubic | MAE vs LR | 5 | 0.0005 | 0.0005 | [0.0002, 0.0009] | n/a | descriptive only |
| tiny_cnn_seed1 - bicubic | PSNR (dB) ↑ | 5 | -0.1913 | -0.2913 | [-0.4039, 0.0282] | n/a | descriptive only |
| tiny_cnn_seed1 - bicubic | SSIM ↑ | 5 | 0.0022 | -0.0006 | [-0.0066, 0.0129] | n/a | descriptive only |
| tiny_cnn_seed1 - bicubic | RMSE ↓ | 5 | 0.0006 | 0.0009 | [-0.0000, 0.0012] | n/a | descriptive only |
| tiny_cnn_seed1 - bicubic | SAM (°) ↓ | 5 | 0.1135 | 0.0866 | [0.0723, 0.1571] | n/a | descriptive only |
| tiny_cnn_seed1 - bicubic | ERGAS ↓ | 5 | 0.2589 | 0.1521 | [0.0392, 0.5702] | n/a | descriptive only |
| tiny_cnn_seed1 - bicubic | NDVI MAE ↓ | 5 | 0.0015 | 0.0012 | [0.0008, 0.0022] | n/a | descriptive only |
| tiny_cnn_seed1 - bicubic | detail rel. error (1 = no detail added) | 5 | 0.0245 | 0.0307 | [0.0037, 0.0451] | n/a | descriptive only |
| tiny_cnn_seed1 - bicubic | MAE vs LR | 5 | 0.0006 | 0.0004 | [0.0002, 0.0010] | n/a | descriptive only |
| tiny_cnn_seed2 - bicubic | PSNR (dB) ↑ | 5 | -0.1980 | -0.2777 | [-0.4084, 0.0160] | n/a | descriptive only |
| tiny_cnn_seed2 - bicubic | SSIM ↑ | 5 | 0.0013 | -0.0016 | [-0.0075, 0.0119] | n/a | descriptive only |
| tiny_cnn_seed2 - bicubic | RMSE ↓ | 5 | 0.0006 | 0.0008 | [-0.0000, 0.0012] | n/a | descriptive only |
| tiny_cnn_seed2 - bicubic | SAM (°) ↓ | 5 | 0.1228 | 0.0978 | [0.0783, 0.1674] | n/a | descriptive only |
| tiny_cnn_seed2 - bicubic | ERGAS ↓ | 5 | 0.2747 | 0.1447 | [0.0453, 0.6086] | n/a | descriptive only |
| tiny_cnn_seed2 - bicubic | NDVI MAE ↓ | 5 | 0.0016 | 0.0013 | [0.0009, 0.0024] | n/a | descriptive only |
| tiny_cnn_seed2 - bicubic | detail rel. error (1 = no detail added) | 5 | 0.0269 | 0.0338 | [0.0068, 0.0468] | n/a | descriptive only |
| tiny_cnn_seed2 - bicubic | MAE vs LR | 5 | 0.0005 | 0.0005 | [0.0002, 0.0009] | n/a | descriptive only |
| tiny_cnn_seed3 - bicubic | PSNR (dB) ↑ | 5 | -0.1364 | -0.1232 | [-0.3277, 0.0549] | n/a | descriptive only |
| tiny_cnn_seed3 - bicubic | SSIM ↑ | 5 | 0.0021 | 0.0024 | [-0.0055, 0.0103] | n/a | descriptive only |
| tiny_cnn_seed3 - bicubic | RMSE ↓ | 5 | 0.0004 | 0.0004 | [-0.0001, 0.0009] | n/a | descriptive only |
| tiny_cnn_seed3 - bicubic | SAM (°) ↓ | 5 | 0.1030 | 0.1009 | [0.0674, 0.1387] | n/a | descriptive only |
| tiny_cnn_seed3 - bicubic | ERGAS ↓ | 5 | 0.2266 | 0.0707 | [0.0160, 0.5485] | n/a | descriptive only |
| tiny_cnn_seed3 - bicubic | NDVI MAE ↓ | 5 | 0.0013 | 0.0011 | [0.0008, 0.0019] | n/a | descriptive only |
| tiny_cnn_seed3 - bicubic | detail rel. error (1 = no detail added) | 5 | 0.0173 | 0.0127 | [0.0014, 0.0340] | n/a | descriptive only |
| tiny_cnn_seed3 - bicubic | MAE vs LR | 5 | 0.0002 | 0.0002 | [0.0002, 0.0003] | n/a | descriptive only |
| tiny_cnn_seed4 - bicubic | PSNR (dB) ↑ | 5 | -0.1606 | -0.2468 | [-0.3648, 0.0524] | n/a | descriptive only |
| tiny_cnn_seed4 - bicubic | SSIM ↑ | 5 | 0.0025 | -0.0003 | [-0.0061, 0.0131] | n/a | descriptive only |
| tiny_cnn_seed4 - bicubic | RMSE ↓ | 5 | 0.0005 | 0.0007 | [-0.0001, 0.0011] | n/a | descriptive only |
| tiny_cnn_seed4 - bicubic | SAM (°) ↓ | 5 | 0.0987 | 0.0735 | [0.0628, 0.1347] | n/a | descriptive only |
| tiny_cnn_seed4 - bicubic | ERGAS ↓ | 5 | 0.2386 | 0.1341 | [0.0311, 0.5240] | n/a | descriptive only |
| tiny_cnn_seed4 - bicubic | NDVI MAE ↓ | 5 | 0.0012 | 0.0010 | [0.0006, 0.0018] | n/a | descriptive only |
| tiny_cnn_seed4 - bicubic | detail rel. error (1 = no detail added) | 5 | 0.0225 | 0.0281 | [0.0016, 0.0435] | n/a | descriptive only |
| tiny_cnn_seed4 - bicubic | MAE vs LR | 5 | 0.0004 | 0.0003 | [0.0001, 0.0008] | n/a | descriptive only |
| sen2sr_mamba - sen2sr_lite | PSNR (dB) ↑ | 5 | 0.0015 | 0.0189 | [-0.0293, 0.0323] | n/a | descriptive only |
| sen2sr_mamba - sen2sr_lite | SSIM ↑ | 5 | 0.0009 | 0.0007 | [-0.0003, 0.0019] | n/a | descriptive only |
| sen2sr_mamba - sen2sr_lite | RMSE ↓ | 5 | 0.0000 | -0.0000 | [-0.0001, 0.0001] | n/a | descriptive only |
| sen2sr_mamba - sen2sr_lite | SAM (°) ↓ | 5 | -0.0089 | -0.0160 | [-0.0173, 0.0024] | n/a | descriptive only |
| sen2sr_mamba - sen2sr_lite | ERGAS ↓ | 5 | -0.0226 | 0.0004 | [-0.0843, 0.0159] | n/a | descriptive only |
| sen2sr_mamba - sen2sr_lite | NDVI MAE ↓ | 5 | -0.0001 | -0.0001 | [-0.0002, 0.0001] | n/a | descriptive only |
| sen2sr_mamba - sen2sr_lite | detail rel. error (1 = no detail added) | 5 | 0.0036 | 0.0040 | [-0.0022, 0.0095] | n/a | descriptive only |
| sen2sr_mamba - sen2sr_lite | MAE vs LR | 5 | 0.0003 | 0.0001 | [0.0001, 0.0005] | n/a | descriptive only |
| sen2sr_lite - sen2sr_lite_no_constraint | PSNR (dB) ↑ | 5 | 0.5358 | 0.4932 | [0.4580, 0.6319] | n/a | descriptive only |
| sen2sr_lite - sen2sr_lite_no_constraint | SSIM ↑ | 5 | 0.0039 | 0.0015 | [-0.0018, 0.0112] | n/a | descriptive only |
| sen2sr_lite - sen2sr_lite_no_constraint | RMSE ↓ | 5 | -0.0015 | -0.0014 | [-0.0017, -0.0013] | n/a | descriptive only |
| sen2sr_lite - sen2sr_lite_no_constraint | SAM (°) ↓ | 5 | -0.3960 | -0.3654 | [-0.5271, -0.2995] | n/a | descriptive only |
| sen2sr_lite - sen2sr_lite_no_constraint | ERGAS ↓ | 5 | -0.3560 | -0.1995 | [-0.6725, -0.1856] | n/a | descriptive only |
| sen2sr_lite - sen2sr_lite_no_constraint | NDVI MAE ↓ | 5 | -0.0083 | -0.0068 | [-0.0126, -0.0055] | n/a | descriptive only |
| sen2sr_lite - sen2sr_lite_no_constraint | detail rel. error (1 = no detail added) | 5 | -0.0048 | -0.0046 | [-0.0103, 0.0006] | n/a | descriptive only |
| sen2sr_lite - sen2sr_lite_no_constraint | MAE vs LR | 5 | -0.0036 | -0.0035 | [-0.0044, -0.0028] | n/a | descriptive only |

### Dataset `opensr_spain_urban` (opensr_test, role independent_benchmark, split test)

Records 20: **evaluated 20**, skipped 0, invalid 0, unreadable 0; 4 scene units; manifest digest `1596f37c40c631a6…`.
Dataset-defined categories (descriptive only): {'urban': 20}

**Reference accuracy (SR vs the independent HR reference)**

| system | n units | PSNR (dB) ↑ | SSIM ↑ | RMSE ↓ | MAE ↓ | SAM (°) ↓ | ERGAS ↓ |
|---|---|---|---|---|---|---|---|
| bicubic | 4 | 29.2855 ± 1.7286 | 0.7001 ± 0.0588 | 0.0357 ± 0.0071 | 0.0247 ± 0.0048 | 3.9187 ± 0.9874 | 6.2182 ± 2.0500 |
| sen2sr_lite | 4 | 29.2793 ± 1.8968 | 0.7103 ± 0.0738 | 0.0358 ± 0.0081 | 0.0246 ± 0.0053 | 3.9518 ± 1.0363 | 6.2690 ± 2.2100 |
| sen2sr_lite_no_constraint | 4 | 28.7937 ± 1.8870 | 0.7056 ± 0.0785 | 0.0378 ± 0.0086 | 0.0263 ± 0.0056 | 4.4526 ± 1.2081 | 6.6893 ± 2.4382 |
| sen2sr_mamba | 4 | 29.2511 ± 1.9081 | 0.7086 ± 0.0747 | 0.0360 ± 0.0081 | 0.0246 ± 0.0053 | 3.9981 ± 1.0633 | 6.2652 ± 2.1834 |
| tiny_cnn_seed0 | 4 | 28.8670 ± 1.9765 | 0.6992 ± 0.0793 | 0.0377 ± 0.0089 | 0.0258 ± 0.0059 | 4.2399 ± 1.0440 | 6.7195 ± 2.4015 |
| tiny_cnn_seed1 | 4 | 28.8621 ± 1.9893 | 0.7005 ± 0.0804 | 0.0377 ± 0.0090 | 0.0258 ± 0.0059 | 4.2403 ± 1.0503 | 6.7147 ± 2.4087 |
| tiny_cnn_seed2 | 4 | 28.8728 ± 1.9563 | 0.6996 ± 0.0796 | 0.0376 ± 0.0088 | 0.0258 ± 0.0058 | 4.2452 ± 1.0581 | 6.7189 ± 2.4290 |
| tiny_cnn_seed3 | 4 | 28.9861 ± 1.9381 | 0.7016 ± 0.0773 | 0.0371 ± 0.0086 | 0.0255 ± 0.0057 | 4.2258 ± 1.0579 | 6.6253 ± 2.4062 |
| tiny_cnn_seed4 | 4 | 28.8904 ± 1.9853 | 0.7004 ± 0.0799 | 0.0376 ± 0.0089 | 0.0258 ± 0.0059 | 4.2076 ± 1.0395 | 6.6965 ± 2.3927 |

**Derived indices and band ratios (vs the HR reference)**

| system | n units | NDVI MAE ↓ | NDVI bias | NDVI r ↑ | NDWI MAE ↓ | log(B08/B04) bias |
|---|---|---|---|---|---|---|
| bicubic | 4 | 0.0623 ± 0.0202 | -0.0035 ± 0.0024 | 0.7750 ± 0.0505 | 0.0517 ± 0.0134 | -0.0015 ± 0.0012 |
| sen2sr_lite | 4 | 0.0628 ± 0.0207 | -0.0027 ± 0.0023 | 0.7821 ± 0.0544 | 0.0521 ± 0.0143 | 0.0018 ± 0.0006 |
| sen2sr_lite_no_constraint | 4 | 0.0735 ± 0.0255 | -0.0074 ± 0.0158 | 0.7829 ± 0.0572 | 0.0593 ± 0.0167 | 0.0188 ± 0.0842 |
| sen2sr_mamba | 4 | 0.0634 ± 0.0213 | -0.0036 ± 0.0030 | 0.7719 ± 0.0597 | 0.0526 ± 0.0147 | -0.0032 ± 0.0060 |
| tiny_cnn_seed0 | 4 | 0.0673 ± 0.0204 | -0.0005 ± 0.0023 | 0.7695 ± 0.0551 | 0.0565 ± 0.0144 | 0.0099 ± 0.0023 |
| tiny_cnn_seed1 | 4 | 0.0674 ± 0.0205 | -0.0003 ± 0.0023 | 0.7697 ± 0.0544 | 0.0565 ± 0.0145 | 0.0112 ± 0.0035 |
| tiny_cnn_seed2 | 4 | 0.0677 ± 0.0209 | -0.0007 ± 0.0026 | 0.7415 ± 0.1003 | 0.0566 ± 0.0147 | 0.0093 ± 0.0016 |
| tiny_cnn_seed3 | 4 | 0.0671 ± 0.0207 | -0.0016 ± 0.0023 | 0.7739 ± 0.0556 | 0.0564 ± 0.0147 | 0.0074 ± 0.0021 |
| tiny_cnn_seed4 | 4 | 0.0667 ± 0.0205 | -0.0005 ± 0.0028 | 0.7531 ± 0.0835 | 0.0561 ± 0.0144 | 0.0098 ± 0.0013 |

**Spatial / detail correctness (vs the HR reference)**

| system | n units | detail rel. error (1 = no detail added) | detail correlation ↑ | detail energy ratio (1 = same amount) | edge correlation ↑ | shift (HR px) | seam / interior error |
|---|---|---|---|---|---|---|---|
| bicubic | 4 | 0.9735 ± 0.0250 | 0.2342 ± 0.0926 | 0.0589 ± 0.0067 | 0.3852 ± 0.0617 | 1.6096 ± 0.9351 | n/a |
| sen2sr_lite | 4 | 0.9775 ± 0.0529 | 0.2517 ± 0.1293 | 0.1404 ± 0.0099 | 0.4191 ± 0.0810 | 1.6522 ± 0.9242 | n/a |
| sen2sr_lite_no_constraint | 4 | 0.9854 ± 0.0614 | 0.2556 ± 0.1293 | 0.1894 ± 0.0134 | 0.4222 ± 0.0813 | 1.6776 ± 0.9088 | n/a |
| sen2sr_mamba | 4 | 0.9849 ± 0.0607 | 0.2360 ± 0.1412 | 0.1584 ± 0.0218 | 0.4161 ± 0.0984 | 1.6380 ± 0.9054 | n/a |
| tiny_cnn_seed0 | 4 | 1.0237 ± 0.0617 | 0.2191 ± 0.1137 | 0.2793 ± 0.0483 | 0.4210 ± 0.0753 | 1.6679 ± 0.9089 | n/a |
| tiny_cnn_seed1 | 4 | 1.0222 ± 0.0652 | 0.2234 ± 0.1172 | 0.2803 ± 0.0422 | 0.4251 ± 0.0750 | 1.6648 ± 0.9067 | n/a |
| tiny_cnn_seed2 | 4 | 1.0219 ± 0.0599 | 0.2197 ± 0.1126 | 0.2746 ± 0.0363 | 0.4219 ± 0.0738 | 1.6766 ± 0.9070 | n/a |
| tiny_cnn_seed3 | 4 | 1.0073 ± 0.0571 | 0.2249 ± 0.1134 | 0.2234 ± 0.0237 | 0.4247 ± 0.0761 | 1.6483 ± 0.9378 | n/a |
| tiny_cnn_seed4 | 4 | 1.0198 ± 0.0633 | 0.2237 ± 0.1159 | 0.2717 ± 0.0450 | 0.4220 ± 0.0765 | 1.6521 ± 0.9112 | n/a |

**Per-band reflectance error** (RMSE ↓ / bias = SR − reference; bands are the RGBN bands every system produces)

| system | B02 RMSE | B03 RMSE | B04 RMSE | B08 RMSE | B02 bias | B03 bias | B04 bias | B08 bias |
|---|---|---|---|---|---|---|---|---|
| bicubic | 0.0272 ± 0.0082 | 0.0306 ± 0.0080 | 0.0380 ± 0.0080 | 0.0432 ± 0.0090 | 0.0001 ± 0.0000 | 0.0001 ± 0.0000 | 0.0001 ± 0.0000 | 0.0001 ± 0.0000 |
| sen2sr_lite | 0.0274 ± 0.0088 | 0.0307 ± 0.0087 | 0.0379 ± 0.0092 | 0.0436 ± 0.0099 | 0.0001 ± 0.0000 | 0.0001 ± 0.0000 | 0.0001 ± 0.0000 | 0.0001 ± 0.0000 |
| sen2sr_lite_no_constraint | 0.0291 ± 0.0092 | 0.0325 ± 0.0092 | 0.0400 ± 0.0098 | 0.0459 ± 0.0105 | 0.0006 ± 0.0022 | 0.0009 ± 0.0025 | 0.0005 ± 0.0033 | -0.0062 ± 0.0018 |
| sen2sr_mamba | 0.0275 ± 0.0087 | 0.0307 ± 0.0087 | 0.0380 ± 0.0091 | 0.0438 ± 0.0102 | 0.0001 ± 0.0000 | 0.0001 ± 0.0000 | 0.0001 ± 0.0000 | 0.0001 ± 0.0000 |
| tiny_cnn_seed0 | 0.0297 ± 0.0092 | 0.0328 ± 0.0095 | 0.0403 ± 0.0103 | 0.0447 ± 0.0101 | -0.0002 ± 0.0001 | -0.0002 ± 0.0001 | -0.0001 ± 0.0001 | -0.0002 ± 0.0001 |
| tiny_cnn_seed1 | 0.0296 ± 0.0093 | 0.0327 ± 0.0096 | 0.0403 ± 0.0104 | 0.0449 ± 0.0104 | -0.0002 ± 0.0001 | -0.0001 ± 0.0000 | -0.0001 ± 0.0000 | 0.0000 ± 0.0000 |
| tiny_cnn_seed2 | 0.0295 ± 0.0091 | 0.0327 ± 0.0093 | 0.0402 ± 0.0102 | 0.0448 ± 0.0102 | -0.0002 ± 0.0001 | -0.0001 ± 0.0001 | -0.0000 ± 0.0001 | -0.0001 ± 0.0001 |
| tiny_cnn_seed3 | 0.0292 ± 0.0091 | 0.0322 ± 0.0092 | 0.0397 ± 0.0101 | 0.0442 ± 0.0098 | -0.0001 ± 0.0001 | -0.0002 ± 0.0001 | -0.0001 ± 0.0001 | -0.0008 ± 0.0003 |
| tiny_cnn_seed4 | 0.0295 ± 0.0092 | 0.0327 ± 0.0095 | 0.0402 ± 0.0103 | 0.0447 ± 0.0103 | -0.0002 ± 0.0001 | -0.0001 ± 0.0001 | -0.0001 ± 0.0001 | 0.0000 ± 0.0002 |

**What was added relative to bicubic, judged against the reference** (element-wise; τ = 0.005 reflectance; fractions of valid elements; the reference is a different sensor, so "unsupported" means "not confirmed by this reference", not "false")

| system | n units | supported synthesis | unsupported detail | omission | MSE skill vs bicubic |
|---|---|---|---|---|---|
| sen2sr_lite | 4 | 0.1474 ± 0.0264 | 0.1452 ± 0.0408 | 0.5448 ± 0.0317 | -0.0031 ± 0.0599 |
| sen2sr_lite_no_constraint | 4 | 0.2087 ± 0.0313 | 0.3462 ± 0.0460 | 0.3413 ± 0.0460 | -0.1232 ± 0.0767 |
| sen2sr_mamba | 4 | 0.1641 ± 0.0340 | 0.1617 ± 0.0394 | 0.5155 ± 0.0218 | -0.0097 ± 0.0633 |
| tiny_cnn_seed0 | 4 | 0.1801 ± 0.0497 | 0.2284 ± 0.0719 | 0.4464 ± 0.0408 | -0.1041 ± 0.0763 |
| tiny_cnn_seed1 | 4 | 0.1839 ± 0.0436 | 0.2342 ± 0.0657 | 0.4381 ± 0.0293 | -0.1057 ± 0.0825 |
| tiny_cnn_seed2 | 4 | 0.1826 ± 0.0453 | 0.2326 ± 0.0669 | 0.4405 ± 0.0320 | -0.1027 ± 0.0748 |
| tiny_cnn_seed3 | 4 | 0.1622 ± 0.0378 | 0.2023 ± 0.0583 | 0.4850 ± 0.0207 | -0.0740 ± 0.0697 |
| tiny_cnn_seed4 | 4 | 0.1759 ± 0.0480 | 0.2221 ± 0.0701 | 0.4552 ± 0.0367 | -0.0986 ± 0.0808 |

**Self-consistency (SR → area-average → compare with the LR)**. This is NOT accuracy against an HR reference: a system can average back to the LR perfectly and still be far from the reference.

| system | n units | MAE vs LR | RMSE vs LR | NDVI MAE vs LR |
|---|---|---|---|---|
| bicubic | 4 | 0.0024 ± 0.0004 | 0.0035 ± 0.0005 | 0.0068 ± 0.0017 |
| sen2sr_lite | 4 | 0.0016 ± 0.0002 | 0.0024 ± 0.0003 | 0.0046 ± 0.0010 |
| sen2sr_lite_no_constraint | 4 | 0.0051 ± 0.0005 | 0.0067 ± 0.0005 | 0.0212 ± 0.0078 |
| sen2sr_mamba | 4 | 0.0022 ± 0.0004 | 0.0034 ± 0.0005 | 0.0093 ± 0.0056 |
| tiny_cnn_seed0 | 4 | 0.0038 ± 0.0009 | 0.0063 ± 0.0012 | 0.0108 ± 0.0015 |
| tiny_cnn_seed1 | 4 | 0.0040 ± 0.0009 | 0.0065 ± 0.0012 | 0.0109 ± 0.0017 |
| tiny_cnn_seed2 | 4 | 0.0038 ± 0.0008 | 0.0062 ± 0.0010 | 0.0110 ± 0.0015 |
| tiny_cnn_seed3 | 4 | 0.0034 ± 0.0006 | 0.0057 ± 0.0008 | 0.0125 ± 0.0019 |
| tiny_cnn_seed4 | 4 | 0.0038 ± 0.0009 | 0.0064 ± 0.0012 | 0.0107 ± 0.0014 |

**By dataset-defined category** (the dataset's own labels, never inferred; descriptive only: few samples per category, no interval or test)

| category | n samples | n units | system | PSNR (dB) | SAM (°) | detail rel. error |
|---|---|---|---|---|---|---|
| urban | 20 | 4 | bicubic | 29.2851 ± 2.6184 | 4.0157 ± 1.4690 | 0.9746 ± 0.0255 |
| urban | 20 | 4 | sen2sr_lite | 29.2663 ± 2.6795 | 4.0478 ± 1.4882 | 0.9800 ± 0.0534 |
| urban | 20 | 4 | sen2sr_lite_no_constraint | 28.7761 ± 2.6165 | 4.5702 ± 1.6339 | 0.9882 ± 0.0625 |
| urban | 20 | 4 | sen2sr_mamba | 29.2374 ± 2.7065 | 4.0982 ± 1.5365 | 0.9876 ± 0.0591 |
| urban | 20 | 4 | tiny_cnn_seed0 | 28.8673 ± 2.7721 | 4.3315 ± 1.6112 | 1.0255 ± 0.0624 |
| urban | 20 | 4 | tiny_cnn_seed1 | 28.8599 ± 2.7644 | 4.3322 ± 1.6158 | 1.0242 ± 0.0658 |
| urban | 20 | 4 | tiny_cnn_seed2 | 28.8704 ± 2.7439 | 4.3381 ± 1.6143 | 1.0238 ± 0.0618 |
| urban | 20 | 4 | tiny_cnn_seed3 | 28.9775 ± 2.7247 | 4.3182 ± 1.6194 | 1.0096 ± 0.0579 |
| urban | 20 | 4 | tiny_cnn_seed4 | 28.8905 ± 2.7814 | 4.2995 ± 1.6030 | 1.0216 ± 0.0647 |

**Paired differences A − B over scene units** (sign convention: A minus B; for error metrics lower is smaller error; a difference is not a ranking)

| A − B | metric | n units | mean difference | median | 95% CI | Wilcoxon p | label |
|---|---|---|---|---|---|---|---|
| sen2sr_lite - bicubic | PSNR (dB) ↑ | 4 | -0.0062 | 0.0056 | n/a | n/a | descriptive only |
| sen2sr_lite - bicubic | SSIM ↑ | 4 | 0.0102 | 0.0127 | n/a | n/a | descriptive only |
| sen2sr_lite - bicubic | RMSE ↓ | 4 | 0.0001 | -0.0001 | n/a | n/a | descriptive only |
| sen2sr_lite - bicubic | SAM (°) ↓ | 4 | 0.0331 | -0.0069 | n/a | n/a | descriptive only |
| sen2sr_lite - bicubic | ERGAS ↓ | 4 | 0.0509 | 0.0399 | n/a | n/a | descriptive only |
| sen2sr_lite - bicubic | NDVI MAE ↓ | 4 | 0.0005 | -0.0002 | n/a | n/a | descriptive only |
| sen2sr_lite - bicubic | detail rel. error (1 = no detail added) | 4 | 0.0040 | 0.0001 | n/a | n/a | descriptive only |
| sen2sr_lite - bicubic | MAE vs LR | 4 | -0.0008 | -0.0008 | n/a | n/a | descriptive only |
| sen2sr_lite_no_constraint - bicubic | PSNR (dB) ↑ | 4 | -0.4918 | -0.5132 | n/a | n/a | descriptive only |
| sen2sr_lite_no_constraint - bicubic | SSIM ↑ | 4 | 0.0055 | 0.0053 | n/a | n/a | descriptive only |
| sen2sr_lite_no_constraint - bicubic | RMSE ↓ | 4 | 0.0021 | 0.0017 | n/a | n/a | descriptive only |
| sen2sr_lite_no_constraint - bicubic | SAM (°) ↓ | 4 | 0.5338 | 0.5538 | n/a | n/a | descriptive only |
| sen2sr_lite_no_constraint - bicubic | ERGAS ↓ | 4 | 0.4712 | 0.4536 | n/a | n/a | descriptive only |
| sen2sr_lite_no_constraint - bicubic | NDVI MAE ↓ | 4 | 0.0112 | 0.0113 | n/a | n/a | descriptive only |
| sen2sr_lite_no_constraint - bicubic | detail rel. error (1 = no detail added) | 4 | 0.0119 | 0.0066 | n/a | n/a | descriptive only |
| sen2sr_lite_no_constraint - bicubic | MAE vs LR | 4 | 0.0028 | 0.0024 | n/a | n/a | descriptive only |
| sen2sr_mamba - bicubic | PSNR (dB) ↑ | 4 | -0.0344 | -0.0194 | n/a | n/a | descriptive only |
| sen2sr_mamba - bicubic | SSIM ↑ | 4 | 0.0085 | 0.0103 | n/a | n/a | descriptive only |
| sen2sr_mamba - bicubic | RMSE ↓ | 4 | 0.0003 | 0.0001 | n/a | n/a | descriptive only |
| sen2sr_mamba - bicubic | SAM (°) ↓ | 4 | 0.0793 | 0.0461 | n/a | n/a | descriptive only |
| sen2sr_mamba - bicubic | ERGAS ↓ | 4 | 0.0470 | 0.0157 | n/a | n/a | descriptive only |
| sen2sr_mamba - bicubic | NDVI MAE ↓ | 4 | 0.0010 | 0.0007 | n/a | n/a | descriptive only |
| sen2sr_mamba - bicubic | detail rel. error (1 = no detail added) | 4 | 0.0115 | 0.0052 | n/a | n/a | descriptive only |
| sen2sr_mamba - bicubic | MAE vs LR | 4 | -0.0001 | -0.0004 | n/a | n/a | descriptive only |
| tiny_cnn_seed0 - bicubic | PSNR (dB) ↑ | 4 | -0.4185 | -0.3423 | n/a | n/a | descriptive only |
| tiny_cnn_seed0 - bicubic | SSIM ↑ | 4 | -0.0009 | 0.0050 | n/a | n/a | descriptive only |
| tiny_cnn_seed0 - bicubic | RMSE ↓ | 4 | 0.0020 | 0.0013 | n/a | n/a | descriptive only |
| tiny_cnn_seed0 - bicubic | SAM (°) ↓ | 4 | 0.3212 | 0.2848 | n/a | n/a | descriptive only |
| tiny_cnn_seed0 - bicubic | ERGAS ↓ | 4 | 0.5013 | 0.5143 | n/a | n/a | descriptive only |
| tiny_cnn_seed0 - bicubic | NDVI MAE ↓ | 4 | 0.0049 | 0.0043 | n/a | n/a | descriptive only |
| tiny_cnn_seed0 - bicubic | detail rel. error (1 = no detail added) | 4 | 0.0503 | 0.0375 | n/a | n/a | descriptive only |
| tiny_cnn_seed0 - bicubic | MAE vs LR | 4 | 0.0014 | 0.0016 | n/a | n/a | descriptive only |
| tiny_cnn_seed1 - bicubic | PSNR (dB) ↑ | 4 | -0.4234 | -0.3537 | n/a | n/a | descriptive only |
| tiny_cnn_seed1 - bicubic | SSIM ↑ | 4 | 0.0003 | 0.0064 | n/a | n/a | descriptive only |
| tiny_cnn_seed1 - bicubic | RMSE ↓ | 4 | 0.0020 | 0.0013 | n/a | n/a | descriptive only |
| tiny_cnn_seed1 - bicubic | SAM (°) ↓ | 4 | 0.3215 | 0.2816 | n/a | n/a | descriptive only |
| tiny_cnn_seed1 - bicubic | ERGAS ↓ | 4 | 0.4965 | 0.5052 | n/a | n/a | descriptive only |
| tiny_cnn_seed1 - bicubic | NDVI MAE ↓ | 4 | 0.0050 | 0.0044 | n/a | n/a | descriptive only |
| tiny_cnn_seed1 - bicubic | detail rel. error (1 = no detail added) | 4 | 0.0488 | 0.0357 | n/a | n/a | descriptive only |
| tiny_cnn_seed1 - bicubic | MAE vs LR | 4 | 0.0016 | 0.0017 | n/a | n/a | descriptive only |
| tiny_cnn_seed2 - bicubic | PSNR (dB) ↑ | 4 | -0.4127 | -0.3640 | n/a | n/a | descriptive only |
| tiny_cnn_seed2 - bicubic | SSIM ↑ | 4 | -0.0005 | 0.0051 | n/a | n/a | descriptive only |
| tiny_cnn_seed2 - bicubic | RMSE ↓ | 4 | 0.0019 | 0.0013 | n/a | n/a | descriptive only |
| tiny_cnn_seed2 - bicubic | SAM (°) ↓ | 4 | 0.3264 | 0.2772 | n/a | n/a | descriptive only |
| tiny_cnn_seed2 - bicubic | ERGAS ↓ | 4 | 0.5007 | 0.5186 | n/a | n/a | descriptive only |
| tiny_cnn_seed2 - bicubic | NDVI MAE ↓ | 4 | 0.0053 | 0.0043 | n/a | n/a | descriptive only |
| tiny_cnn_seed2 - bicubic | detail rel. error (1 = no detail added) | 4 | 0.0484 | 0.0390 | n/a | n/a | descriptive only |
| tiny_cnn_seed2 - bicubic | MAE vs LR | 4 | 0.0014 | 0.0016 | n/a | n/a | descriptive only |
| tiny_cnn_seed3 - bicubic | PSNR (dB) ↑ | 4 | -0.2994 | -0.2598 | n/a | n/a | descriptive only |
| tiny_cnn_seed3 - bicubic | SSIM ↑ | 4 | 0.0015 | 0.0062 | n/a | n/a | descriptive only |
| tiny_cnn_seed3 - bicubic | RMSE ↓ | 4 | 0.0014 | 0.0009 | n/a | n/a | descriptive only |
| tiny_cnn_seed3 - bicubic | SAM (°) ↓ | 4 | 0.3070 | 0.2576 | n/a | n/a | descriptive only |
| tiny_cnn_seed3 - bicubic | ERGAS ↓ | 4 | 0.4072 | 0.4265 | n/a | n/a | descriptive only |
| tiny_cnn_seed3 - bicubic | NDVI MAE ↓ | 4 | 0.0047 | 0.0038 | n/a | n/a | descriptive only |
| tiny_cnn_seed3 - bicubic | detail rel. error (1 = no detail added) | 4 | 0.0339 | 0.0260 | n/a | n/a | descriptive only |
| tiny_cnn_seed3 - bicubic | MAE vs LR | 4 | 0.0010 | 0.0010 | n/a | n/a | descriptive only |
| tiny_cnn_seed4 - bicubic | PSNR (dB) ↑ | 4 | -0.3951 | -0.3264 | n/a | n/a | descriptive only |
| tiny_cnn_seed4 - bicubic | SSIM ↑ | 4 | 0.0003 | 0.0061 | n/a | n/a | descriptive only |
| tiny_cnn_seed4 - bicubic | RMSE ↓ | 4 | 0.0019 | 0.0012 | n/a | n/a | descriptive only |
| tiny_cnn_seed4 - bicubic | SAM (°) ↓ | 4 | 0.2889 | 0.2518 | n/a | n/a | descriptive only |
| tiny_cnn_seed4 - bicubic | ERGAS ↓ | 4 | 0.4784 | 0.4898 | n/a | n/a | descriptive only |
| tiny_cnn_seed4 - bicubic | NDVI MAE ↓ | 4 | 0.0044 | 0.0037 | n/a | n/a | descriptive only |
| tiny_cnn_seed4 - bicubic | detail rel. error (1 = no detail added) | 4 | 0.0463 | 0.0340 | n/a | n/a | descriptive only |
| tiny_cnn_seed4 - bicubic | MAE vs LR | 4 | 0.0014 | 0.0016 | n/a | n/a | descriptive only |
| sen2sr_mamba - sen2sr_lite | PSNR (dB) ↑ | 4 | -0.0282 | -0.0250 | n/a | n/a | descriptive only |
| sen2sr_mamba - sen2sr_lite | SSIM ↑ | 4 | -0.0017 | -0.0011 | n/a | n/a | descriptive only |
| sen2sr_mamba - sen2sr_lite | RMSE ↓ | 4 | 0.0001 | 0.0001 | n/a | n/a | descriptive only |
| sen2sr_mamba - sen2sr_lite | SAM (°) ↓ | 4 | 0.0463 | 0.0378 | n/a | n/a | descriptive only |
| sen2sr_mamba - sen2sr_lite | ERGAS ↓ | 4 | -0.0039 | 0.0138 | n/a | n/a | descriptive only |
| sen2sr_mamba - sen2sr_lite | NDVI MAE ↓ | 4 | 0.0005 | 0.0002 | n/a | n/a | descriptive only |
| sen2sr_mamba - sen2sr_lite | detail rel. error (1 = no detail added) | 4 | 0.0074 | 0.0051 | n/a | n/a | descriptive only |
| sen2sr_mamba - sen2sr_lite | MAE vs LR | 4 | 0.0006 | 0.0004 | n/a | n/a | descriptive only |
| sen2sr_lite - sen2sr_lite_no_constraint | PSNR (dB) ↑ | 4 | 0.4856 | 0.4754 | n/a | n/a | descriptive only |
| sen2sr_lite - sen2sr_lite_no_constraint | SSIM ↑ | 4 | 0.0047 | 0.0036 | n/a | n/a | descriptive only |
| sen2sr_lite - sen2sr_lite_no_constraint | RMSE ↓ | 4 | -0.0020 | -0.0018 | n/a | n/a | descriptive only |
| sen2sr_lite - sen2sr_lite_no_constraint | SAM (°) ↓ | 4 | -0.5008 | -0.4975 | n/a | n/a | descriptive only |
| sen2sr_lite - sen2sr_lite_no_constraint | ERGAS ↓ | 4 | -0.4203 | -0.3460 | n/a | n/a | descriptive only |
| sen2sr_lite - sen2sr_lite_no_constraint | NDVI MAE ↓ | 4 | -0.0107 | -0.0100 | n/a | n/a | descriptive only |
| sen2sr_lite - sen2sr_lite_no_constraint | detail rel. error (1 = no detail added) | 4 | -0.0079 | -0.0064 | n/a | n/a | descriptive only |
| sen2sr_lite - sen2sr_lite_no_constraint | MAE vs LR | 4 | -0.0035 | -0.0033 | n/a | n/a | descriptive only |

## Evidence class: `synthetic`

### Dataset `synthetic_smoke_test` (synthetic_smoke, role smoke_test, split test)

Records 2: **evaluated 2**, skipped 0, invalid 0, unreadable 0; 1 scene units; manifest digest `01bb0e2d18892c71…`.

**Reference accuracy (SR vs the independent HR reference)**

| system | n units | PSNR (dB) ↑ | SSIM ↑ | RMSE ↓ | MAE ↓ | SAM (°) ↓ | ERGAS ↓ |
|---|---|---|---|---|---|---|---|
| bicubic | 1 | 33.0508 | 0.9118 | 0.0223 | 0.0095 | 2.3922 | 4.4817 |
| sen2sr_lite | 1 | 34.6350 | 0.9326 | 0.0185 | 0.0082 | 2.1240 | 3.5379 |
| sen2sr_lite_no_constraint | 1 | 34.8374 | 0.9281 | 0.0181 | 0.0093 | 2.7577 | 3.4210 |
| sen2sr_mamba | 1 | 35.6613 | 0.9460 | 0.0165 | 0.0074 | 1.9100 | 3.2629 |
| tiny_cnn_seed0 | 1 | 36.1108 | 0.9495 | 0.0156 | 0.0066 | 1.7626 | 2.8145 |
| tiny_cnn_seed1 | 1 | 36.1831 | 0.9502 | 0.0155 | 0.0065 | 1.7568 | 2.7903 |
| tiny_cnn_seed2 | 1 | 36.1146 | 0.9494 | 0.0156 | 0.0066 | 1.7616 | 2.8258 |
| tiny_cnn_seed3 | 1 | 35.7514 | 0.9466 | 0.0163 | 0.0068 | 1.8568 | 2.8573 |
| tiny_cnn_seed4 | 1 | 36.2422 | 0.9501 | 0.0154 | 0.0065 | 1.7265 | 2.8109 |

**Derived indices and band ratios (vs the HR reference)**

| system | n units | NDVI MAE ↓ | NDVI bias | NDVI r ↑ | NDWI MAE ↓ | log(B08/B04) bias |
|---|---|---|---|---|---|---|
| bicubic | 1 | 0.0383 | 0.0065 | 0.9621 | 0.0362 | -0.0026 |
| sen2sr_lite | 1 | 0.0352 | 0.0071 | 0.9702 | 0.0328 | 0.0007 |
| sen2sr_lite_no_constraint | 1 | 0.0690 | -0.0236 | 0.9411 | 0.0569 | -0.0152 |
| sen2sr_mamba | 1 | 0.0308 | 0.0075 | 0.9771 | 0.0296 | 0.0007 |
| tiny_cnn_seed0 | 1 | 0.0312 | 0.0041 | 0.9456 | 0.0288 | 0.0049 |
| tiny_cnn_seed1 | 1 | 0.0314 | 0.0035 | 0.9504 | 0.0289 | 0.0048 |
| tiny_cnn_seed2 | 1 | 0.0325 | 0.0033 | 0.8633 | 0.0294 | 0.0044 |
| tiny_cnn_seed3 | 1 | 0.0337 | 0.0068 | 0.6276 | 0.0307 | 0.0081 |
| tiny_cnn_seed4 | 1 | 0.0317 | 0.0032 | 0.8375 | 0.0288 | 0.0030 |

**Spatial / detail correctness (vs the HR reference)**

| system | n units | detail rel. error (1 = no detail added) | detail correlation ↑ | detail energy ratio (1 = same amount) | edge correlation ↑ | shift (HR px) | seam / interior error |
|---|---|---|---|---|---|---|---|
| bicubic | 1 | 0.9211 | 0.4489 | 0.0517 | 0.5962 | 0.0000 | n/a |
| sen2sr_lite | 1 | 0.8082 | 0.6545 | 0.1563 | 0.8273 | 0.0000 | n/a |
| sen2sr_lite_no_constraint | 1 | 0.7954 | 0.6502 | 0.2038 | 0.8252 | 0.0000 | n/a |
| sen2sr_mamba | 1 | 0.7106 | 0.7871 | 0.1966 | 0.9170 | 0.0000 | n/a |
| tiny_cnn_seed0 | 1 | 0.7184 | 0.7160 | 0.3801 | 0.8700 | 0.0000 | n/a |
| tiny_cnn_seed1 | 1 | 0.7145 | 0.7195 | 0.3877 | 0.8732 | 0.0000 | n/a |
| tiny_cnn_seed2 | 1 | 0.7198 | 0.7151 | 0.3717 | 0.8667 | 0.0000 | n/a |
| tiny_cnn_seed3 | 1 | 0.7374 | 0.6977 | 0.3513 | 0.8539 | 0.0000 | n/a |
| tiny_cnn_seed4 | 1 | 0.7116 | 0.7219 | 0.3877 | 0.8731 | 0.0000 | n/a |

**Per-band reflectance error** (RMSE ↓ / bias = SR − reference; bands are the RGBN bands every system produces)

| system | B02 RMSE | B03 RMSE | B04 RMSE | B08 RMSE | B02 bias | B03 bias | B04 bias | B08 bias |
|---|---|---|---|---|---|---|---|---|
| bicubic | 0.0202 | 0.0184 | 0.0233 | 0.0263 | -0.0000 | 0.0000 | -0.0000 | -0.0000 |
| sen2sr_lite | 0.0159 | 0.0142 | 0.0182 | 0.0243 | -0.0000 | 0.0000 | -0.0000 | -0.0000 |
| sen2sr_lite_no_constraint | 0.0153 | 0.0137 | 0.0176 | 0.0241 | -0.0008 | -0.0010 | -0.0025 | -0.0027 |
| sen2sr_mamba | 0.0148 | 0.0132 | 0.0168 | 0.0202 | -0.0000 | 0.0000 | -0.0000 | -0.0000 |
| tiny_cnn_seed0 | 0.0123 | 0.0116 | 0.0144 | 0.0221 | -0.0001 | -0.0000 | -0.0000 | -0.0001 |
| tiny_cnn_seed1 | 0.0122 | 0.0115 | 0.0142 | 0.0219 | -0.0001 | -0.0000 | 0.0000 | -0.0001 |
| tiny_cnn_seed2 | 0.0123 | 0.0117 | 0.0144 | 0.0219 | -0.0001 | -0.0001 | -0.0001 | -0.0001 |
| tiny_cnn_seed3 | 0.0124 | 0.0118 | 0.0145 | 0.0237 | -0.0000 | 0.0000 | -0.0000 | -0.0001 |
| tiny_cnn_seed4 | 0.0123 | 0.0116 | 0.0144 | 0.0213 | -0.0001 | -0.0000 | -0.0000 | -0.0001 |

**What was added relative to bicubic, judged against the reference** (element-wise; τ = 0.005 reflectance; fractions of valid elements; the reference is a different sensor, so "unsupported" means "not confirmed by this reference", not "false")

| system | n units | supported synthesis | unsupported detail | omission | MSE skill vs bicubic |
|---|---|---|---|---|---|
| sen2sr_lite | 1 | 0.1097 | 0.0520 | 0.2017 | 0.3056 |
| sen2sr_lite_no_constraint | 1 | 0.1525 | 0.2447 | 0.1262 | 0.3372 |
| sen2sr_mamba | 1 | 0.1162 | 0.0296 | 0.2031 | 0.4518 |
| tiny_cnn_seed0 | 1 | 0.1240 | 0.0300 | 0.1910 | 0.5057 |
| tiny_cnn_seed1 | 1 | 0.1289 | 0.0315 | 0.1862 | 0.5138 |
| tiny_cnn_seed2 | 1 | 0.1272 | 0.0316 | 0.1877 | 0.5061 |
| tiny_cnn_seed3 | 1 | 0.1151 | 0.0272 | 0.2011 | 0.4630 |
| tiny_cnn_seed4 | 1 | 0.1238 | 0.0291 | 0.1916 | 0.5204 |

**Self-consistency (SR → area-average → compare with the LR)**. This is NOT accuracy against an HR reference: a system can average back to the LR perfectly and still be far from the reference.

| system | n units | MAE vs LR | RMSE vs LR | NDVI MAE vs LR |
|---|---|---|---|---|
| bicubic | 1 | 0.0014 | 0.0027 | 0.0063 |
| sen2sr_lite | 1 | 0.0013 | 0.0021 | 0.0060 |
| sen2sr_lite_no_constraint | 1 | 0.0046 | 0.0059 | 0.0492 |
| sen2sr_mamba | 1 | 0.0014 | 0.0023 | 0.0080 |
| tiny_cnn_seed0 | 1 | 0.0024 | 0.0062 | 0.0118 |
| tiny_cnn_seed1 | 1 | 0.0025 | 0.0065 | 0.0122 |
| tiny_cnn_seed2 | 1 | 0.0024 | 0.0061 | 0.0122 |
| tiny_cnn_seed3 | 1 | 0.0023 | 0.0060 | 0.0116 |
| tiny_cnn_seed4 | 1 | 0.0025 | 0.0065 | 0.0124 |

**Paired differences A − B over scene units** (sign convention: A minus B; for error metrics lower is smaller error; a difference is not a ranking)

| A − B | metric | n units | mean difference | median | 95% CI | Wilcoxon p | label |
|---|---|---|---|---|---|---|---|
| sen2sr_lite - bicubic | PSNR (dB) ↑ | 1 | 1.5841 | 1.5841 | n/a | n/a | descriptive only |
| sen2sr_lite - bicubic | SSIM ↑ | 1 | 0.0208 | 0.0208 | n/a | n/a | descriptive only |
| sen2sr_lite - bicubic | RMSE ↓ | 1 | -0.0037 | -0.0037 | n/a | n/a | descriptive only |
| sen2sr_lite - bicubic | SAM (°) ↓ | 1 | -0.2681 | -0.2681 | n/a | n/a | descriptive only |
| sen2sr_lite - bicubic | ERGAS ↓ | 1 | -0.9438 | -0.9438 | n/a | n/a | descriptive only |
| sen2sr_lite - bicubic | NDVI MAE ↓ | 1 | -0.0031 | -0.0031 | n/a | n/a | descriptive only |
| sen2sr_lite - bicubic | detail rel. error (1 = no detail added) | 1 | -0.1129 | -0.1129 | n/a | n/a | descriptive only |
| sen2sr_lite - bicubic | MAE vs LR | 1 | -0.0001 | -0.0001 | n/a | n/a | descriptive only |
| sen2sr_lite_no_constraint - bicubic | PSNR (dB) ↑ | 1 | 1.7866 | 1.7866 | n/a | n/a | descriptive only |
| sen2sr_lite_no_constraint - bicubic | SSIM ↑ | 1 | 0.0163 | 0.0163 | n/a | n/a | descriptive only |
| sen2sr_lite_no_constraint - bicubic | RMSE ↓ | 1 | -0.0041 | -0.0041 | n/a | n/a | descriptive only |
| sen2sr_lite_no_constraint - bicubic | SAM (°) ↓ | 1 | 0.3656 | 0.3656 | n/a | n/a | descriptive only |
| sen2sr_lite_no_constraint - bicubic | ERGAS ↓ | 1 | -1.0607 | -1.0607 | n/a | n/a | descriptive only |
| sen2sr_lite_no_constraint - bicubic | NDVI MAE ↓ | 1 | 0.0307 | 0.0307 | n/a | n/a | descriptive only |
| sen2sr_lite_no_constraint - bicubic | detail rel. error (1 = no detail added) | 1 | -0.1257 | -0.1257 | n/a | n/a | descriptive only |
| sen2sr_lite_no_constraint - bicubic | MAE vs LR | 1 | 0.0032 | 0.0032 | n/a | n/a | descriptive only |
| sen2sr_mamba - bicubic | PSNR (dB) ↑ | 1 | 2.6105 | 2.6105 | n/a | n/a | descriptive only |
| sen2sr_mamba - bicubic | SSIM ↑ | 1 | 0.0342 | 0.0342 | n/a | n/a | descriptive only |
| sen2sr_mamba - bicubic | RMSE ↓ | 1 | -0.0058 | -0.0058 | n/a | n/a | descriptive only |
| sen2sr_mamba - bicubic | SAM (°) ↓ | 1 | -0.4822 | -0.4822 | n/a | n/a | descriptive only |
| sen2sr_mamba - bicubic | ERGAS ↓ | 1 | -1.2188 | -1.2188 | n/a | n/a | descriptive only |
| sen2sr_mamba - bicubic | NDVI MAE ↓ | 1 | -0.0075 | -0.0075 | n/a | n/a | descriptive only |
| sen2sr_mamba - bicubic | detail rel. error (1 = no detail added) | 1 | -0.2105 | -0.2105 | n/a | n/a | descriptive only |
| sen2sr_mamba - bicubic | MAE vs LR | 1 | -0.0000 | -0.0000 | n/a | n/a | descriptive only |
| tiny_cnn_seed0 - bicubic | PSNR (dB) ↑ | 1 | 3.0600 | 3.0600 | n/a | n/a | descriptive only |
| tiny_cnn_seed0 - bicubic | SSIM ↑ | 1 | 0.0377 | 0.0377 | n/a | n/a | descriptive only |
| tiny_cnn_seed0 - bicubic | RMSE ↓ | 1 | -0.0066 | -0.0066 | n/a | n/a | descriptive only |
| tiny_cnn_seed0 - bicubic | SAM (°) ↓ | 1 | -0.6295 | -0.6295 | n/a | n/a | descriptive only |
| tiny_cnn_seed0 - bicubic | ERGAS ↓ | 1 | -1.6671 | -1.6671 | n/a | n/a | descriptive only |
| tiny_cnn_seed0 - bicubic | NDVI MAE ↓ | 1 | -0.0071 | -0.0071 | n/a | n/a | descriptive only |
| tiny_cnn_seed0 - bicubic | detail rel. error (1 = no detail added) | 1 | -0.2027 | -0.2027 | n/a | n/a | descriptive only |
| tiny_cnn_seed0 - bicubic | MAE vs LR | 1 | 0.0010 | 0.0010 | n/a | n/a | descriptive only |
| tiny_cnn_seed1 - bicubic | PSNR (dB) ↑ | 1 | 3.1323 | 3.1323 | n/a | n/a | descriptive only |
| tiny_cnn_seed1 - bicubic | SSIM ↑ | 1 | 0.0384 | 0.0384 | n/a | n/a | descriptive only |
| tiny_cnn_seed1 - bicubic | RMSE ↓ | 1 | -0.0067 | -0.0067 | n/a | n/a | descriptive only |
| tiny_cnn_seed1 - bicubic | SAM (°) ↓ | 1 | -0.6354 | -0.6354 | n/a | n/a | descriptive only |
| tiny_cnn_seed1 - bicubic | ERGAS ↓ | 1 | -1.6914 | -1.6914 | n/a | n/a | descriptive only |
| tiny_cnn_seed1 - bicubic | NDVI MAE ↓ | 1 | -0.0069 | -0.0069 | n/a | n/a | descriptive only |
| tiny_cnn_seed1 - bicubic | detail rel. error (1 = no detail added) | 1 | -0.2066 | -0.2066 | n/a | n/a | descriptive only |
| tiny_cnn_seed1 - bicubic | MAE vs LR | 1 | 0.0011 | 0.0011 | n/a | n/a | descriptive only |
| tiny_cnn_seed2 - bicubic | PSNR (dB) ↑ | 1 | 3.0638 | 3.0638 | n/a | n/a | descriptive only |
| tiny_cnn_seed2 - bicubic | SSIM ↑ | 1 | 0.0376 | 0.0376 | n/a | n/a | descriptive only |
| tiny_cnn_seed2 - bicubic | RMSE ↓ | 1 | -0.0066 | -0.0066 | n/a | n/a | descriptive only |
| tiny_cnn_seed2 - bicubic | SAM (°) ↓ | 1 | -0.6305 | -0.6305 | n/a | n/a | descriptive only |
| tiny_cnn_seed2 - bicubic | ERGAS ↓ | 1 | -1.6559 | -1.6559 | n/a | n/a | descriptive only |
| tiny_cnn_seed2 - bicubic | NDVI MAE ↓ | 1 | -0.0057 | -0.0057 | n/a | n/a | descriptive only |
| tiny_cnn_seed2 - bicubic | detail rel. error (1 = no detail added) | 1 | -0.2013 | -0.2013 | n/a | n/a | descriptive only |
| tiny_cnn_seed2 - bicubic | MAE vs LR | 1 | 0.0010 | 0.0010 | n/a | n/a | descriptive only |
| tiny_cnn_seed3 - bicubic | PSNR (dB) ↑ | 1 | 2.7005 | 2.7005 | n/a | n/a | descriptive only |
| tiny_cnn_seed3 - bicubic | SSIM ↑ | 1 | 0.0348 | 0.0348 | n/a | n/a | descriptive only |
| tiny_cnn_seed3 - bicubic | RMSE ↓ | 1 | -0.0059 | -0.0059 | n/a | n/a | descriptive only |
| tiny_cnn_seed3 - bicubic | SAM (°) ↓ | 1 | -0.5353 | -0.5353 | n/a | n/a | descriptive only |
| tiny_cnn_seed3 - bicubic | ERGAS ↓ | 1 | -1.6244 | -1.6244 | n/a | n/a | descriptive only |
| tiny_cnn_seed3 - bicubic | NDVI MAE ↓ | 1 | -0.0046 | -0.0046 | n/a | n/a | descriptive only |
| tiny_cnn_seed3 - bicubic | detail rel. error (1 = no detail added) | 1 | -0.1837 | -0.1837 | n/a | n/a | descriptive only |
| tiny_cnn_seed3 - bicubic | MAE vs LR | 1 | 0.0009 | 0.0009 | n/a | n/a | descriptive only |
| tiny_cnn_seed4 - bicubic | PSNR (dB) ↑ | 1 | 3.1913 | 3.1913 | n/a | n/a | descriptive only |
| tiny_cnn_seed4 - bicubic | SSIM ↑ | 1 | 0.0383 | 0.0383 | n/a | n/a | descriptive only |
| tiny_cnn_seed4 - bicubic | RMSE ↓ | 1 | -0.0068 | -0.0068 | n/a | n/a | descriptive only |
| tiny_cnn_seed4 - bicubic | SAM (°) ↓ | 1 | -0.6656 | -0.6656 | n/a | n/a | descriptive only |
| tiny_cnn_seed4 - bicubic | ERGAS ↓ | 1 | -1.6708 | -1.6708 | n/a | n/a | descriptive only |
| tiny_cnn_seed4 - bicubic | NDVI MAE ↓ | 1 | -0.0066 | -0.0066 | n/a | n/a | descriptive only |
| tiny_cnn_seed4 - bicubic | detail rel. error (1 = no detail added) | 1 | -0.2094 | -0.2094 | n/a | n/a | descriptive only |
| tiny_cnn_seed4 - bicubic | MAE vs LR | 1 | 0.0011 | 0.0011 | n/a | n/a | descriptive only |
| sen2sr_mamba - sen2sr_lite | PSNR (dB) ↑ | 1 | 1.0263 | 1.0263 | n/a | n/a | descriptive only |
| sen2sr_mamba - sen2sr_lite | SSIM ↑ | 1 | 0.0134 | 0.0134 | n/a | n/a | descriptive only |
| sen2sr_mamba - sen2sr_lite | RMSE ↓ | 1 | -0.0021 | -0.0021 | n/a | n/a | descriptive only |
| sen2sr_mamba - sen2sr_lite | SAM (°) ↓ | 1 | -0.2141 | -0.2141 | n/a | n/a | descriptive only |
| sen2sr_mamba - sen2sr_lite | ERGAS ↓ | 1 | -0.2750 | -0.2750 | n/a | n/a | descriptive only |
| sen2sr_mamba - sen2sr_lite | NDVI MAE ↓ | 1 | -0.0044 | -0.0044 | n/a | n/a | descriptive only |
| sen2sr_mamba - sen2sr_lite | detail rel. error (1 = no detail added) | 1 | -0.0976 | -0.0976 | n/a | n/a | descriptive only |
| sen2sr_mamba - sen2sr_lite | MAE vs LR | 1 | 0.0001 | 0.0001 | n/a | n/a | descriptive only |
| sen2sr_lite - sen2sr_lite_no_constraint | PSNR (dB) ↑ | 1 | -0.2025 | -0.2025 | n/a | n/a | descriptive only |
| sen2sr_lite - sen2sr_lite_no_constraint | SSIM ↑ | 1 | 0.0045 | 0.0045 | n/a | n/a | descriptive only |
| sen2sr_lite - sen2sr_lite_no_constraint | RMSE ↓ | 1 | 0.0004 | 0.0004 | n/a | n/a | descriptive only |
| sen2sr_lite - sen2sr_lite_no_constraint | SAM (°) ↓ | 1 | -0.6337 | -0.6337 | n/a | n/a | descriptive only |
| sen2sr_lite - sen2sr_lite_no_constraint | ERGAS ↓ | 1 | 0.1169 | 0.1169 | n/a | n/a | descriptive only |
| sen2sr_lite - sen2sr_lite_no_constraint | NDVI MAE ↓ | 1 | -0.0338 | -0.0338 | n/a | n/a | descriptive only |
| sen2sr_lite - sen2sr_lite_no_constraint | detail rel. error (1 = no detail added) | 1 | 0.0128 | 0.0128 | n/a | n/a | descriptive only |
| sen2sr_lite - sen2sr_lite_no_constraint | MAE vs LR | 1 | -0.0033 | -0.0033 | n/a | n/a | descriptive only |

## Multi-seed groups

Systems sharing a group are the same model trained with different seeds. The tables give the mean and standard deviation ACROSS seeds of each seed's unit-level mean.

`tiny_cnn_seeds`: 5 systems (tiny_cnn_seed0, tiny_cnn_seed1, tiny_cnn_seed2, tiny_cnn_seed3, tiny_cnn_seed4)
- sen2neon_random30: PSNR (dB) ↑ 32.8705 ± 0.0627; RMSE ↓ 0.0293 ± 0.0002; SAM (°) ↓ 3.5246 ± 0.0207; detail rel. error (1 = no detail added) 1.9782 ± 0.0771
- sen2neon_phase3_sample: PSNR (dB) ↑ 35.5297 ± 0.1131; RMSE ↓ 0.0169 ± 0.0002; SAM (°) ↓ 1.0768 ± 0.0223; detail rel. error (1 = no detail added) 1.4665 ± 0.0884
- opensr_spot: PSNR (dB) ↑ 32.8963 ± 0.0442; RMSE ↓ 0.0272 ± 0.0002; SAM (°) ↓ 2.4403 ± 0.0279; detail rel. error (1 = no detail added) 0.9721 ± 0.0066
- opensr_spain_crops: PSNR (dB) ↑ 32.3540 ± 0.0271; RMSE ↓ 0.0247 ± 0.0001; SAM (°) ↓ 2.5293 ± 0.0110; detail rel. error (1 = no detail added) 0.9982 ± 0.0042
- opensr_spain_urban: PSNR (dB) ↑ 28.8957 ± 0.0517; RMSE ↓ 0.0375 ± 0.0002; SAM (°) ↓ 4.2318 ± 0.0153; detail rel. error (1 = no detail added) 1.0190 ± 0.0067
- synthetic_smoke_test: PSNR (dB) ↑ 36.0804 ± 0.1917; RMSE ↓ 0.0157 ± 0.0004; SAM (°) ↓ 1.7729 ± 0.0492; detail rel. error (1 = no detail added) 0.7203 ± 0.0101

## How to read this
- **Valid-pixel rule**: an HR pixel is scored only if it is not all-band nodata, no evaluated band equals the HR nodata value, and all values are finite; the fractions are in `metrics.jsonl` (`quality`). Windowed metrics (SSIM, detail, gradients) additionally exclude a border of their own window width so no window touches nodata.
- **Reference accuracy** compares SR with an independent HR reference. For real cross-sensor pairs (SEN2NEON, OpenSR-Test) the reference is a different sensor with its own radiometry and registration error, so no number here is an absolute error against truth; for synthetic pairs the reference is the ground truth the LR was made from.
- **Detail metrics**: relative error 1.0 is exactly the score of adding no detail; below 1 the added detail matches the reference, above 1 it is worse than none. The energy ratio says how much detail was added, not whether it is right.
- **Registration**: the `shift` column is the displacement of the SR relative to the reference (cross-correlation, HR pixels). A non-zero value means the reference is not registered to the Sentinel-2 grid (close to the same for every system, since all start from the same LR grid), and every pixel-wise and detail score of that scene is then pessimistic for a correctly placed detail. `python -m frame.evaluate shift CONFIG --dataset NAME` measures how much a known displacement changes the scores.
- **Statistics**: units are scenes/acquisitions, not pixels; intervals are percentile bootstrap over units; the paired test is Wilcoxon signed-rank; below 5 units no interval and below 6 pairs no test is reported (descriptive only). No multiple-comparison correction.
- ↑ / ↓ mark which direction is smaller error or closer agreement, not a ranking of systems.

Files: `config.json` (as run), `metrics.jsonl` (one row per sample × system, plus skipped/invalid/unreadable rows), `aggregates.json` (all metrics, sample- and unit-level, paired comparisons), `summary.json` (provenance, counts, evaluation matrix).
