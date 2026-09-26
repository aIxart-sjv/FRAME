# Reliability validation of the TTA stability signal (Phase 6)

`frame/reliability/` tests whether FRAME's existing **relative model-stability signal** is informative about reconstruction error, using **only evidence whose reference is registered to the
prediction grid**. It does not call the stability "confidence" or "calibrated uncertainty": that is exactly what is being tested, and where the evidence does not support a claim the
outputs say so.

```
scene ─▶ SR model ─▶ TTA ensemble ─▶ stability map ─▶ reference eligibility gate ─▶ error targets ─▶ association ─▶ risk-coverage · detection · calibration ─▶ report
         (Lite / Mamba, (6 geometric   (per-pixel std)  (valid pixels, finiteness,  (aligned overlap,   (within-tile,   (unit-clustered bootstrap, held-out dev/test units)
          tile engine)   views)                          registration)              strict mask)        tile, scene)
```

```bash
python -m frame.reliability check experiments/uncertainty/configs/reliability_v1.json    # config, datasets, systems, and a PREVIEW of the eligibility gate; runs no model
python -m frame.reliability run   experiments/uncertainty/configs/reliability_v1.json    # writes experiments/uncertainty/runs/reliability_v1/
python -m frame.reliability scene CONFIG.json --system NAME --lr-geotiff FILE --window R0 R1 C0 C1 --output-dir DIR    # the rectangular multi-tile check
```

Exit codes: `0` success; `1` finished but some dataset/system has no eligible evidence; `2` refused or invalid. Outputs (`run`): `config.json`, `metrics.jsonl` (one row per tile and system;
excluded tiles carry a reason and **no numbers**), `summary.json` (provenance, per-tile registration, cost), `correlations.json`, `risk_coverage.json`, `detection.json`, `calibration.json`, `README.md`.
Per-tile pooled arrays (a seeded subsample of cells and pixels) are cached **outside** the repository (`cache_dir`); no rasters are written to the run directory.

## 1. What the existing TTA computes (audit of `frame/uncertainty/`)

`frame.uncertainty.run_tta_ensemble` runs the frozen model once per geometric view of the **same** LR scene, undoes each transform on the SR output, and accumulates the predictions
with Welford's algorithm. In the deployed pipeline (`frame/api/services/pipeline.py`) the model is wrapped in the tile engine (`frame.tiling.TiledModel`), so every view is a full
tiled pass over the (rotated) scene with **its own tile grid**. Exactly:

| Step | Definition |
|---|---|
| Views | `identity`, `hflip`, `vflip`, `rot90`, `rot180`, `rot270` (six of the eight elements of the dihedral group; the two diagonal transposes are omitted). Pure functions on `(C, H, W)` tensors; `rot90` / `rot270` swap H and W. |
| De-augmentation | each member's SR is mapped back by the exact inverse (flips are self-inverse; `rot90` ↔ `rot270`; `rot180` self-inverse), so all members are pixel-aligned in the canonical orientation. |
| Product | the **ensemble mean** (`sr_mean.tif`), not the single pass. |
| Spread | per band and pixel, the **population** standard deviation over the members (divide by N = 6). |
| Stability map | the band mean of the four spreads (`overall_std`, the fifth band of `uncertainty.tif`). This is the quantity validated here. |
| Summary | `scalar_summary` = mean of the stability map. |
| Export | `uncertainty.tif` on the SR grid (`derive_output_metadata`), four `*_std` bands and `overall_std`. |

What it measures: **how much the model's own prediction moves when the same observation is presented in a different orientation** (and, through the tile engine, cut into different tiles).
It is a property of the model; it uses no reference. It cannot see a bias that is the same in every view, so it is a *floor* on the error, not an estimate of it.

**Verified for rectangular scenes** (`frame/tests/test_reliability_tta_audit.py`, 15 tests): every transform round-trips exactly on non-square tensors (5×9, 9×5, 1×7, 130×257) and
`rot90` / `rot270` swap the sides; the transforms equal `numpy.rot90` / flips element for element; the members are exactly "model on the transformed input, then inverse"; a perfectly
D4-equivariant model (pixel replication) has zero spread on square and rectangular scenes, also through the tile engine on a 200×300 scene (6 tiles, rotated views 300×200), so the
de-transformation preserves orientation; the spread is the population (not sample) standard deviation; a position-dependent model gives a spread where the closed form says. **No defect was found in the TTA
code; it was not modified.** A one-member ensemble has zero spread by construction: it is refused as evidence (`too_few_ensemble_members`).

## 2. What is compared with what

**Error targets** (the independent HR reference, on the strict valid mask only; the evaluated product is the ensemble mean, the single pass is recorded separately):

| Level | Target | Why |
|---|---|---|
| pixel (2.5 m) | band-mean absolute reflectance error (E1); per-pixel spectral angle (E2) | the finest resolution; most exposed to registration error |
| cell (10 m = 4×4 HR px, 40 m = 16×16) | mean of the pixel errors and of the stability over the cell's valid pixels (≥ 75% valid) | one Sentinel-2 pixel; averages out sub-pixel noise; a residual misregistration ≤ 0.5 HR px matters little |
| tile | RMSE, MAE, SAM, ERGAS (the Phase 5 definitions) | the unit at which reference registration error is least harmful |
| scene unit | the tile targets averaged over the tiles of one NEON acquisition / source orthophoto | tiles of a scene are not independent |
| element categories | share of the four bands labelled supported synthesis / unsupported detail / omission (Phase 5 element analysis, τ = 0.005) | the hallucination / omission relationship |

**Trivial predictors** reported next to the stability: image **texture** (Sobel gradient of the band-mean bicubic input) and the **added detail** (|prediction − bicubic|). Neither needs an
ensemble; both correlate with error for reasons unrelated to ensemble disagreement, so a **partial Spearman correlation** controlling the stability for both is reported: a stability that only re-encodes them adds nothing.

## 3. The reference eligibility gate

Phase 5 showed that real references are not registered to the Sentinel-2 grid. That is worse than noise for this question: registration error grows with local contrast, and so does
model instability, so the two correlate through texture alone. (`test_misregistration_alone_manufactures_an_association_with_a_texture_like_stability`: a **perfect** prediction against a reference displaced by
2 px produces a positive stability-error correlation from nothing.) Only registered evidence may enter, and every exclusion carries a machine-readable reason.

Evidence is judged in this order; the first failure is recorded (`status = excluded_from_uncertainty_error_analysis`, no metric fields):

| Reason | Meaning |
|---|---|
| `reference_missing`, `reference_geometry_invalid` | no reference / its grid does not match the prediction |
| `reference_not_finite` | non-finite reference pixels beyond 0 of the tile |
| `insufficient_valid_pixels` | strict valid fraction < 50% (an all-nodata tile is this, never "zero error"); checked again after the alignment crop |
| `reference_alignment_invalid` / `reference_alignment_uncertain` | the registration gate below |
| `prediction_not_finite`, `too_few_ensemble_members`, `model_failure`, `tta_member_failure` | after inference: a non-finite prediction or spread on a valid pixel (the tile engine's own NaN/Inf guard counts), a one-member ensemble, the model failing on its first view, or on a later view (the member is named) |

**Registration** (`frame/reliability/alignment.py`; every setting is in `reliability_v1.json` and was declared before any result):

1. The displacement of the **bicubic baseline** against the reference is estimated (system-neutral: no model chooses its own alignment, both models are judged on identical evidence, and no model's error can influence which tiles are analysed). The
   Phase 5 estimator (`frame.evaluate.shift.estimate_alignment`, validated in Phase 5 against a brute-force search) seeds a **sub-pixel refinement** of the normalised correlation peak over the valid overlap
   (`estimate_displacement`, a parabola through the integer neighbours). *Why the refinement:* with nodata the Phase 5 estimator falls back to a masked correlation with whole-pixel resolution, so a true half-pixel displacement read as ±1 px
   and a one-pixel correction moved it to −1 px; a 0.5 px tolerance cannot be judged that way. The refinement recovers known fractional shifts to 0.15 px with and without nodata (tests) and agrees with a brute-force whole-pixel search within 1 px
   on **87 of 87** real tiles (SEN2NEON random-30, OpenSR `spot`, `spain_crops`, `spain_urban`); on the 66 tiles without nodata it differs from the Phase 5 estimate by a median 0.13 px (max 0.51).
2. Within **0.5 HR px** (an eighth of a Sentinel-2 pixel; the Phase 5 sweep first shows a measurable loss at 1 HR px) the pair is accepted as it is. Otherwise a single **whole-pixel translation of at most 4 HR px (one LR pixel)** may be applied by **cropping both grids to their
   overlap** (`displace_pair`): no resampling, no interpolation, no warp. The correction and the kept window are recorded in every row.
3. After the correction the residual displacement is estimated again and in each of the four quadrants (their spread). A global translation is the only misregistration model accepted: quadrants that disagree mean a non-uniform misregistration that a translation cannot repair.
4. **`pixel_level_eligible`**: residual ≤ 0.5 HR px and quadrant spread ≤ 1.0 HR px. **`not_eligible`**: the correction is out of range, or the residual / spread exceeds 2× those tolerances. **`uncertain`**: in between, or an estimate that could not be computed.
   Only `pixel_level_eligible` evidence is analysed. `uncertain` is a separate class, not a soft pass.

The pixel-level correlation of the eligible tiles is also recomputed with the reference displaced by 1, 2 and 4 HR px (`pixel_level.by_reference_displacement`), which shows why the gate exists.

## 4. Statistics (the Phase 5 discipline, extended)

* Pixels and cells of a tile are spatially autocorrelated, and tiles of a scene are correlated. **Within-tile** correlations are computed per tile and summarised over **scene units** (mean of the unit means, share of tiles with a positive correlation); every interval is a percentile bootstrap that resamples **whole units** (2,000 replicates; pooled cell analyses at most 500; seed 0). Fewer than **5 scene units → `descriptive_only`, no interval**; a statistic whose value is itself a correlation or a curve **across** units or tiles (tile/scene reliability, tile-level risk-coverage, the detail relationship across tiles) needs **10 units** (its bootstrap is degenerate below that); a statistic undefined in more than 10% of replicates withdraws its interval.
* Pooled cell analyses use a seeded uniform subsample of 8,192 cells per tile; ties in a ranking are broken by a seeded random permutation, never by input order. Non-finite values are dropped **and counted**; an undefined statistic (constant stability map, zero error) is `not_computable` with a reason, never 0.
* **Risk-coverage**: rank by instability, remove the most unstable, report the error that remains next to random removal and the oracle (ranked by the true error); *selective efficiency* = (AURC_random − AURC) / (AURC_random − AURC_oracle) (1 = oracle, 0 = random). The baselines are ranked the same way. This describes an ordering; it is **not** calibrated selective prediction.
* **High-error detection** at 10 m cells: scene units are split by a **seeded development/test split** (50/50; needs ≥ 12 units, else descriptive only). "High error" = above the 90% quantile of **development** cells' error; that threshold is only *applied* to test units. AUROC, AUPRC (chance = prevalence) and precision/lift of flagging the top 10% / 20%; intervals over test units; the trivial baselines scored the same way.
* **Calibration** is three separate questions: (1) is the raw spread an interval? (coverage of `prediction ± k·spread` against the Gaussian nominal level, and the scale ratio error/spread); (2) could a monotone (isotonic) map from stability to expected error, **fitted on development units, be calibrated on test units**? (held-out reliability table, calibration line slope/intercept with intervals over test units, ECE, skill over the development-mean constant); (3) verdict. No conformal method is added: there is no calibration set that would justify it.
  The verdict is `uncalibrated_stability_evidence` whenever the raw spread is not an interval; a recalibration is `supported_for_this_dataset_and_error_scale` only if the held-out slope interval contains 1, the intercept interval contains 0 and the skill is positive.
* No multiple-comparison correction is applied; a table of many results is exploratory. No result is called significant, and no causal claim is made.
* Datasets are analysed separately (SEN2NEON, OpenSR `spot`, `spain_crops`, `spain_urban`); nothing is pooled across them.

## 5. The multi-tile / rectangular-scene check

`python -m frame.reliability scene` (and `frame.reliability.scene_check`) runs the deployed pipeline on a non-square, non-multiple-of-128 GeoTIFF window and checks, with the numbers recorded beside each verdict: **coverage** (spread and mean have the SR shape, all finite); **orientation** (every de-transformed member is much closer to the identity view
than the same member rotated by 180° would be); **seams** (mean spread within ±8 px of the tile-seam lines over the mean spread elsewhere, for the canonical grid and for the 90°-rotated view's grid mapped back; tolerance 1.10); **georeferencing** (the spread GeoTIFF is written on the SR grid and read back: CRS, origin, bounds, pixel size / 4, identical values).

## 6. Results (measured, descriptive, non-causal; nothing is ranked)

Record: `experiments/uncertainty/runs/reliability_v1/` (`config.json`, `metrics.jsonl`, `summary.json`, `correlations.json`, `risk_coverage.json`, `detection.json`, `calibration.json`, `README.md` with every table). SEN2SR-Lite (published weights, hard constraint) and SEN2SR-Mamba (isolated worker, RGBN,
hard constraint), the deployed six-view TTA over the tile engine, RTX 3050 laptop GPU, git `eb37695` with a **dirty working tree** (commit before quoting). Datasets are analysed separately. "desc." = fewer than 5 scene units (or, for statistics *across* units, fewer than 10), so no interval is given.

### 6.1 Evidence: what entered and what was excluded

Every tile was judged by the reference gate **before any model ran**; Lite and Mamba were analysed on exactly the same tiles. **No tile was excluded after inference** (no model or member failure, no non-finite prediction).

| dataset | records | eligible | uncertain | not eligible | eligible / all scene units | median raw displacement, all tiles (HR px) | median, eligible tiles | eligible tiles corrected by a translation |
|---|---|---|---|---|---|---|---|---|
| SEN2NEON (seeded random 30) | 30 | 12 | 15 | 3 | 11 / 28 | 0.89 | 0.89 | 8 |
| OpenSR `spot` | 9 | 6 | 3 | 0 | 6 / 9 | 0.52 | 0.42 | 2 |
| OpenSR `spain_crops` | 28 | 21 | 7 | 0 | 5 / 5 | 1.73 | 1.97 | 21 |
| OpenSR `spain_urban` | 20 | 13 | 7 | 0 | 4 / 4 | 1.86 | 1.86 | 12 |

Exclusion reasons: every excluded tile is `reference_alignment_uncertain` (7 + 7 + 3 + 15) except three SEN2NEON tiles, `reference_alignment_invalid` (2023_TEAK: a correction of (−3, −6) HR px, beyond the 4 px limit; 2023_RMNP_5: the four quadrants disagree by 2.1 HR px; 2024_HEAL_6: they disagree by far more than that, an unreliable quadrant estimate). No reference was missing, none had non-finite pixels, none fell below the valid-pixel threshold
(the lowest strict valid fraction among the SEN2NEON tiles is 0.517; the eligible tiles' largest residual is 0.48 HR px and largest quadrant spread 0.94). **Only 12 of the 30 SEN2NEON tiles (11 of 28 scene units) are usable for pixel-level analysis**: the rest are registered with too little certainty, which is the finding Phase 5 predicted, now measured with sub-pixel resolution. The eligible OpenSR-Test tiles need
a translation correction (median raw displacement about 2 HR px for `spain_*`); the correction is a recorded integer crop and the residual after it is within 0.5 HR px. `spain_crops` has 5 and `spain_urban` 4 source orthophotos, `spot` 6 eligible scenes.

### 6.2 Is higher instability accompanied by higher error? (within tiles)

Spearman correlation between the stability and the absolute error, **computed within each tile** and summarised over scene units (mean of unit means, 95% bootstrap interval over units), next to the same correlation of the error with the two trivial predictors, and the partial correlation of the stability controlling for both:

| dataset | model | scale | stability vs abs error | texture vs abs error | added detail vs abs error | partial (stability | both) | share of tiles > 0 |
|---|---|---|---|---|---|---|---|
| SEN2NEON (seeded random 30) | Lite | pixel 2.5 m | 0.094 [0.046, 0.143] | 0.050 [0.010, 0.086] | 0.091 [0.044, 0.139] | 0.063 [0.031, 0.098] | 0.83 |
|  |  | cell 10 m | 0.161 [0.089, 0.233] | 0.132 [0.069, 0.188] | 0.121 [0.060, 0.176] | 0.093 [0.047, 0.140] | 0.83 |
|  |  | cell 40 m | 0.260 [0.147, 0.381] | 0.269 [0.159, 0.382] | 0.213 [0.117, 0.316] | 0.110 [0.052, 0.169] | 0.92 |
|  | Mamba | pixel 2.5 m | 0.078 [0.030, 0.124] | 0.051 [0.007, 0.090] | 0.091 [0.037, 0.146] | 0.037 [-0.002, 0.070] | 0.83 |
|  |  | cell 10 m | 0.133 [0.067, 0.199] | 0.126 [0.058, 0.188] | 0.128 [0.058, 0.196] | 0.052 [0.003, 0.096] | 0.83 |
|  |  | cell 40 m | 0.213 [0.105, 0.326] | 0.256 [0.139, 0.367] | 0.218 [0.110, 0.327] | 0.041 [-0.019, 0.101] | 0.83 |
| OpenSR `spot` | Lite | pixel 2.5 m | 0.167 [0.106, 0.235] | 0.129 [0.073, 0.194] | 0.144 [0.092, 0.197] | 0.087 [0.063, 0.116] | 1.00 |
|  |  | cell 10 m | 0.341 [0.256, 0.429] | 0.263 [0.171, 0.364] | 0.318 [0.226, 0.411] | 0.141 [0.108, 0.177] | 1.00 |
|  |  | cell 40 m | 0.510 [0.397, 0.619] | 0.422 [0.310, 0.542] | 0.507 [0.383, 0.625] | 0.131 [0.053, 0.195] | 1.00 |
|  | Mamba | pixel 2.5 m | 0.198 [0.134, 0.267] | 0.133 [0.072, 0.203] | 0.150 [0.095, 0.205] | 0.121 [0.091, 0.151] | 1.00 |
|  |  | cell 10 m | 0.380 [0.289, 0.471] | 0.263 [0.165, 0.374] | 0.329 [0.239, 0.418] | 0.205 [0.166, 0.243] | 1.00 |
|  |  | cell 40 m | 0.527 [0.408, 0.632] | 0.418 [0.300, 0.545] | 0.509 [0.374, 0.633] | 0.192 [0.165, 0.221] | 1.00 |
| OpenSR `spain_crops` | Lite | pixel 2.5 m | 0.263 [0.206, 0.319] | 0.224 [0.179, 0.273] | 0.235 [0.192, 0.279] | 0.113 [0.086, 0.141] | 1.00 |
|  |  | cell 10 m | 0.439 [0.357, 0.517] | 0.402 [0.336, 0.470] | 0.449 [0.387, 0.512] | 0.109 [0.070, 0.153] | 1.00 |
|  |  | cell 40 m | 0.594 [0.499, 0.681] | 0.570 [0.492, 0.649] | 0.621 [0.545, 0.690] | 0.059 [-0.005, 0.139] | 1.00 |
|  | Mamba | pixel 2.5 m | 0.289 [0.230, 0.349] | 0.224 [0.182, 0.272] | 0.256 [0.213, 0.299] | 0.136 [0.101, 0.173] | 1.00 |
|  |  | cell 10 m | 0.471 [0.386, 0.552] | 0.401 [0.336, 0.464] | 0.464 [0.393, 0.528] | 0.156 [0.106, 0.206] | 1.00 |
|  |  | cell 40 m | 0.598 [0.506, 0.682] | 0.567 [0.484, 0.639] | 0.615 [0.537, 0.681] | 0.100 [0.022, 0.182] | 1.00 |
| OpenSR `spain_urban` | Lite | pixel 2.5 m | 0.333 (desc.) | 0.271 (desc.) | 0.295 (desc.) | 0.152 (desc.) | 1.00 |
|  |  | cell 10 m | 0.540 (desc.) | 0.475 (desc.) | 0.541 (desc.) | 0.169 (desc.) | 1.00 |
|  |  | cell 40 m | 0.713 (desc.) | 0.667 (desc.) | 0.731 (desc.) | 0.131 (desc.) | 1.00 |
|  | Mamba | pixel 2.5 m | 0.377 (desc.) | 0.276 (desc.) | 0.322 (desc.) | 0.200 (desc.) | 1.00 |
|  |  | cell 10 m | 0.589 (desc.) | 0.479 (desc.) | 0.558 (desc.) | 0.261 (desc.) | 1.00 |
|  |  | cell 40 m | 0.725 (desc.) | 0.667 (desc.) | 0.732 (desc.) | 0.223 (desc.) | 1.00 |

* The association is **positive in 83-100% of tiles** and grows with the scale (pixel → 10 m → 40 m), but it is **weak on SEN2NEON** (0.09 / 0.16 / 0.26 for Lite; 0.08 / 0.13 / 0.21 for Mamba) and moderate on OpenSR-Test (10 m: 0.34-0.59).
* **It is about as large as the trivial predictors' association**: image texture alone reaches 0.05-0.67 and the amount of detail the model added 0.09-0.73 on the same tiles.
* **What the stability adds beyond them is small but mostly positive**: partial correlation (stability given both) at 10 m 0.093 [0.047, 0.140] (Lite) and 0.052 [0.003, 0.096] (Mamba) on SEN2NEON, 0.14-0.21 on `spot`, 0.11-0.16 on `spain_crops`; at 40 m it is compatible with 0 for Mamba on SEN2NEON and for Lite on `spain_crops`.
* The pooled 10 m / 40 m cell correlations (all cells of all tiles, interval over units; `README.md`) are lower on SEN2NEON (0.106 [-0.041, 0.275] at 10 m for Lite; the partial is 0.011 [-0.058, 0.123]) because the error level differs between tiles by more than the spread does.

**Why registered evidence is required.** The same eligible tiles, correlated with the reference displaced by known whole pixels (mean over units, pixel level):

| dataset | model | 0 HR px | 1 | 2 | 4 |
|---|---|---|---|---|---|
| SEN2NEON (seeded random 30) | Lite | 0.094 | 0.108 | 0.134 | 0.172 |
|  | Mamba | 0.078 | 0.092 | 0.119 | 0.160 |
| OpenSR `spot` | Lite | 0.167 | 0.180 | 0.208 | 0.255 |
|  | Mamba | 0.198 | 0.211 | 0.242 | 0.285 |
| OpenSR `spain_crops` | Lite | 0.263 | 0.274 | 0.300 | 0.349 |
|  | Mamba | 0.289 | 0.303 | 0.332 | 0.374 |
| OpenSR `spain_urban` | Lite | 0.333 | 0.346 | 0.372 | 0.418 |
|  | Mamba | 0.377 | 0.393 | 0.418 | 0.447 |

**The association rises as the reference is displaced** (SEN2NEON Lite 0.094 → 0.172 at 4 HR px; every dataset and model). Registration error grows with local contrast, exactly as instability does, so a misregistered reference would have *manufactured* a stronger association: the numbers above would have looked better on excluded evidence. The eligible tiles still carry a residual
misregistration of up to 0.5 HR px, which by the same slope inflates the association by roughly 0.01-0.02; that is not removed.

### 6.3 Tile and scene reliability (across tiles)

Tile mean stability against tile RMSE and SAM, Spearman across tiles; the interval resamples scene units and is given only with **10 or more units** (a rank correlation across fewer units bootstraps to a degenerate interval: six `spot` tiles ranked identically by every variable gave `1.000 [1.000, 1.000]` in a first analysis, which is why the rule was tightened, §8):

| dataset | model | target | tiles | stability vs tile error (Spearman) | texture | added detail | partial |
|---|---|---|---|---|---|---|---|
| SEN2NEON (seeded random 30) | Lite | rmse | 12 | 0.434 [-0.077, 0.831] | 0.608 | 0.483 | -0.082 |
|  |  | sam_degrees | 12 | 0.441 [-0.022, 0.942] | 0.175 | -0.007 | 0.491 |
|  | Mamba | rmse | 12 | 0.343 [-0.243, 0.846] | 0.545 | 0.434 | -0.037 |
|  |  | sam_degrees | 12 | 0.084 [-0.531, 0.641] | 0.119 | 0.049 | 0.162 |
| OpenSR `spot` | Lite | rmse | 6 | 1.000 (desc.) | 1.000 | 1.000 | 1.000 |
|  |  | sam_degrees | 6 | 0.943 (desc.) | 0.943 | 0.943 | -0.094 |
|  | Mamba | rmse | 6 | 1.000 (desc.) | 1.000 | 1.000 | 1.000 |
|  |  | sam_degrees | 6 | 0.943 (desc.) | 0.943 | 0.943 | -0.094 |
| OpenSR `spain_crops` | Lite | rmse | 21 | 0.757 (desc.) | 0.732 | 0.764 | 0.064 |
|  |  | sam_degrees | 21 | 0.399 (desc.) | 0.219 | 0.518 | 0.242 |
|  | Mamba | rmse | 21 | 0.536 (desc.) | 0.705 | 0.765 | -0.622 |
|  |  | sam_degrees | 21 | 0.300 (desc.) | 0.217 | 0.573 | -0.153 |
| OpenSR `spain_urban` | Lite | rmse | 13 | 0.852 (desc.) | 0.703 | 0.830 | 0.382 |
|  |  | sam_degrees | 13 | 0.363 (desc.) | 0.093 | 0.522 | -0.096 |
|  | Mamba | rmse | 13 | 0.775 (desc.) | 0.665 | 0.813 | 0.116 |
|  |  | sam_degrees | 13 | 0.346 (desc.) | 0.044 | 0.555 | 0.230 |

On SEN2NEON (11 units) the tile-level association is **positive but not distinguishable from zero** for RMSE (Lite 0.434 [-0.077, 0.831]; Mamba 0.343 [-0.243, 0.846]) and SAM; the **texture of the tile predicts the tile RMSE as well or better (0.61 / 0.55)** and the partial correlation is about 0 (-0.08 / -0.04 for RMSE). On the OpenSR-Test subsets the tile-level associations are large (0.55-1.0) but are descriptive (6, 5 and 4 units) and equalled by texture and added detail:
with a handful of scenes, brighter and more textured scenes have both more instability and more error, and the data cannot separate that from an effect of the instability itself. Scene-level values (tiles of a unit averaged) are in `correlations.json`.

### 6.4 Risk-coverage

Remove the most unstable items first; the risk is the error of what remains. *Selective efficiency*: 1 = as good as removing by the true error (oracle), 0 = as good as removing at random. The baselines are ranked the same way. This describes an ordering and is **not** calibrated selective prediction.

| dataset | model | risk | items | selective efficiency (stability) | texture order | added-detail order | risk reduction at 80% coverage |
|---|---|---|---|---|---|---|---|
| SEN2NEON (seeded random 30) | Lite | tile RMSE | 12 | 0.425 [-0.333, 0.929] | 0.495 | 0.084 | 0.157 [-0.091, 0.299] |
|  |  | 10 m cell abs error | 98304 | 0.144 [0.056, 0.313] | 0.123 | 0.110 | 0.067 [0.040, 0.105] |
|  | Mamba | tile RMSE | 12 | 0.340 [-0.283, 0.986] | 0.491 | 0.219 | -0.065 [-0.103, 0.253] |
|  |  | 10 m cell abs error | 98304 | 0.091 [-0.018, 0.264] | 0.117 | 0.105 | 0.044 [0.008, 0.089] |
| OpenSR `spot` | Lite | tile RMSE | 6 | 1.000 (desc.) | 1.000 | 1.000 | 0.192 (desc.) |
|  |  | 10 m cell abs error | 49152 | 0.772 [0.460, 0.835] | 0.704 | 0.741 | 0.272 [0.110, 0.341] |
|  | Mamba | tile RMSE | 6 | 1.000 (desc.) | 1.000 | 1.000 | 0.198 (desc.) |
|  |  | 10 m cell abs error | 49152 | 0.767 [0.493, 0.831] | 0.708 | 0.747 | 0.282 [0.118, 0.356] |
| OpenSR `spain_crops` | Lite | tile RMSE | 21 | 0.670 (desc.) | 0.622 | 0.649 | 0.069 (desc.) |
|  |  | 10 m cell abs error | 172032 | 0.566 [0.499, 0.622] | 0.525 | 0.543 | 0.145 [0.126, 0.155] |
|  | Mamba | tile RMSE | 21 | 0.546 (desc.) | 0.612 | 0.665 | 0.047 (desc.) |
|  |  | 10 m cell abs error | 172032 | 0.584 [0.507, 0.647] | 0.520 | 0.554 | 0.157 [0.130, 0.167] |
| OpenSR `spain_urban` | Lite | tile RMSE | 13 | 0.887 (desc.) | 0.606 | 0.845 | 0.096 (desc.) |
|  |  | 10 m cell abs error | 106496 | 0.644 (desc.) | 0.552 | 0.600 | 0.166 (desc.) |
|  | Mamba | tile RMSE | 13 | 0.654 (desc.) | 0.557 | 0.666 | 0.063 (desc.) |
|  |  | 10 m cell abs error | 106496 | 0.668 (desc.) | 0.549 | 0.613 | 0.175 (desc.) |

At 10 m cells, removing the 20% most unstable cells lowers the mean error by **4-7% on SEN2NEON** (Lite 0.067 [0.040, 0.105], Mamba 0.044 [0.008, 0.089]) and 15-28% on OpenSR-Test; ranking by texture or by the amount of added detail reaches a similar reduction (efficiency on SEN2NEON: stability 0.144 [0.056, 0.313] against texture 0.123 and added detail 0.110 for Lite). Tile-level curves (12, 6, 21 and 13 tiles) are descriptive
or have intervals that include 0 and cannot separate the stability from the baselines.

### 6.5 High-error detection

"High error" = a 10 m cell above the 90% quantile of error; scores are the ensemble spread and the two trivial baselines. **The pre-declared protocol takes that threshold from development scene units only and applies it to held-out test units, which needs at least 12 units; no dataset has that many eligible units (SEN2NEON has 11), so this is descriptive only**: the threshold is the quantile of the same cells, there is no held-out test, and no interval is reported. The threshold was **not** lowered to reach a result.

| dataset | model | status | prevalence | AUROC (stability) | AUPRC (stability) | texture AUROC / AUPRC | added detail AUROC / AUPRC | lift, top 10% flagged |
|---|---|---|---|---|---|---|---|---|
| SEN2NEON (seeded random 30) | Lite | descriptive_only | 0.100 | 0.482 | 0.153 | 0.472 / 0.156 | 0.454 / 0.137 | 1.89 |
|  | Mamba | descriptive_only | 0.100 | 0.456 | 0.126 | 0.468 / 0.156 | 0.444 / 0.147 | 1.54 |
| OpenSR `spot` | Lite | descriptive_only | 0.100 | 0.892 | 0.506 | 0.862 / 0.471 | 0.880 / 0.496 | 4.85 |
|  | Mamba | descriptive_only | 0.100 | 0.893 | 0.512 | 0.868 / 0.477 | 0.885 / 0.507 | 4.95 |
| OpenSR `spain_crops` | Lite | descriptive_only | 0.100 | 0.819 | 0.398 | 0.791 / 0.343 | 0.793 / 0.388 | 4.26 |
|  | Mamba | descriptive_only | 0.100 | 0.826 | 0.464 | 0.787 / 0.337 | 0.792 / 0.409 | 4.76 |
| OpenSR `spain_urban` | Lite | descriptive_only | 0.100 | 0.837 | 0.430 | 0.794 / 0.359 | 0.813 / 0.392 | 4.35 |
|  | Mamba | descriptive_only | 0.100 | 0.841 | 0.450 | 0.791 / 0.352 | 0.813 / 0.389 | 4.58 |

On SEN2NEON the spread does **not** rank high-error cells above chance (AUROC 0.46-0.48, AUPRC 0.13-0.15 against a prevalence of 0.10), and neither do the baselines (0.44-0.47). On OpenSR-Test the AUROC is 0.82-0.89 and flagging the top 10% gives a lift of 4-5, but texture alone reaches 0.79-0.87 and added detail 0.79-0.89: the spread is at best marginally ahead of the better trivial predictor
(+0.01 to +0.03 AUROC). These pooled figures mix between-tile and within-tile differences, and they are not test-set figures.

### 6.6 Calibration

| dataset | model | median error / median spread | coverage k=1 (nominal 0.683) | coverage k=2 (nominal 0.954) | recalibration | verdict |
|---|---|---|---|---|---|---|
| SEN2NEON (seeded random 30) | Lite | 18.2 | 0.038 | 0.075 | not_assessable | uncalibrated_stability_evidence |
|  | Mamba | 31.9 | 0.041 | 0.080 | not_assessable | uncalibrated_stability_evidence |
| OpenSR `spot` | Lite | 18.2 | 0.036 | 0.071 | not_assessable | uncalibrated_stability_evidence |
|  | Mamba | 15.7 | 0.034 | 0.069 | not_assessable | uncalibrated_stability_evidence |
| OpenSR `spain_crops` | Lite | 23.2 | 0.028 | 0.057 | not_assessable | uncalibrated_stability_evidence |
|  | Mamba | 26.4 | 0.024 | 0.048 | not_assessable | uncalibrated_stability_evidence |
| OpenSR `spain_urban` | Lite | 19.6 | 0.031 | 0.061 | not_assessable | uncalibrated_stability_evidence |
|  | Mamba | 19.2 | 0.029 | 0.057 | not_assessable | uncalibrated_stability_evidence |

**Uncalibrated stability evidence.** The median absolute error is **16-32 times the median ensemble spread**, and `prediction ± 2 × spread` contains the reference for only **5-8%** of the elements (nominal 95%). Ensemble members share the model's bias, so their disagreement is a floor under the error, not a measure of it. Whether a monotone recalibration fitted on development units holds on test units could **not be assessed**: it needs held-out units and there are not enough (11 or fewer eligible units per dataset). Nothing here supports calling
the stability a confidence, a probability of error or a calibrated uncertainty, and no conformal method was added.

### 6.7 Relationship with unsupported detail and omission (Phase 5 element categories)

| dataset | model | 10 m: stability vs unsupported detail | vs omission | vs supported synthesis | tile level (unsupported / omission / supported) |
|---|---|---|---|---|---|
| SEN2NEON (seeded random 30) | Lite | 0.503 [0.426, 0.577] | -0.150 [-0.300, -0.007] | 0.367 [0.298, 0.443] | 0.72 / 0.45 / 0.42 |
|  | Mamba | 0.568 [0.495, 0.637] | -0.213 [-0.354, -0.087] | 0.463 [0.387, 0.543] | 0.97 / 0.01 / 0.64 |
| OpenSR `spot` | Lite | 0.491 [0.389, 0.592] | -0.279 [-0.591, 0.004] | 0.540 [0.424, 0.641] | 1.00 / 0.03 / 1.00 |
|  | Mamba | 0.528 [0.410, 0.644] | -0.294 [-0.563, -0.028] | 0.591 [0.485, 0.679] | 1.00 / -0.26 / 1.00 |
| OpenSR `spain_crops` | Lite | 0.628 [0.525, 0.713] | -0.234 [-0.425, -0.092] | 0.674 [0.569, 0.763] | 0.65 / 0.50 / 0.89 |
|  | Mamba | 0.705 [0.639, 0.761] | -0.327 [-0.489, -0.184] | 0.740 [0.663, 0.805] | 0.69 / 0.45 / 0.85 |
| OpenSR `spain_urban` | Lite | 0.674 (desc.) | -0.519 (desc.) | 0.731 (desc.) | 0.86 / 0.51 / 0.83 |
|  | Mamba | 0.738 (desc.) | -0.590 (desc.) | 0.770 (desc.) | 0.75 / 0.43 / 0.93 |

The stability rises with **both** unsupported detail and supported synthesis, i.e. with how much detail the model *added*, and is unrelated or slightly negatively related to omission. It marks where the model synthesises, not where the synthesis is wrong: the association with unsupported detail is stronger than with supported synthesis on SEN2NEON (0.50-0.57 against 0.37-0.46) and weaker on OpenSR-Test, and both are confounded with texture. "Unsupported" means not confirmed by this reference, not false.

### 6.8 Lite and Mamba on the same tiles (descriptive; not a ranking)

| dataset | units | metric (Lite - Mamba) | mean difference | 95% interval |
|---|---|---|---|---|
| SEN2NEON (seeded random 30) | 11 | 10 m within-tile Spearman | +0.02809 | [0.004929, 0.0514] |
|  | 11 | pixel within-tile Spearman | +0.01561 | [-0.002671, 0.03479] |
|  | 11 | tile mean stability | -0.0001827 | [-0.000252, -0.0001236] |
|  | 11 | tile RMSE | +5.382e-05 | [-9.748e-05, 0.0002498] |
|  | 11 | tile SAM (deg) | +0.04958 | [-0.02052, 0.1391] |
| OpenSR `spot` | 6 | 10 m within-tile Spearman | -0.03893 | [-0.07277, -0.01194] |
|  | 6 | pixel within-tile Spearman | -0.03095 | [-0.04935, -0.01407] |
|  | 6 | tile mean stability | -0.000135 | [-0.0003592, 2.234e-05] |
|  | 6 | tile RMSE | -0.0001123 | [-0.0004312, 0.0001065] |
|  | 6 | tile SAM (deg) | +0.01371 | [-0.06587, 0.07106] |
| OpenSR `spain_crops` | 5 | 10 m within-tile Spearman | -0.03193 | [-0.04225, -0.02237] |
|  | 5 | pixel within-tile Spearman | -0.02671 | [-0.03441, -0.01906] |
|  | 5 | tile mean stability | -9.379e-05 | [-0.0001547, -3.293e-05] |
|  | 5 | tile RMSE | +0.0001109 | [9.862e-06, 0.0002138] |
|  | 5 | tile SAM (deg) | -0.002113 | [-0.04531, 0.03215] |
| OpenSR `spain_urban` | 4 | 10 m within-tile Spearman | -0.04909 | (desc.) |
|  | 4 | pixel within-tile Spearman | -0.04431 | (desc.) |
|  | 4 | tile mean stability | -0.0002161 | (desc.) |
|  | 4 | tile RMSE | -3.946e-05 | (desc.) |
|  | 4 | tile SAM (deg) | -0.04869 | (desc.) |

Mamba's tile mean stability is larger on SEN2NEON (by 0.00018 [0.00012, 0.00025]) and its within-tile association with error is slightly weaker there (10 m: Lite − Mamba +0.028 [0.005, 0.051]); on OpenSR-Test the sign reverses (−0.03 to −0.05). Tile RMSE and SAM differences are within their intervals. These are differences in the behaviour of each model's stability signal on 4-11 scene units, not a verdict on either model.

### 6.9 Cost of the ensemble

| dataset | model | members | single pass (s) | TTA (s) | TTA / single pass | tiles timed |
|---|---|---|---|---|---|---|
| SEN2NEON (seeded random 30) | Lite | 6 | 0.096 | 0.828 | 8.60 | 11 |
|  | Mamba | 6 | 14.788 | 88.858 | 6.01 | 11 |
| OpenSR `spot` | Lite | 6 | 0.013 | 0.189 | 14.28 | 5 |
|  | Mamba | 6 | 1.709 | 10.159 | 5.95 | 5 |
| OpenSR `spain_crops` | Lite | 6 | 0.012 | 0.179 | 14.89 | 20 |
|  | Mamba | 6 | 1.719 | 10.423 | 6.06 | 20 |
| OpenSR `spain_urban` | Lite | 6 | 0.015 | 0.183 | 12.44 | 12 |
|  | Mamba | 6 | 1.558 | 9.463 | 6.07 | 12 |

Six members, each a full tiled pass, so the ensemble costs **6x** the single pass for Mamba (89 s against 14.8 s for a 256x256 LR / 1024x1024 SR scene, 9 tiles) and **8-15x** for Lite, where the transforms, de-transformations and Welford accumulation are a visible share of a 0.1 s pass. Mamba is about 150 times slower than Lite per pass on the same scene. Correctness was not traded for speed; nothing was cached or reduced.

### 6.10 The rectangular multi-tile scene (real models)

`python -m frame.reliability scene` on a 180 x 256 LR window of SEN2NEON tile `2022_JORN_5__9_7` (not a multiple of 128, not square; output 720 x 1024 SR px, 2 x 3 = 6 tiles; the rotated views are 256 x 180 on a 3 x 2 grid):

| check | Lite | Mamba |
|---|---|---|
| spread covers the full output, finite | yes (0 non-finite) | yes (0 non-finite) |
| every member aligned with the identity view | yes (MAE to identity 0.0005-0.0006 against 0.024 if mis-oriented) | yes (0.0006-0.0007 against 0.024) |
| seam / interior spread, canonical grid | 0.950 | 0.931 |
| seam / interior spread, rot90 view's grid | 0.986 | 1.041 |
| georeferencing (CRS, origin, bounds, values, pixel size 2.5 m) | identical | identical |
| TTA / single pass | 9.9 | 6.0 |

No stability seam is introduced (all ratios within the declared tolerance of 1.10), transforms and de-transforms preserve orientation on a rectangular scene, and the spread GeoTIFF shares the SR output's grid exactly. Records: `experiments/uncertainty/runs/scene_check_sen2sr_{lite,mamba}/scene_check.json` (the 15 MB GeoTIFFs are not committed).

## 7. What the evidence supports, and what it does not

The three questions, answered only from the measurements above.

1. **Does higher TTA variation correspond to higher actual error?** *Weakly to moderately, within a tile, and not more than image texture does.* On registered evidence the per-tile rank correlation between the spread and the error is positive in 83-100% of tiles, 0.08-0.16 (pixel to 10 m) on SEN2NEON and 0.17-0.59 on OpenSR-Test, growing with the cell size. The two trivial predictors reach similar values. The part of the association that the stability carries beyond them is small (partial 0.05-0.26 at 10 m, mostly with intervals above 0).
   Across tiles the association is not distinguishable from zero on the one dataset with enough units (SEN2NEON, 11 units), and texture predicts the tile error as well.
2. **Does lower TTA variation correspond to lower actual error?** The same association read the other way: the most stable 80% of 10 m cells have 4-7% lower error than all cells on SEN2NEON (28% at most on OpenSR-Test), an amount that ranking by texture achieves too.
3. **Can instability identify high-risk predictions?** *Not shown.* On SEN2NEON the spread ranks high-error cells no better than chance (AUROC 0.46-0.48; so do the baselines). On OpenSR-Test it ranks them well (AUROC 0.82-0.89) but the trivial baselines are almost as good (0.79-0.89), and none of these figures is a held-out test result: no dataset has the 12 eligible scene units the pre-declared development/test protocol needs.

Further, established by measurement: the spread is **not calibrated** (16-32 times smaller than the error; a 2-sigma interval covers 5-8% of the reference instead of 95%); it **follows how much detail the model adds** (unsupported detail and supported synthesis alike) rather than the omitted detail; and a misregistered reference would have overstated all of it.
The honest statement of the current product is the one already in the API and the requirements: *a relative model-stability signal that marks where the model changes its answer when the scene is re-framed, to be read with the error evidence, not instead of it.* This phase found **weak, inconsistent** support for reading it as an error indicator, and none for confidence or calibrated uncertainty. It was not forced into a positive result.

## 8. Defects found and decisions made on the way

* **Registration estimate at whole-pixel resolution where there is nodata.** The first gate preview classified 10 of 30 SEN2NEON tiles as not eligible and only 10 as eligible because the Phase 5 estimator is integer-valued on tiles with nodata (§3). A sub-pixel refinement of the correlation peak was added (tested against known fractional shifts and against a brute-force search on 87 real tiles) and the gate re-run: 12 / 15 / 3. The tolerances themselves were declared before any result and not changed.
* **A degenerate interval was tightened, in the conservative direction, after seeing it.** A first analysis printed `1.000 [1.000, 1.000]` for the tile-level Spearman of six `spot` tiles. Statistics whose value is a correlation or a curve *across* units now need 10 units for an interval (means of unit values keep 5). This suppresses intervals; it cannot create a result. The analysis was re-run.
* **`python -m frame.reliability scene` handed a CPU tensor to the CUDA Lite model**; found by the real run, fixed with a regression test (the tensor is moved to the system's device).
* **The tile engine's own NaN/Inf guard** raises for a non-finite model output; that is recorded as `prediction_not_finite`, not as a crashed model.
* **No defect was found in the TTA layer** (`frame/uncertainty/` was not modified). `frame/evaluate/metrics.py` gained `hallucination_labels` (the Phase 5 analysis now derives its fractions from it; outputs unchanged, tests pass) and `frame/evaluate/systems.py` exposes `tile_model` / `device` / `tiling`; the Phase 5 estimator was reused, not changed.
* `frame.reliability run --reuse-evidence` repeats only the analysis from cached per-tile evidence whose settings, dataset and model digest are unchanged (the ensemble is the expensive part: 45 minutes for this run).

## 9. Limitations

* **Small, registration-limited evidence.** 12 of 30 SEN2NEON tiles (11 scene units), 6 of 9 `spot` scenes, 21 tiles from 5 orthophotos (`spain_crops`) and 13 from 4 (`spain_urban`) were eligible. Intervals rest on 4-11 units; `spain_urban` is descriptive only; nothing supports a claim about all of SEN2NEON, let alone other regions.
* **The eligible tiles are the best-registered ones** (and the OpenSR tiles needed a translation correction, a recorded integer crop that assumes one global displacement). Their residual misregistration (up to 0.5 HR px) still slightly inflates the association (~0.01-0.02 by the displacement sweep).
* **Registration was judged against the bicubic baseline, not against the truth**, and the reference is a different sensor with its own radiometry: the error is an error against that reference.
* **No held-out test.** The development/test protocol (threshold, recalibration) could not run (needs 12 units; 11 or fewer available), so detection is descriptive and recalibration not assessed. The stability therefore **remains uncalibrated stability evidence**.
* **Two models, one scene family per dataset, one ensemble** (six views; a different perturbation, more members or a deep ensemble might behave differently). North America and Spain only: **no Indian reference, no Indian evidence**.
* **No downstream task in Phase 6** (classification, land cover, change detection): whether unstable regions are the ones downstream products get wrong was untested here. Phase 7 (`docs/DOWNSTREAM.md`) tests it for an NDVI-derived vegetation decision on the same gate-eligible evidence: the stability carries at most a small amount of information about the downstream error beyond texture. Land cover was not tested (no region-level labels). The boundary-versus-interior instability analysis suggested by the requirements was not done.
* **Associations, not causes.** Texture, brightness and the amount of detail added are confounded with both instability and error; the trivial baselines and the partial correlation reduce that, they do not remove it.
* The working tree is not committed; the run records the parent revision with a dirty flag.

## 10. Tests and how to reproduce

`frame/tests/test_reliability_{config,association,detection_calibration,eligibility,targets,evidence,analysis,runner,scene_check,tta_audit,cli}.py` (no GPU): controlled monotone relationships, uninformative signals, known AUROC/AUPRC/risk-coverage fixtures, leakage (development-only threshold, held-out recalibration), the gate (aligned accepted, shifted rejected, quadrant and residual rules, sub-pixel refinement, insufficient valid pixels, NaN/Inf, junk under the mask), rectangular TTA, exclusions and failures, provenance, evidence reuse, CLI. Key mutants (removing the gate's tolerance, the quadrant rule, the alignment crop, the development-only threshold, the held-out assessment, the unit-clustering, the risk-coverage direction) each fail a test.

```bash
sen2sr_venv/bin/python -m frame.reliability check experiments/uncertainty/configs/reliability_v1.json --verbose      # the gate preview: seconds
sen2sr_venv/bin/python -m frame.reliability run   experiments/uncertainty/configs/reliability_v1.json --verbose      # ~45 min (Mamba dominates)
sen2sr_venv/bin/python -m frame.reliability run   experiments/uncertainty/configs/reliability_v1.json --reuse-evidence   # after changing only analysis settings, into a new output_dir
sen2sr_venv/bin/python -m frame.reliability scene experiments/uncertainty/configs/reliability_v1.json --system sen2sr_lite \
    --lr-geotiff $FRAME_DATA_ROOT/sen2neon/s2_l2a_10m/2022_JORN_5__9_7.tif --window 0 180 0 256 --output-dir experiments/uncertainty/runs/scene_check_sen2sr_lite
```

## 11. Changes made to this layer by Phase 7 (`docs/DOWNSTREAM.md`)

The eligibility gate is reused **unchanged** by the downstream layer; three small, tested additions make that reuse explicit and did not change any Phase 6 result: `frame.reliability.eligibility.GATE_VERSION` (`frame-reliability-gate/1`, recorded in every gate result),
`frame.reliability.runner.run_tta_for_tile` (the single-tile TTA step of the runner made public, with the same failure taxonomy; `_run_tta` now calls it), and three public aliases in `frame.reliability.analysis` (`aggregate_over_units`, `across_unit_association`, `unit_bootstrap`) for the statistics the downstream association reuses. The gate counts reproduced by the downstream `check` command are identical to the ones above (12 / 6 / 21 / 13 eligible tiles).
