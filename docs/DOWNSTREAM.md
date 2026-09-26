# Downstream analytical utility (Phase 7)

`frame/downstream/` asks two questions, and reports a null or mixed answer as such:

1. **Does super-resolution change the reliability of a downstream analytical decision?** The task is a vegetation-index decision computed from the Red and NIR bands, compared with the same decision made
   on the high-resolution (HR) reference, for the HR reference, the LR input (native pixels), the bicubic baseline, SEN2SR-Lite and SEN2SR-Mamba.
2. **Does the model-stability signal carry information about downstream mistakes once image texture is controlled for?** The stability is the deployed relative TTA spread; Phase 6 measured that it is only
   weakly informative about reconstruction error, is confounded with texture, and is **uncalibrated**. Nothing in Phase 7 turns it into a confidence or a calibrated uncertainty.

```
scene ─▶ reference gate ─▶ bicubic / Lite / Mamba (+ TTA) ─▶ NDVI maps ─▶ common valid mask ─▶ regions ─▶ NDVI + decision error ─▶ association · risk-coverage ─▶ report
         (Phase 6, unchanged) (system abstraction, tile engine)  (B04, B08)   (all systems, same pixels) (10 m / 40 m)  (per scene unit)       (unit-clustered bootstrap)
```

```bash
python -m frame.downstream check experiments/downstream/configs/downstream_v1.json    # config, datasets, systems, a PREVIEW of the reference gate and of the region counts; runs no model
python -m frame.downstream run   experiments/downstream/configs/downstream_v1.json    # writes experiments/downstream/runs/downstream_v1/ (JSON + README.md)
python -m frame.downstream run   CONFIG.json --reuse-regions                          # repeat only the analysis from the cached region tables
python -m frame.downstream smoke [--output-dir DIR]                                   # end-to-end on synthetic scenes and toy models: no dataset, weights or GPU; the numbers mean nothing
```

Exit codes: `0` success; `1` finished but some dataset has no eligible evidence; `2` refused or invalid input. Outputs (`run`): `config.json`, `tiles.jsonl` (one row per tile: eligibility, the reason for an
exclusion, counts; **an excluded tile carries no numbers**), `summary.json` (provenance, evidence accounting, deferred and unavailable tasks), `downstream_metrics.json`, `association.json`,
`risk_coverage.json`, `README.md`. The per-region tables are cached **outside** the repository (`cache_dir`); no rasters are written to the run directory.

## 1. What is and is not in scope

| Item | Status |
|---|---|
| NDVI from B04 / B08 in reflectance, agreement with the HR reference, region mean / median, a fixed threshold decision, decision disagreement | implemented |
| Change relative to bicubic (and to the LR input), for each model | implemented (paired differences per scene unit; descriptive when the unit count is small) |
| Association of stability with the downstream error, texture-controlled | implemented, at the primary threshold and both region scales |
| Risk-coverage with the downstream error as the target, against the texture ranking and an oracle | implemented, at the primary threshold and both region scales |
| Land-cover classification | **deferred**: `secondary_landcover_task_deferred_no_supported_reference` (§7) |
| Indian-region downstream validation | **unavailable**: `india_downstream_validation_unavailable` (§7) |
| Model ranking or a winner, a calibrated uncertainty, causal claims, threshold tuning | never |

## 2. The NDVI and the decision (declared before any result was computed)

| Item | Definition |
|---|---|
| NDVI | `(B08 - B04) / (B08 + B04)` on reflectance, bands addressed **by name** through the sample's band list (`frame.consistency.spectral_ratios.compute_ndvi`) |
| Valid pixel | the Phase 5 strict mask (finite, valid in the reference, not saturated) **and** reference Red + NIR > 0.02 (`min_reflectance_sum`: a near-zero denominator gives an unstable NDVI) **and** `|NIR + Red| > 1e-6` in the reference, the LR input, the bicubic baseline **and every model**. One mask serves all systems, so every system is scored on identical pixels and one system's undefined NDVI removes those pixels for all of them |
| Decision | `vegetation` iff `NDVI >= 0.3` |
| Threshold policy | **0.3 is a conventional value, declared in the config before results existed; the requirements name no threshold.** It is never tuned. 0.2 and 0.4 are reported beside it as a sensitivity check and are never used to choose. Association and risk-coverage run at 0.3 only |
| Region decision | a region is `vegetation` iff its vegetation fraction (over valid pixels) is `>= 0.5` |
| Alignment | the Phase 5/6 discipline: the registration is estimated from the bicubic baseline, a recorded integer translation (<= 4 HR px) is applied to the data, and everything is cropped to the registered overlap |

The reference for every system is the HR image. That reference is a **different sensor** on a different date for SEN2NEON and OpenSR-Test, so "agreement with the reference" contains sensor and acquisition
differences that no model can remove; the comparison to bicubic and to the LR input is what isolates what the super-resolution added.

## 3. Regions

The requirements do not fix a region size for downstream analysis; they do state that one 10 m Sentinel-2 pixel corresponds to 16 cells at 2.5 m, and Phase 6 already defines fixed cells on the aligned grid. Phase 7
reuses them:

| Scale | Size | Role |
|---|---|---|
| primary | 4 x 4 HR px = 10 m (the footprint of one Sentinel-2 pixel) | headline |
| secondary | 16 x 16 HR px = 40 m | shown beside it |

Every region starts on a multiple of its size in prediction coordinates (so it never straddles a prediction pixel), after the translation crop of §2. A region is kept iff at least 75 % of its pixels are valid
(`min_valid_fraction`); the excluded ones are counted (`region_insufficient_valid_pixels`), not dropped silently. Each region keeps its dataset, scene unit, tile id, spatial index (row / column on the
prediction grid and on the HR grid), role and split. **Regions of one scene are never pooled as independent replicates**: every summary and interval is over scene units.

Per region: valid-pixel count; NDVI summaries (mean, median) of the reference, LR, bicubic, Lite and Mamba; the absolute NDVI error of the region mean; the vegetation fraction and its error; the pixel-level
decision disagreement rate; the majority-decision error; the texture baseline (region mean Sobel gradient of the bicubic input); the added detail (`|prediction - bicubic|`); and the Phase 6 stability summary
(region mean of the model's TTA spread).

## 4. Metrics

**NDVI fidelity** (per scene unit, then over units): MAE, RMSE, signed bias, median absolute error of the region-mean NDVI; the Pearson correlation only when at least 30 regions with a non-constant reference
exist (otherwise `not_computable`); valid coverage (kept / candidate regions).

**Decision:** accuracy, balanced accuracy, false-positive rate and false-negative rate (from confusion counts pooled **within** a scene unit; a rate that needs a class the unit does not contain is
`not_computable`, never 0), and the disagreement rate `(fp + fn) / n_valid`. All at the primary threshold, and at the two sensitivity thresholds.

**Change relative to bicubic** (and to the LR input): the paired per-unit difference `system - bicubic` in each metric. A Wilcoxon signed-rank test only with at least 6 pairs; with fewer the paired
difference is `descriptive_only`. The Lite-minus-Mamba difference is descriptive. **There is no ranking and no winner.**

Non-finite values are dropped and counted; an undefined statistic is `not_computable`.

## 5. Association and risk-coverage (the Phase 6 discipline, downstream target)

* **Targets:** the region's absolute NDVI error and its pixel decision disagreement rate.
* **Within each tile** the Spearman correlation of stability, texture and added detail with the target over the tile's regions, and the **partial** Spearman correlation of the stability given texture, and given
  texture and added detail. The texture-only relationship is reported as the baseline the stability has to beat. Tiles of one scene unit are averaged first; the summary is over **scene units**.
* **Intervals:** a percentile bootstrap that resamples whole scene units (>= 5 units for an interval on a mean; fewer is `descriptive_only`). Correlations **across** tiles / units (tile mean stability against
  tile mean error) need >= 10 units. The pooled-regions correlation uses a seeded subsample of each tile and also resamples whole units. There is no development/test split unless there are >= 12 units.
* **Risk-coverage:** within each scene unit, regions are ranked by instability; the 20 / 40 / 60 % most unstable are dropped (retention 0.8 / 0.6 / 0.4, plus full retention 1.0), and the relative reduction of the
  remaining mean downstream error is reported, next to the same reduction when ranking by texture and by the true error (the oracle). It is an ordering, **not calibrated selective prediction**.

## 6. The reference eligibility gate (reused unchanged)

The Phase 6 gate (`frame-reliability-gate/1`, `frame.reliability.eligibility.evaluate_reference`) is applied to the bicubic baseline **before any model runs**, with the Phase 6 settings: tolerance 0.5 HR px
after an optional recorded whole-pixel translation of at most 4 px, quadrant spread <= 1.0 HR px, >= 50 % strict-valid pixels. It was not relaxed because the sample is small. A tile whose reference is
`uncertain` or `not_eligible` is written to `tiles.jsonl` with `excluded_from_downstream_primary_analysis` and a machine-readable reason; **no NDVI, region table or error is derived from it, no model is run on
it, and it cannot enter any summary** (a regression test counts the calls to the NDVI and region code and checks the cache). If the model fails on a tile (`model_failure`, `tta_member_failure`,
`prediction_not_finite`) the tile is excluded for every system, so all systems share the same tiles.

The gate is the reason this phase can be trusted at all: Phase 6 measured that misregistration *manufactures* association between stability and error.

## 7. Deferred and unavailable tasks (recorded, not weakened)

* **Secondary land-cover task: `secondary_landcover_task_deferred_no_supported_reference`.** A land-cover *change* or *class* decision needs region-level labels or a defensible reference segmentation. None of the datasets
  in the repository has one: the only label is the NEON land-cover superclass of a **whole tile** (it says what the tile mostly is, not what a 10 m region is), OpenSR-Test carries none, and SEN2NAIPv2 was assessed from
  metadata only (Phase 5). The thresholded NDVI decision is a *vegetation proxy*; its agreement with the HR reference is **not** a classification accuracy against labels, and no land-cover performance is claimed. No further
  dataset was downloaded to obtain one.
* **Indian regions: `india_downstream_validation_unavailable`.** There is no Indian HR reference and no Indian downstream label in the repository. No synthetic "Indian truth" was created and no Indian claim is made.

## 8. Results (measured, descriptive, non-causal; nothing is ranked)

Record: `experiments/downstream/runs/downstream_v1/` (`config.json`, `tiles.jsonl`, `summary.json`, `downstream_metrics.json`, `association.json`, `risk_coverage.json`, `README.md` with every table). Config
`experiments/downstream/configs/downstream_v1.json` (digest `cfc6c1a2c91b624f…`). SEN2SR-Lite (published weights, hard constraint) and SEN2SR-Mamba (isolated worker, RGBN, hard constraint), the deployed six-view TTA over the
tile engine, RTX 3050 laptop GPU, git `eb37695` with a **dirty working tree** (commit before quoting). The run was made in two passes over identical inputs: the full pass (gate, models, TTA, regions, analysis; about 41 minutes) and an
analysis-only pass with `--reuse-regions` (region tables reused by digest) that added the paired *stability − texture* column of §8.4 — every other number was verified byte-identical between the passes. Datasets are
analysed separately and never pooled. Intervals are 95 % percentile bootstraps over **scene units**; "desc." = fewer than 5 scene units (or too few units with both classes), so no interval is given.

### 8.1 Evidence: what entered and what was excluded

The Phase 6 gate was applied unchanged; the counts match Phase 6 exactly (a check that the gate was not touched).

| dataset | tiles | eligible | uncertain | not eligible | eligible scene units | candidate 10 m regions | in eligible tiles | analysed |
|---|---|---|---|---|---|---|---|---|
| `sen2neon_random30` | 30 | **12** | 15 | 3 | **11** (of 28) | 1,966,080 | 786,432 | 711,979 |
| `opensr_spot` | 9 | **6** | 3 | 0 | **6** (of 9) | 147,456 | 98,304 | 97,921 |
| `opensr_spain_crops` | 28 | **21** | 7 | 0 | **5** (of 5) | 458,752 | 344,064 | 339,852 |
| `opensr_spain_urban` | 20 | **13** | 7 | 0 | **4** (of 4) | 327,680 | 212,992 | 210,440 |

Exclusion reasons: `reference_alignment_uncertain` (SEN2NEON 15, spot 3, crops 7, urban 7) and `reference_alignment_invalid` (SEN2NEON 3). Inside eligible tiles, regions were excluded for
`region_insufficient_valid_pixels` (SEN2NEON 71,385; the others none) and for falling on the crop edge after the recorded translation (`alignment_crop_edge_cells`: 3,068 / 383 / 4,212 / 2,552). No dataset had a
model failure. `spain_urban` has 4 units, so **everything in it is descriptive**. Excluded tiles have **no** numbers in any table; their regions are counted from the grid geometry alone, so the loss of coverage is visible
(only 40 % of the SEN2NEON candidate regions lie in an eligible tile).

**A check against a second derivation.** For one eligible tile from each of the SEN2NEON, `spot` and `spain_crops` sets, 400 random regions were recomputed **without** `frame.downstream` (band indices by hand, NDVI, the
declared window on the recorded grid). The reference region means agreed to 0.0000; the bicubic region means agreed to a median of 0.0001–0.0006 NDVI (largest 0.009, where the independent bicubic resampling and the
common mask of the run differ). This validates the band selection, the windows and the alignment bookkeeping on real data, not the science.

### 8.2 NDVI fidelity and decision agreement (10 m regions, threshold 0.3)

Mean over scene units. **Most of the error is a floor that no system removes**: the HR reference is a different sensor on a different date, and NDVI (a ratio of two bands) is sensitive to that. `lr_native` is the LR
input itself, the bar below which "SR added nothing" applies.

| dataset (units) | quantity | `lr_native` | `bicubic` | `sen2sr_lite` | `sen2sr_mamba` |
|---|---|---|---|---|---|
| `sen2neon_random30` (11) | NDVI MAE | 0.047 | 0.046 | 0.047 | 0.046 |
| | pixel decision disagreement | 0.055 | 0.053 | 0.055 | 0.055 |
| | balanced accuracy | 0.900 | 0.911 | 0.913 | 0.907 |
| `opensr_spot` (6) | NDVI MAE | 0.020 | 0.018 | 0.020 | 0.019 |
| | pixel decision disagreement | 0.033 | 0.027 | 0.029 | 0.036 |
| | balanced accuracy | 0.878 desc. | 0.893 desc. | 0.898 desc. | 0.879 desc. |
| `opensr_spain_crops` (5) | NDVI MAE | 0.038 | 0.038 | 0.037 | 0.038 |
| | pixel decision disagreement | 0.086 | 0.083 | 0.080 | 0.080 |
| | balanced accuracy | 0.832 | 0.838 | 0.851 | 0.851 |
| `opensr_spain_urban` (4, desc.) | NDVI MAE | 0.041 | 0.041 | 0.040 | 0.041 |
| | pixel decision disagreement | 0.086 | 0.081 | 0.077 | 0.078 |
| | balanced accuracy | 0.864 | 0.872 | 0.884 | 0.883 |

(The README carries the intervals, RMSE, signed bias — between −0.004 and 0.000 everywhere —, median error, vegetation-fraction error, region-majority decision error, accuracy, FPR/FNR and coverage for every cell, at
both region scales and both sensitivity thresholds.)

**Change relative to bicubic** (paired over the same scene units; `system − bicubic`; a negative error difference is a smaller error; not a ranking):

| dataset | model | NDVI MAE | pixel decision disagreement | balanced accuracy |
|---|---|---|---|---|
| `sen2neon_random30` (11) | `sen2sr_lite` | **+0.0011** [0.0007, 0.0015] | **+0.0019** [0.0009, 0.0030] | +0.0022 [−0.0024, 0.0068] |
| | `sen2sr_mamba` | +0.0002 [−0.0008, 0.0012] | +0.0017 [0.0000, 0.0035] | −0.0047 [−0.0118, 0.0019] |
| `opensr_spot` (6) | `sen2sr_lite` | **+0.0024** [0.0010, 0.0042] | **+0.0020** [0.0002, 0.0049] | +0.0045 desc. |
| | `sen2sr_mamba` | **+0.0014** [0.0004, 0.0029] | **+0.0083** [0.0003, 0.0202] | −0.0142 desc. |
| `opensr_spain_crops` (5) | `sen2sr_lite` | −0.0007 [−0.0016, −0.0000] | **−0.0027** [−0.0064, −0.0003] | **+0.0125** [0.0070, 0.0209] |
| | `sen2sr_mamba` | −0.0002 [−0.0005, 0.0001] | **−0.0033** [−0.0065, −0.0009] | **+0.0131** [0.0088, 0.0185] |
| `opensr_spain_urban` (4, desc.) | `sen2sr_lite` | −0.0006 | −0.0042 | +0.0124 |
| | `sen2sr_mamba` | +0.0003 | −0.0028 | +0.0110 |

(bold = the interval excludes zero; with 5–6 units these intervals are wide-tailed and no multiplicity correction was applied.)

*Not comparable with §8 of `docs/EVALUATION.md`:* that document scores NDVI **per pixel** on every tile with the Phase 5 strict mask (SEN2NEON NDVI MAE 0.054–0.072); here the NDVI of fixed 10 m / 40 m **regions** is scored on registration-eligible tiles
only, so the values are smaller (region means average out pixel noise, and the misregistered tiles are gone). Neither replaces the other.

**What this says, and does not say.**

* **Super-resolution did not consistently improve the downstream decision.** On SEN2NEON and `spot` the models were, if anything, marginally *worse* than bicubic (NDVI MAE +0.0002 to +0.0024; disagreement +0.2 to +0.8
  percentage points); on `spain_crops` (and directionally `spain_urban`) they were marginally *better* (disagreement −0.3 percentage points, balanced accuracy about +1.3 points, a lower false-positive rate). The sign
  depends on the dataset, not on the model. This is a **mixed / null** result.
* **The effects are small next to the error floor:** NDVI MAE 0.02–0.05 and disagreement 3–9 % for every system including the LR input, against differences of ~0.001 NDVI and ≤ 0.8 percentage points.
* **At 40 m the systems are practically indistinguishable** (NDVI MAE within 0.001 of each other in every dataset). At 10 m a region is one LR pixel, and the Fourier hard constraint of both models ties the low-frequency content (the
  block mean) to the LR observation, so a region *mean* has little room to change; the extra information of SR lives inside the region, where the pixel-level decision disagreement is the only metric that can see it. That the hard
  constraint explains the near-equality is plausible but was not tested here (the Phase 5 no-constraint variant was not re-run for NDVI).
* **No transfer to Indian data, no land-cover claim, and no statement about which system is better** follow from these numbers.
* **Threshold dependence.** At 0.2 and 0.4 the systems again differ by at most about 1 percentage point and the direction still depends on the dataset (for `spot`, `bicubic` is lowest at 0.3 and 0.4; for `spain_crops` the SR models are lowest at all
  three). The primary threshold was fixed in advance and is not selected among these.

### 8.3 Is instability associated with downstream mistakes, once texture is controlled? (10 m)

Within-tile Spearman correlations of a region's stability with its error, averaged over scene units (interval over units; `partial` = controlling for texture, and for texture + added detail). Texture is the trivial predictor
the stability has to add to.

| dataset | model | error (units) | stability | texture only | added detail only | partial (given texture) | partial (given texture + added detail) |
|---|---|---|---|---|---|---|---|
| `sen2neon_random30` | Lite | \|NDVI error\| (11) | 0.095 [0.030, 0.157] | 0.073 [0.035, 0.111] | 0.077 [0.016, 0.128] | 0.063 [0.006, 0.115] | 0.054 [0.015, 0.095] |
|  | Mamba | \|NDVI error\| (11) | 0.108 [0.034, 0.180] | 0.090 [0.036, 0.152] | 0.105 [0.029, 0.185] | 0.066 [-0.000, 0.126] | 0.046 [0.006, 0.095] |
|  | Lite | disagreement (10) | 0.105 [0.019, 0.194] | 0.117 [0.030, 0.196] | 0.068 [-0.020, 0.153] | 0.029 [-0.037, 0.085] | 0.050 [0.016, 0.084] |
|  | Mamba | disagreement (10) | 0.082 [-0.015, 0.180] | 0.113 [0.026, 0.196] | 0.073 [-0.012, 0.159] | 0.008 [-0.069, 0.079] | 0.028 [-0.018, 0.077] |
| `opensr_spot` | Lite | \|NDVI error\| (6) | 0.143 [0.037, 0.263] | 0.103 [0.013, 0.201] | 0.158 [0.061, 0.260] | 0.109 [0.033, 0.183] | 0.054 [-0.000, 0.108] |
|  | Mamba | \|NDVI error\| (6) | 0.183 [0.057, 0.312] | 0.123 [0.033, 0.221] | 0.173 [0.068, 0.288] | 0.142 [0.047, 0.238] | 0.087 [0.018, 0.154] |
|  | Lite | disagreement (4) | 0.145 desc. | 0.126 desc. | 0.144 desc. | 0.074 desc. | 0.042 desc. |
|  | Mamba | disagreement (4) | 0.154 desc. | 0.141 desc. | 0.174 desc. | 0.075 desc. | 0.012 desc. |
| `opensr_spain_crops` | Lite | \|NDVI error\| (5) | 0.039 [-0.087, 0.165] | 0.035 [-0.075, 0.156] | 0.052 [-0.064, 0.168] | 0.028 [-0.027, 0.090] | 0.004 [-0.035, 0.053] |
|  | Mamba | \|NDVI error\| (5) | 0.058 [-0.066, 0.182] | 0.049 [-0.065, 0.166] | 0.063 [-0.040, 0.172] | 0.040 [-0.021, 0.115] | 0.013 [-0.044, 0.077] |
|  | Lite | disagreement (5) | 0.151 [-0.009, 0.287] | 0.144 [-0.003, 0.269] | 0.193 [0.050, 0.308] | 0.066 [-0.004, 0.125] | -0.015 [-0.077, 0.048] |
|  | Mamba | disagreement (5) | 0.216 [0.039, 0.368] | 0.145 [-0.002, 0.270] | 0.203 [0.066, 0.320] | 0.170 [0.067, 0.273] | 0.095 [-0.019, 0.197] |
| `opensr_spain_urban` | Lite | \|NDVI error\| (4) | 0.165 desc. | 0.125 desc. | 0.174 desc. | 0.110 desc. | 0.044 desc. |
|  | Mamba | \|NDVI error\| (4) | 0.185 desc. | 0.144 desc. | 0.198 desc. | 0.118 desc. | 0.038 desc. |
|  | Lite | disagreement (4) | 0.235 desc. | 0.212 desc. | 0.270 desc. | 0.113 desc. | 0.011 desc. |
|  | Mamba | disagreement (4) | 0.304 desc. | 0.213 desc. | 0.296 desc. | 0.223 desc. | 0.106 desc. |


**Reading the association.**

* Every stability-only correlation with the error is **positive but small** (0.04–0.22 at 10 m in the datasets with intervals), and **so is the texture-only correlation next to it** (0.03–0.15): the stability and a plain image-gradient
  map order regions by their downstream error about equally well. This is the Phase 6 finding carried to the downstream target.
* **Controlling for texture removes most of it.** The partial correlation is 0.01–0.17 at 10 m; where an interval exists it excludes zero for \|NDVI error\| on SEN2NEON (Lite) and `spot` (both models) and for the decision
  disagreement on `spain_crops` (Mamba), and includes zero elsewhere (SEN2NEON Mamba \|NDVI error\| and `spain_crops` Lite disagreement by a hair; SEN2NEON disagreement and `spain_crops` \|NDVI error\| clearly). With **added detail** also
  controlled the partial falls to −0.02–0.10; what survives is small and dataset- and model-dependent (its interval excludes zero for \|NDVI error\| on SEN2NEON with both models, on `spot` with Mamba, and for the SEN2NEON Lite disagreement).
  These are about twenty intervals from five to eleven units with no multiplicity correction: read them as "a small residual signal appears in some cells", not as a demonstrated independent information source.
* **The decision-disagreement target is zero for most regions** (the oracle in §8.4 removes all of it after dropping 40 %), so its correlations are coarse and computable for fewer units (10 of 11 on SEN2NEON, 4 of 6 on `spot`).
* At **40 m** (README) the \|NDVI error\| correlations on SEN2NEON are 0.03–0.04 with intervals that include zero; the disagreement correlations are larger (0.18–0.34) but so are the texture-only ones (0.26–0.27 on SEN2NEON), and the partial given texture is near zero on SEN2NEON (0.04, −0.01) while it stays positive on `spain_crops` (Lite 0.14 [0.07, 0.20], Mamba 0.25 [0.12, 0.37]).
* **Across tiles** (tile mean stability against tile mean error) only SEN2NEON has 11 units, enough for an interval, and every one includes zero (Lite \|NDVI error\| 0.413 [−0.050, 0.972]); the other datasets have too few units.
  The **pooled regions** correlations (a seeded subsample of each tile; intervals over units) are larger on `spot` (0.39–0.56) but are dominated by between-tile differences in brightness and texture, not by the within-tile ordering; they are shown for completeness only.

### 8.4 Risk-coverage on the downstream error (10 m, threshold 0.3)

Within each scene unit, regions are ordered by instability (or by texture, or by the true error = oracle) and the 20 / 40 / 60 % worst are dropped. The value is the relative reduction of the mean downstream error that remains
(mean over units [interval]). `stability − texture` is the paired per-unit difference of the two reductions. **It is an ordering, not calibrated selective prediction, and the paired column was added after the first pass
because two overlapping intervals of separate means cannot answer "does the stability beat texture"; no setting was changed and nothing was selected on it.**

**Target: \|NDVI error\|**

| dataset (units) | model | kept | reduction, stability order | reduction, texture order | stability − texture (paired) | reduction, oracle |
|---|---|---|---|---|---|---|
| `sen2neon_random30` (11) | Lite | 80% | 0.057 [0.027, 0.090] | 0.040 [0.015, 0.066] | +0.017 [+0.005, +0.030] | 0.401 [0.337, 0.489] |
|  | Lite | 60% | 0.071 [0.012, 0.127] | 0.042 [-0.020, 0.089] | +0.029 [+0.009, +0.051] | 0.579 [0.529, 0.641] |
|  | Lite | 40% | 0.067 [-0.056, 0.170] | 0.023 [-0.121, 0.125] | +0.043 [+0.011, +0.081] | 0.731 [0.702, 0.770] |
|  | Mamba | 80% | 0.049 [0.008, 0.088] | 0.047 [0.016, 0.079] | +0.001 [-0.020, +0.021] | 0.406 [0.341, 0.495] |
|  | Mamba | 60% | 0.063 [-0.025, 0.137] | 0.054 [-0.014, 0.116] | +0.009 [-0.022, +0.041] | 0.585 [0.533, 0.648] |
|  | Mamba | 40% | 0.048 [-0.122, 0.181] | 0.037 [-0.113, 0.154] | +0.011 [-0.027, +0.054] | 0.735 [0.706, 0.774] |
| `opensr_spot` (6) | Lite | 80% | 0.058 [0.023, 0.093] | 0.042 [0.014, 0.069] | +0.016 [+0.001, +0.030] | 0.353 [0.318, 0.398] |
|  | Lite | 60% | 0.111 [0.042, 0.184] | 0.075 [0.022, 0.128] | +0.036 [+0.014, +0.057] | 0.548 [0.514, 0.591] |
|  | Lite | 40% | 0.161 [0.056, 0.281] | 0.108 [0.022, 0.201] | +0.054 [+0.019, +0.086] | 0.709 [0.684, 0.741] |
|  | Mamba | 80% | 0.077 [0.036, 0.121] | 0.052 [0.025, 0.077] | +0.025 [+0.002, +0.053] | 0.361 [0.325, 0.405] |
|  | Mamba | 60% | 0.135 [0.059, 0.209] | 0.091 [0.038, 0.144] | +0.044 [+0.017, +0.073] | 0.556 [0.521, 0.599] |
|  | Mamba | 40% | 0.192 [0.065, 0.329] | 0.129 [0.044, 0.222] | +0.064 [+0.007, +0.113] | 0.716 [0.690, 0.748] |
| `opensr_spain_crops` (5) | Lite | 80% | 0.067 [0.003, 0.129] | 0.056 [0.004, 0.112] | +0.011 [-0.002, +0.030] | 0.370 [0.313, 0.421] |
|  | Lite | 60% | 0.087 [-0.017, 0.191] | 0.072 [-0.015, 0.159] | +0.015 [-0.004, +0.042] | 0.559 [0.494, 0.609] |
|  | Lite | 40% | 0.096 [-0.031, 0.235] | 0.064 [-0.058, 0.191] | +0.032 [+0.013, +0.052] | 0.715 [0.659, 0.756] |
|  | Mamba | 80% | 0.073 [0.008, 0.138] | 0.060 [0.010, 0.119] | +0.013 [-0.017, +0.054] | 0.373 [0.317, 0.423] |
|  | Mamba | 60% | 0.093 [-0.012, 0.203] | 0.080 [-0.006, 0.166] | +0.013 [-0.017, +0.058] | 0.562 [0.500, 0.611] |
|  | Mamba | 40% | 0.099 [-0.035, 0.232] | 0.076 [-0.047, 0.199] | +0.022 [-0.002, +0.055] | 0.717 [0.664, 0.758] |
| `opensr_spain_urban` (4) | Lite | 80% | 0.075 desc. | 0.050 desc. | +0.025 desc. | 0.368 desc. |
|  | Lite | 60% | 0.129 desc. | 0.091 desc. | +0.038 desc. | 0.556 desc. |
|  | Lite | 40% | 0.170 desc. | 0.139 desc. | +0.031 desc. | 0.711 desc. |
|  | Mamba | 80% | 0.083 desc. | 0.056 desc. | +0.027 desc. | 0.371 desc. |
|  | Mamba | 60% | 0.139 desc. | 0.104 desc. | +0.035 desc. | 0.560 desc. |
|  | Mamba | 40% | 0.185 desc. | 0.155 desc. | +0.030 desc. | 0.714 desc. |

**Target: decision disagreement, dropping 20 %** (zero for most regions, so the curves saturate quickly and the oracle reaches 1.0)

| dataset (units) | model | kept | reduction, stability order | reduction, texture order | stability − texture (paired) | reduction, oracle |
|---|---|---|---|---|---|---|
| `sen2neon_random30` (11) | Lite | 80% | 0.352 [0.131, 0.605] | 0.267 [0.100, 0.448] | +0.085 [-0.040, +0.271] | 0.948 [0.844, 1.000] |
|  | Mamba | 80% | 0.314 [0.088, 0.571] | 0.262 [0.100, 0.427] | +0.053 [-0.074, +0.229] | 0.948 [0.844, 1.000] |
| `opensr_spot` (6) | Lite | 80% | 0.543 desc. | 0.352 desc. | +0.191 desc. | 0.903 desc. |
|  | Mamba | 80% | 0.531 desc. | 0.346 desc. | +0.184 desc. | 0.893 desc. |
| `opensr_spain_crops` (5) | Lite | 80% | 0.220 [-0.013, 0.552] | 0.186 [-0.011, 0.489] | +0.035 [-0.007, +0.076] | 0.885 [0.777, 0.992] |
|  | Mamba | 80% | 0.302 [0.062, 0.591] | 0.187 [-0.010, 0.489] | +0.116 [+0.035, +0.226] | 0.885 [0.778, 0.992] |
| `opensr_spain_urban` (4) | Lite | 80% | 0.161 desc. | 0.129 desc. | +0.031 desc. | 0.904 desc. |
|  | Mamba | 80% | 0.230 desc. | 0.133 desc. | +0.097 desc. | 0.903 desc. |

* Dropping the 20 % most unstable regions removes **about 5–8 % of the mean \|NDVI error\|** (the oracle removes 35–41 %); the texture order removes 4–6 %. The gain is real but modest.
* **The stability order beats the texture order by a small margin in the point estimates for \|NDVI error\| in all 24 cells** (+0.001 to +0.064 in the reduction). The interval excludes zero for SEN2NEON Lite (all three retentions, Wilcoxon
  p = 0.03 / 0.02 / 0.04 over 11 units), for `spot` (both models; only 6 units, exact Wilcoxon p ≥ 0.06) and for `spain_crops` Lite at 40 %; it includes zero for SEN2NEON Mamba and for `spain_crops` Mamba. With 48 comparisons and no
  correction, this is **weak evidence of a small advantage of the stability ordering over texture on this target, dataset- and model-dependent**, not a demonstrated one.
* For the disagreement target the point difference is positive in 23 of 24 cells, its interval excludes zero only for `spain_crops` Mamba (5 units, no test possible), and includes zero on SEN2NEON (11 units).

### 8.5 What the evidence supports, and what it does not

**Supported (descriptive, non-causal, per dataset):**

* On registration-checked evidence, super-resolution with either model leaves the region-level NDVI and the vegetation decision **close to those of the bicubic baseline and of the LR input**: differences are ≈ 0.001 NDVI and
  ≤ 0.8 percentage points of decisions, against an error floor of 0.02–0.05 NDVI and 3–9 % disagreement set largely by the cross-sensor reference.
* The direction of the change is **dataset-dependent** (slightly worse than bicubic on SEN2NEON and `spot`; slightly better on `spain_crops` and, descriptively, `spain_urban`) and not separable from that dataset's sensor/registration
  characteristics with four datasets.
* The stability carries **a small amount of information about downstream error beyond texture** in some cells (partial correlations up to ≈ 0.17; a 1–6 point larger risk reduction than the texture order for \|NDVI error\|).

**Not supported:**

* That super-resolution **improves** the reliability of a downstream decision — not shown; the result is mixed to null.
* That either model is **better** than the other or than bicubic — no ranking is made; the differences are within a fraction of a percentage point and change sign with the dataset.
* That the stability is **calibrated, a confidence, or a usable threshold for accepting a region** — it is an ordering with a weak, texture-confounded link to the downstream error (Phase 6 measured its magnitude 16–32× smaller than the error).
* That the stability **causes** or explains the error, or that the residual after controlling for texture and added detail is independent of other image properties.
* Anything about **land cover** or **Indian regions** (§7).
* Anything **pooled across datasets**, or a statement with an interval for `spain_urban` (4 units).

## 9. Limitations

* **Few eligible scene units.** The registration gate leaves 11 / 6 / 5 / 4 units; 40 % of the SEN2NEON candidate regions are excluded with their tiles. Intervals from 5–6 units are wide and bootstrap-based; `spain_urban` is descriptive only;
  no development/test split was possible (needs ≥ 12 units).
* **The reference is a different sensor on a different date.** NDVI and the threshold decision inherit its radiometry and its own classification noise (it is not ground truth); this is the floor of §8.2. Residual misregistration up
  to the gate tolerance (0.5 HR px) remains, and OpenSR-Test tiles needed a recorded whole-pixel translation that assumes one global displacement.
* **Geographic scope.** North America (SEN2NEON), Spain and a SPOT-based subset (OpenSR-Test). Nothing Indian.
* **One index, one decision rule, one threshold family, two models, one perturbation set.** The threshold 0.3 is conventional and predeclared, not validated for these landscapes; the sensitivity thresholds (0.2, 0.4) show the dependence. NDVI
  is not land cover.
* **Regions are a modelling choice.** 10 m (one LR pixel) and 40 m cells; the requirements define no analysis region. At 10 m the hard constraint leaves little room for the region mean to change, so the region-level comparison is
  conservative about what SR adds inside a pixel; the pixel-level decision disagreement is where sub-pixel content shows.
* **Multiplicity.** The tables contain many intervals (2 models × 4 datasets × 2 targets × several statistics); none is corrected. A few "significant" cells are expected by chance.
* **Confounding is reduced, not removed.** Texture and added detail were controlled by partial rank correlation; brightness, land-cover type and sensor differences were not.
* **Two-pass run and a dirty working tree.** See §8; commit before quoting the git revision.

## 10. Tests and how to reproduce

```bash
TMPDIR=~/.cache/frame_tmp sen2sr_venv/bin/python -m pytest frame/tests/test_downstream_*.py --basetemp=~/.cache/frame_pytest_tmp     # the Phase 7 unit tests (no GPU, no dataset)
sen2sr_venv/bin/python -m frame.downstream smoke                                                                                     # synthetic end-to-end run
sen2sr_venv/bin/python -m frame.downstream check experiments/downstream/configs/downstream_v1.json                                   # gate + region preview, no model
sen2sr_venv/bin/python -m frame.downstream run   experiments/downstream/configs/downstream_v1.json                                   # the real run (about 40 minutes); the output directory must not exist
```

The unit tests cover: config strictness and the declared threshold policy; NDVI (bands by name, formula, mask rule, undefined denominators); regions (grid alignment after a recorded translation, valid-pixel rule, exclusion counts,
mixed sizes refused); decisions (confusion counts, rates that need a missing class, balanced accuracy); fidelity metrics and paired differences; association (known relationships: informative, null, texture-confounded, stability
beyond texture; unit counting; pooled and across-tile behaviour) and risk-coverage (including the paired stability-versus-texture difference); the runner (identical pixels for every system, model-failure exclusion, cache digest,
no reuse across a different model); provenance; the report; the CLI. **The regression that matters — an ineligible reference cannot enter the primary analysis — counts the calls to the NDVI and region code and checks the cache, the tile rows and the unit counts; a companion test counts the model calls
(none for an ineligible tile).** The Phase 6 code touched by Phase 7 (a gate version string, a public single-tile TTA function, three public aliases) has its own tests, and the Phase 6 behaviour is unchanged.
