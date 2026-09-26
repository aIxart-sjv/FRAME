# FRAME evaluation layer (Phase 5)

`frame/evaluate/` produces independent, auditable evidence about reconstruction accuracy, spectral fidelity and spatial correctness, and records exactly what was
evaluated. It **measures**; it does not train, does not rank systems, and does not merge synthetic and real results.

```
dataset ─▶ reference / geometry validation ─▶ frozen system ─▶ inference ─▶ mask-aware metrics ─▶ aggregation ─▶ statistics ─▶ provenance
(manifest    role safety, strict evaluation     bicubic, Lite,    tile engine     per sample × system     over scene    bootstrap CI,   git, weights
 / subset)   mask, invalid ones LISTED          Mamba, frozen     (frame.tiling)  in separate groups      units, not    paired tests,   hashes, digests,
                                                checkpoints                                               pixels        labels          configs
```

```bash
python -m frame.evaluate check experiments/evaluation/configs/benchmarks_v1.json    # validate config, datasets (roles, geometry, files), systems; evaluate nothing
python -m frame.evaluate run   experiments/evaluation/configs/benchmarks_v1.json    # writes experiments/evaluation/benchmarks_v1/
```

Exit codes: `0` success; `1` finished but nothing could be evaluated for some dataset; `2` refused or invalid (bad config, role violation, train/eval overlap, existing results). Relative paths in a
config resolve against the config file's directory; results go to `output_dir` (`config.json`, `metrics.jsonl`, `aggregates.json`, `summary.json`, `README.md`; large rasters are never written).

## 0. What existed, what was reused, what was missing

The audit of the existing code found a sound single-sample toolbox and no evaluation layer:

| Existing | Reused as |
|---|---|
| `frame.validation.compute_reference_metrics` (PSNR, SSIM, RMSE, SAM, ERGAS; SAM and RMSE via opensr-test's own `SAD` / `L2`) | PSNR, RMSE, SAM, ERGAS are called unchanged. SSIM is **recomputed**: the existing function evaluates SSIM over the whole image and masks the resulting map afterwards, so a window near nodata still contains nodata values; the new one only scores windows that lie entirely inside the valid mask. |
| `frame.validation.bicubic_upsample` | the bicubic baseline, unchanged |
| `frame.validation.compute_opensr_test_metrics` | the `opensr_native` group, unchanged (its own vocabulary, no mask) |
| `frame.consistency` (SR → area-average → compare with LR) | the self-consistency group, unchanged, in its own table |
| `frame.data` adapters, manifests, roles, geometry checks (Phase 3) | dataset loading, role safety, geometry validation, checksums |
| `frame.tiling` (Phase 2), `frame.models` (Phase 1), `frame.train` checkpoints (Phase 4) | inference of all network systems; provenance (weights hashes, checkpoint metadata) |
| `frame.analysis.indices` | not reused: it computes NDVI of one stack (Phase 6 of the original plan); the evaluation needs NDVI and NDWI of the SR **and** the reference on the same pixels, with an exclusion rule decided from the reference alone |

Missing before Phase 5: a multi-sample, multi-system, multi-dataset run; a strict valid-pixel rule (a spectrum with a zero in one band was scored); per-band and detail metrics; any measure of spatial correctness beyond PSNR/SSIM; aggregation over scenes and any
statistics; provenance per result; role/overlap safety at evaluation time; a record of which samples were skipped or invalid. `frame/validation/` and `frame/consistency/` were not modified.

## 1. What is evaluated

**Systems** (all frozen; `frame/evaluate/systems.py`):

| System | What it is | Hard constraint | Tiling |
|---|---|---|---|
| `bicubic` | `frame.validation.bicubic_upsample` (antialiased bicubic, negatives clamped) | n/a | whole scene |
| `sen2sr_lite` | the published SEN2SR-Lite inference system (CNN + clamp ≥ 0 + low-frequency Fourier constraint), weights hashed | **yes** | tile engine |
| `sen2sr_lite_no_constraint` | the same CNN with only the clamp; an ablation, **not** the intended system | **no** | tile engine |
| `sen2sr_mamba` | SEN2SR-Mamba RGBN through its isolated worker (`frame.models`), weights and worker load report recorded | **yes** | tile engine |
| `tiny_cnn_seed0…4` | the Phase 4 primary model (47,424 params, trained on **synthetic** scenes only) for five seeds, checkpoints outside the repo | no | tile engine |

All network systems use the same tile engine and configuration (128 px tiles, 32 px overlap, reflect padding, linear blending); the hard constraint acts inside each 512×512 tile output as designed.
The input convention is common: reflectance fraction = stored DN / the dataset's reflectance scale (10,000), bands B04, B03, B02, B08 selected **by name**, no BOA offset, no clipping by FRAME.
A checkpoint system is refused if its training summary shows it saw the evaluated scenes or was trained on the benchmark (`train_eval_overlap`, `trained_on_benchmark`).
The Mamba worker serves only the constrained system, so a Mamba ablation without the constraint was **not** evaluated.

**Datasets** (`frame/evaluate/datasets.py`), each with an evidence class that is never mixed:

| Dataset | Class | Role | Records | Scene unit | Own categories |
|---|---|---|---:|---|---|
| `sen2neon_random30` | real cross-sensor | independent benchmark, test only | 30 (seeded uniform draw, seed 0, no filtering) | NEON acquisition (28 units) | NEON land-cover superclass |
| `sen2neon_phase3_sample` | real cross-sensor | independent benchmark | 3 (hand-picked in Phase 3; reported separately, smoke only) | acquisition (2) | same |
| `opensr_spot`, `opensr_spain_crops`, `opensr_spain_urban` | real cross-sensor | independent benchmark, test only | 9 / 28 / 20 | SPOT ROI (9) / source orthophoto (5, 4) | subset: mixed / crops / urban |
| `synthetic_smoke_test` | synthetic | held-out test split of FRAME's own smoke set | 2 | region (1) | none |

`sen2naipv2` **cannot** be evaluated: SEN2SR was trained on it, so it is not independent of the systems under test (the config refuses it). Benchmark records labelled train/val refuse the whole
evaluation. OpenSR-Test uses the harmonised reference (`HRharm`, the benchmark's own default); SEN2NEON's LR is a convenience 10 m grid of all 12 bands (the 20 m / 60 m bands were not measured at 10 m).

## 2. Valid pixels and nodata

*An HR pixel is scored only if it is not all-band nodata (the dataset's rule), none of the evaluated bands equals the HR nodata value, and all evaluated values are finite.* The dataset's own rule alone is not enough:
on the seeded random SEN2NEON sample **21 of 30 tiles** contain "valid" pixels with a zero in *some* bands (up to 1.0% of a tile's pixels), mostly around nodata boundaries, and 16 of 30 tiles have real nodata (up to 48% of the tile); such a spectrum is not valid for SAM / ERGAS / indices and
made SAM undefined before the stricter rule existed. OpenSR-Test and the synthetic set contain no nodata at all. Both fractions are recorded per sample (`quality.hr_dataset_nodata_fraction`, `hr_partial_nodata_fraction`, `hr_valid_fraction`). A tile with strict valid fraction below
`min_valid_fraction` (0.05) is **skipped and listed** with the rule; nothing is silently dropped. Windowed metrics (SSIM, high-frequency filter, gradients) additionally erode the mask by their window radius
(border counted invalid) so no window ever touches nodata. Junk placed under the mask changes no metric (tested).

## 3. Metrics (all in `frame/evaluate/metrics.py`; groups are never merged)

**Reference accuracy** (SR vs the independent HR reference; PSNR, RMSE, SAM and ERGAS reuse `frame.validation.compute_reference_metrics`, which reuses opensr-test's own distances):
PSNR (data range 1.0, reflectance as a fraction; infinite PSNR is reported as missing), SSIM (skimage, eroded mask), RMSE, MAE, SAM (degrees), ERGAS, and **per band** RMSE / MAE / bias / percent bias / PSNR / Pearson r / SSIM.
SAM is a float32 per-pixel arccos, so identical spectra score ~0.006° (a measured numerical floor, negligible next to real values).
*Behaviour differs by evidence class:* for synthetic pairs the reference is the ground truth the LR was made from; for real cross-sensor pairs it is a **different sensor** with its own radiometry and registration error, so no
number is an absolute error against truth.

**Derived indices and ratios** (vs the HR reference; MAE, RMSE, bias, Pearson r, Wasserstein-1 distance of the two distributions): NDVI = (B08−B04)/(B08+B04) and NDWI = (B03−B08)/(B03+B08), scored where the *HR* reflectance sum
exceeds 0.02 (decided from the reference only, so every system is scored on the same pixels; the excluded fraction is reported). MNDWI, NDBI and BSI need SWIR B11, which the RGBN systems do not produce, and are listed as
`skipped` with the reason, never approximated. Band ratios B08/B04 and B03/B04 are compared as log-ratios (bias, MAE, median |error|) where both HR bands exceed 0.005.

**Spatial / detail correctness** (vs the HR reference): high-frequency detail is `x − gaussian_blur(x, σ = 2 HR px)` per band (finer than half an LR pixel). `hf_correlation`; `hf_energy_ratio` = Σ SR_hf² / Σ HR_hf² (how *much* detail, not
whether it is right); `hf_relative_error` = rms(SR_hf − HR_hf) / rms(HR_hf), where **exactly 1.0 is the score of adding no detail at all**, below 1 the added detail matches the reference and above 1 it is worse than none;
`gradient_correlation` of Sobel edge magnitudes; `phase_shift_px`: the global displacement (dy, dx) of the SR relative to the reference by cross-correlation of the band-mean images (1/10 px; masked correlation when there is nodata;
**not** phase-whitened, see §7): non-zero means misregistration of the SR and/or of the reference itself, and a constant image is reported as not computable instead of receiving an arbitrary shift. `seam_error`: reference-based |SR − HR| within 8 px of tile seams (the middle of each blend overlap) versus the interior; complements `frame.tiling`'s no-reference seam diagnostic.

**What was added, judged against the reference** (`hallucination_analysis`; defined for a system relative to the bicubic baseline, so not for bicubic itself): per valid (band, pixel) element, `A = SR − bicubic`, `T = HR − bicubic`, threshold τ = 0.005 reflectance
(sensitivity at 0.0025 and 0.01 is in the outputs). The elements are partitioned into **supported synthesis** (added, needed, same sign as T, error reduced), **unsupported detail** (added but not confirmed; sub-breakdown into wrong
direction / overshoot / where the reference has none), **omission** (the reference has detail and none was added) and **neutral**; plus `mse_skill_vs_baseline` = 1 − MSE_SR/MSE_bicubic. It uses no semantics and invents no
labels; "unsupported" means "not confirmed by this reference", not "false". It is a quantitative, element-wise description; **no visual inspection was scored**.

**Self-consistency** (`frame.consistency`): SR → area-average downsample → compare with the LR (per band, pooled, NDVI, B08/B04). **This is not accuracy against an HR reference**: a system can average back to the LR exactly and still be far from the
reference (tested). It is reported in its own table and never combined with reference accuracy. **opensr-test native** values (reflectance / spectral / spatial consistency, synthesis, hallucination, omission, improvement) come from the
installed package with its default config and are their own group; they take no mask, so they are computed only for fully valid references and otherwise recorded as `not_computed`.

## 4. Statistics (`frame/evaluate/stats.py`)

Requirements 142 (Req. 9 §28-32): pixels are not independent samples and tiles of one scene are correlated, so **the unit of analysis is the scene** (NEON acquisition / source orthophoto / region): each metric is averaged within a unit first,
and statistics are taken over units, with sample-level (tile) descriptives beside them. Reported: n, mean ± standard deviation, median; percentile bootstrap CI over units (10,000 resamples, seeded); for comparisons the paired
difference A − B of unit means, its paired-bootstrap CI, a Wilcoxon signed-rank test and an effect size (d_z). **Below 5 units no interval, and below 6 pairs no test, is reported** (with n pairs the smallest exact two-sided
Wilcoxon p is 2/2ⁿ, which is 0.0625 for n = 5), and the result is labelled `descriptive_only` with the reason. No multiple-comparison correction is applied and none is implied; a table of many comparisons is exploratory. Paired
differences are always taken against the reference system (bicubic) or a named pair; they are not rankings. Seeds of one model are additionally summarised across seeds.

## 5. Reproducibility and provenance

Every `summary.json` records: git revision and dirty flag, config digest, metric configuration and version (`frame-eval-metrics/1`), tiling, environment, per-system provenance (model, weights and their SHA-256, hard constraint,
preprocessing, band order, tiling; for checkpoints also the training step, seed, manifest digest, git revision, config digest and the training summary's hash), per-dataset manifest digest, role, split, scene units, sample ids, the counts
of evaluated / skipped / invalid / unreadable samples with reasons, inference failures, and the evaluation matrix (system × dataset × bands × resolutions × hard constraint × tiling × weights × metric configuration). Two runs of
one configuration produce identical metrics (tested).

## 6. Spatial-shift sensitivity (requirements 142 §27)

```bash
python -m frame.evaluate shift CONFIG.json --dataset NAME [--systems A B ...] [--lr-shifts 0 0.25 0.5 1 2] [--output-dir DIR]
```

Real cross-sensor references are not registered to the Sentinel-2 grid (§7 measures how badly), and a pixel-wise metric punishes a correctly placed detail that is one pixel off as much as an invented one. The `shift` command
measures that directly: the SR of each system is **displaced** against the reference by known amounts and the same accuracy, NDVI and detail metrics are recomputed on the overlap. The default sweep is the requirements' list, 0, 0.25, 0.5, 1 and 2
LR pixels = 0, 1, 2, 4 and 8 HR pixels at x4. Shifts are **whole HR pixels along the column axis, nothing is resampled** (a shift that is not a whole number of HR pixels is refused, so no interpolation smoothing enters the comparison).
The extra condition `aligned_to_bicubic` removes the whole-pixel displacement that cross-correlation finds between the **bicubic baseline** and the reference: a system-neutral estimate, identical for every system on a sample (no system aligns itself).
It goes through the same dataset-role and train/eval-overlap checks as `run`, writes `config.json`, `rows.jsonl`, `aggregates.json`, `summary.json`, `README.md` and refuses an existing directory. Classification accuracy and area error, which §27 also names, need a
downstream task and were **not** computed. It is an interpretation aid: it says how large a difference misregistration alone produces, so that a difference between systems can be read against it. It is not a ranking.

## 7. Defects found while building the evaluation

These were found by auditing real results, not by unit tests alone; each has a regression test.

1. **Phase-whitened registration estimate was wrong on real tiles.** The first real run reported a median global displacement of about 1 HR pixel (mean magnitude 0.98) for bicubic on SEN2NEON. A brute-force search (best whole-pixel displacement by pooled MSE over the valid overlap)
   disagreed on the tiles with no nodata: e.g. 2021_JORN_4__5_4 was reported as (0, 0) while displacing by (-3, -1) HR px lowered the RMSE by 8.9%. skimage's default whitening gives every frequency the same weight, and a blurry SR has no energy where the sharper
   reference does, so those frequencies become noise. Without whitening the estimate agreed with the search within 1 px on **87 of 87** tiles (SEN2NEON random-30, OpenSR spot / spain_crops / spain_urban). The first run was discarded and repeated with the corrected estimator.
   A constant image also received a made-up shift of (1, 1) with status `ok`; it is now `not_computable`.
2. **Rows could not be attributed to their configured dataset.** `sample_result` rows carried the dataset *kind* (`sen2neon`, `opensr_test`), so the SEN2NEON random-30 rows could not be told from the Phase 3 sample's rows in `metrics.jsonl`. They now carry the configured name
   (`dataset`) and the kind (`dataset_kind`); skipped / invalid / unreadable / failed rows already did.
3. **Strict valid mask.** 21 of 30 random SEN2NEON tiles contain 'valid' pixels with a zero in some bands (§2); without the strict mask SAM was undefined.
4. **SAM float32 floor.** Identical inputs score about 0.006 degrees, the numerical floor of the reused opensr-test spectral angle; tests allow 0.02.
5. **tacoreader 0.4.5 returns the same rows in a different order on every load** (measured), so an index drawn from the loaded order is not reproducible; the SEN2NAIPv2 helper sorts by the unique `tortilla:id` before its seeded draw.
6. **A test that compares a self-consistency claim can pass for the wrong reason** (the README merely contained the word); the separation of self-consistency from HR accuracy is now checked structurally.
7. **The registration shift of tiles WITH nodata has whole-pixel resolution** (found in Phase 6). With nodata the estimator falls back to a masked correlation that returns integers, so the `shift (HR px)` column of §8.1 and the `aligned_to_bicubic` correction of §6 are integer-valued
   for the 16 of 30 SEN2NEON tiles that contain nodata (a true half-pixel displacement reads as 0 or ±1; a one-pixel correction can then move it to ∓1). The OpenSR-Test tiles and the synthetic set have no nodata and 0.1 px resolution. The qualitative conclusions of §8.4 do not depend on it (the displacement sweep is unaffected and the median displacement
   stays about one HR pixel), but the per-tile numbers are coarse. Phase 6 adds a sub-pixel refinement of the correlation peak (`frame.reliability.alignment.estimate_displacement`, validated against a brute-force search on 87 of 87 real tiles) and uses it for its registration gate; the Phase 5 outputs were not recomputed.

## 8. Results (measured; descriptive; **not a ranking**)

Record: `experiments/evaluation/benchmarks_v1/` (`config.json`, `metrics.jsonl` with one row per sample × system, `aggregates.json`, `summary.json` with the provenance of every system and dataset, `README.md` with every table). Run on the RTX 3050 laptop GPU (torch 2.14, ~35 minutes).
Git revision `eb37695` **with a dirty working tree** (Phases 1-5 are uncommitted; the summary records the flag): commit before quoting the numbers. Every number is the mean over **scene units** (tiles of one scene are averaged first); standard deviations,
sample-level descriptives and bootstrap intervals are in `aggregates.json`. Datasets and evidence classes are never pooled: there is deliberately no combined score.

| Dataset (evidence class) | Samples | Scene units | Invalid / skipped / unreadable / failed | Inference tiles per network system |
|---|---:|---:|---|---:|
| SEN2NEON seeded random 30 (real cross-sensor) | 30 | 28 | 0 / 0 / 0 / 0 | 270 (9 per 1024² scene) |
| SEN2NEON Phase 3 hand-picked (real cross-sensor; smoke only) | 3 | 2 | 0 / 0 / 0 / 0 | 27 |
| OpenSR-Test `spot` (real cross-sensor) | 9 | 9 | 0 / 0 / 0 / 0 | 9 |
| OpenSR-Test `spain_crops` (real cross-sensor) | 28 | 5 | 0 / 0 / 0 / 0 | 28 |
| OpenSR-Test `spain_urban` (real cross-sensor) | 20 | 4 | 0 / 0 / 0 / 0 | 20 |
| synthetic smoke set, held-out `test` split (synthetic) | 2 | 1 | 0 / 0 / 0 / 0 | 2 |

92 samples × 9 systems = 828 result rows; all systems were available; the strict mask left every tile above the 5% threshold (lowest valid fraction 0.517).
Metric definitions: §3. The systems and their weights: §1 and `summary.json`. All numbers below use HR-pixel reflectance as a fraction (data range 1).

### 8.1 The numbers

Mean over scene units; the last row of each table is the mean ± standard deviation **across the five seeds** of the tiny model (trained on synthetic data only). ↑ / ↓ only say which direction is smaller error or closer agreement.
"detail rel. error" is 1.0 for adding no detail at all; "shift" is the displacement of the SR against the reference (§3, §7).

#### SEN2NEON, seeded random 30 tiles: 30 samples, 28 scene units

Reference accuracy:

| system | PSNR (dB) | SSIM | RMSE | MAE | SAM (°) | ERGAS |
|---|---|---|---|---|---|---|
| bicubic | 33.6867 | 0.9188 | 0.0271 | 0.0176 | 3.1892 | 5.5151 |
| SEN2SR-Lite (hard constraint) | 33.2307 | 0.9034 | 0.0282 | 0.0184 | 3.3482 | 5.7883 |
| SEN2SR-Lite, no hard constraint (ablation) | 32.1263 | 0.8799 | 0.0314 | 0.0212 | 4.0892 | 6.7321 |
| SEN2SR-Mamba (hard constraint) | 33.2234 | 0.9077 | 0.0282 | 0.0183 | 3.2623 | 5.6290 |
| tiny model, 5 seeds (mean ± sd across seeds) | 32.8705 ± 0.0627 | 0.8966 ± 0.0022 | 0.0293 ± 0.0002 | 0.0192 ± 0.0001 | 3.5246 ± 0.0207 | 6.1952 ± 0.0420 |

Spectral fidelity (indices vs the reference, per-band RMSE):

| system | NDVI MAE | NDWI MAE | log(B08/B04) bias | B02 RMSE | B03 RMSE | B04 RMSE | B08 RMSE |
|---|---|---|---|---|---|---|---|
| bicubic | 0.0541 | 0.0419 | 0.0217 | 0.0168 | 0.0193 | 0.0226 | 0.0380 |
| SEN2SR-Lite (hard constraint) | 0.0567 | 0.0442 | 0.0245 | 0.0178 | 0.0203 | 0.0238 | 0.0393 |
| SEN2SR-Lite, no hard constraint (ablation) | 0.0715 | 0.0538 | 0.1275 | 0.0201 | 0.0227 | 0.0270 | 0.0434 |
| SEN2SR-Mamba (hard constraint) | 0.0551 | 0.0433 | 0.0177 | 0.0176 | 0.0200 | 0.0234 | 0.0396 |
| tiny model, 5 seeds (mean ± sd across seeds) | 0.0595 ± 0.0003 | 0.0466 ± 0.0003 | 0.0290 ± 0.0018 | 0.0194 ± 0.0001 | 0.0217 ± 0.0001 | 0.0255 ± 0.0002 | 0.0398 ± 0.0003 |

Spatial detail (vs the reference):

| system | detail rel. error | detail corr. | detail energy ratio | edge corr. | shift (HR px) |
|---|---|---|---|---|---|
| bicubic | 1.138 | 0.529 | 1.828 | 0.593 | 1.503 |
| SEN2SR-Lite (hard constraint) | 1.748 | 0.463 | 4.712 | 0.549 | 1.421 |
| SEN2SR-Lite, no hard constraint (ablation) | 2.007 | 0.468 | 6.315 | 0.549 | 1.421 |
| SEN2SR-Mamba (hard constraint) | 1.833 | 0.415 | 4.877 | 0.513 | 1.439 |
| tiny model, 5 seeds (mean ± sd across seeds) | 1.978 ± 0.077 | 0.416 ± 0.012 | 5.945 ± 0.451 | 0.521 ± 0.008 | 1.426 ± 0.037 |

#### SEN2NEON, Phase 3 hand-picked 3 tiles (smoke only): 3 samples, 2 scene units

Reference accuracy:

| system | PSNR (dB) | SSIM | RMSE | MAE | SAM (°) | ERGAS |
|---|---|---|---|---|---|---|
| bicubic | 36.3391 | 0.9739 | 0.0156 | 0.0080 | 0.9316 | 2.3066 |
| SEN2SR-Lite (hard constraint) | 35.7513 | 0.9608 | 0.0166 | 0.0085 | 1.0016 | 2.5636 |
| SEN2SR-Lite, no hard constraint (ablation) | 34.5531 | 0.9423 | 0.0190 | 0.0110 | 1.5409 | 3.8746 |
| SEN2SR-Mamba (hard constraint) | 35.6728 | 0.9609 | 0.0167 | 0.0084 | 0.9968 | 2.4994 |
| tiny model, 5 seeds (mean ± sd across seeds) | 35.5297 ± 0.1131 | 0.9639 ± 0.0022 | 0.0169 ± 0.0002 | 0.0086 ± 0.0001 | 1.0768 ± 0.0223 | 3.0326 ± 0.0442 |

Spectral fidelity (indices vs the reference, per-band RMSE):

| system | NDVI MAE | NDWI MAE | log(B08/B04) bias | B02 RMSE | B03 RMSE | B04 RMSE | B08 RMSE |
|---|---|---|---|---|---|---|---|
| bicubic | 0.0153 | 0.0149 | 0.0015 | 0.0037 | 0.0052 | 0.0048 | 0.0299 |
| SEN2SR-Lite (hard constraint) | 0.0165 | 0.0160 | 0.0025 | 0.0043 | 0.0059 | 0.0055 | 0.0316 |
| SEN2SR-Lite, no hard constraint (ablation) | 0.0310 | 0.0220 | 0.2143 | 0.0059 | 0.0073 | 0.0087 | 0.0354 |
| SEN2SR-Mamba (hard constraint) | 0.0163 | 0.0160 | -0.0005 | 0.0043 | 0.0058 | 0.0054 | 0.0318 |
| tiny model, 5 seeds (mean ± sd across seeds) | 0.0175 ± 0.0003 | 0.0167 ± 0.0002 | 0.0063 ± 0.0016 | 0.0058 ± 0.0001 | 0.0070 ± 0.0001 | 0.0070 ± 0.0001 | 0.0312 ± 0.0005 |

Spatial detail (vs the reference):

| system | detail rel. error | detail corr. | detail energy ratio | edge corr. | shift (HR px) |
|---|---|---|---|---|---|
| bicubic | 1.008 | 0.689 | 1.791 | 0.639 | 1.548 |
| SEN2SR-Lite (hard constraint) | 1.635 | 0.591 | 4.710 | 0.581 | 1.336 |
| SEN2SR-Lite, no hard constraint (ablation) | 1.912 | 0.598 | 6.429 | 0.582 | 1.336 |
| SEN2SR-Mamba (hard constraint) | 1.737 | 0.519 | 4.898 | 0.542 | 1.401 |
| tiny model, 5 seeds (mean ± sd across seeds) | 1.466 ± 0.088 | 0.536 ± 0.010 | 3.357 ± 0.444 | 0.552 ± 0.003 | 1.480 ± 0.132 |

#### OpenSR-Test `spot`: 9 samples, 9 scene units

Reference accuracy:

| system | PSNR (dB) | SSIM | RMSE | MAE | SAM (°) | ERGAS |
|---|---|---|---|---|---|---|
| bicubic | 33.1171 | 0.8176 | 0.0261 | 0.0183 | 1.8458 | 3.7674 |
| SEN2SR-Lite (hard constraint) | 33.2229 | 0.8307 | 0.0258 | 0.0180 | 2.0710 | 3.6846 |
| SEN2SR-Lite, no hard constraint (ablation) | 32.4468 | 0.8295 | 0.0279 | 0.0197 | 2.6792 | 3.9759 |
| SEN2SR-Mamba (hard constraint) | 33.1966 | 0.8248 | 0.0261 | 0.0182 | 2.0153 | 3.7279 |
| tiny model, 5 seeds (mean ± sd across seeds) | 32.8963 ± 0.0442 | 0.8185 ± 0.0012 | 0.0272 ± 0.0002 | 0.0191 ± 0.0001 | 2.4403 ± 0.0279 | 3.9657 ± 0.0304 |

Spectral fidelity (indices vs the reference, per-band RMSE):

| system | NDVI MAE | NDWI MAE | log(B08/B04) bias | B02 RMSE | B03 RMSE | B04 RMSE | B08 RMSE |
|---|---|---|---|---|---|---|---|
| bicubic | 0.0253 | 0.0253 | -0.0008 | 0.0185 | 0.0217 | 0.0260 | 0.0345 |
| SEN2SR-Lite (hard constraint) | 0.0295 | 0.0282 | 0.0017 | 0.0181 | 0.0210 | 0.0253 | 0.0347 |
| SEN2SR-Lite, no hard constraint (ablation) | 0.0415 | 0.0374 | -0.0222 | 0.0196 | 0.0230 | 0.0278 | 0.0368 |
| SEN2SR-Mamba (hard constraint) | 0.0282 | 0.0275 | -0.0027 | 0.0183 | 0.0212 | 0.0255 | 0.0351 |
| tiny model, 5 seeds (mean ± sd across seeds) | 0.0357 ± 0.0005 | 0.0337 ± 0.0004 | 0.0092 ± 0.0008 | 0.0197 ± 0.0001 | 0.0225 ± 0.0002 | 0.0272 ± 0.0002 | 0.0357 ± 0.0002 |

Spatial detail (vs the reference):

| system | detail rel. error | detail corr. | detail energy ratio | edge corr. | shift (HR px) |
|---|---|---|---|---|---|
| bicubic | 0.946 | 0.333 | 0.080 | 0.385 | 0.472 |
| SEN2SR-Lite (hard constraint) | 0.932 | 0.378 | 0.188 | 0.445 | 0.472 |
| SEN2SR-Lite, no hard constraint (ablation) | 0.936 | 0.383 | 0.251 | 0.450 | 0.472 |
| SEN2SR-Mamba (hard constraint) | 0.936 | 0.381 | 0.213 | 0.463 | 0.503 |
| tiny model, 5 seeds (mean ± sd across seeds) | 0.972 ± 0.007 | 0.330 ± 0.005 | 0.288 ± 0.026 | 0.410 ± 0.006 | 0.471 ± 0.010 |

#### OpenSR-Test `spain_crops`: 28 samples, 5 scene units

Reference accuracy:

| system | PSNR (dB) | SSIM | RMSE | MAE | SAM (°) | ERGAS |
|---|---|---|---|---|---|---|
| bicubic | 32.5305 | 0.8137 | 0.0242 | 0.0163 | 2.4172 | 3.9377 |
| SEN2SR-Lite (hard constraint) | 32.5494 | 0.8188 | 0.0242 | 0.0162 | 2.4420 | 3.9670 |
| SEN2SR-Lite, no hard constraint (ablation) | 32.0136 | 0.8149 | 0.0256 | 0.0177 | 2.8380 | 4.3230 |
| SEN2SR-Mamba (hard constraint) | 32.5509 | 0.8197 | 0.0242 | 0.0162 | 2.4332 | 3.9444 |
| tiny model, 5 seeds (mean ± sd across seeds) | 32.3540 ± 0.0271 | 0.8156 ± 0.0006 | 0.0247 ± 0.0001 | 0.0165 ± 0.0000 | 2.5293 ± 0.0110 | 4.1916 ± 0.0207 |

Spectral fidelity (indices vs the reference, per-band RMSE):

| system | NDVI MAE | NDWI MAE | log(B08/B04) bias | B02 RMSE | B03 RMSE | B04 RMSE | B08 RMSE |
|---|---|---|---|---|---|---|---|
| bicubic | 0.0391 | 0.0315 | -0.0009 | 0.0159 | 0.0192 | 0.0254 | 0.0315 |
| SEN2SR-Lite (hard constraint) | 0.0393 | 0.0318 | 0.0006 | 0.0159 | 0.0191 | 0.0251 | 0.0316 |
| SEN2SR-Lite, no hard constraint (ablation) | 0.0476 | 0.0374 | 0.0182 | 0.0171 | 0.0205 | 0.0268 | 0.0332 |
| SEN2SR-Mamba (hard constraint) | 0.0393 | 0.0316 | -0.0015 | 0.0159 | 0.0191 | 0.0251 | 0.0316 |
| tiny model, 5 seeds (mean ± sd across seeds) | 0.0405 ± 0.0002 | 0.0332 ± 0.0001 | 0.0037 ± 0.0009 | 0.0169 ± 0.0001 | 0.0199 ± 0.0001 | 0.0259 ± 0.0001 | 0.0318 ± 0.0001 |

Spatial detail (vs the reference):

| system | detail rel. error | detail corr. | detail energy ratio | edge corr. | shift (HR px) |
|---|---|---|---|---|---|
| bicubic | 0.974 | 0.225 | 0.049 | 0.404 | 1.494 |
| SEN2SR-Lite (hard constraint) | 0.976 | 0.240 | 0.110 | 0.431 | 1.544 |
| SEN2SR-Lite, no hard constraint (ablation) | 0.980 | 0.244 | 0.147 | 0.434 | 1.560 |
| SEN2SR-Mamba (hard constraint) | 0.979 | 0.233 | 0.126 | 0.435 | 1.575 |
| tiny model, 5 seeds (mean ± sd across seeds) | 0.998 ± 0.004 | 0.215 ± 0.004 | 0.185 ± 0.020 | 0.425 ± 0.003 | 1.563 ± 0.013 |

#### OpenSR-Test `spain_urban`: 20 samples, 4 scene units

Reference accuracy:

| system | PSNR (dB) | SSIM | RMSE | MAE | SAM (°) | ERGAS |
|---|---|---|---|---|---|---|
| bicubic | 29.2855 | 0.7001 | 0.0357 | 0.0247 | 3.9187 | 6.2182 |
| SEN2SR-Lite (hard constraint) | 29.2793 | 0.7103 | 0.0358 | 0.0246 | 3.9518 | 6.2690 |
| SEN2SR-Lite, no hard constraint (ablation) | 28.7937 | 0.7056 | 0.0378 | 0.0263 | 4.4526 | 6.6893 |
| SEN2SR-Mamba (hard constraint) | 29.2511 | 0.7086 | 0.0360 | 0.0246 | 3.9981 | 6.2652 |
| tiny model, 5 seeds (mean ± sd across seeds) | 28.8957 ± 0.0517 | 0.7003 ± 0.0009 | 0.0375 ± 0.0002 | 0.0257 ± 0.0001 | 4.2318 ± 0.0153 | 6.6950 ± 0.0400 |

Spectral fidelity (indices vs the reference, per-band RMSE):

| system | NDVI MAE | NDWI MAE | log(B08/B04) bias | B02 RMSE | B03 RMSE | B04 RMSE | B08 RMSE |
|---|---|---|---|---|---|---|---|
| bicubic | 0.0623 | 0.0517 | -0.0015 | 0.0272 | 0.0306 | 0.0380 | 0.0432 |
| SEN2SR-Lite (hard constraint) | 0.0628 | 0.0521 | 0.0018 | 0.0274 | 0.0307 | 0.0379 | 0.0436 |
| SEN2SR-Lite, no hard constraint (ablation) | 0.0735 | 0.0593 | 0.0188 | 0.0291 | 0.0325 | 0.0400 | 0.0459 |
| SEN2SR-Mamba (hard constraint) | 0.0634 | 0.0526 | -0.0032 | 0.0275 | 0.0307 | 0.0380 | 0.0438 |
| tiny model, 5 seeds (mean ± sd across seeds) | 0.0672 ± 0.0004 | 0.0564 ± 0.0002 | 0.0095 ± 0.0014 | 0.0295 ± 0.0002 | 0.0326 ± 0.0003 | 0.0401 ± 0.0003 | 0.0447 ± 0.0003 |

Spatial detail (vs the reference):

| system | detail rel. error | detail corr. | detail energy ratio | edge corr. | shift (HR px) |
|---|---|---|---|---|---|
| bicubic | 0.973 | 0.234 | 0.059 | 0.385 | 1.610 |
| SEN2SR-Lite (hard constraint) | 0.978 | 0.252 | 0.140 | 0.419 | 1.652 |
| SEN2SR-Lite, no hard constraint (ablation) | 0.985 | 0.256 | 0.189 | 0.422 | 1.678 |
| SEN2SR-Mamba (hard constraint) | 0.985 | 0.236 | 0.158 | 0.416 | 1.638 |
| tiny model, 5 seeds (mean ± sd across seeds) | 1.019 ± 0.007 | 0.222 ± 0.003 | 0.266 ± 0.024 | 0.423 ± 0.002 | 1.662 ± 0.012 |

#### synthetic smoke set, held-out `test` split: 2 samples, 1 scene units

Reference accuracy:

| system | PSNR (dB) | SSIM | RMSE | MAE | SAM (°) | ERGAS |
|---|---|---|---|---|---|---|
| bicubic | 33.0508 | 0.9118 | 0.0223 | 0.0095 | 2.3922 | 4.4817 |
| SEN2SR-Lite (hard constraint) | 34.6350 | 0.9326 | 0.0185 | 0.0082 | 2.1240 | 3.5379 |
| SEN2SR-Lite, no hard constraint (ablation) | 34.8374 | 0.9281 | 0.0181 | 0.0093 | 2.7577 | 3.4210 |
| SEN2SR-Mamba (hard constraint) | 35.6613 | 0.9460 | 0.0165 | 0.0074 | 1.9100 | 3.2629 |
| tiny model, 5 seeds (mean ± sd across seeds) | 36.0804 ± 0.1917 | 0.9492 ± 0.0015 | 0.0157 ± 0.0004 | 0.0066 ± 0.0001 | 1.7729 ± 0.0492 | 2.8197 ± 0.0246 |

Spectral fidelity (indices vs the reference, per-band RMSE):

| system | NDVI MAE | NDWI MAE | log(B08/B04) bias | B02 RMSE | B03 RMSE | B04 RMSE | B08 RMSE |
|---|---|---|---|---|---|---|---|
| bicubic | 0.0383 | 0.0362 | -0.0026 | 0.0202 | 0.0184 | 0.0233 | 0.0263 |
| SEN2SR-Lite (hard constraint) | 0.0352 | 0.0328 | 0.0007 | 0.0159 | 0.0142 | 0.0182 | 0.0243 |
| SEN2SR-Lite, no hard constraint (ablation) | 0.0690 | 0.0569 | -0.0152 | 0.0153 | 0.0137 | 0.0176 | 0.0241 |
| SEN2SR-Mamba (hard constraint) | 0.0308 | 0.0296 | 0.0007 | 0.0148 | 0.0132 | 0.0168 | 0.0202 |
| tiny model, 5 seeds (mean ± sd across seeds) | 0.0321 ± 0.0010 | 0.0293 ± 0.0008 | 0.0050 ± 0.0019 | 0.0123 ± 0.0001 | 0.0117 ± 0.0001 | 0.0144 ± 0.0001 | 0.0222 ± 0.0009 |

Spatial detail (vs the reference):

| system | detail rel. error | detail corr. | detail energy ratio | edge corr. | shift (HR px) |
|---|---|---|---|---|---|
| bicubic | 0.921 | 0.449 | 0.052 | 0.596 | 0.000 |
| SEN2SR-Lite (hard constraint) | 0.808 | 0.654 | 0.156 | 0.827 | 0.000 |
| SEN2SR-Lite, no hard constraint (ablation) | 0.795 | 0.650 | 0.204 | 0.825 | 0.000 |
| SEN2SR-Mamba (hard constraint) | 0.711 | 0.787 | 0.197 | 0.917 | 0.000 |
| tiny model, 5 seeds (mean ± sd across seeds) | 0.720 ± 0.010 | 0.714 ± 0.010 | 0.376 ± 0.015 | 0.867 ± 0.008 | 0.000 ± 0.000 |

### 8.2 Paired differences against bicubic

**SEN2NEON random 30 (28 units)** — paired difference (system − bicubic) of scene-unit means, 95% percentile-bootstrap interval; Wilcoxon p where ≥ 6 units:

| system − bicubic | PSNR (dB) | SAM (°) | detail corr. |
|---|---|---|---|
| Lite | -0.456 [-0.593, -0.335], p=1.49e-08 | +0.159 [+0.085, +0.263], p=1.42e-07 | -0.066 [-0.084, -0.046], p=5.68e-06 |
| Lite, no constraint | -1.560 [-1.861, -1.281], p=7.45e-09 | +0.838 [+0.677, +1.021], p=2.98e-08 | -0.062 [-0.079, -0.042], p=7.96e-06 |
| Mamba | -0.463 [-0.638, -0.314], p=1.49e-08 | +0.073 [+0.028, +0.121], p=0.00344 | -0.115 [-0.145, -0.084], p=8.2e-07 |
| tiny (seed 0) | -0.811 [-1.128, -0.548], p=7.45e-09 | +0.327 [+0.163, +0.559], p=1.49e-08 | -0.126 [-0.158, -0.094], p=8.2e-07 |

**OpenSR `spot` (9 units)** — paired difference (system − bicubic) of scene-unit means, 95% percentile-bootstrap interval; Wilcoxon p where ≥ 6 units:

| system − bicubic | PSNR (dB) | SAM (°) | detail corr. |
|---|---|---|---|
| Lite | +0.106 [+0.026, +0.181], p=0.0391 | +0.225 [+0.106, +0.365], p=0.00391 | +0.045 [+0.033, +0.058], p=0.00391 |
| Lite, no constraint | -0.670 [-0.949, -0.439], p=0.00391 | +0.833 [+0.587, +1.139], p=0.00391 | +0.050 [+0.038, +0.062], p=0.00391 |
| Mamba | +0.079 [-0.033, +0.194], p=0.25 | +0.170 [+0.070, +0.280], p=0.00781 | +0.048 [+0.037, +0.061], p=0.00391 |
| tiny (seed 0) | -0.265 [-0.507, -0.052], p=0.0547 | +0.615 [+0.211, +1.076], p=0.00391 | -0.009 [-0.030, +0.013], p=0.426 |

**OpenSR `spain_crops` (5 units)** — paired difference (system − bicubic) of scene-unit means; the interval is shown because 5 units is the minimum the rule accepts, but with 5 units treat it as descriptive; no test is run below 6 units:

| system − bicubic | PSNR (dB) | SAM (°) | detail corr. |
|---|---|---|---|
| Lite | +0.019 [-0.168, +0.229] | +0.025 [+0.008, +0.052] | +0.015 [-0.014, +0.044] |
| Lite, no constraint | -0.517 [-0.794, -0.267] | +0.421 [+0.314, +0.553] | +0.019 [-0.011, +0.047] |
| Mamba | +0.020 [-0.186, +0.251] | +0.016 [-0.003, +0.039] | +0.008 [-0.032, +0.047] |
| tiny (seed 0) | -0.196 [-0.390, +0.004] | +0.122 [+0.080, +0.167] | -0.014 [-0.037, +0.013] |

**OpenSR `spain_urban` (4 units)** — paired difference (system − bicubic) of scene-unit means; 4 units is below the minimum for an interval (5) and for a test (6), so these are **descriptive only**:

| system − bicubic | PSNR (dB) | SAM (°) | detail corr. |
|---|---|---|---|
| Lite | -0.006 (descriptive only) | +0.033 (descriptive only) | +0.018 (descriptive only) |
| Lite, no constraint | -0.492 (descriptive only) | +0.534 (descriptive only) | +0.021 (descriptive only) |
| Mamba | -0.034 (descriptive only) | +0.079 (descriptive only) | +0.002 (descriptive only) |
| tiny (seed 0) | -0.418 (descriptive only) | +0.321 (descriptive only) | -0.015 (descriptive only) |

No multiple-comparison correction is applied to any table: a table with many comparisons is exploratory. `inferential` in the aggregates means only that there were enough units for an interval or a test.

### 8.3 Spatial-shift sensitivity (§6)

Same tiles, same SR, the reference displaced by known whole HR pixels (`python -m frame.evaluate shift`); the `0` row equals the ordinary evaluation of §8.1. Mean over scene units; systems as in §8.1 (a subset was swept).

**SEN2NEON random 30 (28 scene units)** — estimated displacement of the bicubic baseline against the reference: median 1.00, mean 1.54, max 6.50 HR px.

| reference displaced by | PSNR bicubic | PSNR Lite | PSNR Mamba | PSNR tiny (seed 0) | detail corr. bicubic | detail corr. Lite | detail corr. Mamba | detail corr. tiny (seed 0) |
|---|---|---|---|---|---|---|---|---|
| 0 (as evaluated) | 33.69 | 33.23 | 33.22 | 32.88 | 0.529 | 0.463 | 0.415 | 0.403 |
| 0.25 LR px (1 HR px) | 33.43 | 32.96 | 32.97 | 32.63 | 0.456 | 0.387 | 0.347 | 0.334 |
| 0.5 LR px (2 HR px) | 33.01 | 32.51 | 32.55 | 32.23 | 0.313 | 0.248 | 0.229 | 0.214 |
| 1 LR px (4 HR px) | 32.13 | 31.62 | 31.69 | 31.39 | 0.042 | -0.011 | 0.007 | -0.002 |
| 2 LR px (8 HR px) | 31.00 | 30.62 | 30.69 | 30.38 | -0.038 | -0.008 | -0.015 | -0.020 |
| `aligned_to_bicubic` | 33.84 | 33.40 | 33.38 | 33.03 | 0.547 | 0.503 | 0.452 | 0.442 |

**OpenSR `spain_crops` (5 units)** — estimated displacement of the bicubic baseline against the reference: median 1.54, mean 1.63, max 3.26 HR px.

| reference displaced by | PSNR bicubic | PSNR Lite | PSNR Mamba | PSNR tiny (seed 0) | detail corr. bicubic | detail corr. Lite | detail corr. Mamba | detail corr. tiny (seed 0) |
|---|---|---|---|---|---|---|---|---|
| 0 (as evaluated) | 32.53 | 32.55 | 32.55 | 32.33 | 0.225 | 0.240 | 0.233 | 0.211 |
| 0.25 LR px (1 HR px) | 32.22 | 32.10 | 32.08 | 31.83 | 0.172 | 0.159 | 0.146 | 0.134 |
| 0.5 LR px (2 HR px) | 31.76 | 31.51 | 31.49 | 31.20 | 0.103 | 0.069 | 0.061 | 0.057 |
| 1 LR px (4 HR px) | 30.80 | 30.46 | 30.46 | 30.12 | -0.007 | -0.035 | -0.026 | -0.022 |
| 2 LR px (8 HR px) | 29.53 | 29.32 | 29.32 | 28.97 | -0.009 | 0.008 | 0.006 | 0.001 |
| `aligned_to_bicubic` | 32.92 | 33.11 | 33.15 | 32.94 | 0.290 | 0.350 | 0.361 | 0.317 |

**OpenSR `spain_urban` (4 units)** — estimated displacement of the bicubic baseline against the reference: median 1.63, mean 1.67, max 3.44 HR px.

| reference displaced by | PSNR bicubic | PSNR Lite | PSNR Mamba | PSNR tiny (seed 0) | detail corr. bicubic | detail corr. Lite | detail corr. Mamba | detail corr. tiny (seed 0) |
|---|---|---|---|---|---|---|---|---|
| 0 (as evaluated) | 29.29 | 29.28 | 29.25 | 28.87 | 0.234 | 0.252 | 0.236 | 0.219 |
| 0.25 LR px (1 HR px) | 28.95 | 28.78 | 28.75 | 28.33 | 0.181 | 0.169 | 0.151 | 0.145 |
| 0.5 LR px (2 HR px) | 28.45 | 28.13 | 28.10 | 27.63 | 0.107 | 0.070 | 0.056 | 0.058 |
| 1 LR px (4 HR px) | 27.43 | 26.99 | 27.02 | 26.46 | -0.020 | -0.056 | -0.047 | -0.044 |
| 2 LR px (8 HR px) | 26.21 | 25.93 | 25.96 | 25.42 | -0.019 | 0.001 | -0.001 | -0.004 |
| `aligned_to_bicubic` | 29.72 | 29.92 | 29.90 | 29.56 | 0.304 | 0.365 | 0.358 | 0.323 |

**OpenSR `spot` (9 units)** — estimated displacement of the bicubic baseline against the reference: median 0.40, mean 0.47, max 0.85 HR px.

| reference displaced by | PSNR bicubic | PSNR Lite | PSNR Mamba | PSNR tiny (seed 0) | detail corr. bicubic | detail corr. Lite | detail corr. Mamba | detail corr. tiny (seed 0) |
|---|---|---|---|---|---|---|---|---|
| 0 (as evaluated) | 33.12 | 33.22 | 33.20 | 32.85 | 0.333 | 0.378 | 0.381 | 0.324 |
| 0.25 LR px (1 HR px) | 33.03 | 33.07 | 33.04 | 32.70 | 0.315 | 0.349 | 0.349 | 0.301 |
| 0.5 LR px (2 HR px) | 32.61 | 32.45 | 32.41 | 32.08 | 0.248 | 0.248 | 0.237 | 0.217 |
| 1 LR px (4 HR px) | 31.39 | 30.93 | 30.90 | 30.55 | 0.053 | 0.001 | -0.007 | 0.009 |
| 2 LR px (8 HR px) | 29.72 | 29.36 | 29.37 | 28.95 | -0.055 | -0.038 | -0.031 | -0.039 |
| `aligned_to_bicubic` | 33.14 | 33.26 | 33.23 | 32.88 | 0.337 | 0.383 | 0.388 | 0.327 |

### 8.4 What the measurements show, and what they do not

* **SEN2NEON (28 acquisitions).** Every network system scores *below* bicubic on the pixel-wise and spectral metrics against this reference, and the difference is clear over 28 units (PSNR, system − bicubic: Lite −0.46 dB [−0.59, −0.34], Mamba −0.46 dB [−0.64, −0.31], tiny model −0.81 dB;
  SAM +0.16° / +0.07° / +0.33°). Against the same reference the added detail is worse than none for every network system (detail rel. error 1.75-2.0 versus 1.14 for bicubic), and the detail correlation is lower (0.40-0.46 versus 0.53).
* **The reference is not registered to the Sentinel-2 grid, and that matters as much as the differences above.** The displacement of the bicubic baseline against the SEN2NEON references has a median of 1.0 HR pixel (mean 1.5, max 6.5; validated against a brute-force search, §7). A displacement of 0.25 LR pixel (1 HR pixel)
  lowers every system's PSNR by about 0.26 dB and the detail correlation from 0.53 to 0.46, the same size as the differences between systems (0.46 dB); a displacement of 2 LR pixels lowers PSNR by 2.7 dB and removes the detail correlation entirely. After removing the estimated displacement (`aligned_to_bicubic`) every system gains about 0.15 dB
  and **the ordering of the pixel-wise metrics is unchanged**: misregistration lowers all scores but does not explain the gap on SEN2NEON. What does: this evaluation cannot say. Candidates it cannot separate are the radiometric difference between Sentinel-2 and the AVIRIS-NG-derived reference, texture
  and shadow in the reference that a 10 m input does not contain, and the land-cover mix of the sample.
* **OpenSR-Test.** On the pixel metrics the systems are close to bicubic (PSNR difference ≤ 0.11 dB on `spot`, ≤ 0.02 dB on `spain_crops`, ≤ 0.04 dB on `spain_urban` for Lite and Mamba). On `spot`, where the reference is best registered (median displacement 0.4 HR px), Lite and Mamba
  have a **higher** detail correlation than bicubic (+0.045 [0.033, 0.058] and +0.048 [0.037, 0.061]; 9 units) and a slightly worse SAM (+0.23° and +0.17°). On `spain_crops` and `spain_urban` (median displacement 1.5-1.6 HR px) they equal bicubic as evaluated and lie about 0.2 dB above it, with a detail correlation of 0.35-0.37
  versus 0.29-0.30, once the estimated displacement is removed. That is **descriptive**: 5 and 4 scene units. The alignment is estimated from the bicubic baseline, so it cannot favour the other systems by construction.
* **The hard constraint.** SEN2SR-Lite without it is worse than with it on PSNR, SSIM, RMSE, MAE, SAM, ERGAS and NDVI error on every real dataset (PSNR 1.10 dB [0.93, 1.29] lower on SEN2NEON) and biased in the NIR band (B08 bias −0.0075 against +0.0001) with a log(B08/B04) bias of 0.13 (0.02 with it); reduced back to the LR grid it deviates by 0.0054 MAE instead of 0.0012. Its detail correlation is marginally *higher* (0.468 versus 0.463 on SEN2NEON), the other side of adding much more unsupported detail (below). This is the constrained system's purpose,
  and it is measured here for Lite only: the Mamba worker serves the constrained system, so a Mamba ablation was not run. On the 2-sample synthetic set it is not worse (PSNR 34.84 versus 34.64 dB), which is one scene unit and no evidence.
* **Spectral fidelity.** The error is largest in the NIR band (B08 RMSE 0.038-0.043 on SEN2NEON against 0.017-0.027 in the visible bands) for every system, so NDVI error (MAE 0.054-0.072) follows; NDVI correlation with the reference is 0.72-0.75. Differences between the constrained systems in SAM and NDVI are small compared with the spread across scenes (SAM standard deviation across
  units ≈ 2.5°).
* **What was added, judged against this reference** (element-wise, τ = 0.005 reflectance; "unsupported" means *not confirmed by this reference*, not false): for Lite 4.8% of the valid elements are supported synthesis, 9.4% unsupported detail and 47.7% omission on SEN2NEON (11.7% / 10.8% / 52.6% on `spot`); the MSE skill against bicubic is −0.11 on SEN2NEON,
  +0.02 on `spot` and about 0 on `spain_crops` and `spain_urban`. For the constrained systems and the tiny model 45-59% of all valid elements are omissions (the reference has detail that was not added); the unconstrained Lite omits less (28-36%) and adds far more unsupported detail (30-36% of the elements). No image was inspected qualitatively and no scene-category conclusion is drawn (SEN2NEON's own land-cover labels are in `aggregates.json` but have 1-16 samples per category).
* **Self-consistency (not accuracy).** The constrained systems reduce back to the LR within 0.0012-0.0019 MAE (bicubic 0.0016). This says the SR is consistent with its own input; it says nothing about the reference (a system can be self-consistent and far from it, tested).
* **Seeds.** Over five seeds the tiny model's PSNR varies by a standard deviation of 0.03-0.06 dB on the real datasets (0.19 dB on the synthetic set), far below the differences between systems, so a single seed is representative *for this model*; the published Lite and Mamba weights are single checkpoints and have no seed spread.
* **Tile seams.** The reference-based error within 8 px of the tile seams divided by the interior error is 0.99 for Lite, Mamba and the tiny model on SEN2NEON: no measurable seam penalty.
* **Synthetic set.** The tiny model, trained on the same synthetic degradation, scores highest on its own held-out synthetic scenes; that is in-distribution by construction, is one region and two scenes, and is not evidence about real data. The same model is below bicubic on SEN2NEON.

## 9. What remains unproven

* **No absolute accuracy.** Every real reference is a different sensor with its own radiometry and a registration error of median 1-1.6 HR pixels (max 6.5); the numbers are comparisons against that reference, not errors against ground truth.
* **Small and geographically narrow samples.** 28 SEN2NEON acquisitions (North America), OpenSR-Test's `spot` subset (9 scenes) and its Spanish subsets (5 and 4 source orthophotos). `spain_crops` and `spain_urban` have 5 and 4 source orthophotos: their numbers are descriptive. Tiles of one scene are correlated, so tile-level counts overstate the evidence.
* **No Indian evidence at all.** There is no Indian HR reference in this evaluation and no claim about Indian landscapes (requirements §26).
* **No downstream task.** Classification accuracy, area error and change detection (the rest of §27) were not evaluated.
* **No calibrated uncertainty or OOD evaluation**, and no uncertainty-versus-error analysis: that is the next phase.
* **RGBN ×4 only.** The full 12-band / 10-band cascade is not evaluated, so NDWI needs only B03/B08 but MNDWI, NDBI and BSI (which need B11) are not computed.
* **Trained systems are not the intended ones.** The tiny model is a 47k-parameter CNN trained for 600 steps on a synthetic set; it exists to test the pipeline. No real training data was used and no training on SEN2NAIPv2 has been done (§10).
* **One harmonisation of OpenSR-Test.** The harmonised `HRharm` reference (the benchmark default) was used; the raw `HR` was not evaluated. The tile size, overlap, padding and blending were fixed by the Phase 2 defaults (128 / 32 / reflect / linear) and not varied here.
* **Mamba** was evaluated with its constraint only, and locally only for inference. No Mamba ablation, no Mamba fine-tuning.
* **Working tree not committed** at the time of the runs; the git revision in the record is the parent commit plus a dirty flag.

## 10. SEN2NAIPv2 acquisition decision

No whole part was downloaded. Facts checked from metadata alone: 61,282 pairs (`unet` and `histmatch`) in four parts of 20.0 + 20.0 + 20.0 + 10.28 GB (`unet`) and 8,000 real cross-sensor pairs in a single 9.72 GB file (`crosssensor`), 56 states, CC0-1.0, read with `tacoreader` in a
separate environment (`~/.venvs/frame_taco`, 516 MB) that is not a FRAME dependency; 124 GB of disk free. A seeded random 130-pair sample (287 MB, 0 failures) verified the real format (130×130 → 520×520, RGBN order, no nodata, alignment to 0.3 mm), a leakage-free state-level split and the dataset's geography (100 `unet` pairs: 42 states, 95 counties). A part is longitude-banded
(part 0003: 19 states, east coast), so a national geographic split needs at least two parts; the `crosssensor` file is the only single download that covers the whole country. SEN2SR was trained on this dataset, so it is **not** an independent benchmark and the evaluation refuses it.
Full reasoning, numbers and the recommended parts for a later training phase: `experiments/evaluation/data_acquisition/SEN2NAIPV2_DECISION.md`.

## 11. Reproduce

```bash
sen2sr_venv/bin/python -m frame.evaluate check experiments/evaluation/configs/benchmarks_v1.json
sen2sr_venv/bin/python -m frame.evaluate run   experiments/evaluation/configs/benchmarks_v1.json --verbose
sen2sr_venv/bin/python -m frame.evaluate shift experiments/evaluation/configs/benchmarks_v1.json --dataset sen2neon_random30 \
    --systems bicubic sen2sr_lite sen2sr_mamba tiny_cnn_seed0 --output-dir ../shift_sensitivity/sen2neon_random30
```

Tests: `TMPDIR=~/.cache/frame_tmp sen2sr_venv/bin/python -m pytest frame/tests/test_evaluate_*.py --basetemp=~/.cache/frame_pytest_tmp` (none needs the GPU; `test_evaluate_real.py` skips itself if the real data are absent). Set `FRAME_DATA_ROOT` for the data location.

## 12. Downstream analytical utility (Phase 7)

`docs/DOWNSTREAM.md` asks whether the SR changes an NDVI-derived vegetation decision, on the Phase 6 gate-eligible evidence and per scene unit. It is **not** a re-run of §8's pixel-level NDVI metrics: the numbers differ by design (this document scores NDVI per pixel on all tiles of a set with the Phase 5 strict mask; Phase 7 scores the mean NDVI of fixed 10 m / 40 m regions on registration-eligible tiles only, and the decision at a fixed predeclared threshold). Neither replaces the other, and neither ranks the systems.
