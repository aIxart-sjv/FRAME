# FRAME training layer (Phase 4)

`frame/train/` is the smallest reproducible training capability that consumes the Phase 3 paired-data layer:

```
Manifest (frame.data, digest) ─▶ geographic split + leakage gate ─▶ PairedPatchDataset ─▶ model ─▶ loss ─▶ optimizer ─▶ checkpoint ─▶ validation
        │                                (scene / region / role)         (random crops,      sr =     L1 / Charbonnier   AdamW        resumable     deterministic,
        └── digest recorded in every checkpoint and summary              flips + rot90,      model(lr)  [+ spectral]       + schedule                  vs bicubic / Lite
                                                                          masks)
```

It is deliberately not an MLOps platform: one loop, one config file, one command, one experiment directory. It trains nothing at scale
(no long runs, no hyper-parameter search, no distributed training) and it does **not** train on SEN2NEON, OpenSR-Test or any other benchmark.

## 1. Use

```bash
python -m frame.train check CONFIG.json     # validate the config, the manifest and the leakage gate; build the model; train nothing
python -m frame.train run   CONFIG.json     # train, validate, checkpoint, record
python -m frame.train run   CONFIG.json --resume latest        # continue an interrupted run (or --resume path/to/step_000200.pt)
```

Relative paths inside a config resolve against the directory of the config file. Exit codes: `0` success; `1` the run failed while training
(non-finite loss, out of GPU memory; the failure is written to `summary.json`); `2` refused or invalid input (bad config, leakage or role
violation, missing or edited manifest, incompatible resume, unbuildable model), in which case nothing was trained and nothing overwritten.

The full flow from nothing, exactly as run for the smoke experiments (all data is synthetic and generated on the spot):

```bash
python -m frame.data synthetic --out unsplit.jsonl --regions 5 --scenes-per-region 2 --seed 0     # writes files under $FRAME_DATA_ROOT
python -m frame.data split unsplit.jsonl --out experiments/training/manifests/synthetic_5x2_seed0.jsonl --level region --seed 0 --fractions 0.6 0.2 0.2
python -m frame.train check experiments/training/smoke_tiny_cnn/config.json
python -m frame.train run   experiments/training/smoke_tiny_cnn/config.json
```

## 2. Modules

| Module | Purpose | Runs in |
|---|---|---|
| `config.py` | frozen, strict, serialisable `TrainConfig`; `resume_signature`, `digest` | both envs |
| `losses.py` | L1, Charbonnier, spectral-angle, downsample-consistency, `CompositeLoss` (mask-aware, float32) | both |
| `models.py` | registry: `tiny_cnn`, `sen2sr_lite`, `sen2sr_mamba`; everything is `sr = model(lr)` | both (Mamba only where `mamba_ssm` exists) |
| `memory.py` | FRAME-side activation checkpointing for MambaSR | both |
| `checkpoint.py` | atomic, weights-only-loadable, compatibility-checked checkpoints; RNG state; pruning | both |
| `trainer.py` | the loop: steps, accumulation, AMP with fallback, scheduler, validation hook, resume | both |
| `validate.py` | deterministic evaluation of any `model(lr)`; basic and reference metrics | both (`reference_metrics` main env) |
| `baselines.py` | bicubic (FRAME's convention) and the cached SEN2SR-Lite | both (Lite main env) |
| `data.py` | manifest → train/val datasets behind the leakage gate | main env |
| `record.py`, `run.py`, `cli.py` | environment/git capture, `summary.json`, README, the command | main env |

The **core is torch-only** so the same `Trainer` drives the small CNN in the main environment and `MambaSR` in the isolated Mamba environment
(which has torch and `mamba_ssm` but not rasterio / scikit-image; nothing was installed there). The data layer and the evaluation libraries
are imported lazily and only exist in the main environment, which is why the Mamba fine-tune is a two-step script (§9) and not `frame.train run`.

## 3. Configuration

One JSON file; unknown keys are refused (a typo must not silently take a default) and every value is range-checked, with errors naming the dotted field.

| Field | Meaning (default) |
|---|---|
| `name`, `output_dir` | run name; experiment directory (`.` = the config's own directory) |
| `model` | `{name: tiny_cnn \| sen2sr_lite \| sen2sr_mamba, params: {...}, pretrained: false}` |
| `data.manifest`, `data.data_root`, `data.datasets` | Phase 3 manifest; dataset root (default `$FRAME_DATA_ROOT`); optional dataset filter |
| `data.train_split`, `data.val_split` | manifest split **names** (`train`, `val`); `test` is refused for either |
| `data.lr_patch`, `data.val_lr_patch`, `data.patches_per_pair` | random LR training crop (32); validation tile (128, the models' native size); crops per pair per epoch (4) |
| `data.augment`, `data.min_valid_fraction`, `data.lr_bands`, `data.hr_bands` | same flip/rotation on LR and HR (true); redraw crops that are mostly nodata (0.8); band **names**, default FRAME's RGBN order |
| `data.require_region_disjoint` | true: a region may not span splits; false allows scene-level splitting (scene leakage is refused either way) |
| `steps`, `batch_size`, `grad_accum` | optimizer updates; micro-batch; micro-batches per update |
| `optim` | `adamw` (default) / `adam` / `sgd`; `lr`, `weight_decay`, `betas`, `grad_clip_norm` |
| `scheduler` | `none` (default), `cosine`, `step`; linear `warmup_steps` from lr/10 |
| `loss` | `reconstruction: l1 \| charbonnier`, `charbonnier_eps`, `spectral_weight` (0), `consistency_weight` (0) |
| `precision.amp` | `off` (default, float32), `fp16`, `bf16` |
| `checkpoint_every`, `validate_every`, `log_every`, `keep_last_checkpoints` | cadence (0 = only at the end for the first two) |
| `seed`, `deterministic`, `device` | one seed drives model init, data order and crop draws; deterministic kernels requested; `auto` \| `cpu` \| `cuda[:N]` |
| `evaluate_train`, `baselines` | also score the training scenes (an over-fit check); reference upsamplers on the same validation patches (`bicubic`, `lite`) |

`steps` counts optimizer updates. `resume_signature` (everything except the step budget, output directory, run name, baselines and cadence) must
match for a checkpoint to be resumed.

## 4. Leakage safety

Training **never splits patches or scenes**. The train and validation records are exactly the manifest's own `split` labels (assigned per region by
`python -m frame.data split`). Before any file is touched, `frame.train.data` runs Phase 3's `validate_manifest` over every record of the (filtered)
manifest and **refuses to start** on `scene_leakage`, `region_leakage` (unless the config opts into a scene-level split) or `role_violation`, and on any other
QC error. A second, independent check refuses any record whose dataset **role** is an independent benchmark or holdout in the train or validation
split, even if a profile were misconfigured to allow it. Records in the `test` split may share the manifest, but they are never opened: their files are
not even checked for existence (a test deletes them and trains anyway). The manifest's digest is verified against its own header (an edited manifest is
refused) and recorded in every checkpoint and summary. The leakage gate and the second role layer were each disabled in turn to confirm their tests fail (mutation-checked).

## 5. Models

* **`tiny_cnn`**: a small residual CNN (47,424 parameters at width 32, depth 4) over **FRAME's bicubic** (`frame.validation.bicubic_upsample`: antialiased
  kernel, negatives clamped). Its output head is zero-initialised, so *before training it is exactly the bicubic baseline* (tested) and any later gain
  is attributable to training. LeakyReLU body (see §10).
* **`sen2sr_lite`**: the published `CNNSR(4,4,24,4,True,True,6)` from `sen2sr` (572,336 parameters, 472,496 trainable), optionally with the cached published weights
  (`pretrained: true`, never downloads). Built in **train mode**: the published inference build (`train_mode=False`) rebuilds its fused convolutions under `.detach()` on
  every forward, so only 4 tensors (16,216 parameters) would receive gradients (§10). It is the `sr_model` *without* the low-frequency hard constraint.
* **`sen2sr_mamba`**: `MambaSR` (13,759,444 parameters). Needs `mamba_ssm` and CUDA, i.e. the isolated Mamba environment; in the main environment building it fails with a
  clear error. `activation_checkpointing: true` wraps each `VSSBlock` (§8). Also without the hard constraint.

The **hard constraint is not part of training**: its mask is fixed at 512×512, so it cannot be applied to small crops. A model trained without it and
evaluated with it is a different system from the one that was trained; that is a limitation to close before any claim about SEN2SR-equivalent output (§11).

## 6. Losses (training objectives, not evaluation metrics)

Grounded in Requirements 142 Req. 4 §33 and Req. 5 §24-25 (a pixel loss, optionally a spectral term; no perceptual or adversarial term, because in Earth
observation "perceptual hallucination can be dangerous"): `L = L_rec + λ_s·L_spectral + λ_c·L_consistency`. All are mask-aware (masked pixels contribute nothing to value or
gradient; a fully masked batch gives zero loss and zero finite gradients), computed in float32, and independently tested.

* `l1` mean absolute error. `charbonnier` = `sqrt(d² + ε²) − ε`: smooth, exactly 0 for identical inputs.
* `spectral` = mean `(1 − cos θ)` between predicted and true spectral vectors: brightness-invariant, 0 for identical direction, finite gradient at 0 (arccos would not be). A surrogate of the SAM
  angle, not SAM.
* `consistency` = L1 between the area-average downsampling of the SR and the LR input. **Only meaningful when the LR really is an area average of the HR** (synthetic pairs, roughly);
  for cross-sensor pairs it would penalise correct outputs, so it defaults to 0 and was **not** used in any experiment.

The training spectral term does **not** prove spectral fidelity; SAM/ERGAS in validation are diagnostics on a small synthetic set. SAM and ERGAS are already reported (from `frame.validation`), so Phase 5 can build on them.

## 7. Optimization, AMP, accumulation

AdamW; schedules `none` / `cosine` / `step` with optional linear warm-up; optional gradient clipping (the pre-clip norm is logged). A NaN/Inf loss stops training before the weights change.
**AMP is opt-in and CUDA-only**; on CPU (or bf16 without hardware support) it falls back to float32 and records why. **Gradient accumulation raises the effective batch size at
constant activation memory; it does not reduce the activation memory of one micro-batch** (tested: accumulation 4 × batch 1 equals batch 4 for a batch-independent model, to float rounding).
Batch order is a pure function of `(seed, epoch)`, so a resumed run continues on exactly the batch an uninterrupted one would see next.

## 8. Checkpoints and resume

Each checkpoint holds model, optimizer, scheduler and scaler state, step and micro-step counters, RNG states, the full config and its resume signature, the seed, the
**manifest digest**, the git revision and the torch version. Only tensors and primitives are stored, so it loads with `torch.load(weights_only=True)` (this caught a real problem: `torch.__version__`
is a `TorchVersion` object, which that loader refuses). Writes are atomic; resuming is refused if the config (apart from the exempt fields) or the manifest digest differs. **Tested**: save/load, exact
continuation, and a resumed run reproducing the uninterrupted one bit for bit (CPU), including discarding the log lines the resumed run will redo. It was also exercised on the real 13.7 M-parameter
Mamba (159 MiB checkpoint, resumed from step 20).

## 9. Experiments (all measured; synthetic data, one machine)

Hardware/software: RTX 3050 Laptop GPU (3,759 MiB usable, ~600 MiB used by the desktop), 15.2 GiB RAM, Linux 7.1.5, Python 3.11.9, torch 2.14.0+cu130 (main) and 2.6.0+cu118 (Mamba env).
Data: `synthetic_smoke` (Phase 3 data layer; generated RGBN scenes, LR from `frame_default_v1`), 10 scenes in 5 regions split by region: 3 train regions (6 scenes), 1 validation region (2 scenes), 1 test region (2 scenes, never read).
Manifest digest `01bb0e2d…`. Validation = the 2 validation scenes as full 128×128 → 512×512 tiles. **Every number below is a mean over 2 validation patches (1 for the over-fit set); read them with the seed spread in mind.**

| Run (`experiments/training/…`) | Loss | Val PSNR (dB) | SSIM | RMSE | SAM (°) | ERGAS |
|---|---|---:|---:|---:|---:|---:|
| baseline: bicubic (= `tiny_cnn` at step 0) | – | 33.267 | 0.9126 | 0.02173 | 2.467 | 4.001 |
| baseline: SEN2SR-Lite (published, with hard constraint) | – | 34.846 | 0.9327 | 0.01812 | 2.161 | 3.166 |
| `smoke_tiny_cnn` (600 steps, batch 4×2) | L1 | 36.555 | 0.9516 | 0.01487 | 1.729 | 2.487 |
| `smoke_tiny_cnn_charbonnier` (ε = 0.01) | Charbonnier | 36.691 | 0.9515 | 0.01464 | 1.723 | 2.463 |
| `smoke_tiny_cnn_spectral` (λ_s = 0.05) | L1 + spectral | 36.571 | 0.9512 | 0.01485 | 1.680 | 2.509 |

**Seed spread** (`seed_sweep/`, seeds 0-4, same data and config): L1 36.495 ± 0.187 dB (36.18-36.67), Charbonnier 36.654 ± 0.051, L1 + spectral 36.588 ± 0.066. Every seed learns (min 36.18 vs bicubic 33.27).
The differences *between the loss variants (≤ 0.16 dB) are within the L1 seed spread*: these runs cannot say whether Charbonnier or the spectral term helps.
**Nothing here is a ranking, and no model is a winner.** The Lite row is a different kind of object (pretrained on real imagery, with its hard constraint) evaluated on out-of-domain synthetic scenes.

**Over-fit check** (`overfit_one_scene`: 1 training scene, 1 validation scene, batch size 1, 2,000 steps): training-scene PSNR 37.75 dB vs bicubic 33.74 dB (the model fits its training scene); validation scene 36.72 dB vs bicubic 33.11 dB. In `smoke_tiny_cnn` the training scenes score
36.50 dB against 36.56 dB on validation: this 47 k-parameter model does not memorise.

**Reproducibility** (`reproducibility/compare_runs.py`; same data, seed and config, run twice): the loss trajectory, validation metrics and final weights were **bit-for-bit identical** on CPU, and on this GPU both with deterministic kernels requested
and with default kernels. CPU vs GPU is a different experiment: validation PSNR 36.392 vs 36.555 dB, weights differing by up to 0.27. So: same-machine, same-device reproducibility was verified for *this* experiment; bitwise GPU determinism in general, or across devices/software, is **not** claimed.

**Hardware evidence** (`feasibility/measure_memory.py`, the real Trainer, random tensors, batch 1, process capped ~350 MiB below the free GPU memory so a probe cannot exhaust the desktop; an OOM is a recorded result):

| Model | LR crop → HR | precision, checkpointing | peak VRAM | time / step |
|---|---|---|---:|---:|
| `tiny_cnn` | 128 → 512 | fp32 / fp16 / bf16 | 27.4 / 26.1 / 26.1 MiB | ~4 ms |
| `sen2sr_lite` (fine-tunable, 572 k) | 128 → 512 | fp32 | 121.9 MiB | 22.7 ms |
| `sen2sr_lite` | 128 → 512 | fp16 / bf16 | 71.6 MiB | 12.9 / 13.7 ms |
| `sen2sr_mamba` (13.76 M) | 16 → 64 | fp32, off | 528 MiB | 0.10 s |
| `sen2sr_mamba` | 32 → 128 | fp32, off | 1,549 MiB | 0.32 s |
| `sen2sr_mamba` | 64 → 256 | fp32, off | **out of memory** | – |
| `sen2sr_mamba` | 32 → 128 | fp32, **activation checkpointing** | 280 MiB | 0.39 s |
| `sen2sr_mamba` | 64 → 256 | fp32, checkpointing | 424 MiB | 1.48 s |
| `sen2sr_mamba` | **128 → 512 (native tile)** | fp32, checkpointing | **1,164 MiB** | **7.51 s** |
| `sen2sr_mamba` | 128 → 512, batch 2 | fp32, checkpointing | 2,345 MiB | 15.6 s |
| `sen2sr_mamba` | 32 / 64, fp16 or bf16, checkpointing | | 280 / 420 MiB (same as fp32) | same as fp32 |

Findings: without checkpointing Mamba cannot train beyond a 32×32 crop on this card; **with FRAME-side block checkpointing the native 128×128 tile fits at batch 1 (1.16 GiB)**, at ~5× the time of the 64-px crop. **AMP runs without error on Mamba but saves nothing measurable**
(the selective scan runs in float32; upstream asserts it). Batch 2 at the native tile fits only just (2.35 GiB of the ~2.7 GiB cap). Host RSS stayed near 1.3 GiB.

**Mamba fine-tuning smoke** (`mamba_finetune_smoke/`, in the Mamba env through the same Trainer): pretrained `MambaSR`, activation checkpointing, fp32, batch 1 × accumulation 2, 64-px crops, AdamW 1e-5, L1, 30 optimizer steps (20, then resumed from a real 159 MiB checkpoint), peak 779 MiB, 2.97 s/step, training loss 0.0109 → 0.0078.
On the 2 synthetic validation tiles the pretrained model *before* fine-tuning scored 28.54 dB (worse than bicubic 33.27: out-of-domain data, no hard constraint) and 35.43 dB after 30 steps. **This proves the boundary and the loop, not a fine-tuning result**: 32 synthetic patches say nothing about real imagery.

### Local training decision

* **Smoke tests, `tiny_cnn`, and SEN2SR-Lite fine-tuning at the native tile: LOCAL** (≤ 122 MiB, tens of milliseconds per step).
* **Mamba: LOCAL BUT MEMORY-CONSTRAINED** for debugging and small runs (batch 1, checkpointing, fp32; 7.5 s per 128-px micro-step, 1.5 s at 64-px crops).
* **Any Mamba fine-tune beyond a few thousand steps: CLOUD GPU RECOMMENDED** (not required). Extrapolating from the measured per-step time, 10,000 optimizer steps at effective batch 8 (64-px crops) would take about 33 hours locally
  (80,000 micro-steps × 1.48 s) and about 7 days at the native tile (× 7.5 s), with the desktop's ~600 MiB competing for the GPU; these are extrapolations, not measurements. No cloud automation was launched.

## 10. Failures found and fixed on the way (kept, because they change what the numbers mean)

1. **Dead-ReLU stall in my own smoke model.** The first `tiny_cnn` (ReLU body, zero-initialised head) sat at *exactly* the bicubic output for some seeds (2 of 6 at batch 1, 3 of 6 at batch 4; 94% of the last layer's channels dead on stalled seeds). I first attributed this to L1's median-seeking
   behaviour, because L2 / Charbonnier learned on the same seed. **That explanation was wrong**: with a LeakyReLU body plain L1 learns on every seed. Fixed, with a regression test that fails on the old body (33.106 dB, unchanged) and passes on the new (≈36 dB).
2. **Two different "bicubic"s.** Torch's default bicubic and FRAME's `bicubic_upsample` (antialiased, PIL-style kernel) differ (RMSE 0.0069 vs 0.0100 on the same scenes). The model's base now uses FRAME's operator, so "step 0 = the bicubic baseline" is true, not approximate.
3. **The published Lite build cannot be fine-tuned** (`train_mode=False` detaches the fused convs): 4 tensors would train. Also, upstream's `CNNSR.forward` feeds every block the same input, so blocks 1-4 never reach the output (68 of 164 trainable tensors receive gradients even in train mode). `sen2sr/` is untouched.
4. **Upstream's `use_checkpoint` flag is broken** (`BasicLayer` calls `checkpoint(blk, x)` without `x_size`), so `frame.train.memory` wraps the blocks from outside.
5. **The validation scenes are not untouched held-out data.** While developing, I looked at these same two validation scenes when moving from 300 to 600 steps and from L1 to Charbonnier (before the dead-ReLU cause was found). Nothing was tuned per model for the comparison table and the seed sweep re-ran the final configs, but an unbiased final
   number would need the test split, which this phase deliberately never opens.

## 11. Real training data: decision (metadata only; nothing downloaded)

`experiments/training/data_acquisition/inspect_sources.py` (output `sources.json`) queries the services' own listings:

* **SEN2NAIPv2** (`tacofoundation/SEN2NAIPv2`, CC0, revision `c3705042…`): **149.4 GB** in 13 files (the Phase 3 figure of 139.1 GB is superseded). Multi-part `.taco` containers of up to 20 GB: `unet` 70.3 GB (4 parts, ~62 k pairs), `histmatch` 69.4 GB (4 parts),
  `crosssensor` 9.7 GB (1 file, 8,000 real S2/NAIP pairs). The **smallest useful piece is one part**: the last `unet` part, **10.28 GB, ≈ 9,100 pairs (an estimate scaled from the card's totals)**, i.e. ≈ 1.1 MB per compressed pair. Free disk is 125 GB, so this fits, but reading `.taco` needs `tacoreader`
  (not installed; adding a dependency needs a decision) and the export recipe in `docs/DATA.md` is still untested, and **the region metadata a geographic split needs was not inspected**, so it is unknown whether one part spans enough regions for a region-disjoint split. **Recommendation:** decide on installing `tacoreader`, fetch that single part, and check its
  geography first; do not start with the full set.
* **SEN2VENµS** (Zenodo 14603764): 139.5 GB in 29 per-site zips, licence `other-nc` (**the VENµS half is non-commercial**, so a model trained on it inherits a constraint to be checked). Sites are natural regions: the four smallest (FGMANAUS 0.12, SO2 0.79, ESTUAMAR 0.89, SUDOUE-4 0.95 GB) total **2.76 GB**.
  Its pairs are 5 m HR, 8 bands, ×2 (RGBN 10 m) and ×4 (red-edge 20 m): supplementary data, **not the RGBN 10 m → 2.5 m ×4 task**.

So Phase 4's actual training runs are synthetic by decision, not by oversight; the first real-data run should be one SEN2NAIPv2 part, after the two open questions above.

## 12. Known limitations

* **Synthetic data only**; no real SEN2NAIPv2 / SEN2VENµS data was used. Nothing measured here is evidence about real imagery, and the validation sets are 1-2 patches (see the seed spread).
* Training runs **without the low-frequency hard constraint**, which cannot be applied to small crops; Lite/Mamba are evaluated in that form (Lite's *baseline* row includes its constraint).
* No full benchmark campaign, no test-split evaluation (the test split is deliberately never opened), no uncertainty-vs-error or hallucination analysis, no Indian evaluation, no downstream-task validation, no 10-band cascade, no LDSR-S2/GAN/Swin.
* One optimizer family, three schedules, no hyper-parameter search, single GPU, no distributed training.
* GPU determinism was verified for one experiment on one machine; the loader re-reads GeoTIFFs with a small cache (fine at this scale, not for large data).
* The Mamba fine-tune is a two-step script because the Mamba environment cannot import the data layer; `frame.train run` covers the main environment.

## 13. Tests

`frame/tests/test_train_*.py` (249 tests), `test_data_synthetic.py` (17) and 5 new mask tests in `test_data_loader.py`, all offline: config 64, losses 29, models 19, memory 6, checkpoint 19, trainer 36, data wiring and leakage gate 25, validation and baselines 16, record 10, CLI end-to-end 25. Tests that need cached local data (the published Lite weights) skip themselves when it is absent.
Mutation-checked (the guard disabled, the test confirmed to fail): the leakage gate, the second role layer, the fresh-output and finished-run guards, and the dead-ReLU regression. The one-command reproduction of the full suite is in `experiments/training/README.md`.
