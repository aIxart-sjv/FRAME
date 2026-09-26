# FRAME registry: models and artifacts, datasets and evidence, measured cost

The single place that says **what is executed, on what data, and what it cost**. Every number here is copied from a record made in Phases 1–8 (the source is named in each row);
nothing was benchmarked again for this page except the Phase 8 smoke records under `experiments/final_readiness/`. Where a manifest and the executable disagree, the **executable truth** is stated and the
disagreement is left visible. Read [`CLAIMS.md`](CLAIMS.md) before using any of it as a statement about quality.

## 1. Models and artifacts

Two models are selectable (`POST /sr/run {"model": "lite" | "mamba"}`, default `lite`). Both are **RGBN, 10 m → 2.5 m pixel grid (×4)**, take exactly **128 × 128** input tiles (scenes of any size are cut into
tiles by `frame.tiling`), use the channel order **B04, B03, B02, B08**, and expect **surface reflectance as a fraction** (the API converts raw digital numbers by ÷ 10000 only when the caller says so).
Neither is trained, fine-tuned or modified by FRAME; `sen2sr/` has no diff. There is no third model and no fallback between them.

| | SEN2SR-Lite | SEN2SR-Mamba |
|---|---|---|
| API id / canonical name | `lite` / `SEN2SRLite/NonReference_RGBN_x4` | `mamba` / `SEN2SR/MambaSR_RGBN_x4` |
| **Executable** architecture | `sen2sr.models.opensr_baseline.cnn.CNNSR(4, 4, 24, 4, True, False, 6)` (a SPAN-derived CNN), wrapped by upstream `sen2sr.nonreference.srmodel` | `sen2sr.models.opensr_baseline.mamba.MambaSR` (`embed_dim 96`, `depths [8]*6`, `num_heads [8]*6`, `sigmoid_02` attention, pixel-shuffle upsampler), same wrapper |
| Parameters (measured) | **572,336** (204 float32 tensors; strict load: 0 missing, 0 unexpected) | **13,759,444** (1,228 tensors; strict load: 0 missing, 0 unexpected) |
| Hard constraint | yes: `hard_constraint.safetensor` (Fourier low-pass, low frequencies taken from the bicubic-upsampled input) | yes: `sr_hard_constraint.safetensor`. **Byte-identical** to Lite's mask file (same SHA-256) |
| Runs | in the main process, CPU or CUDA (`mlstac.load(...).compiled_model`) | in an **isolated worker process** in a separate environment; **CUDA only**, CPU is refused, never substituted |
| Executed weights (SHA-256) | `model.safetensor` `479aa796d5068d0b1206118ccbca27bd3223df0214db1a9b31a1e18349ed1c7e`; `hard_constraint.safetensor` `fbad981519066387c413ead1d6af7ef3e0d2947c34147ba90163fc79ae539239` | `sr_model.safetensor` `11e551b03663e873c5dcd4f02ec3fc8cdfa41ef46677540295ee4c6686ddc854` (recorded in every Mamba result's `metadata.model_runtime`); the same constraint hash as Lite |
| Artifact location | `$SEN2SR_BASELINE_WEIGHTS_DIR`, default `~/.cache/sen2sr_baseline/SEN2SRLite_RGBN` (fetched by `mlstac` on first use if absent; **not versioned**) | `$FRAME_MAMBA_WEIGHTS_DIR`, default `<repo>/models/SEN2SR/` (~367 MB, `/models/` is git-ignored). Only `sr_model.safetensor` and `sr_hard_constraint.safetensor` are used |
| Upstream source | `huggingface.co/tacofoundation/sen2sr/resolve/main/SEN2SRLite/NonReference_RGBN_x4/mlm.json` | `huggingface.co/tacofoundation/SEN2SR/resolve/main/SEN2SR/main/…` (from the artifact's own manifest) |
| Isolated environment | none (main env `sen2sr_venv`, torch 2.14.0) | `sen2sr_mamba_venv`: Python 3.11.9, torch 2.6.0+cu118, `mamba-ssm 2.3.2.post1`, `causal-conv1d 1.5.2`. Not mergeable with the main env (the prebuilt CUDA extension is tied to that torch) |
| Runtime, this machine (§3) | ms per tile; a 200 × 300 scene with the six-view stability ensemble in ≈ 1–2.5 s | ≈ 1.7 s per tile; the same scene ≈ 62–68 s |
| GPU memory (§3) | ≈ 100 MiB | ≈ 606 MiB allocator peak, 724 MiB process footprint |
| Known limitations | RGBN ×4 only; exactly 128 × 128 per model call; not fine-tunable as shipped (`train_mode=False` detaches the fused convolutions) | RGBN ×4 only; CUDA required; POSIX only; one worker, requests are serialised; output values above 1.0 are produced by the model and are not clipped by FRAME |

**Manifest versus executable (not resolved, only recorded).**

* *Lite.* The artifact's `mlm.json` calls it `CNN_Light_SR`, `mlm:architecture: SPAN`, `mlm:total_parameters: 472496`, `file:size: 1889984`. The shipped `model.safetensor` is 2,308,216 bytes and holds **572,336**
  parameters that load strictly into `CNNSR(4, 4, 24, 4, True, False, 6)`. The manifest's counts describe a different build than the file that runs. The executable class's own docstring is SPAN, so the architecture *label* is not contradicted; the counts and the size are.
* *Mamba.* The artifact's `mlm.json` says `Swin2SR` (that is its primary 10-band cascade component); the RGBN stage that FRAME runs is `MambaSR` (13,759,444 parameters, not the manifest's 12,894,526). FRAME records both labels
  (`executable_architecture: "MambaSR"`, `artifact_metadata_label: "Swin2SR"`). Details: [`MAMBA_INTEGRATION.md`](MAMBA_INTEGRATION.md).
* *Neither manifest was modified, and neither URL is pinned to a revision (`…/resolve/main/…`).* The SHA-256 values above are what every Phase 5–8 record ran. A later re-download could differ; compare hashes before comparing results.
* *Not integrated:* the 10-band cascade (`model.safetensor`, `f2_*` in `models/SEN2SR/`), LDSR-S2, any GAN / Swin / diffusion model. `ldsrs2_venv/` in the repository root is an unused exploratory environment (Python 3.14); nothing in `frame/` imports it.

The **weights of the evaluated product** are therefore exactly the two rows above. The tiny CNN of Phase 4 (`frame.train.models.tiny_cnn`, 47 k parameters, trained for 600 steps on synthetic data) and the Mamba fine-tune smoke checkpoint are
**not** part of the product; they exist to test the training loop (§4).

### The API and what it records about a run

`POST /sr/run` returns `model_id`, `model_name`, `input_shape`, `output_shape` (always exactly 4× in height and width), the resolution block (10 m → 2.5 m, scale 4), the seed and the six TTA transform names, the device,
`inference_seconds`, the tiling record (grid, overlap, counts, seam diagnostic), the output georeferencing, the valid-pixel coverage, and (Mamba only) `model_runtime`. The GeoTIFFs carry the CRS, the transform and a `FRAME_SR_VARIANT`
tag naming the model that ran. Lite results carry no weights hash in the API response; the Phase 8 smoke record for Lite (`experiments/final_readiness/smoke_lite/`) does.

## 2. Datasets and evidence

"Read for real" means FRAME opened real files of that dataset and ran something on them; "format only" means the reader follows the dataset's published format and was tested on synthetic files written in that format.
**No dataset was downloaded whole.** Data live outside the repository under `$FRAME_DATA_ROOT` (default `~/.cache/frame_data`) and `~/.config/opensr_test/`.

| Dataset | Intended role | What FRAME actually has | Validation status in FRAME | Bands / scale | Trained on (by FRAME) | Used for validation | Independent test | Real evidence run | Major caveats |
|---|---|---|---|---|---|---|---|---|---|
| **SEN2NAIPv2** (`unet`, `histmatch`) | primary training | **130 real pairs** (100 `unet`, 30 `crosssensor`) exported unmodified by ranged reads from 149.4 GB / 61,282 + 8,000 pairs, 287 MB. No whole part | format **verified on real files** (band order, sizes, origin agreement); seeded state-level split of the 100 `unet` pairs: 79 / 21 pairs, 37 / 5 disjoint states, no leakage | RGBN → RGBN, 10 m → 2.5 m, 130 × 130 → 520 × 520, `uint16` (scale ÷ 10000 **inferred**, not documented) | **no** (Phase 4 training was synthetic only) | no | **no** and it cannot be: the published SEN2SR weights were trained on it | format and split checks only; **no model metric** | tacoreader needed (separate env). One part is the smallest sensible download (≈ 10.3 GB, ≈ 9,100 pairs, an estimate); the region metadata a geographic split needs was not inspected |
| **SEN2NAIPv2** `crosssensor` | development validation | the 30 pairs above | format verified | as above | no | not run | no | none | as above |
| **SEN2VENµS** | supplementary training | **nothing downloaded**; metadata only (139.5 GB, 29 site zips, licence non-commercial) | **format adapter only**, tested on synthetic files; no real file was ever read | 5 m HR, 8 bands, ×2 (10 m) and ×4 (20 m): not the RGBN 10 m → 2.5 m task | no | no | no | none | LR is Theia L2A, not ESA L2A; the VENµS half is CC-BY-NC |
| **SEN2NEON** | independent benchmark, test only | a seeded uniform random **30 of 2,269 tiles** (seed 0; 28 acquisitions), 423 MB with the Phase 3 tiles | read for real; LR checksums verified; role enforced (train/val rejected by QC) | 12 → 12 bands on one 10 m grid (60 / 20 m bands were not measured at 10 m); 256 × 256 → 1024 × 1024; RGBN used | **never** | no | yes (all systems) | Phase 5 metrics on 30 tiles; Phases 6–7 on the **12 registration-eligible tiles (11 scene units)** | North America only; HR is AVIRIS-NG-derived (a different sensor); median registration error of the bicubic baseline 1.0 HR px (max 6.5) |
| **OpenSR-Test** `spot`, `spain_crops`, `spain_urban` | independent benchmark, test only | all of the three subsets: 9 + 28 + 20 scenes, 240 MB, MIT | read for real; role enforced | 12 → RGBN; 10 m → 2.5 m; 128 × 128 → 512 × 512; the harmonised `HRharm` reference | **never** | no | yes | Phase 5 on all 57; Phases 6–7 on **6 / 21 / 13 eligible tiles (6 / 5 / 4 scene units)** | Spain and SPOT only; `spain_crops` (5) and `spain_urban` (4) have few source orthophotos, so every number is descriptive; the raw `HR` (not `HRharm`) was not evaluated |
| **India holdout** | domain holdout, test only | **nothing**: a profile and a record builder | planned, **no data**, no synthetic truth was made | 12 → (no HR yet) | no | no | no | **none** | no Indian HR reference and no Indian downstream label exist in this repository |
| `synthetic_smoke` | smoke test | generated by `python -m frame.data synthetic` (10 scenes, 5 regions) | generated fixtures | RGBN → RGBN, ×4 | tiny CNN and the Mamba fine-tune smoke only | 2 validation patches | no | not evidence about real imagery | never a benchmark |
| Baseline 0 scene | demonstration / integration scene | one real L2A window (AOI 39.49, −0.43; 2023-01-15; EPSG:32630; saved tensor `experiments/baseline/outputs/input_tensor.pt`) | used by the API integration test and the Phase 0–9 experiments | RGBN, 128 × 128 | no | no | no | self-consistency only | one scene; the STAC catalogue has since drifted (a re-query returns 3 items where 1 was indexed), so it is reused, never re-fetched |
| Phase 8 smoke scene | pipeline invariants | synthetic, generated in memory (200 × 300, seeded) | `python -m frame.smoke` | RGBN, digital numbers | no | no | no | pipeline only | smooth random fields, not Sentinel-2 |

Splits, roles and leakage controls: [`DATA.md`](DATA.md) §2, §6; the gate that decides which reference tiles may enter a pixel-level analysis: [`RELIABILITY.md`](RELIABILITY.md) §3.

## 3. Measured cost (only values recorded in Phases 1–8; RTX 3050 Laptop GPU, 4 GB, ≈ 600 MiB used by the desktop; 15.2 GiB RAM)

Mamba is **substantially more expensive than Lite** (about two orders of magnitude, 90–300× depending on what is timed). That is a statement about cost, not about quality: nothing here ranks the models.

| Measurement | Lite | Mamba | Source |
|---|---|---|---|
| Single 128 × 128 tile, warm | ≈ 6 ms | 1.74 s (median of 10; 1.64–1.91 s) | `MAMBA_INTEGRATION.md` |
| Mean tile inference inside the HTTP pipeline | 19 ms | 1.739 s | `TILING.md` |
| First use (model load + first call) | 0.55 s | worker start 2.12 s (load 1.38 s); first call 1.92 s | `MAMBA_INTEGRATION.md` |
| GPU memory | ≈ 100.5 MiB | 605.6 MiB allocator peak; 724 MiB process (`nvidia-smi`); unchanged for a 40-tile scene | `MAMBA_INTEGRATION.md`, `TILING.md` |
| One tiled pass, real scene 511 × 777 (40 tiles) | – | 68.6 s, bit-identical on repeat | `TILING.md` |
| Full stability ensemble (6 views), one 128 tile | – | 10.6 s | `MAMBA_INTEGRATION.md` |
| HTTP `/sr/run`, real 256 × 384 crop (12 tiles × 6 passes = 72 inferences) | 3.75 s | 128.3 s | `TILING.md` |
| UI, real scenes | 511 × 777 in 8.4 s | 200 × 300 in 68.5 s (36 inferences) | `TILING.md` |
| Ensemble cost against one pass (Phase 6, 256 × 256 LR scenes) | 8.6–14.9× (0.096 s → 0.83 s on SEN2NEON) | 6.0× (14.8 s → 88.9 s); Mamba ≈ 150× slower than Lite per pass | `RELIABILITY.md` §6.9 |
| **Phase 8 smoke, 200 × 300 scene, 6 tiles × 6 passes = 36 inferences** (`experiments/final_readiness/smoke_*`) | `inference_seconds` 0.98 s, HTTP 2.43 s (incl. first load) | 66.6 s, HTTP 68.9 s | this phase (final records; earlier runs of the same scene on the same machine: 62–66 s for Mamba) |
| Peak host memory of the whole pipeline, measured with a cheap stand-in model (independent of which SR model is selected) | 870 MiB (256 × 256) → 2.2 GiB (511 × 777) → 4.7 GiB (1024 × 1024) | (same measurement) | `TILING.md` |
| Full evaluation / reliability / downstream runs (Mamba dominates) | – | ≈ 35–45 min / ≈ 45 min / ≈ 41 min | `EVALUATION.md`, `RELIABILITY.md`, `DOWNSTREAM.md` |

* **The API is synchronous**, in-process and in-memory; a job is the sum `tiles × 6 passes × per-tile time`. Extrapolating the measured Mamba per-tile time to the 1024 × 1024 input cap (121 tiles) gives about 20 minutes per request:
  **that is arithmetic, not a measurement**. Concurrent requests are not bounded. **No production throughput is claimed.**
* **Training (Phase 4, smoke scale, synthetic data):** `tiny_cnn` ≤ 27 MiB and ≈ 4 ms per step; Lite fine-tune 122 MiB (fp32) and 22.7 ms per step at the native tile; Mamba needs FRAME-side activation checkpointing (1,164 MiB and 7.5 s per micro-step at 128 → 512, batch 1; out of memory beyond a 32 × 32 crop without it);
  the Mamba fine-tune smoke ran 30 optimizer steps at 2.97 s / step. Extrapolations to real training (10,000 steps ≈ 33 h locally at 64-px crops) are labelled as such in `TRAINING.md`; a cloud GPU is recommended, none was used.

## 4. What exists but is not part of the evaluated product

| Component | Status |
|---|---|
| `frame/train` (config, leakage gate, losses, checkpoints, resume, validation) | implemented and tested; **used only for synthetic-data smoke experiments**; no real-data training was run ([`TRAINING.md`](TRAINING.md)) |
| Trained checkpoints under `experiments/training/` | `tiny_cnn` (47 k parameters, synthetic) and a 30-step Mamba fine-tune smoke; **not selectable in the API** and not evaluated as a product |
| `frame.data` adapters for SEN2NAIPv2 / SEN2VENµS / India | format adapters; see §2 |
| `frame.evaluate`, `frame.reliability`, `frame.downstream` | offline research tooling with their own CLIs; not called by the API |
