# FRAME release readiness (Phase 8)

A plain statement of the state of the repository: what works, what was measured, what is limited, what was not done. It describes the code and the evidence; it does not score them. The machine-readable counterpart is
`experiments/final_readiness/status.json`. Read [`CLAIMS.md`](CLAIMS.md) for what may be said about any of it.

**State of the tree.** Revision `eb3769559525ea81e902422253f8950c36b075a5` plus **uncommitted work** (the whole gap-closure roadmap, Phases 1–8, is uncommitted; `sen2sr/` has no diff). Every record names that parent revision with `dirty: true`. Nothing was committed by Phase 8.

## 1. What is ready (engineering, tested)

* **Scene in, georeferenced 4× product out.** A four-band RGBN Sentinel-2 L2A GeoTIFF of any height and width (up to 1,048,576 input pixels) is validated, scaled, masked, tiled, run through the selected model, blended, and written as a GeoTIFF whose CRS, origin and footprint equal the input's and whose pixel size is 2.5 m. Checked on read-back from disk for a rectangular scene that is not a multiple of 128 (200 × 300 → 800 × 1200), on the toy stand-in, the real SEN2SR-Lite and the real SEN2SR-Mamba.
* **Explicit model selection.** `lite` (default) or `mamba`. Unknown → 422, unavailable → 503, no silent substitution in either direction; the model that ran is named in the result and in the GeoTIFF tag.
* **Input validation with named errors and a uniform JSON error body**: not a GeoTIFF, wrong or missing bands, no CRS, no valid pixel, a declared scale the values contradict, oversized scene, model unavailable, model contract broken. No traceback or server path is ever returned.
* **A stability diagnostic and an NDVI demonstration** in the API and the UI, each labelled for what it is (an uncalibrated TTA reconstruction-variation diagnostic; a demonstration that is not evidence of a downstream advantage).
* **Offline research tooling with one shared reference gate**: `frame.evaluate`, `frame.reliability`, `frame.downstream` (each with `check` / `run` and a machine-readable record); `frame.data` (paired data, splits, leakage controls); `frame.train` (training and checkpointing, synthetic smoke only).
* **A deterministic end-to-end smoke path** (`python -m frame.smoke`) for the toy, Lite and Mamba models with a JSON record, and a demo-input generator (`--demo-scenes`).
* **Provenance**: config digests, seeds, weight hashes, git revision and dirty flag, manifests, environment capture, per-tile records; a bit-identical repeat verified on this machine for the toy, Lite and Mamba models.

## 2. What is demonstrated (measured on real data, with limits)

Numbers and intervals are in [`EVALUATION.md`](EVALUATION.md), [`RELIABILITY.md`](RELIABILITY.md) and [`DOWNSTREAM.md`](DOWNSTREAM.md); the summary of each is in [`CLAIMS.md`](CLAIMS.md) §2.

* Against independent (different-sensor) references, SEN2SR-Lite and SEN2SR-Mamba differ from bicubic by small amounts whose sign depends on the dataset. Both are below bicubic on the pixel-wise and spectral metrics on SEN2NEON (PSNR −0.46 dB for each, 28 scene units) and within ≈ 0.1 dB of it on OpenSR-Test. **No ranking is made.**
* The reference registration error is a first-order effect (median 1.0 HR px on SEN2NEON), so a gate admits only registered tiles to any pixel-level analysis: 12 of 30 SEN2NEON tiles (11 units), 6 / 21 / 13 OpenSR-Test tiles (6 / 5 / 4 units).
* The TTA stability is weakly associated with reconstruction error, about as strongly as image texture; near chance for high-error detection on SEN2NEON; **uncalibrated** (its spread is 16–32× smaller than the error).
* On an NDVI vegetation decision (fixed regions, a predeclared threshold of 0.3) super-resolution changed region-level NDVI by about 0.001 and pixel decisions by at most 0.8 percentage points, with a dataset-dependent sign: **mixed to null**. The stability carries at most a small, dataset- and model-dependent residual association with the downstream error beyond texture.
* Mamba costs about two orders of magnitude more than Lite (≈ 1.7 s against ≈ 6–19 ms per tile; the 200 × 300 smoke scene with the six-view ensemble: ≈ 62–69 s against ≈ 1 s) on this machine.

## 3. What is evidence-limited

* **Small, narrow, cross-sensor evidence.** 4–11 registration-eligible scene units per dataset; North America and Spain only; the references are a different sensor; two of the four datasets are descriptive only (fewer than 5 units).
* **Not a test of downstream benefit on labels.** The NDVI decision is a vegetation proxy; there are no land-cover labels.
* **The stability's usefulness** is a weak, texture-like association, not a tested selective-prediction tool.
* **One machine.** All timings and the bit-identity verification are for the RTX 3050 Laptop GPU (4 GB) of the development machine.
* **Artifacts are not revision-pinned** and the environments have no lockfile (freeze files record what ran); the model manifests disagree with the executables in parameter counts and file sizes (recorded, not resolved).

## 4. What remains deferred

Real-data training (the first run would be one SEN2NAIPv2 part, after `tacoreader` and the geography of that part are checked); any Indian data or domain-shift experiment; land-cover and other downstream tasks needing region-level labels; the 12-band / SWIR products, LDSR-S2 and any other architecture;
alternative or calibrated uncertainty methods; STAC acquisition inside the product; asynchronous jobs, authentication, containers, deployment and load testing. None is started; none is claimed.

## 5. How to run the demo

Full flow: [`DEMO_RUNBOOK.md`](DEMO_RUNBOOK.md). In short:

```bash
sen2sr_venv/bin/python -m frame.smoke --demo-scenes ~/.cache/frame_demo   # demo inputs (real ones if cached)
sen2sr_venv/bin/python -m frame.smoke                                     # pipeline check, ~1 s, no weights
sen2sr_venv/bin/uvicorn frame.api.app:app --port 8000                     # backend
(cd frontend && npm run dev)                                              # http://localhost:5173
```

## 6. How to reproduce the key analyses

[`REPRODUCIBILITY.md`](REPRODUCIBILITY.md) §6 lists every command with its runtime and prerequisites. The three research analyses each take 35–45 minutes with Mamba and read datasets that live outside the repository; each `check` command validates everything and previews the reference gate in seconds without running a model.
The committed run records reproduce the numbers in the documents: `experiments/evaluation/benchmarks_v1/`, `experiments/uncertainty/runs/reliability_v1/`, `experiments/downstream/runs/downstream_v1/`.

## 7. Known resource constraints

* **GPU.** 4 GB laptop GPU shared with the desktop (≈ 600 MiB): Mamba inference needs ≈ 0.6–0.7 GiB; Mamba training needs FRAME-side activation checkpointing (1.2 GiB per micro-step at the native tile) and is impractically slow locally beyond a few thousand steps (a cloud GPU is recommended; none was used).
* **Host memory.** The pipeline peaks at ≈ 4.7 GiB for a 1,024 × 1,024-pixel input (the API's default cap); concurrent requests are not bounded.
* **Time.** A synchronous request costs `tiles × 6 ensemble passes × per-tile time`: seconds with Lite, about a minute for the 200 × 300 scene with Mamba, about 20 minutes (arithmetic, not measured) at the input cap.
* **Disk.** Nothing large is in git. Outside it: datasets ≈ 1.2 GB (`~/.cache/frame_data` 773 MB, `~/.config/opensr_test` 417 MB), the Mamba weights 367 MB (`models/`, git-ignored), research caches ≈ 130 MB, the cached real scene 17 MB, and the **API workspace, which is never cleaned** (923 MB here after the development runs: each job writes two rasters and three tensors, ≈ 16× the input pixels). `experiments/` is 105 MB on disk, of which the largest single files are two evaluation JSON records of 14 MB and 7 MB.
* **Platform.** The Mamba worker is POSIX-only and CUDA-only.

## 8. Final test status

| Suite | Command | Result |
|---|---|---|
| Backend unit (whole `frame/tests`, GPU-free) | `pytest frame/tests` | **2044 passed**, 0 failed, 0 skipped; 33 deselected as `integration` |
| Backend integration (real Lite and Mamba, the GPU, the real API and tile engine) | `pytest frame/tests -m integration` | **33 passed**, 0 failed, 0 skipped |
| Frontend | `cd frontend && npx vitest run` | **70 passed** in 13 files; `tsc -b` clean; `npm run build` succeeds; `oxlint`: one warning, in `useRaster.ts`, which Phase 8 did not touch |
| End-to-end smoke, same synthetic 200 × 300 scene | `python -m frame.smoke --model toy\|lite\|mamba --repeat` | **22 / 22 checks each**; repeat run bit-identical for all three; `sr/run` 0.6 s / 2.4 s / 68.9 s |
| Downstream smoke | `python -m frame.downstream smoke` | completed (6 synthetic scenes, 6,144 regions) |
| Geospatial, tile-engine geometry, integration-contract, **ineligible-reference** and reference-gate regressions (run again on their own) | `pytest test_geospatial_geotiff test_geospatial_transform test_tiling_geospatial test_integration_contract test_downstream_runner test_reliability_eligibility` | **140 passed** |
| Live UI against the real backend (Playwright / Chromium, not part of the suites) | – | Lite on the real 200 × 300 crop (5.8 s; 4 × 200 × 300 → 4 × 800 × 1200) and Mamba on the real 128 × 128 scene (14.3 s) both completed; a real reflectance file with "Raw digital number" was refused inline, then accepted after switching; the Stability and NDVI tabs showed the new wording; no page errors |

Everything above was run after the last code change of Phase 8. No test was weakened or deleted; one real-model integration test that had been declaring a reflectance scene as raw digital numbers (and passing on a near-black image because it only checked shapes) was corrected and given an assertion on the output scale.

## 9. Changes made in Phase 8

Integration hardening (each with tests): non-finite pixels excluded from the mask and the reported coverage; opt-in content validation (no valid pixel, a contradicted input scale) used by the API at upload and run time for both models; a named `UnreadableRasterError`; a JSON error handler for unexpected exceptions; a per-call model-output geometry check;
missing or corrupt Lite artifacts as named errors (nothing cached on failure); server paths removed from error messages. Wording: the stability relabelled and its Phase 6 finding stated in the API and UI; an NDVI demonstration note. New: `frame/smoke.py`, `experiments/final_readiness/`, and the documents listed in [`README.md`](README.md) §4.
A `.gitignore` defect was fixed (its `downloads/` and `lib/` patterns had excluded the front end's `components/downloads/` and `lib/` source directories from git, so a clean clone could not build). Superseding banners were added to `FINAL_SCIENTIFIC_AUDIT.md` and `experiments/validation/README.md`; their bodies are unchanged.
