# Reproducibility specification and runbook

What is needed to repeat FRAME's results, what was recorded for each, and what is **not** reproducible from the repository alone. Nothing here is a package version invented for tidiness: environment facts come from the environments as they exist on the development machine
(`experiments/final_readiness/environments/*.freeze.txt`) and from the records the runs themselves wrote.

## 1. Four levels, from cheap to expensive

| Level | Needs | Time here | Command |
|---|---|---|---|
| **A. Unit tests** | main environment only; no GPU, no weights, no network | ≈ 6 min | `TMPDIR=~/.cache/frame_tmp sen2sr_venv/bin/python -m pytest frame/tests -q --basetemp=~/.cache/frame_pytest_tmp` |
| **B. End-to-end smoke** | main environment; the toy model needs nothing else; Lite needs its cached weights; Mamba needs a CUDA GPU and the Mamba environment | toy ≈ 1 s · Lite ≈ 3 s · Mamba ≈ 65–70 s | `sen2sr_venv/bin/python -m frame.smoke [--model toy\|lite\|mamba]` |
| **C. Integration tests** | the real weights and (for Mamba) the GPU; they **skip**, never download, when something is absent | ≈ 3.5 min | `sen2sr_venv/bin/python -m pytest frame/tests -m integration -q` |
| **D. Research runs** | the datasets (outside the repository), the GPU, and 35–45 minutes each | ≈ 35–45 min | `python -m frame.{evaluate,reliability,downstream} run CONFIG` |

The repository's own `pyproject.toml` sets `testpaths = ["tests"]` (the upstream `sen2sr` tests) and `addopts = "-m 'not integration'"`. A bare `pytest` therefore runs the upstream tests only; **FRAME's tests are under `frame/tests` and must be named**. The frontend has its own suite: `cd frontend && npx vitest run`.

## 2. Environments

There are three Python environments, all in the repository root and all git-ignored; the main and the Mamba one are **deliberately separate** and must not be merged (the prebuilt `mamba-ssm` CUDA extension is tied to torch 2.6.0+cu118, and rebuilding it exhausted this machine's RAM).

| | Main (`sen2sr_venv`) | Mamba (`sen2sr_mamba_venv`) | `ldsrs2_venv` |
|---|---|---|---|
| Python | 3.11.9 | 3.11.9 | 3.14.6 |
| torch | 2.14.0 (`torch.__version__` = `2.14.0+cu130`, recorded in every run) | 2.6.0+cu118 | 2.14.0 |
| What runs there | API, Lite, tiling, data, training (main), evaluation, reliability, downstream, smoke, all tests | the Mamba worker only (`python -m frame.models.mamba_worker`, started by the API); Mamba fine-tune smoke | **unused by FRAME**: an exploratory environment for LDSR-S2; nothing imports it |
| Key packages | numpy 2.4.6 · scipy 1.17.1 · rasterio 1.4.4 · fastapi 0.141.1 · pydantic 2.13.5 · starlette 1.6.0 · httpx 0.28.1 · uvicorn 0.52.4 · mlstac 0.4.9 · scikit-image 0.26.0 · opensr-test 1.3.3 · safetensors 0.8.0 · sen2sr 0.8.5 · pytest 9.1.1 · pystac-client 0.9.0 · cubo 2026.2.0 | numpy 2.4.6 · safetensors 0.8.0 · einops 0.8.2 · timm 1.0.30 · **mamba-ssm 2.3.2.post1** · **causal-conv1d 1.5.2** · triton 3.5.0 (no rasterio, no pytest, no `frame` data layer) | – |
| Full list | `experiments/final_readiness/environments/main.freeze.txt` (141 distributions) | `…/mamba.freeze.txt` (47) | – |
| Lockfile | **none**: the repository has no dependency manifest for FRAME's own packages (`requirements.txt` is the upstream documentation toolchain; `pyproject.toml` declares only `sen2sr`'s minimal dependencies). The freeze files are a record of what ran, generated with `importlib.metadata`, **not** an install recipe | same | – |

Front end: Node 26.5.1; `frontend/package-lock.json` pins React 19.3.0, Vite 8.3.0, Vitest 5.0.0, TypeScript 6.0.3, geotiff 3.0.5 (`npm ci` reproduces it).

**Hardware and OS of record** (every timing in this repository): NVIDIA GeForce RTX 3050 Laptop GPU, 4096 MiB (3,759 MiB usable, ≈ 600 MiB used by the desktop), driver 610.43.03 (the main environment's torch is built for CUDA 13.0, the Mamba worker's for 11.8), Arch Linux, kernel 7.1.5, x86_64, 20 CPUs, 15.2 GiB RAM.
The Mamba path is **POSIX-only** and **CUDA-only**. Nothing was tried on another machine.

## 3. External artifacts, caches and where things live

| What | Default location | Override | In git? | Recorded as |
|---|---|---|---|---|
| SEN2SR-Lite weights | `~/.cache/sen2sr_baseline/SEN2SRLite_RGBN/` (fetched by `mlstac` on first use) | `SEN2SR_BASELINE_WEIGHTS_DIR` | no | SHA-256 in [`REGISTRY.md`](REGISTRY.md) §1 and in `smoke_lite/smoke_record.json` |
| SEN2SR-Mamba weights | `<repo>/models/SEN2SR/` | `FRAME_MAMBA_WEIGHTS_DIR` | no (`/models/`) | `metadata.model_runtime.weights_sha256` in every Mamba result |
| Mamba interpreter | `<repo>/sen2sr_mamba_venv/bin/python` | `FRAME_MAMBA_PYTHON` | no | `model_runtime.worker_python`, `worker_torch` |
| Datasets | `$FRAME_DATA_ROOT`, default `~/.cache/frame_data` (`sen2neon/`, `sen2naipv2/`, `synthetic_smoke/`); OpenSR-Test in `~/.config/opensr_test/` | `FRAME_DATA_ROOT` | no | manifests: relative paths only, content digest, dataset revision, LR checksums |
| API workspace (uploads, jobs, GeoTIFFs, tensors) | `~/.cache/frame_api/workspace/` | `FRAME_API_WORKSPACE_DIR` | no | job ids; nothing is auto-deleted |
| Research caches | `~/.cache/frame_eval`, `frame_reliability`, `frame_downstream` (per-tile evidence / region tables) | in each config's `cache_dir` | no | per-tile SHA-256 in the run record |
| Experiment records | `experiments/<area>/…` | – | **yes** (JSON / Markdown; rasters and checkpoints excluded by `.gitignore`) | configs, digests, README per run |

The Lite and Mamba manifests point at `…/resolve/main/…` on Hugging Face, **not at a pinned revision**, and the Baseline 0 STAC window has drifted. A re-download or a re-fetch can differ from what the records ran; compare the SHA-256 values before comparing results.

## 4. Seeds, digests and what identifies a run

| Item | Value or rule |
|---|---|
| TTA ensemble seed | 42 (`FRAME_API_UNCERTAINTY_SEED`; overridable per request); six views: identity, hflip, vflip, rot90, rot180, rot270 |
| Bootstrap | seed 0, 2000 resamples, α = 0.05 (Phases 5–7) |
| Sample and split seeds | SEN2NEON random-30 sample seed 0 (manifest `experiments/evaluation/manifests/sen2neon_random30_seed0.jsonl`); geographic splits are seeded and recorded in the manifest |
| Config digest | SHA-256 of the canonical JSON of the config (`frame.{evaluate,reliability,downstream}.config`, `frame.train.config`); recorded in `config.json` / `summary.json` |
| Manifest digest | content digest of the JSONL manifest (byte-reproducible), recorded by training and evaluation |
| Weights | SHA-256 of the executed files ([`REGISTRY.md`](REGISTRY.md) §1) |
| Smoke run | `settings_digest`, `scene_sha256`, `output_sha256`, seed, git revision and dirty flag (`smoke_record.json`) |
| Reference gate | `frame-reliability-gate/1`, recorded on every tile row; metrics `frame-eval-metrics/1` |
| Code | parent git revision **plus a dirty flag** |

**The working tree is not committed.** Every record that captures the git state (Phases 1–8) carries revision `eb3769559525ea81e902422253f8950c36b075a5` (the last commit) with `dirty: true`, because the whole gap-closure work (Mamba, tiling, data, training, evaluation, reliability, downstream, this phase) is uncommitted.
That fact is preserved in the records and was not edited. After the work is committed, new runs will carry the new revision; the historical ones will still name the parent. **Commit before quoting a revision.** FRAME has no version string of its own (`GET /health` reports `frame_version: null`; the API version is `0.1.0`).

## 5. Determinism, as verified

* **Same machine, same device, same seed, same scene: bit-identical output.** Verified for the toy model, the real SEN2SR-Lite and the real SEN2SR-Mamba on the 200 × 300 smoke scene (a repeat run's raster hash equals the first; `identical_repeat: true`), for the tile engine with Mamba on a real scene (`TILING.md`), for the Phase 9 end-to-end run, and for the training loop on CPU and GPU (`TRAINING.md`).
* **Not claimed:** bit-identity across GPUs, drivers, CUDA or library versions, or CPU versus GPU (a CPU / GPU training comparison differed by up to 0.27 in weights). Floating-point accumulation order is a known source of cross-hardware difference.
* Tile order, blending and padding are fixed; the tile engine adds no randomness. The only randomness in the product path is the TTA seed, which is explicit.

## 6. Principal commands

```bash
# ---- one-time: the two environments already exist here; see §2. Weights: Lite is fetched on first use; Mamba's are placed in models/SEN2SR/.

# ---- demo path (details in DEMO_RUNBOOK.md)
sen2sr_venv/bin/uvicorn frame.api.app:app --port 8000        # backend (Mamba worker starts on first use)
cd frontend && npm install && npm run dev                    # front end at http://localhost:5173

# ---- end-to-end smoke (writes smoke_record.json; the record is JSON only)
sen2sr_venv/bin/python -m frame.smoke                        # toy model: no weights, no GPU
sen2sr_venv/bin/python -m frame.smoke --model lite --repeat  # real Lite
sen2sr_venv/bin/python -m frame.smoke --model mamba          # real Mamba (≈ 65–70 s)

# ---- tests
sen2sr_venv/bin/python -m pytest frame/tests -q                    # unit
sen2sr_venv/bin/python -m pytest frame/tests -m integration -q     # real weights / GPU; skips if absent
(cd frontend && npx vitest run)                                     # front end

# ---- research analyses (each writes a NEW output directory; an existing one is refused)
sen2sr_venv/bin/python -m frame.evaluate    check experiments/evaluation/configs/benchmarks_v1.json    # then `run` (35-45 min)
sen2sr_venv/bin/python -m frame.reliability check experiments/uncertainty/configs/reliability_v1.json  # then `run` (≈ 45 min); `--reuse-evidence` repeats only the analysis
sen2sr_venv/bin/python -m frame.downstream  check experiments/downstream/configs/downstream_v1.json    # then `run` (≈ 41 min); `--reuse-regions` repeats only the analysis
sen2sr_venv/bin/python -m frame.downstream  smoke                                                       # synthetic, no data

# ---- data and training (synthetic smoke set; no download)
sen2sr_venv/bin/python -m frame.data synthetic --out unsplit.jsonl --regions 5 --scenes-per-region 2 --seed 0
sen2sr_venv/bin/python -m frame.train check experiments/training/smoke_tiny_cnn/config.json
```

The `check` commands validate the config, the datasets, the systems and (for reliability and downstream) preview the reference gate **without running a model**; run them first. Exit codes of every CLI: `0` success · `1` finished but a dataset had no usable evidence (or a smoke check failed) · `2` refused or invalid input.

## 7. What is not reproducible from the repository alone

* **The datasets** (SEN2NEON, OpenSR-Test, the SEN2NAIPv2 sample): fetched separately; manifests and the records name exactly what was read, but the files are not in git.
* **The exact weights**, unless the same revision is fetched (§3); compare hashes.
* **The STAC scene of the early experiments** (catalogue drift): reuse the saved tensor.
* **A clean install of the environments**: there is no lockfile (§2); the freeze files list what worked.
* **Timings** on other hardware; **any GPU number** on other GPUs.
* **A record's git revision** until the tree is committed (§4).

## 8. Provenance audit (Phase 8): inconsistencies found and what was done

Historical records were **not** edited to look cleaner. Where a statement is superseded, a pointer was added and the original left in place.

| Finding | Action |
|---|---|
| The root `README.md` is the upstream `sen2sr` README; nothing pointed to FRAME | [`README.md`](README.md) in `docs/` is the FRAME entry point; the root README is unchanged |
| Two phase-numbering sequences (the original nine-phase build and the gap-closure roadmap) are both called "Phase N" | explained in [`README.md`](README.md) §3; a banner in `FRAME_TECHNICAL_SPEC.md` |
| `FINAL_SCIENTIFIC_AUDIT.md` claimed final authority yet predates Mamba, tiling and Phases 5–7, and reads the model as "marginally better than bicubic" from 9 unregistered samples | banner at its top pointing to [`CLAIMS.md`](CLAIMS.md); body unchanged |
| `experiments/validation/README.md` reads "outperforms bicubic" (9 samples, before the registration analysis) | pointer to `CLAIMS.md` §5 added at its top; body unchanged |
| The Mamba input-range check applied to the Mamba path only; Lite accepted raw digital numbers as reflectance silently; a scene called raw DN that already held reflectance produced a black image with no error | Phase 8: one content validation at upload and at run time for **both** models (`frame.preprocessing`, `frame/api/services/pipeline.py`), with tests |
| A file with a `.tif` name that is not a GeoTIFF, an unexpected exception, and a model returning the wrong number of channels produced a **plain-text 500**, contradicting the documented JSON error body; some messages carried server paths | Phase 8: named `UnreadableRasterError`; a JSON `internal_error` handler; per-call output check; file names only in messages; tests |
| A missing or corrupt Lite artifact surfaced as a raw network / library exception | Phase 8: `ModelUnavailableError` (503) / `ModelLoadError`, nothing cached on failure |
| NaN/Inf pixels were zero-filled but counted as valid, so the reported coverage was 100 % | Phase 8: non-finite pixels are excluded from the mask and the coverage |
| The stability was labelled "relative model-stability uncertainty" although Phase 6 measured it as weakly informative and uncalibrated; the NDVI view had no such caveat | Phase 8: relabelled "TTA stability — reconstruction-variation diagnostic" (API, UI, tests, docs); an NDVI demonstration note added. The JSON field is still named `uncertainty` for compatibility |
| The Lite manifest (`SPAN`, 472,496 parameters, 1,889,984 bytes) and the Mamba manifest (`Swin2SR`, 12,894,526 parameters) differ from the executable artifacts | **not resolvable here**: recorded in [`REGISTRY.md`](REGISTRY.md) §1 with the executable truth |
| Mamba versus Lite cost quoted as ≈ 300× (per tile, Phase 1), ≈ 90× (HTTP pipeline, Phase 2) and ≈ 150× (per pass, Phase 6) | not a contradiction: different quantities timed. All three are kept in [`REGISTRY.md`](REGISTRY.md) §3 |
| The frontend README still said the input must be 128 × 128 | corrected |
| `.gitignore` (the Python-packaging template's `downloads/` and `lib/` patterns) silently excluded two **FRAME frontend source directories**, `frontend/src/components/downloads/` and `frontend/src/lib/` (colormap, GeoTIFF decoding, rendering): they were never in the commit, so a clean clone could not build the front end | re-included with two negation rules in `.gitignore`; the directories now show as untracked and are ready to be committed (nothing was committed) |
| Test counts quoted in older documents (531, 1087, 1358, …) | historical: each was true when written; the current count is in [`RELEASE_READINESS.md`](RELEASE_READINESS.md) |
| Not changed: unpinned artifact revisions; the missing lockfile; the uncommitted tree; `pytest`'s default `testpaths` | documented above |
