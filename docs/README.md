# FRAME documentation

FRAME (SIH 2026, problem statement 26142, NTRO) is an orchestration, validation and demonstration layer **around** the unmodified, published SEN2SR super-resolution models (`sen2sr/` has no diff). It takes a Sentinel-2 L2A RGBN scene,
produces an **SR-derived product on a 2.5 m pixel grid** with its georeferencing intact, adds a test-time-augmentation **stability diagnostic** and a small NDVI demonstration, and ships offline tooling that measures how far any of this can be trusted.
It is a demo-grade, single-process prototype. Sentinel-2 has never observed the ground at 2.5 m; nothing here is native 2.5 m imagery.

**Read [`CLAIMS.md`](CLAIMS.md) before you present or quote anything.** The measured results are mixed on purpose and are reported as such: the stability diagnostic is uncalibrated and only weakly informative; super-resolution changed a downstream NDVI decision only slightly and with a dataset-dependent sign.

## 1. Where to start

| I want to… | Read / run |
|---|---|
| know what is ready, what is evidence-limited, what is deferred | [`RELEASE_READINESS.md`](RELEASE_READINESS.md) |
| give the SIH demonstration | [`DEMO_RUNBOOK.md`](DEMO_RUNBOOK.md) |
| check the pipeline works right now, without data or a GPU | `sen2sr_venv/bin/python -m frame.smoke` |
| know which models and datasets exist and what they cost | [`REGISTRY.md`](REGISTRY.md) |
| know exactly what may and may not be claimed | [`CLAIMS.md`](CLAIMS.md) |
| see each of the 12 requirements against what was built and measured | [`TRACEABILITY.md`](TRACEABILITY.md) |
| reproduce something | [`REPRODUCIBILITY.md`](REPRODUCIBILITY.md) |
| use the HTTP API | [`../frame/api/README.md`](../frame/api/README.md) |
| use the front end | [`../frontend/README.md`](../frontend/README.md) |

## 2. How the pieces fit

```
                         PRODUCT PATH (frame/api FastAPI  ·  frontend/ React)
 Sentinel-2 L2A GeoTIFF ─▶ preprocessing ─▶ model selection ─▶ tiling / reconstruction ─▶ geospatial output ─▶ TTA stability ─▶ self-consistency ─▶ NDVI demo
 (RGBN, any H x W)         frame.preprocessing frame.models       frame.tiling               frame.geospatial      frame.uncertainty  frame.consistency    frame.analysis
                           validate, scale,    lite | mamba,      128 px tiles, overlap 32,  CRS / origin kept,    six geometric      SR back to the       NDVI of the SR
                           mask, refuse        no fallback        reflect pad, blend         exactly 4x, 2.5 m     views, spread      input grid, not      product vs its own
                           bad input                                                                                                  accuracy             input; a demonstration

                         RESEARCH PATH (offline CLIs; nothing here is called by the API)
 frame.data ─▶ frame.train ─▶ frame.evaluate ─▶ frame.reliability ─▶ frame.downstream
 paired data,  training       accuracy vs        does the stability   does SR change an NDVI decision,
 splits, QC    (synthetic     independent        inform about error?  and does the stability track its
               smoke only)    references         (reference gate)     mistakes?  (the same reference gate)
```

* **One public entry path.** For a scene: `POST /upload` → `POST /sr/run` (`"model": "lite" | "mamba"`) → download GeoTIFFs → optional `POST /analysis/ndvi`; the front end drives exactly this. For a check without a server: `python -m frame.smoke`.
* **The heavy research tooling is separate on purpose.** `frame.evaluate`, `frame.reliability` and `frame.downstream` are commands that read datasets and take tens of minutes; they are not part of a request. They share one reference eligibility gate (`frame-reliability-gate/1`): a tile whose reference is not registered to the prediction grid cannot enter any pixel-level analysis.
* **Models.** SEN2SR-Lite runs in the API process; SEN2SR-Mamba runs in an isolated worker in its own environment (CUDA only). Selection is explicit; there is no fallback and no third model ([`REGISTRY.md`](REGISTRY.md)).

## 3. Two "Phase" numbering systems (a source of confusion, explained once)

The repository was built in two sequences, and both call their steps "Phase N".

| Original nine-phase build (`FRAME_TECHNICAL_SPEC.md` table; `frame/*/README.md`, `frame/api/README.md`, `frontend/README.md`, `FINAL_SCIENTIFIC_AUDIT.md`) | Gap-closure roadmap (`docs/MAMBA_INTEGRATION.md` … `DOWNSTREAM.md`, this page) |
|---|---|
| 0 baseline · 1 preprocessing · 2 geospatial · 3 consistency · 4 external-reference validation · 5 TTA stability · 6 NDVI demonstration · 7 FastAPI backend · 8 React front end · 9 final integration and hardening | 1 Mamba integration · 2 arbitrary-size tiling · 3 paired data layer · 4 training · 5 evaluation · 6 reliability / uncertainty validation · 7 downstream utility · **8 final integration and readiness** |

So "Phase 6" in `frame/analysis` (the NDVI demonstration) and "Phase 6" in `RELIABILITY.md` (whether the stability tells anything about error) are different things. When a document says which sequence it uses, believe it; the roadmap documents say so in their first lines.

## 4. The documents

| Document | What it is | Sequence |
|---|---|---|
| [`Requirements 142.txt`](<Requirements 142.txt>) | the twelve requirements and the research memo behind them (not rewritten) | – |
| [`FRAME_TECHNICAL_SPEC.md`](FRAME_TECHNICAL_SPEC.md) | the original design rationale, tagged `[FACT]` / `[PROPOSAL]` / `[OPEN QUESTION]` as of when each section was written | original |
| [`FINAL_SCIENTIFIC_AUDIT.md`](FINAL_SCIENTIFIC_AUDIT.md) | the audit that closed the original build; **partly superseded** by `CLAIMS.md` | original |
| [`MAMBA_INTEGRATION.md`](MAMBA_INTEGRATION.md) | SEN2SR-Mamba behind an isolated worker; contract, measurements | roadmap 1 |
| [`TILING.md`](TILING.md) | the tile engine for scenes of any size; measurements | roadmap 2 |
| [`DATA.md`](DATA.md) | the paired LR–HR data layer, dataset roles, splits | roadmap 3 |
| [`TRAINING.md`](TRAINING.md) | the training pipeline and the synthetic smoke experiments | roadmap 4 |
| [`EVALUATION.md`](EVALUATION.md) | accuracy and spectral / spatial correctness against real references | roadmap 5 |
| [`RELIABILITY.md`](RELIABILITY.md) | the reference gate and whether the stability informs about error | roadmap 6 |
| [`DOWNSTREAM.md`](DOWNSTREAM.md) | an NDVI decision task with the same gate | roadmap 7 |
| [`REGISTRY.md`](REGISTRY.md) · [`CLAIMS.md`](CLAIMS.md) · [`TRACEABILITY.md`](TRACEABILITY.md) · [`REPRODUCIBILITY.md`](REPRODUCIBILITY.md) · [`DEMO_RUNBOOK.md`](DEMO_RUNBOOK.md) · [`RELEASE_READINESS.md`](RELEASE_READINESS.md) | the consolidation: artifacts and evidence, claims, requirements, reproduction, the demo, readiness | roadmap 8 |

**Which document wins.** For *what may be claimed*: `CLAIMS.md`. For *what a run measured*: the run's own record under `experiments/` and the roadmap document that describes it. For *how a module behaves*: its package README and tests.
The design spec's tags describe understanding at the time; each experiment README and package README is the authoritative record of what was actually built and verified for its scope. The repository's root `README.md` is the upstream `sen2sr` project's and is unchanged.

## 5. Repository map

```
frame/            the FRAME packages (preprocessing, geospatial, consistency, validation, uncertainty, analysis, api | models, tiling, data, train, evaluate, reliability, downstream | smoke)
frame/tests/      FRAME's tests (run them with an explicit path; see REPRODUCIBILITY.md §1)
frontend/         the React console
experiments/      configs, run records (JSON / Markdown) and small metadata; datasets, caches and checkpoints live outside git
docs/             this documentation
sen2sr/           upstream, unmodified
models/           git-ignored model artifacts (Mamba)
```
