# SEN2SR-Mamba integration (gap-closure roadmap, Phase 1)

Status: **implemented and verified on the development machine** (RTX 3050 Laptop GPU, 4 GB).
Scope: the actual SEN2SR **Mamba RGBN 10 m → 2.5 m** model is usable from FRAME through a
tested, isolated path, selectable next to the existing SEN2SR-Lite baseline.

> Numbering note: this is Phase 1 of the *post-Phase-9 gap-closure roadmap*, not the original
> "Phase 1 — preprocessing" of `FRAME_TECHNICAL_SPEC.md`.

## What this phase does and does not establish

| Established | Not established (out of scope here) |
|---|---|
| The real `MambaSR` weights load strictly (0 missing / 0 unexpected keys) and run on CUDA | The full 10-band cascade (`f2_model`, `model.safetensor`, both constraint files are **not used**) |
| 4 × 128 × 128 → 4 × 512 × 512, deterministic, finite | ~~Tiling of scenes larger than 128 × 128~~ — done in Phase 2 of the roadmap, see [`TILING.md`](TILING.md) |
| Isolated from the main environment; Lite is untouched | Any training or fine-tuning |
| Selectable via API and UI, with clear errors when unavailable | Accuracy against reference imagery, Indian generalisation, calibrated uncertainty |
| Measured latency and GPU memory on this machine | LDSR-S2, STAC, COG, async jobs, Docker |

The output is an **SR-derived product on a 2.5 m pixel grid**. Sentinel-2 has never observed the
ground at 2.5 m; this is not a native 2.5 m measurement.

## The model artifact

Location: `models/SEN2SR/` (configurable with `FRAME_MAMBA_WEIGHTS_DIR`). Not versioned
(`/models/` is git-ignored; ~367 MB). Only three of its files are used:

| File | Used | Role |
|---|---|---|
| `sr_model.safetensor` | yes | `MambaSR` RGBN weights (52.6 MB, 1228 tensors, 13,759,444 parameters) |
| `sr_hard_constraint.safetensor` | yes | 512 × 512 low-pass mask for the Fourier hard constraint |
| `example_data.safetensor` | tests / validation only | a 10-band 128 × 128 tile (`lr`, float32 reflectance) |
| `load.py`, `mlm.json` | reference | executable loader / manifest (see "Metadata inconsistency") |
| `model.safetensor`, `hard_constraint.safetensor`, `f2_*.safetensor` | **no** | the 10-band Swin2SR cascade — not integrated |

`sr_model.safetensor` SHA-256: `11e551b03663e873c5dcd4f02ec3fc8cdfa41ef46677540295ee4c6686ddc854`
(recorded in every Mamba result's `metadata.model_runtime.weights_sha256`).

### Architecture (single source of truth: `frame/models/config.py`)

`sen2sr.models.opensr_baseline.mamba.MambaSR(**MAMBA_ARCHITECTURE.kwargs())`:

```
img_size=(128, 128)   in_channels=4    out_channels=4     embed_dim=96
depths=[8,8,8,8,8,8]  num_heads=[8,8,8,8,8,8]             mlp_ratio=4
upscale=4             attention_type="sigmoid_02"         upsampler="pixelshuffle"
resi_connection="1conv"                                   operation_attention="sum"
```

`frame/tests/test_models_config.py` parses `models/SEN2SR/load.py` and fails if these ever
differ from the artifact's own loader.

## Input / output contract

Enforced in code (`frame/models/contract.py`), on both sides of the process boundary.

| | Input | Output |
|---|---|---|
| Shape | `(4, 128, 128)` or `(B, 4, 128, 128)` — exactly 128 × 128 | `(4, 512, 512)` or `(B, 4, 512, 512)` — same rank as the input |
| Channels / order | **B04, B03, B02, B08** (red, green, blue, NIR) | same order |
| dtype | `float32` (anything else is rejected) | `float32` |
| Values | surface reflectance as a fraction, `[-0.1, 6.5535]` and finite | finite; measured `[0.022, 1.353]` on the artifact tile, `[0.080, 0.727]` on the Baseline 0 scene |

**Band order is verified, not assumed.** The upstream cascade (`sen2sr/referencex4.py`) slices its
10-band tensor with `bands_10m = [2, 1, 0, 6]` — with the artifact's band list
`B02,B03,B04,B05,B06,B07,B08,B8A,B11,B12` that is B04, B03, B02, B08 — before calling this exact
model, and only reorders the *output* (`[2, 1, 0, 3]`) to restore Sentinel-2 file order for its own
10-band stack. FRAME already used B04,B03,B02,B08 (`RGBN_BANDS`), so **no reordering happens
anywhere in the adapter**; `test_models_config.py` re-derives the order from those two sources.
The order matters to the network: swapping red and blue changes its output (integration test).

Value bounds come from the L2A encoding (largest uint16 DN ÷ 10000; smallest BOA-offset-corrected
DN), not tuning. Raw digital numbers (thousands) fall far outside and are **rejected with an
actionable message** instead of silently producing a garbage SR image. Non-finite input is rejected
too — verified: a single NaN input pixel turns the entire affected band of the output NaN (all
512 × 512 pixels), because the hard constraint works in the Fourier domain.
The check applies to the Mamba path; the Lite path is unchanged and has no such check. *(Phase 8: the HTTP API now validates the scene's values against the declared input scale and refuses a scene with no valid pixel at upload and again at run time, for **both** models, so a Lite request can no longer treat raw digital numbers as reflectance or the reverse; the Lite model call itself is still unchecked.)*

## The hard constraint

Reused, not re-implemented. FRAME calls `sen2sr.nonreference.srmodel(sr_model, hard_constraint, device)`
with `sen2sr.models.tricks.HardConstraint(low_pass_mask=…)` — exactly what `models/SEN2SR/load.py`
does for its RGBN stage. Forward: `clamp(MambaSR(x), min=0)` then the Fourier recombination
(low frequencies from the bicubic-upsampled input, high frequencies from the network).

Measured effect: SR area-averaged back to the 10 m grid matches the input to RMSE 0.0031 (artifact
tile) / 0.0035 (Baseline 0 scene); a plain bicubic upsample scores 0.0036 on the artifact tile.

## Environment strategy: isolated worker, not a merged environment

| | Main FRAME env (`sen2sr_venv`) | Mamba env (`sen2sr_mamba_venv`) |
|---|---|---|
| torch | 2.14.0 | 2.6.0+cu118 |
| mamba-ssm / causal-conv1d | absent | 2.3.2.post1 / 1.5.2 (prebuilt CUDA extensions) |
| rasterio, fastapi, pydantic, opensr-test, scikit-image, mlstac, scipy | present | absent |

A merged environment is **not safe**: the prebuilt `mamba-ssm` extension is compiled against
torch 2.6.0+cu118, and rebuilding it from source against another torch previously exhausted this
machine's RAM. So the two stay separate and talk over a narrow tensor contract:

```
FRAME API (main env)
  └─ model selector: "lite" | "mamba"                       frame/api/services/model.py
       ├─ lite  → mlstac model, in-process                  (unchanged)
       └─ mamba → MambaWorkerClient                         frame/models/mamba_client.py
                    │  stdin/stdout, framed messages        frame/models/protocol.py
                    ▼
                  worker process (Mamba env)                frame/models/mamba_worker.py
                    └─ MambaRGBNModel                       frame/models/mamba_adapter.py
                         └─ MambaSR + HardConstraint  → 4 × 512 × 512
```

* **Long-lived worker**, started lazily on first use, reused across requests, stopped when the API
  process exits (or when its stdin closes). Loading the model per request would repeat a ~1.4 s load.
* **Protocol**: `[4-byte length][JSON header][raw little-endian float32 tensor]`. No pickle. The
  worker moves its real stdout to a private descriptor and points fd 1 at stderr, so a library that
  prints cannot corrupt a frame.
* **The main process never imports the Mamba runtime** (`test_models_isolation.py` checks
  `mamba_ssm`, `causal_conv1d`, `selective_scan_cuda`, `sen2sr` in a fresh interpreter). The worker
  needs only `torch`, `numpy`, `safetensors`, `einops`, `timm`, `mamba_ssm`, and the repository root
  on `PYTHONPATH` (the client sets it; `sen2sr` is not installed in that environment).
* **Failures are reported, not hung on**: worker startup errors, crashes, hangs (request timeout,
  the worker is killed) and CUDA OOM come back as typed errors; a crashed worker is restarted on the
  next request. Requests are serialised by a lock — one worker, one GPU.
* **CPU is refused, never fallen back to**: the selective-scan kernel is CUDA-only.
* Two CUDA contexts coexist (main process + worker); measured footprint below fits the 4 GB card.

Configuration (all in `frame/models/config.py`, overridable by environment variable):
`FRAME_MAMBA_WEIGHTS_DIR`, `FRAME_MAMBA_PYTHON`, `FRAME_MAMBA_STARTUP_TIMEOUT_S`,
`FRAME_MAMBA_REQUEST_TIMEOUT_S`.

## Model selection

* **API**: `POST /sr/run {"upload_id": …, "model": "lite" | "mamba"}` — `model` optional, default
  `"lite"`, case-insensitive; unknown value → `422` listing the supported ids. `GET /health` lists
  both models with an availability verdict and a user-facing reason (checked without starting a model).
  Unavailable → `503 model_unavailable`; input violating the contract → `422 model_input_invalid`;
  runtime failure → generic message (technical detail goes to the server log, never the response).
* **Provenance**: the result records `model_id` / `model_name`; the output GeoTIFFs' `FRAME_SR_VARIANT`
  tag is set to the model that actually ran (previously it was always the Lite name — a Mamba result
  would have been mislabelled); Mamba results add `metadata.model_runtime` (weights hash, parameter
  count, runtime versions).
* **UI**: a "Model — SEN2SR-Lite | SEN2SR-Mamba" control above the Run button. A model the backend
  reports unavailable is disabled with the backend's reason. Runtime library names are never shown
  (a test checks the copy).
* **Code**: `frame.models.selection` (ids, labels, validation) and `frame.api.services.model.get_model(device, model_name=…)`.

## Metadata inconsistency (documented, not rewritten)

`models/SEN2SR/mlm.json` was **not modified**. It labels the artifact `"Swin2SR model"` /
`mlm:architecture: "Swin2SR"`, but `models/SEN2SR/load.py` builds `MambaSR` for the RGBN stage.
Reading both: `mlm.json` describes the artifact's *primary* 10-band cascade entry — whose
`model.safetensor` is, per `load.py`, a Swin2SR — and lists the RGBN model only as an
"Auxiliar RGBN super-resolution model" asset. So the label is not wrong for that component, but it
is misleading as a description of the RGBN `MambaSR`, and several fields are stale for it: input
`data_type: float16` (the model asserts float32 internally), `file:size: 40455416` (the actual
`model.safetensor` is 188,160,632 bytes), `mlm:total_parameters: 12894526` (the RGBN model has
13,759,444), `mlm:framework_version: 2.1.2+cu121`.

FRAME keeps the two labels apart everywhere it records provenance:
`executable_architecture: "MambaSR"` and `artifact_metadata_label: "Swin2SR"`
(`frame/models/config.py`, `metadata.model_runtime`).

## Measured GPU validation

Reproduce: `sen2sr_venv/bin/python experiments/mamba_integration/run_validation.py`
(writes `experiments/mamba_integration/metadata/run_metadata.json`). Machine: RTX 3050 Laptop GPU,
4096 MiB (≈600 MiB in use by the desktop before the run), driver 610.43.03; worker torch 2.6.0+cu118.
Produced from the Phase 1 working tree on top of commit `eb37695`.

| Measurement | Value |
|---|---|
| Worker start (spawn + imports + weight load) | 2.12 s (model load 1.38 s of that) |
| First (cold) inference | 1.92 s |
| Warm inference, median of 10 (min – max) | 1.74 s (1.64 – 1.91 s) artifact tile; 1.73 s (1.64 – 2.13 s) Baseline 0 scene |
| Peak GPU memory, allocator, inside worker | 605.6 MiB |
| Worker process GPU footprint (`nvidia-smi`, incl. CUDA context) | 126 MiB after load → 724 MiB after inference |
| Input / output | `(1, 4, 128, 128)` float32 → `(1, 4, 512, 512)` float32 |
| Output range | artifact tile [0.0223, 1.3535]; Baseline 0 scene [0.0803, 0.7270] |
| NaN / Inf | none |
| Deterministic across calls | yes (bit-identical) |
| Full FRAME uncertainty ensemble (6 transforms) | 10.6 s |
| SEN2SR-Lite, same machine, for context | first 0.55 s; warm ≈ 6 ms; 100.5 MiB |

Notes: no kernel compilation occurs on first use (the selective scan is a prebuilt CUDA extension;
the cold/warm gap is ~0.2 s). Standalone runs measured ≈ 1.5 s warm; the laptop GPU's clocks vary, so
treat 1.5 – 1.9 s as the range. **Mamba is roughly 300× slower than Lite per tile here.** Output values
above 1.0 (max 1.35 on the artifact tile) are what the upstream model + constraint produce and are not
clipped by FRAME. On the Baseline 0 scene the ensemble's stability summary is 1.7 × 10⁻³ (Lite's
recorded value for the same AOI is 1.2 × 10⁻³); it is a relative model-stability signal, not a
calibrated error, and is not comparable across models without validation.

GPU execution is evidenced by the worker process holding GPU memory in `nvidia-smi`, a non-zero
`torch.cuda` peak allocation, and the adapter refusing any non-CUDA device.

## Tests

Default run (`pytest frame/tests`): **531 passed, 21 deselected** — 370 pre-existing (unchanged) + 161
new. All new unit tests run without a GPU or the Mamba environment.

| File | Covers |
|---|---|
| `test_models_config.py` | exact architecture; equals `load.py`; band order derived from source; L2A-derived bounds |
| `test_models_contract.py` | channels, size, rank, dtype, NaN/Inf, raw-DN rejection, output contract |
| `test_models_selection.py`, `test_api_model_service_selection.py`, `test_api_model_selection.py` | `lite`/`mamba` selectable, invalid fails clearly, caching per (model, device), provenance, error mapping, health |
| `test_models_protocol.py` | framing and tensor codec |
| `test_models_mamba_client.py` | worker lifecycle against a stub worker: start, reuse, startup error, crash + restart, hang + timeout, concurrent requests; the real worker's startup-failure path |
| `test_models_mamba_worker.py` | the real request loop, in-process with a fake model |
| `test_models_mamba_adapter.py` | contract handling, tile-by-tile batching, no band reordering |
| `test_models_isolation.py` | main process never imports the Mamba runtime |

Integration (`pytest frame/tests -m integration`): **21 passed** — 16 in `test_models_mamba_integration.py`
(real weights on the real GPU: strict load, shapes, determinism, hard-constraint consistency, band-order
sensitivity, memory, contract rejection, worker survives a rejected request), 4 in
`test_api_mamba_integration.py` (real HTTP API → real worker → GeoTIFFs, alongside Lite), and the
pre-existing Lite integration test. They skip — never fail or download — when the GPU, environment or
files are absent.

Frontend: **66 passed** (52 pre-existing + 14 new), `tsc` clean, production build succeeds.
The real UI flow (upload → select Mamba → run on the GPU → results → repeat with Lite) was also driven
in a real browser against the real backend; the unavailable-model states were checked with intercepted
responses.

## Known limitations

* **4-band RGBN only**, and the model itself still consumes exactly **128 × 128** input. Scenes of any
  size are handled *in front of* the model by the tile engine (Phase 2, [`TILING.md`](TILING.md)); a
  non-128 × 128 tensor handed directly to the model is still rejected.
* The output is a **2.5 m reconstructed grid** (SR-derived), not native resolution.
* The **full 10-band cascade is not integrated**; nor LDSR-S2.
* **No training/fine-tuning**; the shipped weights are used as-is.
* **No accuracy validation** against reference imagery for this model, **no Indian-region
  generalisation** evaluation, **no uncertainty calibration** (self-consistency and ensemble stability
  only).
* **Latency**: ~1.7 s per tile and ~10.6 s per job with the ensemble on this GPU. Jobs are still
  synchronous (`/sr/run` blocks).
* **One worker, serialised requests**; POSIX only (the client uses `select` on pipes).
* **CUDA required** for Mamba; CPU is refused. Peak memory was measured for a single tile (batches run
  tile-by-tile in the worker so it does not grow with batch size, but only B = 1 and 2 were exercised).
* The input-scale gap from the requirements analysis is **unchanged**: `raw_digital_number` divides by
  10000 without the −1000 BOA offset for processing baseline ≥ 04.00. Phase 1 did not touch it.
* The Mamba environment has no `pytest`; the worker's own code is tested from the main environment
  (stub worker, in-process loop) and through the real worker in the integration tests.
