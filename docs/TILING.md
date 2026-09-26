# Arbitrary-size tiling (gap-closure roadmap, Phase 2)

Status: **implemented and verified on the development machine** (RTX 3050 Laptop GPU, 4 GB; 16 GB RAM).
Scope: any scene size — rectangular, not a multiple of 128 — through the two 128 × 128 models
(SEN2SR-Lite and SEN2SR-Mamba) with one reusable, model-agnostic tile engine (`frame/tiling/`).

```
arbitrary scene (C, H, W)
   │  plan_tiles                 explicit TileSpec list, row-major          frame/tiling/plan.py
   ▼
overlapping 128 × 128 tiles      edge tiles reflect-padded to full size     frame/tiling/padding.py
   │  model(tile[None])          the caller's callable: Lite, Mamba, or a fake
   ▼
tile outputs (4 × 512 × 512)     cropped to the valid part of the tile
   │  weighted accumulate        separable linear ramps, normalised         frame/tiling/blend.py
   ▼
SR raster (C, 4H, 4W)            georeferenced by the existing frame.geospatial
```

The output is an **SR-derived product on a 2.5 m pixel grid**, not a native 2.5 m measurement.

## Contract

| | |
|---|---|
| Tile size | **128** (the models' native input; `frame.tiling.DEFAULT_TILE_SIZE`) |
| Overlap | **32 px** by default (stride 96); configurable **0 – 64**. `FRAME_API_TILE_OVERLAP` for the API |
| Padding | **reflect** — edge tiles are mirrored about the scene edge without repeating the edge pixel |
| Blend | **linear** ramps over the overlap, normalised by the accumulated weights |
| Scale | 4 (10 m → 2.5 m); `output H = 4 × H`, `output W = 4 × W` |
| Scenes | any `H, W ≥ 1` (a scene smaller than one tile is one padded tile); the API additionally caps `H × W` at `FRAME_API_MAX_INPUT_PIXELS` (default 1,048,576) |
| Channels | preserved and in order (the engine is channel-count agnostic; the models require RGBN = B04, B03, B02, B08) |
| Batching | **none** — one tile at a time, in a fixed row-major order |
| Model | any `model(x) -> y` with `x` = `(1, C, 128, 128)`, `y` = `(1, C_out, 512, 512)` |

Invalid configuration is rejected on construction (`TilingConfig`): non-integer or `< 1` `tile_size`/`scale`,
negative or non-integer `overlap`, an overlap whose stride `tile_size − overlap` is not positive, unknown
padding/blend modes — and **an overlap above half a tile** (beyond that the two blend ramps meet inside one
tile and the tile count grows quadratically; stride 1 on a 300 × 500 scene would be ~64,000 tiles).

### Planning (all coordinate arithmetic lives in `plan.py`)

Tile starts along an axis are `0, stride, 2·stride, …` until `start + tile_size ≥ size`. Every pixel is
therefore inside at least one tile, no tile starts beyond the scene, every tile that has a successor is fully
valid, consecutive tiles overlap by exactly `overlap` pixels, and the last tile is always wider than
`overlap`. Each `TileSpec` carries `row_start/row_end`, `col_start/col_end`, `valid_height/valid_width`,
`padded_height/padded_width` (always the tile size), which sides border a neighbour, and the same tile on
the SR grid. Examples at overlap 32: 128 × 256 → 1 × 3 tiles; 300 × 500 → 3 × 5 = 15; 511 × 777 → 5 × 8 = 40.

### Border handling

The last tile of a row/column may extend past the scene. Its in-scene part is its *valid* extent; the rest is
filled by reflect padding, the model runs on the full 128 × 128 tile, and the result is **cropped back to the
valid extent** — the padded strip never reaches the output and nothing is resized or distorted. Reflect keeps
the local texture and spectral values of the real neighbourhood; zero-fill would invent a hard black edge
(which the models' Fourier hard constraint would ring across the whole tile) and edge-replication makes streaks.
The mirror is periodic, so it also works for a scene smaller than a tile.

### Blending

Each tile's output is multiplied by a weight window (1 in the interior; a ramp `(k + 0.5) / R` over
`R = overlap × 4` SR pixels on each side that borders another tile; **no ramp on a side that lies on the scene
boundary**), summed into a canvas together with the weights, and divided by the summed weights. The ramps of
two neighbours cover the same pixels in opposite directions and sum to exactly 1, every weight is strictly
positive, and every output pixel is a convex combination of the predictions covering it: no last-write-wins,
no brightness change. The ramp sits on each tile's border zone, where the model is least reliable. With
`overlap = 0` the tiles abut and nothing is blended (a seam is possible — measured below).

### Seam diagnostic

For every horizontally or vertically adjacent tile pair, the two tiles' *raw* predictions are compared on the
pixels they share, **before blending**, using `frame.consistency.tiles.compare_tile_overlap` (written in an
earlier phase for exactly this and waiting for a FRAME-owned tiler), and pooled: mean/RMSE/max absolute
difference. Only two rows of tile outputs are held at a time. It is a **tiling consistency diagnostic, not an
accuracy measure**, and it is biased upward (the shared pixels are the border zone), so compare runs at equal
overlap only. For a single tile or `overlap = 0` it is reported as `NOT_COMPUTABLE`, never as 0.

## Errors (nothing partial is ever returned)

| Situation | Result |
|---|---|
| empty raster, wrong rank, non-float dtype, zero channels, NaN/Inf | `InvalidSceneError` → HTTP 400 |
| invalid tile size / overlap / scale / modes | `InvalidTilingConfigError` (API: fails at startup) |
| model failure, worker crash, worker timeout, CUDA OOM | the model's own typed error (`ModelWorkerError`, `ModelInferenceError`, …), unchanged → HTTP 503/500 with a generic message; the scene is aborted at that tile |
| a tile output with the wrong shape, channel count, type, or NaN/Inf | `ModelInferenceError` naming the tile |
| any output pixel not covered by a tile | `ReconstructionError` |
| scene above the API pixel limit | HTTP 413 `scene_too_large`, checked from the file header, rejected file not kept |

NaN/Inf in an *uploaded* scene never reaches the tiler: `frame.preprocessing.to_reflectance` has always
replaced them with 0 (as the upstream README does).

## Worker reuse, memory, determinism

* **One Mamba worker serves the whole scene** — every tile and every ensemble pass. The engine only ever
  calls the injected `MambaWorkerClient`; nothing in `frame/tiling/` names a model (a test parses its source
  for model names and model imports). Measured: one process for all 40 tiles of the real scene; unit tests
  count process spawns.
* **GPU memory is the single-tile figure.** Whole 40-tile scene: allocator peak **605.6 MiB**, worker
  process **724 MiB** (`nvidia-smi`) — identical to Phase 1's single tile.
* **Host memory** grows with the scene (the SR raster is 16× the input pixels; the ensemble holds several
  full-size arrays). Peak resident memory of the whole pipeline (cheap model, fresh process per size):

  | Scene | Input px | Peak RSS | Pipeline time (cheap model) |
  |---|---|---|---|
  | 256 × 256 | 65,536 | 870 MiB | 0.8 s |
  | 511 × 777 | 397,047 | 2,208 MiB | 3.6 s |
  | 768 × 768 | 589,824 | 2,962 MiB | 6.7 s |
  | 1024 × 1024 | 1,048,576 | 4,756 MiB (of 15,602 MiB) | 14.4 s |

  That is the basis for the API default cap of 1024 × 1024 input pixels. Concurrent requests are **not**
  bounded (synchronous routes run in a thread pool): two simultaneous 1024 × 1024 jobs would need roughly
  twice that.
* **Deterministic.** Tile order is fixed, arithmetic is float32 in a fixed order. With the real Mamba model two
  identical whole-scene runs were **bit-identical** (max abs difference 0.0), so the documented tolerance is
  exactly zero. A single-tile scene is bit-identical to calling the model directly.

## Geospatial behaviour

The tiler produces a raster; georeferencing is the existing `frame.geospatial.derive_output_metadata`
(unchanged) and `write_geotiff` (no second writer). For a 10 m input the output transform is
`(a/4, b/4, c, d/4, e/4, f)`: CRS unchanged, upper-left origin unchanged, footprint unchanged, pixel 2.5 m.
`test_tiling_geospatial.py` runs the real pipeline and reads the GeoTIFFs back for a square scene, a
rectangular one and two whose sides are not multiples of 128 (128×128, 128×256, 300×500, 261×389): output
dimensions exactly 4H × 4W, CRS equal, bounds equal to 1e-9, transform `(2.5, 0, c, 0, −2.5, f)` to 1e-12,
the far corner of the SR grid on the input's far corner, each input pixel exactly a 4 × 4 SR block, a second
CRS/origin preserved verbatim, and — with a position-consistent model — the SR value at random world
coordinates (including points hugging every scene edge) equal to the input value there, i.e. no shift or seam
from tiling.

## Why not upstream's `predict_large`

`sen2sr/utils.py` was read and **run unmodified** on a perfectly behaved fake model (nearest ×4, whose correct
answer is known exactly). It was not reusable:

| Input (overlap) | Result |
|---|---|
| 128 × 128 (32) | shape right, **23.4 % of the output pixels wrong** (an unfilled strip: a single tile is cropped as if it had a neighbour) |
| 256 × 256 (32 or 0) | correct — only because the fake model is position-consistent |
| 128 × 256, 256 × 128, 300 × 500, 511 × 777 (32) | **crash** |

Cause, from the code: the output is allocated square from `X.shape[1]`; rows and columns are indexed
crosswise for non-square input; the border test compares an input-space size with an output-space offset; and
the hard-crop of middle tiles overwrites its neighbours. FRAME's engine keeps a regular grid, explicit
padding and weighted blending instead, and `sen2sr/` is untouched.

## The overlap default was checked, not copied

Sweep on the real scene (`experiments/tiling`, section A). Each reconstruction is compared with the
largest-overlap one (64), the most heavily blended reconstruction available; that is a stability check, not
accuracy against any reference.

SEN2SR-Lite, real 511 × 777 scene:

| Overlap | Tiles | Time | vs overlap 64: RMSE | max abs | Seam RMSE (pre-blend) |
|---|---|---|---|---|---|
| 0 | 28 | 0.54 s | 2.0e-3 | **0.300** | — |
| 8 | 35 | 0.39 s | 3.5e-4 | 0.042 | 6.0e-3 |
| 16 | 35 | 0.44 s | 2.7e-4 | 0.042 | 4.3e-3 |
| **32** | **40** | 0.61 s | **6.6e-5** | **0.006** | 3.1e-3 |
| 48 | 60 | 1.05 s | 2.4e-4 | 0.042 | 2.5e-3 |
| 64 | 84 | 1.59 s | (reference) | | 2.2e-3 |

SEN2SR-Mamba, real 256 × 384 crop:

| Overlap | Tiles | Time | vs overlap 64: RMSE | max abs | Seam RMSE (pre-blend) |
|---|---|---|---|---|---|
| 0 | 6 | 10.3 s | 2.7e-3 | 0.227 | — |
| 16 | 12 | 20.4 s | 2.9e-3 | 0.193 | 6.1e-3 |
| **32** | **12** | 20.3 s | 2.6e-3 | 0.191 | 4.7e-3 |
| 64 | 15 | 25.5 s | (reference) | | 3.6e-3 |

What this supports, and what it does not:

* **Overlap 0 is clearly worse** for Lite: a hard hand-over with a maximum deviation of 0.30, against ≤ 0.042
  with any overlap. It is kept for completeness, not recommended.
* **Beyond 16 – 32 there is no monotonic improvement** for either model, while cost climbs (Lite 40 → 84
  tiles; Mamba 20 s → 26 s). The Lite figure is non-monotonic (48 is farther from 64 than 32 is), which is a
  grid-alignment effect, so it is not evidence that 32 is systematically best — only that nothing larger buys
  anything measurable. Mamba's differences (2.6 – 2.9e-3) are within what tile-to-tile disagreement alone
  produces.
* The seam-RMSE column falls as overlap grows only because a wider shared region contains more interior
  (less border-affected) pixels; the note above about comparing at equal overlap applies. At overlap 32 Mamba's
  tiles disagree more than Lite's (4.7e-3 vs 3.1e-3, different scenes), which is consistent with Mamba's
  scan covering the whole tile, but that is an observation, not a tested cause.

**Decision:** keep 32 (upstream's convention), now with a measurement behind "no larger overlap is worth its
cost". The maximum deviations of ~0.2 seen for Mamba at every overlap are extremes on a real scene whose values
reach 1.87 and which contains zero-filled NaN pixels; where they occur, and why, was not investigated.

## Measured on the real scene (RTX 3050, 4 GB)

Scene: one real Sentinel-2 L2A window fetched with `cubo` (AOI/date of `experiments/baseline`, 1024 × 1024 px,
catalogue returned 3 items, item 0 used, acquisition 2023-01-15T10:54:11, EPSG:32630), cached outside the
repository; ~1.7 % of the raw values were NaN (outside the swath) and were set to 0. SHA-256 of the cached
raster in the metadata. Nothing here is an accuracy claim.

**Engine, real Mamba, 511 × 777 (4 × 2044 × 3108 output):**

| Measurement | Value |
|---|---|
| Tile size / overlap / stride | 128 / 32 / 96 |
| Tile grid / tile count / padded tiles | 5 × 8 / 40 / 12 |
| Worker start (spawn + imports + load) | 3.74 s |
| Total wall time (whole scene, one pass) | 68.6 s |
| Tile time: mean / median / min / max | 1.715 / 1.708 / 1.650 / 2.084 s (max = the first tile) |
| Engine overhead outside the tile loop (planning + normalisation) | 0.015 s |
| Peak GPU memory, allocator / worker process | 605.6 / 724 MiB |
| One worker for all tiles | yes |
| Output | `(4, 2044, 3108)` float32, finite, range [−0.022, 2.295] |
| Repeat run | bit-identical |
| Hard-constraint consistency (SR area-averaged to 10 m vs input) | RMSE 0.0063 (single tile in Phase 1: 0.0031) |
| Seam diagnostic (67 pairs, 4.3 M px) | mean 2.7e-3, RMSE 5.8e-3, max 0.285 |

The per-tile time is flat across the 40 tiles and equals Phase 1's warm single-tile time (1.74 s), so the
tiler adds no measurable cost on top of the model. Only these scene sizes were run; larger ones were not.

**HTTP API end-to-end, real 256 × 384 crop** (the ensemble runs the tile engine 6 times — once per test-time
transform — and two of the six transposed passes tile the scene as 384 × 256):

| | Mamba | Lite |
|---|---|---|
| Tiles per pass / passes / tile inferences | 12 / 6 / 72 | 12 / 6 / 72 |
| HTTP wall time (incl. first model load) | **128.3 s** | 3.75 s |
| Mean time per tile inference | 1.739 s | 19 ms |
| Output GeoTIFF | 4 × 1024 × 1536, EPSG:32630, 2.5 m, footprint equal to the input, `FRAME_SR_VARIANT` = the model that ran | same |
| Worker footprint after the job | 724 MiB | — |

The real UI was also driven in a browser with real rectangular scenes: Lite on 511 × 777 finished in 8.4 s and
Mamba on 200 × 300 in 68.5 s (36 tile inferences), the correct model was recorded, and the comparison slider,
uncertainty map and NDVI panels all keep the scene's true aspect ratio.

### The bottleneck (reported, not solved)

Tiling multiplies the number of model calls, and the uncertainty ensemble multiplies them by 6 again. With
Mamba at ~1.7 s per tile, a job costs `tiles × 6 × ~1.7 s` on this GPU (the 12-tile scene above took 128 s).
Extrapolating that arithmetic to the 1024 × 1024 cap (11 × 11 = 121 tiles) gives a synchronous request of
about 20 minutes; **that figure is arithmetic from the measured per-tile time, not a measurement**. The API is
synchronous, so a large Mamba job holds an HTTP request open for that long. Lite is not affected (72 tile
inferences in 1.9 s). This is what an async job model (with progress from the `on_tile` hook) or a reduced
ensemble for large scenes would address; neither is part of this phase.

## Tests

Default run (`pytest frame/tests`): **756 passed, 33 deselected** — 531 before this phase + 225 new; all run
without a GPU. Integration (`pytest frame/tests -m integration`): **33 passed** (1 pre-existing Lite, 20 from
Phase 1, 12 new). Frontend: **68 passed** (66 + 2 new), `tsc` clean, build succeeds.

| File | Covers |
|---|---|
| `test_tiling_plan.py` | coverage/counts for 128×128, 128×256, 256×128, 300×500, 511×777 and a 15 × 15 × 5 grid of sizes × overlaps; closed-form tile-count oracle; overlap 0 / default / others; invalid overlap, tile size, scale, modes, scenes |
| `test_tiling_padding.py` | reflect indices vs numpy; interior tiles untouched; padded edge tiles; no invented values |
| `test_tiling_blend.py` | ramp values; neighbouring ramps sum to 1; boundary sides not ramped; canvas normalisation; incomplete reconstruction refused |
| `test_tiling_engine.py` | reconstruction of a position-consistent model on all shapes × overlaps; exact for overlap 0 and a single tile; **blending proven against the closed-form ramp with a position-dependent model**; determinism; error paths incl. model failure mid-scene, malformed output, incomplete reconstruction; seam diagnostic on known differences; `TiledModel` inside the 6-transform ensemble (transposed passes); no model names or model imports in `frame/tiling/` |
| `test_tiling_geospatial.py` | the real pipeline + GeoTIFF read-back checks listed above |
| `test_tiling_worker_reuse.py` | real client + real engine against a stub worker: one process per scene and across ensemble passes, crash mid-scene then recovery, hang → timeout, contract enforced before any spawn |
| `test_api_tiling.py` | HTTP: arbitrary sizes accepted, both models, recorded configuration and tile counts, configured overlap, size guard (413, header-first, file not kept, inclusive boundary), NaN handling, worker failure mid-scene leaves no job or artifacts |
| `test_tiling_mamba_integration.py`, `test_api_tiling_mamba_integration.py` | real Mamba + GPU: multi-tile rectangular scene, one worker, bit-identical repeat, single tile = direct call, consistency, seam, overlap 0, memory, ensemble; the full API for Mamba then Lite |

## Known limitations

* **RGBN only**, and the models still consume **128 × 128 tiles** internally; the full 10-band cascade is not
  integrated. Nothing was trained; no HR-reference validation; no Indian-region generalisation.
* **Synchronous jobs**, with the Mamba latency above; no progress reporting to the user (the engine has an
  `on_tile` hook, unused by the API).
* **Scene size is capped** by `FRAME_API_MAX_INPUT_PIXELS` for host memory; concurrency is unbounded.
* **Edge quality:** the left/top scene edges lie on a tile border (nothing is mirrored there), the right/bottom
  edges are interior to a padded tile. Not measured separately.
* Reflect padding needs no data beyond the scene, but it is a mirror, not observed imagery; only the valid part
  is kept.
* The seam diagnostic is a tiling check, biased upward, not comparable across overlaps, and not an accuracy
  measure. Overlap 0 is not blended and can show seams.
* The multipart-upload path was confirmed to have no size cap of its own; an upload failure seen while
  developing this phase was the machine's `/tmp` quota being exhausted (a spooled temp file failing to roll
  over), not an API limit.
