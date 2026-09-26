# SIH demonstration runbook

A repeatable flow for showing FRAME, built so that nothing in it depends on a large dataset, a network, or a long live GPU run. Every step below was exercised on the development machine
(see the records under `experiments/final_readiness/`). If something differs on the demo machine, stop and use the fallback in §8 rather than improvising.

## 1. Prepare (do this the day before, not on stage)

```bash
cd ~/Documents/Projects/FRAME
# 1. the demo inputs: a synthetic rectangular scene, a real 128 x 128 Sentinel-2 window, and a real rectangular crop (if cached)
sen2sr_venv/bin/python -m frame.smoke --demo-scenes ~/.cache/frame_demo
# 2. the pipeline check with the deterministic stand-in model (no weights, no GPU, ~1 s): must print "smoke passed: 22/22 checks"
sen2sr_venv/bin/python -m frame.smoke
# 3. the same with the real models you intend to show
sen2sr_venv/bin/python -m frame.smoke --model lite      # ~3 s   (needs the cached Lite weights)
sen2sr_venv/bin/python -m frame.smoke --model mamba     # ~65 s  (needs the GPU, models/SEN2SR/ and the Mamba environment)
# 4. the front end dependencies
(cd frontend && npm install)
```

If step 3 prints `unavailable: …` for Mamba, Mamba cannot be shown on this machine: use Lite and say why (§7). Do not let a live run be the first time Mamba starts: the first request in a fresh worker pays a few seconds of start-up.

The demo inputs written in step 1:

| File | "Pixel values" setting in the UI | Shape | What it shows |
|---|---|---|---|
| `real_baseline0_128_reflectance.tif` | **Reflectance (0–1)** | 4 × 128 × 128, real Sentinel-2 L2A, 2023-01-15 | the simplest real input: one tile; well under a second with Lite, about 10 s with Mamba |
| `real_crop_200x300_reflectance.tif` | **Reflectance (0–1)** | 4 × 200 × 300, a crop of a real scene | the **rectangular tiling path**: 2 × 3 = 6 tiles per pass, output 4 × 800 × 1200 |
| `synthetic_200x300_raw_dn.tif` | **Raw digital number** | 4 × 200 × 300, generated | the raw-DN path and a nodata corner (valid coverage 99.8 %); **not** Sentinel-2 |

**Set "Pixel values" to match the file** (it defaults to Raw digital number; changing it re-validates the file). Choosing wrongly is refused with a message that says what to pick: a real reflectance file with "Raw digital number" is a 400, not a black image. Show that once if there is time: it is a real safeguard.

## 2. Start FRAME

```bash
# terminal 1 — backend (Mamba's worker starts on the first Mamba request and stays)
sen2sr_venv/bin/uvicorn frame.api.app:app --port 8000
# terminal 2 — front end
cd frontend && npm run dev          # open http://localhost:5173
```

Check `http://127.0.0.1:8000/health`: `available_models` lists `lite` and `mamba`; Mamba shows `available: true` only if a CUDA GPU and its environment are usable. The API is **synchronous**: a request blocks until it finishes.

## 3. Load an input

In the UI: choose the file, set "Pixel values" as in the table, and wait for the input preview (dimensions, CRS, resolution, bands; the upload is validated before anything runs). Point out what was checked: the four bands, the CRS, the size, and that the scene has valid pixels.

## 4. Choose the model and generate the 4× output

* Select **SEN2SR-Lite** or **SEN2SR-Mamba** in the model control (an unavailable model is disabled with its reason; nothing falls back silently).
* Press **Run FRAME**. Expected: Lite finishes in seconds; Mamba takes about a minute for the 200 × 300 scene (36 model calls: 6 tiles × 6 ensemble passes) and about 10 s for the 128 × 128 one (one tile, six ensemble passes).
* Say what is running: the selected model over 128 × 128 tiles, blended into one raster, and six test-time views for the stability diagnostic.

## 5. Inspect the result

| Tab | What to point at |
|---|---|
| **Overview** | the comparison slider (10 m input against the **SR-derived product — 2.5 m pixel grid**); the input and output shapes (e.g. 4 × 200 × 300 → 4 × 800 × 1200: exactly 4×); the model name |
| **Stability** | the **TTA stability — reconstruction-variation diagnostic**. Read the two notes on the page aloud: it is uncalibrated, and in FRAME's own validation it was only weakly associated with error, about as much as image texture. It is something to inspect, not a reliability score |
| **NDVI** | an optional **demonstration**: NDVI from the SR product against NDVI from its own low-resolution input. It says what it is on the page; do not present it as a result |
| **Metadata** | the reproducibility record: model, device, seed, the six transforms, timing, CRS, valid-pixel coverage, self-consistency (against the model's *own input*, not ground truth) |
| Downloads | the SR GeoTIFF and the stability GeoTIFF; open one in QGIS if available: same CRS and footprint as the input, 2.5 m pixels |

Result metadata to quote: `metadata.tiling` (tile grid, overlap 32, reflect padding, linear blending, tile and inference counts, seam diagnostic), `metadata.output_geospatial` (CRS, transform, bounds), and for Mamba `metadata.model_runtime` (weights SHA-256, parameter count, worker environment).

## 6. Demonstrate the rectangular tiling path

Use `real_crop_200x300_reflectance.tif` (or the synthetic one). Show in **Metadata** that the scene is 200 × 300 (not a multiple of 128, not square), that `tile_grid` is `[2, 3]`, and that the output is exactly 800 × 1200 with the input's footprint. If a terminal is open, this one command shows the same invariants on every model with no UI:

```bash
sen2sr_venv/bin/python -m frame.smoke --model lite     # 21 checks (22 with --repeat), including CRS, origin, footprint, 2.5 m pixel size, model tag, tile plan
```

## 7. What not to claim (read before presenting)

* **Do not** call the output native or true 2.5 m imagery, or say it is validated as accurate at 2.5 m. Say: *"an SR-derived product on a 2.5 m pixel grid; Sentinel-2 has not observed the ground at 2.5 m."*
* **Do not** call the stability map uncertainty, confidence, a probability of error or a reliability score. Say: *"a TTA stability diagnostic; we tested it against error and found a weak association, about as strong as image texture."*
* **Do not** say super-resolution improves NDVI, land-cover or any downstream result, or that either model is better than bicubic or than the other. Say: *"reference-based tests found small, dataset-dependent differences and no consistent advantage."*
* **Do not** say it generalises, that it has been tested in India, that it is real-time or production-ready. Say what it is: a synchronous, single-process prototype.
* **Do not** compare Lite and Mamba on quality. Their **cost** differs by about two orders of magnitude on this machine (Lite in seconds, Mamba about a minute for the same scene); that is a cost fact, not a ranking.
* If asked what the evidence is: North America (SEN2NEON, 30 tiles sampled of 2,269) and Spain (OpenSR-Test), against different-sensor references, with a registration gate that removes about 60 % of the SEN2NEON tiles, 4–11 usable scene units per dataset, no Indian data, no land-cover labels. Full statement: [`CLAIMS.md`](CLAIMS.md).

## 8. If something goes wrong

| Symptom | Cause and what to do |
|---|---|
| Upload rejected: "input_scale is … but …" | the wrong "Pixel values" setting for the file; pick the one in the table |
| Upload rejected: "no valid pixel" | an all-nodata or all-NaN file; use a demo scene |
| Mamba greyed out, `503 model_unavailable` | no CUDA GPU or the Mamba environment is missing; use Lite and say so. **Nothing falls back on its own** |
| A run takes minutes | it is Mamba on a large scene (cost ≈ tiles × 6 × 1.7 s); use the 128 × 128 scene, or Lite |
| `413 scene_too_large` | above 1,048,576 input pixels; use a demo scene |
| The backend restarted and a result is "not found" | the job registry is in memory and is lost on restart; upload again |
| No network at all | nothing above needs it once the Lite weights are cached (`~/.cache/sen2sr_baseline/`) |
| Anything unexplained | run `sen2sr_venv/bin/python -m frame.smoke`; if it passes, the pipeline is fine and the problem is in the UI or the input |

## 9. Records to have open (all JSON, small)

`experiments/final_readiness/smoke_{toy,lite,mamba}/smoke_record.json` (the pipeline invariants and timings for each model on the same scene), `experiments/final_readiness/status.json` (revision, dirty flag, test counts, caveats), and the three analysis reports:
`experiments/evaluation/benchmarks_v1/README.md`, `experiments/uncertainty/runs/reliability_v1/README.md`, `experiments/downstream/runs/downstream_v1/README.md`.
