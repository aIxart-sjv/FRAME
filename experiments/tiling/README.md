# experiments/tiling

Measured validation of the arbitrary-size tile engine (gap-closure roadmap, Phase 2) on a **real
Sentinel-2 scene**. Full write-up: [`docs/TILING.md`](../../docs/TILING.md).

```
sen2sr_venv/bin/python experiments/tiling/run_validation.py [--sections A,B,C,D] [--metadata-dir DIR]
```

Sections are independent (results are merged into `metadata/run_metadata.json`):

| | What it measures |
|---|---|
| **A** | The overlap default, checked rather than copied: an overlap sweep on the real rectangular scene (Lite at full size, real Mamba on a smaller crop) — tile count, time, the tile-seam diagnostic, and distance from the most-blended reconstruction. |
| **B** | The real Mamba model through the tile engine on a real multi-tile rectangular scene: worker start, per-tile time, total time, peak GPU memory, one-worker-for-all-tiles, output shape, hard-constraint consistency, bit-identical repeat. |
| **C** | The same through the real HTTP API (Mamba, then Lite): the 6-member ensemble multiplies the tile calls by 6. |
| **D** | Peak resident memory of the whole pipeline at growing scene sizes (cheap model), which is the measurement behind `FRAME_API_MAX_INPUT_PIXELS`. |

**Data.** The script fetches one real 1024 × 1024 px RGBN scene with `cubo` (same AOI and date as
`experiments/baseline`), caches it **outside the repository** (`~/.cache/frame_tiling/`, override with
`FRAME_TILING_CACHE_DIR`; 16 MB) and records the scene's identity — AOI, date window, catalogue items
returned, acquisition timestamp, bounds and a SHA-256 of the cached raster — in the metadata. Sentinel-2
catalogue contents drift, so the hash is what says whether a later run used the same pixels. About 1.7 % of
the raw values were NaN (outside the swath) and were set to 0, as `experiments/baseline` does.

Needs the Mamba environment, a CUDA GPU and `models/SEN2SR/` for sections B and C (they are skipped with
a warning otherwise), and the cached Lite weights for A, C and D. Trains nothing, does not modify `sen2sr/`.
Point `TMPDIR` at a disk with room if `/tmp` is small (the API section writes large rasters).

Timings are for the machine recorded in the metadata. Nothing here is an accuracy claim; the seam numbers
are a tiling diagnostic.
