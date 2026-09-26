# Final readiness records (Phase 8)

Small, machine-readable records of the state of FRAME at the end of the gap-closure roadmap. Nothing here is scientific evidence about any model; it says what runs, what was checked, and what is (not) established.
Read `docs/RELEASE_READINESS.md` and `docs/CLAIMS.md`.

| Path | What it is | How it was made |
|---|---|---|
| `status.json` | revision, dirty flag, supported models / scale / bands, datasets and their real evidence, **test counts parsed from the run logs**, smoke summaries, caveats, non-claims, deferred work | `python experiments/final_readiness/build_status.py --unit-log … --integration-log … --frontend-log …` |
| `smoke_toy/`, `smoke_lite/`, `smoke_mamba/` | `smoke_record.json` of the end-to-end smoke path on the same deterministic 200 × 300 synthetic scene: every check, the shapes expected and produced, timings, the tile plan, provenance (scene and output SHA-256, seed, settings digest, environment, git revision and dirty flag; for Lite the SHA-256 of the weight files; for Mamba the worker's runtime block) | `python -m frame.smoke --model toy\|lite\|mamba [--repeat]` |
| `environments/main.freeze.txt`, `mamba.freeze.txt` | the distributions installed in the two environments the records were made in (`importlib.metadata`; **not** a lockfile) | see `docs/REPRODUCIBILITY.md` §2 |
| `build_status.py` | the builder of `status.json` | – |

The toy model is a deterministic stand-in served through the `lite` selection, so its record says so; the smoke scene is synthetic. Timings are for the development machine (RTX 3050 Laptop GPU, 4 GB).
The records were made from a working tree with uncommitted changes (`dirty: true`), and that fact is preserved.
