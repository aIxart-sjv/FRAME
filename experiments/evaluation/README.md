# Phase 5 evaluation experiments

Everything needed to reproduce the Phase 5 evidence. Methods, metric definitions and caveats: [`docs/EVALUATION.md`](../../docs/EVALUATION.md). **These are measurements, not a ranking.**

| Path | What |
|---|---|
| `configs/benchmarks_v1.json` | the evaluation: 9 systems (bicubic, SEN2SR-Lite with/without the hard constraint, SEN2SR-Mamba, 5 seeds of the tiny model) on 6 datasets (SEN2NEON seeded random-30 and the Phase 3 sample, OpenSR-Test `spot` / `spain_crops` / `spain_urban`, the synthetic smoke test split) |
| `shift_sensitivity/<dataset>/` | spatial-shift sensitivity (requirements 142 §27) for `sen2neon_random30`, `opensr_spain_crops`, `opensr_spain_urban`, `opensr_spot`: bicubic, Lite, Mamba and one tiny seed scored against a reference displaced by 0-2 LR pixels, plus an `aligned_to_bicubic` condition (`rows.jsonl`, `aggregates.json`, `summary.json`, `README.md`) |
| `benchmarks_v1/` | the results: `config.json` (resolved), `metrics.jsonl` (one row per sample × system), `aggregates.json`, `summary.json` (provenance), `README.md` (readable tables) |
| `manifests/sen2neon_random30_seed0.jsonl` | 30 SEN2NEON tiles drawn uniformly at random (seed 0) from the full metadata table by `data_acquisition/select_sen2neon_random.py`; no filtering |
| `manifests/sen2naipv2_*_sample*.jsonl` | Phase 3 records of the 100 + 30 SEN2NAIPv2 pairs of the acquisition check (**not** evaluated: SEN2SR trained on that dataset) |
| `train_tiny_seeds.py` | trains the Phase 4 tiny model for seeds 0-4 on the synthetic smoke set (checkpoints go **outside** the repo: `~/.cache/frame_eval/checkpoints/`) |
| `data_acquisition/` | dataset acquisition helpers and the SEN2NAIPv2 decision (`SEN2NAIPV2_DECISION.md`, `naipv2_sample_report.json`) |

## Reproduce

```bash
export FRAME_DATA_ROOT=~/.cache/frame_data            # SEN2NEON tiles live in $FRAME_DATA_ROOT/sen2neon/, OpenSR-Test in ~/.config/opensr_test/

# 0. data (sizes: SEN2NEON random-30 ~0.4 GB, OpenSR-Test subsets 0.24 GB; nothing large is committed)
sen2sr_venv/bin/python experiments/evaluation/data_acquisition/select_sen2neon_random.py --fetch   # seeded draw of 30 of the 2,269 tiles + checksummed download

# 1. the frozen trained models (Phase 4 config; ~minutes on CPU/GPU)
sen2sr_venv/bin/python experiments/evaluation/train_tiny_seeds.py

# 2. validate everything without evaluating anything, then evaluate (SEN2SR-Mamba runs in its isolated worker; the GPU is used when present)
sen2sr_venv/bin/python -m frame.evaluate check experiments/evaluation/configs/benchmarks_v1.json
sen2sr_venv/bin/python -m frame.evaluate run   experiments/evaluation/configs/benchmarks_v1.json --verbose
```

```bash
sen2sr_venv/bin/python -m frame.evaluate shift experiments/evaluation/configs/benchmarks_v1.json --dataset sen2neon_random30 \
    --systems bicubic sen2sr_lite sen2sr_mamba tiny_cnn_seed0 --output-dir ../shift_sensitivity/sen2neon_random30      # ~37 min for 30 tiles; the OpenSR subsets take 2-7 min
```

`run` refuses to overwrite an existing `benchmarks_v1/` (results are never mixed between runs); choose a new `name`/`output_dir` for a new run. The whole configuration takes about 35 minutes on the RTX 3050 laptop GPU (about one minute per 1024×1024 SEN2NEON tile for all nine systems; the metrics, not the networks, dominate).

## Reading the results

* The unit of statistics is the **scene**, not the pixel or the tile; a table shows n (scenes), mean ± std, median and, when there are enough scenes, a bootstrap CI (`descriptive_only` otherwise).
* Synthetic, real cross-sensor and independent-reference results are separate sections and datasets are never pooled.
* Every skipped, invalid or unreadable sample is listed with its reason in `summary.json` and the README; none is dropped silently.
* Self-consistency (SR → downsample → compare with LR) is its own table and is **not** accuracy against the HR reference.
