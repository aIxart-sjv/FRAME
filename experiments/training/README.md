# Phase 4 training experiments

Everything here was produced by `python -m frame.train` (or the scripts named below) on SYNTHETIC paired data; nothing is evidence about real
imagery. See `docs/TRAINING.md` for the method, the measured results, the failures found on the way, and the local-training decision.

| Directory | What it is |
|---|---|
| `manifests/` | the Phase 3 manifests the runs used (`synthetic_5x2_seed0.jsonl`: 10 scenes in 5 regions split 6/2/2; `synthetic_2x1_seed1.jsonl`: the 1+1-scene over-fit set) |
| `smoke_tiny_cnn/` | primary run: tiny residual CNN, L1, 600 steps. `config.json` (input) + `metrics.jsonl`, `summary.json`, `README.md` (outputs); `checkpoint/` is git-ignored |
| `smoke_tiny_cnn_charbonnier/`, `smoke_tiny_cnn_spectral/` | loss variants of the same run (Charbonnier ε = 0.01; L1 + spectral λ = 0.05) |
| `overfit_one_scene/` | 1 training scene, 1 validation scene, batch size 1, 2,000 steps |
| `seed_sweep/` | seed-to-seed spread of the three configs (`run_seed_sweep.py`, `results.json`) |
| `reproducibility/` | same config run twice on CPU / GPU, compared bit for bit (`compare_runs.py`, `results.json`) |
| `feasibility/` | GPU memory and time of one training step for `tiny_cnn`, `sen2sr_lite` and `sen2sr_mamba` (`measure_memory.py`, `results/*.json`) |
| `mamba_finetune_smoke/` | a real tiny fine-tune of the pretrained SEN2SR-Mamba in its own environment (`export_patches.py` in the main env, `run_mamba_smoke.py` in the Mamba env) |
| `data_acquisition/` | metadata-only survey of the real training-data sources (`inspect_sources.py`, `sources.json`); nothing downloaded |

Reproduce a run from a clean checkout (data and checkpoints live outside the repository):

```bash
export FRAME_DATA_ROOT=~/.cache/frame_data
python -m frame.data synthetic --out /tmp/unsplit.jsonl --regions 5 --scenes-per-region 2 --seed 0
python -m frame.data split /tmp/unsplit.jsonl --out experiments/training/manifests/synthetic_5x2_seed0.jsonl --level region --seed 0 --fractions 0.6 0.2 0.2
python -m frame.train check experiments/training/smoke_tiny_cnn/config.json
python -m frame.train run   experiments/training/smoke_tiny_cnn/config.json      # output_dir is the config's own directory; refuses if a run is already there

sen2sr_venv/bin/python        experiments/training/feasibility/measure_memory.py --model sen2sr_lite --out /tmp/lite.json
sen2sr_mamba_venv/bin/python  experiments/training/feasibility/measure_memory.py --model sen2sr_mamba --lr-sizes 16 32 64 128 --checkpointing off on --out /tmp/mamba.json
```

Feasibility results for Mamba with checkpointing were produced first by an inline copy of the wrapper and re-verified with the library function
`frame.train.memory.enable_block_checkpointing` (`results/sen2sr_mamba_ckpt_recheck.json`, identical to 0.1 MiB).
