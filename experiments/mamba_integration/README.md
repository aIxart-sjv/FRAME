# experiments/mamba_integration

Measured GPU validation of the real SEN2SR-Mamba RGBN model through FRAME's isolated worker
(gap-closure roadmap, Phase 1). Full write-up: [`docs/MAMBA_INTEGRATION.md`](../../docs/MAMBA_INTEGRATION.md).

```
sen2sr_venv/bin/python experiments/mamba_integration/run_validation.py [--warm-runs N] [--metadata-dir DIR]
```

Needs the Mamba environment (`sen2sr_mamba_venv`), a CUDA GPU and `models/SEN2SR/`; exits with a
clear message if any is missing. Trains nothing, downloads nothing, does not modify `sen2sr/`.

It records, to `metadata/run_metadata.json`: worker start time, first (cold) and warm inference
latency, peak GPU memory (allocator peak and `nvidia-smi` process footprint), output
shape/dtype/range, NaN/Inf, determinism, the hard-constraint consistency RMSE, the 6-member
uncertainty ensemble time, and SEN2SR-Lite timings on the same tile for context.

Two real inputs: the tile shipped with the model artifact and the deterministic Baseline 0 scene
used by every earlier phase. Timings are for the machine recorded in the metadata; nothing here is
an accuracy claim.
