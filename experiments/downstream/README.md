# Downstream analytical utility (Phase 7)

`python -m frame.downstream` asks whether super-resolution changes an NDVI-derived vegetation decision, and whether the TTA stability signal is informative about downstream mistakes once texture is controlled for. Full description,
results and limits: `docs/DOWNSTREAM.md`.

* `configs/downstream_v1.json`: the experiment (SEN2NEON random-30, OpenSR-Test `spot`, `spain_crops`, `spain_urban`; SEN2SR-Lite and SEN2SR-Mamba; the Phase 6 reference gate unchanged; NDVI threshold 0.3 declared in advance, 0.2/0.4 as sensitivity;
  regions of 10 m and 40 m).
* `runs/downstream_v1/`: the machine-readable record and the Markdown report (`README.md` has every table): `config.json`, `tiles.jsonl` (one row per tile, with the eligibility decision and the exclusion reason), `summary.json` (provenance, evidence
  accounting, deferred and unavailable tasks), `downstream_metrics.json`, `association.json`, `risk_coverage.json`. Per-region tables are cached outside the repository (`~/.cache/frame_downstream`) and are not versioned.

```bash
sen2sr_venv/bin/python -m frame.downstream check experiments/downstream/configs/downstream_v1.json   # gate and region preview, no model
sen2sr_venv/bin/python -m frame.downstream run   experiments/downstream/configs/downstream_v1.json   # about 40 minutes; the output directory must not exist
sen2sr_venv/bin/python -m frame.downstream smoke                                                      # synthetic end-to-end run, no dataset, no weights
```

One line: the region-level NDVI and vegetation decision of Lite and Mamba stay close to bicubic and the LR input (dataset-dependent sign, a mixed to null result, no ranking); the stability carries at most a small amount of information
about the downstream error beyond texture and is uncalibrated; the land-cover task is deferred (no region-level labels) and Indian validation is unavailable.
