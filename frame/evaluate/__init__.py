"""FRAME evaluation layer (Phase 5): independent, auditable evidence about reconstruction accuracy, spectral fidelity and spatial correctness.

    dataset -> reference / geometry validation -> frozen system -> inference -> mask-aware metrics -> aggregation -> statistics -> provenance

Nothing here trains anything or ranks anything. Separate concepts stay separate in every output:

    reference accuracy      SR vs an independent HR reference          (PSNR, SSIM, RMSE, MAE, SAM, ERGAS, per band, indices)
    spatial / detail        is fine detail supported by the reference?  (high-frequency error, edge agreement, registration shift, seams,
                                                                         improvement / supported synthesis / omission / unsupported detail)
    self-consistency        SR -> area-average -> compare with the LR   (NOT accuracy against a reference)
    opensr-test native      the benchmark's own consistency / synthesis / hallucination / omission vocabulary, kept as its own group
    data quality            valid-pixel and nodata fractions, skipped / invalid samples

Synthetic and real cross-sensor datasets are reported in separate sections and never combined into one headline number.

    python -m frame.evaluate check CONFIG.json     validate config, datasets, geometry and role safety; evaluate nothing
    python -m frame.evaluate run   CONFIG.json     run the evaluation and write experiments/evaluation/<name>/
    python -m frame.evaluate shift CONFIG.json --dataset NAME    spatial-shift sensitivity of the reference metrics (requirements 142 section 27)

See docs/EVALUATION.md.
"""
