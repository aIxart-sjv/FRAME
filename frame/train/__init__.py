"""FRAME training layer (Phase 4): the smallest reproducible loop that consumes the Phase 3 paired data.

    manifest -> geographic train/val split -> PairedPatchDataset -> model -> loss -> optimizer -> checkpoint -> validation

The core (`config`, `losses`, `models`, `checkpoint`, `trainer`) imports only torch, numpy and the standard
library, so it runs in BOTH environments (main and the isolated Mamba one). Anything that needs the data
layer or evaluation libraries (`data`, `validate`, `baselines`, `cli`) imports them lazily and runs in the
main environment. Importing this package imports nothing heavy.

    python -m frame.train run CONFIG.json          train (and validate) as configured
    python -m frame.train check CONFIG.json        validate the config and the manifest; train nothing

See docs/TRAINING.md.
"""
