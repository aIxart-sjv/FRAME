"""FRAME-side activation checkpointing for MambaSR (Phase 4). Torch only.

Why this exists: fine-tuning ``MambaSR`` on a 4 GB GPU is limited by activation memory, and gradient checkpointing (recompute
activations in the backward pass instead of storing them) is the standard remedy. Upstream ships a ``use_checkpoint`` flag, but it cannot
be used: ``BasicLayer.forward`` calls ``checkpoint.checkpoint(blk, x)`` without the ``x_size`` argument ``VSSBlock.forward`` requires.
``sen2sr/`` is not modified; this wraps each block's ``forward`` from the outside.

Measured on the RTX 3050 Laptop GPU (docs/TRAINING.md): at batch 1, native 128x128 LR tile, fp32, peak memory falls from out-of-memory
(without it even 64x64 does not fit) to 1,164 MiB, at the price of about one extra forward pass of compute per step.

Enable it through the model config: ``{"name": "sen2sr_mamba", "params": {"activation_checkpointing": true}}``.
"""

from __future__ import annotations

import torch.nn as nn
from torch.utils.checkpoint import checkpoint


def enable_block_checkpointing(model: nn.Module, block_class: str = "VSSBlock") -> int:
    """Wrap ``forward`` of every ``block_class`` module so its activations are recomputed in the backward pass.

    Returns how many blocks were newly wrapped (0 for a model without such blocks, and for blocks wrapped before: calling it twice does
    not nest). Outputs and gradients are identical to the unwrapped model; only memory and compute change.
    """
    wrapped = 0
    for module in model.modules():
        if type(module).__name__ != block_class or getattr(module, "_frame_checkpointed", False):
            continue
        original = module.forward

        def forward(x, x_size, _original=original):
            return checkpoint(_original, x, x_size, use_reentrant=False)

        module.forward = forward
        module._frame_checkpointed = True
        wrapped += 1
    return wrapped
