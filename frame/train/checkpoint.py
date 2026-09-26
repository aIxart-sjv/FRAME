"""Resumable checkpoints (Phase 4). Torch, numpy and the standard library only.

A checkpoint holds what is needed to CONTINUE a run exactly and to AUDIT it later: model, optimizer,
scheduler and AMP-scaler state, the optimizer-step and micro-step counters, the RNG states, the full
training config and its resume signature, the seed, the Phase 3 manifest digest (which data index the run
used), the git revision, and the torch version.

* Only tensors and Python primitives are stored (RNG states are converted), so a checkpoint loads with
  ``torch.load(..., weights_only=True)`` and never unpickles arbitrary objects.
* Writes are atomic (temp file + rename): a crash mid-save leaves the previous checkpoint intact.
* Resuming is refused when the config (apart from the step budget / output location / cadence) or the
  manifest digest differs, because the continued run would not be the same experiment.
* Data order is a pure function of ``(seed, micro_step)`` (see frame.train.trainer), so no loader state needs
  saving: the counters are enough to resume on exactly the batch an uninterrupted run would see next.
"""

from __future__ import annotations

import os
import random
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch

from frame.train.config import TrainConfig
from frame.train.errors import CheckpointError

FORMAT_VERSION = 1
_REQUIRED = ("format_version", "model", "optimizer", "scheduler", "scaler", "step", "micro_step", "config", "resume_signature",
             "seed", "manifest_digest", "torch_version", "rng")
_NAME = re.compile(r"^step_(\d+)\.pt$")

PathLike = Union[str, Path]


def checkpoint_name(step: int) -> str:
    return f"step_{step:06d}.pt"


# ---------------------------------------------------------------------------------------------------------------
# RNG
# ---------------------------------------------------------------------------------------------------------------

def capture_rng() -> Dict[str, Any]:
    version, internal, gauss = random.getstate()
    name, keys, pos, has_gauss, cached = np.random.get_state()
    state: Dict[str, Any] = {
        "python": [version, list(internal), gauss],
        "numpy": [name, [int(k) for k in keys], int(pos), int(has_gauss), float(cached)],
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available() and torch.cuda.is_initialized():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng(state: Dict[str, Any]) -> None:
    version, internal, gauss = state["python"]
    random.setstate((version, tuple(internal), gauss))
    name, keys, pos, has_gauss, cached = state["numpy"]
    np.random.set_state((name, np.array(keys, dtype=np.uint32), pos, has_gauss, cached))
    torch.set_rng_state(state["torch"].cpu())
    if "cuda" in state and torch.cuda.is_available() and torch.cuda.is_initialized():
        torch.cuda.set_rng_state_all([s.cpu() for s in state["cuda"]])


# ---------------------------------------------------------------------------------------------------------------
# save / load / apply
# ---------------------------------------------------------------------------------------------------------------

def save_checkpoint(
    path: PathLike,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    scaler: Any,
    step: int,
    micro_step: int,
    config: TrainConfig,
    manifest_digest: str,
    git_revision: Optional[str],
    extra: Optional[Dict[str, Any]] = None,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: Dict[str, Any] = {
        "format_version": FORMAT_VERSION,
        "step": int(step),
        "micro_step": int(micro_step),
        "seed": config.seed,
        "config": config.to_dict(),
        "resume_signature": config.resume_signature(),
        "manifest_digest": manifest_digest,
        "git_revision": git_revision,
        "torch_version": str(torch.__version__),   # TorchVersion is a str subclass that weights_only loading refuses
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(),
        "rng": capture_rng(),
        "extra": dict(extra or {}),
    }
    temporary = path.with_name(path.name + ".tmp")
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return path


def load_checkpoint(path: PathLike, *, map_location: Union[str, torch.device] = "cpu") -> Dict[str, Any]:
    path = Path(path)
    if not path.is_file():
        raise CheckpointError(f"Checkpoint not found: {path}")
    try:
        payload = torch.load(path, map_location=map_location, weights_only=True)
    except Exception as exc:  # torch raises several unrelated types for a corrupt or non-torch file
        raise CheckpointError(f"Checkpoint {path} could not be read ({type(exc).__name__}: {exc}).") from exc
    if not isinstance(payload, dict):
        raise CheckpointError(f"Checkpoint {path} does not contain a checkpoint dictionary.")
    missing = [key for key in _REQUIRED if key not in payload]
    if missing:
        raise CheckpointError(f"Checkpoint {path} is missing required entries {missing}.")
    return payload


def check_resumable(payload: Dict[str, Any], config: TrainConfig, manifest_digest: str) -> None:
    """Raise `CheckpointError` unless ``payload`` can be continued by a run with ``config`` on ``manifest_digest``."""
    if payload.get("format_version") != FORMAT_VERSION:
        raise CheckpointError(f"Unsupported checkpoint format version {payload.get('format_version')!r} (this code reads {FORMAT_VERSION}).")
    if payload["resume_signature"] != config.resume_signature():
        raise CheckpointError(
            "The training config differs from the one that produced this checkpoint. Only the step budget, output directory, "
            "run name, baselines and logging/checkpoint cadence may change on resume."
        )
    if payload["manifest_digest"] != manifest_digest:
        raise CheckpointError(
            f"The data manifest changed: the checkpoint was made on manifest {payload['manifest_digest'][:12]}..., "
            f"this run uses {manifest_digest[:12]}..."
        )


def apply_checkpoint(payload: Dict[str, Any], model: torch.nn.Module, optimizer: torch.optim.Optimizer, scheduler: Any, scaler: Any) -> None:
    """Load the model / optimizer / scheduler / scaler state and the RNG streams from ``payload``."""
    try:
        model.load_state_dict(payload["model"])
    except RuntimeError as exc:
        raise CheckpointError(f"The checkpoint's model state does not match this model's architecture: {exc}") from exc
    try:
        optimizer.load_state_dict(payload["optimizer"])
        scheduler.load_state_dict(payload["scheduler"])
        scaler.load_state_dict(payload["scaler"])
    except (ValueError, KeyError, RuntimeError) as exc:
        raise CheckpointError(f"The checkpoint's optimizer/scheduler state is incompatible with this run: {exc}") from exc
    restore_rng(payload["rng"])


# ---------------------------------------------------------------------------------------------------------------
# directory management
# ---------------------------------------------------------------------------------------------------------------

def _numbered(directory: PathLike) -> List[Path]:
    directory = Path(directory)
    if not directory.is_dir():
        return []
    return sorted((p for p in directory.iterdir() if _NAME.match(p.name)), key=lambda p: int(_NAME.match(p.name).group(1)))


def latest_checkpoint(directory: PathLike) -> Optional[Path]:
    found = _numbered(directory)
    return found[-1] if found else None


def prune_checkpoints(directory: PathLike, keep_last: int) -> List[Path]:
    """Delete all but the newest ``keep_last`` numbered checkpoints; nothing else in the directory is touched."""
    found = _numbered(directory)
    doomed = found[: max(0, len(found) - keep_last)]
    for path in doomed:
        path.unlink()
    return doomed
