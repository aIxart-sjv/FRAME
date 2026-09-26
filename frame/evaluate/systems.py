"""Frozen systems under evaluation (Phase 5): bicubic, SEN2SR-Lite (with and without its hard constraint), SEN2SR-Mamba (through its isolated worker),
and frozen trained checkpoints, behind one interface with full provenance.

    system.infer(lr)      (C, h, w) reflectance -> InferenceResult with the (C, 4h, 4w) SR on the CPU
    system.provenance()   what exactly was evaluated: model, weights and their hashes, hard constraint, preprocessing, band order, tiling
    system.close()        release a worker / GPU memory

Fairness rules encoded here
---------------------------
* Every network system is evaluated through the SAME tile engine and tiling configuration (frame.tiling: 128 px tiles, reflect padding, linear blending),
  which is also how the API runs it. The hard constraint of Lite/Mamba acts inside each 512 x 512 tile output, as designed. Bicubic needs no tiling and is applied to
  the whole scene; that is recorded, not hidden.
* "With" and "without" the intended hard constraint are DIFFERENT named systems: ``lite`` is the published inference system, ``lite`` with ``hard_constraint: false``
  is the same CNN with only the clamp-to-non-negative the published wrapper applies before its constraint. The Mamba worker serves the constrained system only.
* A checkpoint system is FROZEN (eval mode, no gradients) and its training summary supplies the scenes it saw, so evaluation on those scenes, or of a model trained on a
  benchmark, is refused (`check_no_training_overlap`).
* Input convention for all systems: reflectance fraction = stored DN / the dataset's reflectance scale, bands B04,B03,B02,B08, no BOA offset, no clipping by FRAME.
"""

from __future__ import annotations

import hashlib
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import torch

from frame.evaluate.config import SystemSpec, resolve_path
from frame.evaluate.errors import RoleSafetyError, SystemUnavailableError
from frame.models.config import LITE_MODEL_NAME, RGBN_BAND_ORDER
from frame.models.mamba_client import MambaWorkerClient, check_mamba_availability
from frame.tiling import TilingConfig, run_tiled

PREPROCESSING = {
    "input_bands": list(RGBN_BAND_ORDER),
    "value_convention": "reflectance fraction = stored DN / the dataset's reflectance_scale (10000); no BOA offset applied, no clipping by FRAME",
}
BENCHMARK_KINDS = ("sen2neon", "opensr_test")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _home_relative(path: Path) -> str:
    try:
        return "~/" + str(Path(path).resolve().relative_to(Path.home()))
    except ValueError:
        return str(path)


@dataclass
class InferenceResult:
    sr: torch.Tensor                      # (C, 4h, 4w) float32, CPU
    seconds: float
    tile_count: int
    plan: Any = None                      # frame.tiling TilePlan, None when not tiled
    seam_diagnostic: Optional[Dict[str, Any]] = None


def seam_lines(plan: Any, config: TilingConfig) -> Tuple[List[int], List[int]]:
    """Seam row/column positions on the SR grid: the middle of each blend overlap between consecutive tiles."""
    half = (config.overlap * config.scale) // 2
    rows = sorted({t.sr_row_start + half for t in plan.tiles if t.sr_row_start > 0})
    cols = sorted({t.sr_col_start + half for t in plan.tiles if t.sr_col_start > 0})
    return rows, cols


class System:
    name: str
    kind: str
    hard_constraint: Optional[bool]

    def infer(self, lr: torch.Tensor) -> InferenceResult:
        raise NotImplementedError

    def provenance(self) -> Dict[str, Any]:
        raise NotImplementedError

    def close(self) -> None:
        pass


class BicubicSystem(System):
    kind = "bicubic"
    hard_constraint = None

    def __init__(self, name: str):
        self.name = name

    def infer(self, lr: torch.Tensor) -> InferenceResult:
        from frame.validation import bicubic_upsample

        started = time.perf_counter()
        sr = bicubic_upsample(lr.float().cpu(), 4)
        return InferenceResult(sr=sr, seconds=time.perf_counter() - started, tile_count=1)

    def provenance(self) -> Dict[str, Any]:
        return {"name": self.name, "kind": "bicubic", "hard_constraint": None, "weights": None, "model_name": None,
                "operator": "frame.validation.bicubic_upsample: bicubic interpolation with antialiasing, negatives clamped to 0", "tiling": "none (whole scene)", **PREPROCESSING}


class CallableSystem(System):
    """A tile model ``model(x[1, C, tile, tile]) -> y[1, C, 4*tile, 4*tile]`` run through the tile engine."""

    def __init__(self, name: str, model: Callable[[torch.Tensor], torch.Tensor], tiling: TilingConfig, *, hard_constraint: Optional[bool], provenance: Dict[str, Any],
                 device: str = "cpu", kind: str = "callable"):
        self.name, self.kind, self.hard_constraint = name, kind, hard_constraint
        self._model, self._tiling, self._provenance, self._device = model, tiling, dict(provenance), torch.device(device)

    def infer(self, lr: torch.Tensor) -> InferenceResult:
        result = run_tiled(self._model, lr.float().to(self._device), self._tiling, collect_seam_diagnostic=True)
        return InferenceResult(sr=result.sr, seconds=result.total_seconds, tile_count=result.plan.tile_count, plan=result.plan,
                               seam_diagnostic=result.seam_diagnostic.as_dict() if result.seam_diagnostic is not None else None)

    @property
    def tile_model(self) -> Callable[[torch.Tensor], torch.Tensor]:
        """The single-tile model ``model(x[1, C, tile, tile]) -> y`` (frame.reliability wraps it in the tile engine and the TTA ensemble)."""
        return self._model

    @property
    def device(self) -> str:
        return str(self._device)

    @property
    def tiling(self) -> TilingConfig:
        return self._tiling

    def provenance(self) -> Dict[str, Any]:
        return {"name": self.name, "kind": self.kind, "hard_constraint": self.hard_constraint, "tiling": self._tiling.describe(), "device": str(self._device),
                **PREPROCESSING, **self._provenance}


# ---------------------------------------------------------------------------------------------------------------
# Lite
# ---------------------------------------------------------------------------------------------------------------

def _lite(spec: SystemSpec, tiling: TilingConfig, device: str) -> System:
    directory = Path(spec.params.get("weights_dir") or os.environ.get("SEN2SR_BASELINE_WEIGHTS_DIR") or Path.home() / ".cache" / "sen2sr_baseline" / "SEN2SRLite_RGBN")
    files = ("model.safetensor", "hard_constraint.safetensor", "mlm.json")
    missing = [f for f in files if not (directory / f).is_file()]
    if missing:
        raise SystemUnavailableError(f"SEN2SR-Lite weights not found ({missing}) in {directory}. Nothing is downloaded automatically (the API downloads them on first use).")
    import mlstac

    compiled = mlstac.load(str(directory)).compiled_model(device=device)
    constrained = bool(spec.hard_constraint)
    if constrained:
        model: Callable = compiled
        note = "the published inference system: CNN + clamp to >= 0 + low-frequency Fourier hard constraint (512x512 mask) inside every tile"
    else:
        def model(x: torch.Tensor) -> torch.Tensor:
            return compiled.sr_model(x).clamp(min=0.0)

        note = "the same CNN with its output clamped to >= 0 exactly as the published wrapper does before its constraint; the hard constraint is NOT applied (an ablation, not the intended system)"
    params = sum(p.numel() for p in compiled.sr_model.parameters())
    provenance = {"model_name": LITE_MODEL_NAME, "executable_architecture": "CNNSR", "weights_dir": _home_relative(directory), "weights": {f: sha256_file(directory / f) for f in files},
                  "parameters": params, "note": note}
    return CallableSystem(spec.name, model, tiling, hard_constraint=constrained, provenance=provenance, device=device, kind="lite")


# ---------------------------------------------------------------------------------------------------------------
# Mamba
# ---------------------------------------------------------------------------------------------------------------

class MambaSystem(CallableSystem):
    def __init__(self, name: str, client: Any, tiling: TilingConfig):
        report = client.describe()
        provenance = {"model_name": report.get("model_name"), "executable_architecture": report.get("executable_architecture"), "parameters": report.get("parameter_count"),
                      "weights": {k: report.get(k) for k in ("weights_file", "weights_sha256", "hard_constraint_file", "hard_constraint_sha256")},
                      "architecture": report.get("architecture"), "worker_device": report.get("device"),
                      "note": "SEN2SR-Mamba RGBN through the isolated worker: MambaSR + clamp to >= 0 + low-frequency hard constraint, float32 (no autocast)"}
        super().__init__(name, client, tiling, hard_constraint=True, provenance=provenance, device="cpu", kind="mamba")
        self._client = client

    def close(self) -> None:
        self._client.close()


def _mamba(spec: SystemSpec, tiling: TilingConfig, device: str, factory: Optional[Callable[..., Any]]) -> System:
    if factory is None:
        availability = check_mamba_availability(device)
        if not availability.available:
            raise SystemUnavailableError(f"SEN2SR-Mamba is unavailable: {availability.message} ({availability.reason_code})")
        factory = lambda **kw: MambaWorkerClient(device=device, **kw)      # noqa: E731
    client = factory(**({"weights_dir": spec.params["weights_dir"]} if spec.params.get("weights_dir") else {}))
    try:
        client.start()
    except Exception as exc:
        raise SystemUnavailableError(f"The SEN2SR-Mamba worker could not start: {type(exc).__name__}: {exc}") from exc
    return MambaSystem(spec.name, client, tiling)


# ---------------------------------------------------------------------------------------------------------------
# frozen trained checkpoints
# ---------------------------------------------------------------------------------------------------------------

class CheckpointSystem(CallableSystem):
    def __init__(self, name: str, model: torch.nn.Module, tiling: TilingConfig, provenance: Dict[str, Any], device: str, training: Dict[str, Any]):
        super().__init__(name, model, tiling, hard_constraint=False, provenance=provenance, device=device, kind="checkpoint")
        self.training_info = training

    @property
    def _model_module(self) -> torch.nn.Module:
        return self._model


def _checkpoint(spec: SystemSpec, tiling: TilingConfig, device: str, base_dir: Optional[Path]) -> System:
    import json

    from frame.train import checkpoint as ckpt
    from frame.train.config import TrainConfig
    from frame.train.errors import TrainError
    from frame.train.models import build_model, count_parameters

    path = resolve_path(spec.checkpoint, base_dir)
    try:
        payload = ckpt.load_checkpoint(path)
        train_config = TrainConfig.from_dict(payload["config"])
        model = build_model(train_config.model, seed=train_config.seed)
        model.load_state_dict(payload["model"])
    except (TrainError, RuntimeError, KeyError) as exc:
        raise SystemUnavailableError(f"Checkpoint {path} could not be used ({type(exc).__name__}: {exc}).") from exc
    model = model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)

    training: Dict[str, Any] = {"step": int(payload["step"]), "seed": int(payload["seed"]), "manifest_digest": payload["manifest_digest"], "git_revision": payload.get("git_revision"),
                                "config_digest": train_config.digest(), "torch_version": payload.get("torch_version"), "training_datasets": None, "train_scenes": None, "val_scenes": None}
    if spec.train_summary:
        summary_path = resolve_path(spec.train_summary, base_dir)
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            data = summary["dataset"]
            training.update(training_datasets=list(data["datasets"]), train_scenes=list(data["train_scenes"]), val_scenes=list(data["val_scenes"]),
                            summary_sha256=sha256_file(summary_path), overlap_check="performed against the train and validation scenes listed in train_summary")
        except (OSError, KeyError, ValueError) as exc:
            raise SystemUnavailableError(f"The training summary {summary_path} could not be read ({type(exc).__name__}: {exc}).") from exc
    else:
        training["overlap_check"] = "not performed: no train_summary was given"
    provenance = {"model_name": train_config.model.name, "model_params": train_config.model.params, "parameters": count_parameters(model), "group": spec.group,
                  "weights": {"checkpoint": _home_relative(path), "checkpoint_sha256": sha256_file(path)}, "training": training,
                  "note": "a frozen model trained by frame.train WITHOUT the low-frequency hard constraint; not the published SEN2SR system"}
    return CheckpointSystem(spec.name, model, tiling, provenance, device, training)


def check_no_training_overlap(system: System, *, dataset_kind: str, scene_ids: Sequence[str]) -> None:
    """Refuse an evaluation that would contaminate it: a model trained on the benchmark, or scored on scenes it saw in training or validation."""
    info = getattr(system, "training_info", None)
    if not info or info.get("training_datasets") is None:
        return
    if dataset_kind in BENCHMARK_KINDS and dataset_kind in info["training_datasets"]:
        raise RoleSafetyError(f"Evaluation refused: system {system.name!r} was trained on {dataset_kind!r}, which is an independent benchmark.", codes=("trained_on_benchmark",))
    seen = set(info["train_scenes"] or []) | set(info["val_scenes"] or [])
    overlap = sorted(seen & set(scene_ids))
    if overlap:
        raise RoleSafetyError(f"Evaluation refused: system {system.name!r} saw scene(s) {overlap[:5]} in training/validation; they cannot be used to evaluate it.", codes=("train_eval_overlap",))


def build_system(spec: SystemSpec, tiling: TilingConfig, *, device: str, base_dir: Optional[Path] = None, mamba_client_factory: Optional[Callable[..., Any]] = None) -> System:
    if spec.kind == "bicubic":
        return BicubicSystem(spec.name)
    if spec.kind == "lite":
        return _lite(spec, tiling, device)
    if spec.kind == "mamba":
        return _mamba(spec, tiling, device, mamba_client_factory)
    if spec.kind == "checkpoint":
        return _checkpoint(spec, tiling, device, base_dir)
    raise SystemUnavailableError(f"Unknown system kind {spec.kind!r}.")
