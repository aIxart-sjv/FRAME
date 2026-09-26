"""SEN2SR-Mamba RGBN adapter: the real ``MambaSR`` + its hard constraint behind
a plain tensor-in / tensor-out callable (Phase 1).

This module runs in the **Mamba environment** (it needs ``mamba_ssm``); the
main FRAME environment never imports it -- it talks to it through
frame.models.mamba_client / mamba_worker. It is nonetheless *importable*
anywhere: every heavy import (``sen2sr``, ``mamba_ssm``, ``safetensors``)
happens inside `MambaRGBNModel.load`, so unit tests can import the module and
exercise the checks that do not need the model.

What it wraps -- reused from upstream, not re-implemented:

    sen2sr.models.opensr_baseline.mamba.MambaSR   (architecture: config.MAMBA_ARCHITECTURE)
    sr_model.safetensor                           (weights)
    sen2sr.models.tricks.HardConstraint           (low-frequency Fourier constraint)
    sr_hard_constraint.safetensor                 (its low-pass mask)
    sen2sr.nonreference.srmodel                   (clamp >= 0, then the hard constraint)

This is the same construction ``models/SEN2SR/load.py`` uses for the RGBN
stage of the cascade, minus the 20 m / SWIR stages, which are out of scope.

Contract (see frame.models.contract): float32 reflectance,
(4|B, 4, 128, 128) in B04,B03,B02,B08 order -> float32 (4|B, 4, 512, 512) in
the same order. No band reordering happens here: the upstream cascade feeds
this exact model ``x[:, [2, 1, 0, 6]]`` (B04,B03,B02,B08) and only reorders
the *output* back to Sentinel-2 file order for its own 10-band stack.

Runs in float32 with no autocast: ``MambaSR``'s selective-scan asserts
float32 internally, and the artifact manifest's ``float16`` is stale.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch

from frame.models import config
from frame.models.contract import validate_input, validate_output
from frame.models.errors import ModelInferenceError, ModelLoadError, ModelUnavailableError

PathLike = Union[str, Path]


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class MambaLoadReport:
    """What was actually loaded -- recorded for provenance and tests."""

    model_name: str
    executable_architecture: str
    artifact_metadata_label: str
    architecture: Dict[str, Any]
    parameter_count: int
    state_tensors: int
    missing_keys: List[str]
    unexpected_keys: List[str]
    weights_file: str
    weights_sha256: str
    hard_constraint_file: str
    hard_constraint_sha256: str
    device: str
    load_seconds: float
    input_band_order: List[str] = field(default_factory=lambda: list(config.RGBN_BAND_ORDER))

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


class MambaRGBNModel:
    """Loaded SEN2SR-Mamba RGBN 4x model. Build with `MambaRGBNModel.load`."""

    def __init__(self, module: torch.nn.Module, device: str, report: MambaLoadReport):
        self._module = module
        self.device = device
        self.report = report

    # ------------------------------------------------------------------ load

    @classmethod
    def load(cls, weights_dir: Optional[PathLike] = None, device: str = "cuda") -> "MambaRGBNModel":
        """Construct ``MambaSR``, load ``sr_model.safetensor`` strictly, attach
        the hard constraint, freeze, and move to ``device``.

        Raises `ModelUnavailableError` if CUDA / the weights / the runtime are
        missing, `ModelLoadError` if the weights do not match the architecture.
        """
        started = time.time()
        weights_dir = Path(weights_dir) if weights_dir is not None else config.MAMBA_WEIGHTS_DIR

        if not str(device).startswith("cuda") or not torch.cuda.is_available():
            # MambaSR's selective scan is a CUDA kernel: there is no CPU
            # implementation, so refuse rather than fall back.
            raise ModelUnavailableError(
                "SEN2SR-Mamba requires a CUDA-capable GPU.",
                technical_detail=f"device={device!r}, torch.cuda.is_available()={torch.cuda.is_available()}",
            )

        weights_file = weights_dir / config.MAMBA_SR_WEIGHTS_FILENAME
        constraint_file = weights_dir / config.MAMBA_HARD_CONSTRAINT_FILENAME
        for required in (weights_file, constraint_file):
            if not required.is_file():
                raise ModelUnavailableError(
                    "SEN2SR-Mamba model files were not found on this server.",
                    technical_detail=f"missing {required}",
                )

        try:
            import safetensors.torch as safetensors_torch
            from sen2sr.models.opensr_baseline.mamba import MambaSR
            from sen2sr.models.tricks import HardConstraint
            from sen2sr.nonreference import srmodel
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise ModelUnavailableError(
                "The SEN2SR-Mamba runtime is not installed on this server.",
                technical_detail=f"{type(exc).__name__}: {exc}",
            ) from exc

        state_dict = safetensors_torch.load_file(str(weights_file))
        module = MambaSR(**config.MAMBA_ARCHITECTURE.kwargs())
        result = module.load_state_dict(state_dict, strict=False)
        missing, unexpected = list(result.missing_keys), list(result.unexpected_keys)
        if missing or unexpected:
            raise ModelLoadError(
                "The SEN2SR-Mamba weights do not match the expected architecture.",
                technical_detail=(
                    f"{len(missing)} missing keys (e.g. {missing[:3]}), "
                    f"{len(unexpected)} unexpected keys (e.g. {unexpected[:3]})"
                ),
            )

        module = module.to(device).eval()
        for parameter in module.parameters():
            parameter.requires_grad = False

        mask = safetensors_torch.load_file(str(constraint_file))["weights"]
        if tuple(mask.shape) != (config.OUTPUT_SIZE, config.OUTPUT_SIZE):
            raise ModelLoadError(
                "The SEN2SR-Mamba hard-constraint mask has an unexpected size.",
                technical_detail=f"mask shape {tuple(mask.shape)}, expected {(config.OUTPUT_SIZE,) * 2}",
            )
        hard_constraint = HardConstraint(low_pass_mask=mask.to(device), device=device)
        wrapped = srmodel(sr_model=module, hard_constraint=hard_constraint, device=device)

        report = MambaLoadReport(
            model_name=config.MAMBA_MODEL_NAME,
            executable_architecture=config.MAMBA_EXECUTABLE_ARCHITECTURE,
            artifact_metadata_label=config.MAMBA_ARTIFACT_METADATA_LABEL,
            architecture=config.MAMBA_ARCHITECTURE.kwargs(),
            parameter_count=sum(p.numel() for p in module.parameters()),
            state_tensors=len(state_dict),
            missing_keys=missing,
            unexpected_keys=unexpected,
            weights_file=weights_file.name,
            weights_sha256=sha256_of(weights_file),
            hard_constraint_file=constraint_file.name,
            hard_constraint_sha256=sha256_of(constraint_file),
            device=str(device),
            load_seconds=round(time.time() - started, 3),
        )
        return cls(wrapped, str(device), report)

    # ------------------------------------------------------------- inference

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Super-resolve ``x`` (contract in the module docstring).

        Tiles of a batch are run one at a time so peak GPU memory stays at the
        single-tile figure measured for a 4 GB card regardless of batch size.
        The result is on this model's device and has the same rank as ``x``.
        """
        validate_input(x)
        batched = x if x.ndim == 4 else x[None]

        outputs = []
        with torch.no_grad():
            for tile in batched:
                try:
                    outputs.append(self._module(tile[None].to(self.device)))
                except torch.cuda.OutOfMemoryError as exc:
                    torch.cuda.empty_cache()
                    raise ModelInferenceError(
                        "SEN2SR-Mamba ran out of GPU memory.", technical_detail=str(exc)
                    ) from exc
        result = torch.cat(outputs, dim=0)
        result = result if x.ndim == 4 else result[0]

        try:
            validate_output(result, input_shape=tuple(x.shape))
        except Exception as exc:
            raise ModelInferenceError(
                "SEN2SR-Mamba produced an invalid output.", technical_detail=str(exc)
            ) from exc
        return result

    # --------------------------------------------------------------- helpers

    def describe(self) -> Dict[str, Any]:
        return self.report.as_dict()

    def peak_memory_mib(self) -> float:
        return torch.cuda.max_memory_allocated(self.device) / 2**20

    def reset_peak_memory(self) -> None:
        torch.cuda.reset_peak_memory_stats(self.device)
