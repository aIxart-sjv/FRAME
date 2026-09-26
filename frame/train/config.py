"""The training configuration: one frozen, validated, serialisable dataclass tree (Phase 4).

Design points
-------------
* Stdlib only, so it imports in every environment.
* Strict: unknown keys are refused (a typo must not silently fall back to a default), every value is
  range-checked, and errors name the dotted field.
* Splits are NAMES from the Phase 3 manifest (``train`` / ``val``), never a recipe for splitting: this
  layer never splits patches. ``test`` is refused as either split, so test-only data cannot enter the
  loop through the config.
* ``resume_signature`` is the part of the config a checkpoint must match to be resumed. It excludes what
  may legitimately change between a run and its resumption (the step budget, the output location, the
  logging/checkpoint cadence, the run name).
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import MISSING, dataclass, field, fields, is_dataclass, replace
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple, Union

from frame.models.config import RGBN_BAND_ORDER
from frame.train.errors import ConfigError

MODEL_NAMES = ("tiny_cnn", "sen2sr_lite", "sen2sr_mamba")
OPTIMIZERS = ("adamw", "adam", "sgd")
SCHEDULERS = ("none", "cosine", "step")
RECONSTRUCTION_LOSSES = ("l1", "charbonnier")
AMP_MODES = ("off", "fp16", "bf16")
BASELINES = ("bicubic", "lite")
SPLIT_NAMES = ("train", "val")          # 'test' is deliberately absent: it can never be trained or selected on
#: Canonical Sentinel-2 L2A band names (same set as frame.data.bands.L2A_BANDS; a test keeps them equal).
KNOWN_BANDS = ("B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B11", "B12")

_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_DEVICE = re.compile(r"^(auto|cpu|cuda(:\d+)?)$")
#: Fields a resumed run may change without invalidating its checkpoint.
_RESUME_EXEMPT = ("steps", "output_dir", "checkpoint_every", "validate_every", "log_every", "keep_last_checkpoints", "name", "baselines")


def resolve_path(value: Union[str, Path], base_dir: Optional[Path]) -> Path:
    """Relative paths in a config are resolved against the directory of the config file (the current directory if none is given)."""
    path = Path(value).expanduser()
    return path if path.is_absolute() or base_dir is None else Path(base_dir) / path


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_num(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _require(condition: bool, field_name: str, message: str) -> None:
    if not condition:
        raise ConfigError(message, field=field_name)


@dataclass(frozen=True)
class ModelConfig:
    name: str
    params: Dict[str, Any] = field(default_factory=dict)   # constructor arguments of the chosen model (see frame.train.models)
    pretrained: bool = False                                # start from the published SEN2SR weights (lite / mamba only)

    def validate(self) -> None:
        _require(self.name in MODEL_NAMES, "model.name", f"must be one of {MODEL_NAMES}, got {self.name!r}")
        _require(isinstance(self.params, dict), "model.params", "must be an object")
        _require(isinstance(self.pretrained, bool), "model.pretrained", "must be true or false")


@dataclass(frozen=True)
class DataConfig:
    manifest: str
    data_root: Optional[str] = None            # default: $FRAME_DATA_ROOT
    datasets: Tuple[str, ...] = ()             # empty = every dataset in the manifest
    train_split: str = "train"
    val_split: str = "val"
    lr_patch: int = 32                         # LR training crop (HR crop = this x scale)
    val_lr_patch: int = 128                    # LR validation patch; 128 is the models' native tile
    patches_per_pair: int = 4
    min_valid_fraction: float = 0.8            # random crops with less valid (non-nodata) coverage are redrawn
    augment: bool = True                       # same flip / rotation on LR and HR (frame.data loader)
    lr_bands: Tuple[str, ...] = tuple(RGBN_BAND_ORDER)
    hr_bands: Tuple[str, ...] = ()             # empty = same as lr_bands (stored resolved)
    require_region_disjoint: bool = True

    def __post_init__(self) -> None:
        if len(self.hr_bands) == 0:
            object.__setattr__(self, "hr_bands", tuple(self.lr_bands))

    def validate(self) -> None:
        _require(isinstance(self.manifest, str) and self.manifest.strip() != "", "data.manifest", "must be a non-empty path")
        for name, value in (("data.train_split", self.train_split), ("data.val_split", self.val_split)):
            _require(value != "test", name, "the 'test' split can never be used for training or validation")
            _require(value in SPLIT_NAMES, name, f"must be one of {SPLIT_NAMES}, got {value!r}")
        _require(self.train_split != self.val_split, "data.val_split", "must differ from data.train_split")
        _require(_is_int(self.lr_patch) and self.lr_patch >= 8, "data.lr_patch", "must be an integer >= 8")
        _require(_is_int(self.val_lr_patch) and self.val_lr_patch >= 8, "data.val_lr_patch", "must be an integer >= 8")
        _require(_is_int(self.patches_per_pair) and self.patches_per_pair >= 1, "data.patches_per_pair", "must be an integer >= 1")
        _require(_is_num(self.min_valid_fraction) and 0 < self.min_valid_fraction <= 1, "data.min_valid_fraction", "must be in (0, 1]")
        for name, bands in (("data.lr_bands", self.lr_bands), ("data.hr_bands", self.hr_bands)):
            _require(name == "data.hr_bands" or len(bands) > 0, name, "must list at least one band")
            _require(len(set(bands)) == len(bands), name, f"has duplicate bands: {list(bands)}")
            unknown = [b for b in bands if b not in KNOWN_BANDS]
            _require(not unknown, name, f"unknown band name(s) {unknown}; use canonical names {KNOWN_BANDS}")
        _require(len(self.hr_bands) == len(self.lr_bands), "data.hr_bands", "must list the same number of bands as data.lr_bands")
        _require(isinstance(self.augment, bool) and isinstance(self.require_region_disjoint, bool), "data.augment", "flags must be true or false")


@dataclass(frozen=True)
class LossConfig:
    reconstruction: str = "l1"
    charbonnier_eps: float = 1e-3
    spectral_weight: float = 0.0       # lambda for the spectral-angle term (0 = off)
    consistency_weight: float = 0.0    # lambda for the downsample-consistency term (0 = off)

    def validate(self) -> None:
        _require(self.reconstruction in RECONSTRUCTION_LOSSES, "loss.reconstruction", f"must be one of {RECONSTRUCTION_LOSSES}, got {self.reconstruction!r}")
        _require(_is_num(self.charbonnier_eps) and self.charbonnier_eps > 0, "loss.charbonnier_eps", "must be a positive finite number")
        for name in ("spectral_weight", "consistency_weight"):
            _require(_is_num(getattr(self, name)) and getattr(self, name) >= 0, f"loss.{name}", "must be a finite number >= 0")


@dataclass(frozen=True)
class OptimConfig:
    name: str = "adamw"
    lr: float = 1e-3
    weight_decay: float = 0.0
    betas: Tuple[float, float] = (0.9, 0.999)
    momentum: float = 0.9                      # sgd only
    grad_clip_norm: Optional[float] = None

    def validate(self) -> None:
        _require(self.name in OPTIMIZERS, "optim.name", f"must be one of {OPTIMIZERS}, got {self.name!r}")
        _require(_is_num(self.lr) and self.lr > 0, "optim.lr", "must be a positive finite number")
        _require(_is_num(self.weight_decay) and self.weight_decay >= 0, "optim.weight_decay", "must be a finite number >= 0")
        _require(len(self.betas) == 2 and all(_is_num(b) and 0 <= b < 1 for b in self.betas), "optim.betas", "must be two numbers in [0, 1)")
        _require(_is_num(self.momentum) and 0 <= self.momentum < 1, "optim.momentum", "must be in [0, 1)")
        _require(self.grad_clip_norm is None or (_is_num(self.grad_clip_norm) and self.grad_clip_norm > 0), "optim.grad_clip_norm", "must be null or a positive number")


@dataclass(frozen=True)
class SchedulerConfig:
    name: str = "none"
    warmup_steps: int = 0                      # linear warm-up from lr/10 (in optimizer steps)
    step_size: int = 100                       # 'step': decay every N optimizer steps
    gamma: float = 0.5                         # 'step': decay factor
    min_lr_fraction: float = 0.0               # 'cosine': final lr = lr * this

    def validate(self, steps: int) -> None:
        _require(self.name in SCHEDULERS, "scheduler.name", f"must be one of {SCHEDULERS}, got {self.name!r}")
        _require(_is_int(self.warmup_steps) and 0 <= self.warmup_steps < steps, "scheduler.warmup_steps", f"must be an integer in [0, steps={steps})")
        _require(_is_int(self.step_size) and self.step_size >= 1, "scheduler.step_size", "must be an integer >= 1")
        _require(_is_num(self.gamma) and 0 < self.gamma <= 1, "scheduler.gamma", "must be in (0, 1]")
        _require(_is_num(self.min_lr_fraction) and 0 <= self.min_lr_fraction <= 1, "scheduler.min_lr_fraction", "must be in [0, 1]")


@dataclass(frozen=True)
class PrecisionConfig:
    amp: str = "off"                           # off = float32 (the conservative default); fp16 needs a GPU; bf16 needs Ampere or newer

    def validate(self) -> None:
        _require(self.amp in AMP_MODES, "precision.amp", f"must be one of {AMP_MODES}, got {self.amp!r}")


@dataclass(frozen=True)
class TrainConfig:
    name: str
    model: ModelConfig
    data: DataConfig
    steps: int                                  # optimizer updates (each consumes grad_accum micro-batches)
    output_dir: str
    seed: int = 0
    batch_size: int = 1
    grad_accum: int = 1
    optim: OptimConfig = field(default_factory=OptimConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    precision: PrecisionConfig = field(default_factory=PrecisionConfig)
    checkpoint_every: int = 0                   # 0 = only the final checkpoint
    validate_every: int = 0                     # 0 = only at the end
    log_every: int = 1
    keep_last_checkpoints: int = 2
    device: str = "auto"
    deterministic: bool = True
    evaluate_train: bool = False                # also evaluate the training scenes at the end (an over-fit check)
    baselines: Tuple[str, ...] = ("bicubic",)   # reference upsamplers evaluated on the same validation patches

    def __post_init__(self) -> None:
        _require(isinstance(self.name, str) and bool(_NAME.match(self.name)), "name", "must be non-empty and made of letters, digits, '.', '_' or '-'")
        self.model.validate()
        self.data.validate()
        self.loss.validate()
        self.optim.validate()
        self.precision.validate()
        _require(_is_int(self.steps) and self.steps >= 1, "steps", "must be an integer >= 1")
        self.scheduler.validate(self.steps)
        _require(isinstance(self.output_dir, str) and self.output_dir.strip() != "", "output_dir", "must be a non-empty path")
        _require(_is_int(self.seed) and self.seed >= 0, "seed", "must be an integer >= 0")
        _require(_is_int(self.batch_size) and self.batch_size >= 1, "batch_size", "must be an integer >= 1")
        _require(_is_int(self.grad_accum) and self.grad_accum >= 1, "grad_accum", "must be an integer >= 1")
        for name, minimum in (("checkpoint_every", 0), ("validate_every", 0), ("log_every", 1), ("keep_last_checkpoints", 1)):
            _require(_is_int(getattr(self, name)) and getattr(self, name) >= minimum, name, f"must be an integer >= {minimum}")
        _require(isinstance(self.device, str) and bool(_DEVICE.match(self.device)), "device", "must be 'auto', 'cpu', 'cuda' or 'cuda:N'")
        _require(isinstance(self.deterministic, bool) and isinstance(self.evaluate_train, bool), "deterministic", "flags must be true or false")
        _require(all(b in BASELINES for b in self.baselines), "baselines", f"must only contain {BASELINES}, got {list(self.baselines)}")
        _require("lite" not in self.baselines or self.data.val_lr_patch == 128, "baselines",
                 "'lite' only accepts 128x128 LR tiles (its hard-constraint mask is 512x512): set data.val_lr_patch to 128 or drop it")

    # ------------------------------------------------------------------ derived

    @property
    def effective_batch_size(self) -> int:
        return self.batch_size * self.grad_accum

    # ------------------------------------------------------------------ serialisation

    def to_dict(self) -> Dict[str, Any]:
        return _to_plain(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "TrainConfig":
        return _build(cls, data, "")

    @classmethod
    def from_json(cls, text: str) -> "TrainConfig":
        try:
            return cls.from_dict(json.loads(text))
        except json.JSONDecodeError as exc:
            raise ConfigError(f"not valid JSON ({exc.msg} at line {exc.lineno})") from None

    @classmethod
    def load(cls, path: Union[str, Path]) -> "TrainConfig":
        path = Path(path)
        if not path.is_file():
            raise ConfigError(f"config file not found: {path}")
        try:
            return cls.from_dict(json.loads(path.read_text(encoding="utf-8")))
        except json.JSONDecodeError as exc:
            raise ConfigError(f"{path}: not valid JSON ({exc.msg} at line {exc.lineno})") from None

    def with_output_dir(self, output_dir: str) -> "TrainConfig":
        return replace(self, output_dir=str(output_dir))

    # ------------------------------------------------------------------ identity

    def digest(self) -> str:
        return hashlib.sha256(self.to_json().encode("utf-8")).hexdigest()

    def resume_signature(self) -> str:
        """Digest of everything a checkpoint must agree with to be resumed (see the module docstring)."""
        plain = self.to_dict()
        for key in _RESUME_EXEMPT:
            plain.pop(key, None)
        return hashlib.sha256(json.dumps(plain, sort_keys=True).encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------------------------------------------
# (de)serialisation helpers
# ---------------------------------------------------------------------------------------------------------------

def _to_plain(value: Any) -> Any:
    if is_dataclass(value):
        return {f.name: _to_plain(getattr(value, f.name)) for f in fields(value)}
    if isinstance(value, (tuple, list)):
        return [_to_plain(v) for v in value]
    if isinstance(value, dict):
        return {k: _to_plain(v) for k, v in value.items()}
    return value


_TUPLE_FIELDS = {"datasets", "lr_bands", "hr_bands", "betas", "baselines"}
_NESTED = {"model": ModelConfig, "data": DataConfig, "optim": OptimConfig, "scheduler": SchedulerConfig, "loss": LossConfig, "precision": PrecisionConfig}


def _build(cls: Any, data: Mapping[str, Any], prefix: str) -> Any:
    if not isinstance(data, Mapping):
        raise ConfigError("must be an object", field=prefix.rstrip(".") or None)
    known = {f.name: f for f in fields(cls)}
    unknown = sorted(set(data) - set(known))
    if unknown:
        raise ConfigError(f"unknown key(s) {unknown}; valid keys: {sorted(known)}", field=prefix.rstrip(".") or None)
    kwargs: Dict[str, Any] = {}
    for name, f in known.items():
        path = f"{prefix}{name}"
        if name in data:
            value = data[name]
            if cls is TrainConfig and name in _NESTED:
                value = _build(_NESTED[name], value, f"{path}.")
            elif name in _TUPLE_FIELDS and isinstance(value, (list, tuple)):
                value = tuple(value)
            elif isinstance(value, dict):
                value = dict(value)                       # never alias the caller's object
            kwargs[name] = value
        elif f.default is MISSING and f.default_factory is MISSING:
            raise ConfigError("is required", field=path)
    try:
        return cls(**kwargs)
    except TypeError as exc:                              # pragma: no cover - guarded by the checks above
        raise ConfigError(str(exc)) from None
