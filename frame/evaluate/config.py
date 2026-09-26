"""The evaluation configuration: frozen, strict, serialisable (Phase 5). Errors name the dotted field; unknown keys are refused.

Safety is part of the config: benchmarks are evaluated on their ``test`` split only, ``sen2naipv2`` is not an evaluable dataset (SEN2SR was trained on
it, so it is not independent of the systems under test), a synthetic set is evaluated on its held-out ``test`` split only, and a checkpoint system must
name its file. Relative paths resolve against the directory of the config file.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import MISSING, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple, Union

from frame.evaluate.errors import EvalConfigError
from frame.evaluate.metrics import METRICS_VERSION, MetricConfig
from frame.models.config import RGBN_BAND_ORDER

SYSTEM_KINDS = ("bicubic", "lite", "mamba", "checkpoint")
DATASET_KINDS = ("sen2neon", "opensr_test", "synthetic_smoke")
OPENSR_SUBSETS = ("spot", "spain_crops", "spain_urban")
HR_VARIANTS = ("HR", "HRharm")
EVIDENCE_CLASS = {"sen2neon": "real_cross_sensor", "opensr_test": "real_cross_sensor", "synthetic_smoke": "synthetic"}
DEVICES = re.compile(r"^(auto|cpu|cuda(:\d+)?)$")
_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


def _fail(message: str, field_name: Optional[str] = None):
    raise EvalConfigError(message, field=field_name)


@dataclass(frozen=True)
class SystemSpec:
    name: str
    kind: str
    checkpoint: Optional[str] = None        # kind 'checkpoint': a frame.train checkpoint (a frozen trained model)
    train_summary: Optional[str] = None     # its training run's summary.json: provenance and the train/eval overlap check
    group: Optional[str] = None             # systems sharing a group (e.g. seeds of one model) are also summarised together
    hard_constraint: Optional[bool] = None  # resolved default: lite/mamba True, checkpoint False, bicubic None (not applicable)
    params: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not _NAME.match(str(self.name)):
            _fail("must be non-empty and made of letters, digits, '.', '_' or '-'", "systems.name")
        if self.hard_constraint is None:
            object.__setattr__(self, "hard_constraint", {"lite": True, "mamba": True, "checkpoint": False}.get(self.kind))


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    kind: str
    manifest: Optional[str] = None          # sen2neon / synthetic_smoke: a Phase 3 manifest
    subset: Optional[str] = None            # opensr_test: spot | spain_crops | spain_urban
    hr_variant: str = "HRharm"              # opensr_test: HRharm (harmonised, the benchmark's own default) or HR
    split: str = "test"
    data_root: Optional[str] = None         # default $FRAME_DATA_ROOT

    @property
    def evidence_class(self) -> str:
        return EVIDENCE_CLASS[self.kind]


@dataclass(frozen=True)
class TilingSpec:
    tile_size: int = 128
    overlap: int = 32
    padding_mode: str = "reflect"
    blend_mode: str = "linear"


@dataclass(frozen=True)
class StatisticsSpec:
    n_boot: int = 10_000
    alpha: float = 0.05
    seed: int = 0
    reference_system: str = "bicubic"       # the system every paired difference is taken against (a reference point, not a claim of rank)
    pairs: Tuple[Tuple[str, str], ...] = () # further (A, B) comparisons, each reported as the paired difference A - B


@dataclass(frozen=True)
class EvalConfig:
    name: str
    output_dir: str
    systems: Tuple[SystemSpec, ...]
    datasets: Tuple[DatasetSpec, ...]
    tiling: TilingSpec = field(default_factory=TilingSpec)
    statistics: StatisticsSpec = field(default_factory=StatisticsSpec)
    metrics: Dict[str, Any] = field(default_factory=dict)
    bands: Tuple[str, ...] = tuple(RGBN_BAND_ORDER)
    min_valid_fraction: float = 0.05        # a tile with less valid reference coverage is SKIPPED (and listed with the reason), never silently dropped
    device: str = "auto"
    seed: int = 0

    def __post_init__(self) -> None:
        if not _NAME.match(str(self.name)):
            _fail("must be non-empty and made of letters, digits, '.', '_' or '-'", "name")
        if not isinstance(self.output_dir, str) or not self.output_dir.strip():
            _fail("must be a non-empty path", "output_dir")
        if not self.systems:
            _fail("at least one system is required", "systems")
        if not self.datasets:
            _fail("at least one dataset is required", "datasets")
        if len({s.name for s in self.systems}) != len(self.systems):
            _fail("system names must be unique", "systems")
        if len({d.name for d in self.datasets}) != len(self.datasets):
            _fail("dataset names must be unique", "datasets")
        for i, s in enumerate(self.systems):
            self._check_system(i, s)
        for i, d in enumerate(self.datasets):
            self._check_dataset(i, d)
        if tuple(self.bands) != tuple(RGBN_BAND_ORDER):
            _fail(f"must be FRAME's RGBN order {list(RGBN_BAND_ORDER)} (what the systems consume); got {list(self.bands)}", "bands")
        try:
            from frame.tiling.plan import TilingConfig

            TilingConfig(tile_size=self.tiling.tile_size, overlap=self.tiling.overlap, padding_mode=self.tiling.padding_mode, blend_mode=self.tiling.blend_mode)
        except Exception as exc:
            _fail(str(exc), "tiling")
        st = self.statistics
        if not isinstance(st.n_boot, int) or isinstance(st.n_boot, bool) or st.n_boot < 100:
            _fail("must be an integer >= 100", "statistics.n_boot")
        if not (isinstance(st.alpha, (int, float)) and 0 < st.alpha <= 0.2):
            _fail("must be in (0, 0.2]", "statistics.alpha")
        if st.reference_system not in {s.name for s in self.systems}:
            _fail(f"{st.reference_system!r} is not one of the configured systems {[s.name for s in self.systems]}", "statistics.reference_system")
        names = {s.name for s in self.systems}
        for pair in st.pairs:
            if len(pair) != 2 or pair[0] == pair[1] or not set(pair) <= names:
                _fail(f"each pair must name two different configured systems, got {list(pair)}", "statistics.pairs")
        if not (isinstance(self.min_valid_fraction, (int, float)) and 0 < self.min_valid_fraction <= 1):
            _fail("must be in (0, 1]", "min_valid_fraction")
        if not DEVICES.match(str(self.device)):
            _fail("must be 'auto', 'cpu', 'cuda' or 'cuda:N'", "device")
        if not isinstance(self.seed, int) or isinstance(self.seed, bool) or self.seed < 0:
            _fail("must be an integer >= 0", "seed")
        object.__setattr__(self, "metrics", self._normalise_metrics(self.metrics))

    # ------------------------------------------------------------------ validation helpers

    @staticmethod
    def _check_system(i: int, s: SystemSpec) -> None:
        p = f"systems[{i}]"
        if s.kind not in SYSTEM_KINDS:
            _fail(f"must be one of {SYSTEM_KINDS}, got {s.kind!r}", f"{p}.kind")
        if s.kind == "checkpoint" and not s.checkpoint:
            _fail("a 'checkpoint' system needs a checkpoint file", f"{p}.checkpoint")
        if s.kind != "checkpoint" and (s.checkpoint or s.train_summary):
            _fail("checkpoint / train_summary only apply to kind 'checkpoint'", f"{p}.checkpoint")
        if s.hard_constraint is False and s.kind not in ("lite", "checkpoint"):
            _fail("hard_constraint=false can only be evaluated for 'lite' (the Mamba worker serves the constrained system only; bicubic has none)", f"{p}.hard_constraint")

    @staticmethod
    def _check_dataset(i: int, d: DatasetSpec) -> None:
        p = f"datasets[{i}]"
        if d.kind == "sen2naipv2":
            _fail("sen2naipv2 is in-distribution for SEN2SR (it was trained on it), so it is not an independent benchmark and cannot be evaluated", f"{p}.kind")
        if d.kind not in DATASET_KINDS:
            _fail(f"must be one of {DATASET_KINDS}, got {d.kind!r}", f"{p}.kind")
        if d.split != "test":
            _fail("only the 'test' split can be evaluated (benchmarks are test-only; synthetic sets use their held-out test split)", f"{p}.split")
        if d.kind in ("sen2neon", "synthetic_smoke") and not d.manifest:
            _fail("needs a Phase 3 manifest", f"{p}.manifest")
        if d.kind == "opensr_test":
            if d.subset not in OPENSR_SUBSETS:
                _fail(f"must be one of {OPENSR_SUBSETS}, got {d.subset!r} (the 'naip' and 'venus' subsets have other grids)", f"{p}.subset")
            if d.hr_variant not in HR_VARIANTS:
                _fail(f"must be one of {HR_VARIANTS}", f"{p}.hr_variant")

    @staticmethod
    def _normalise_metrics(given: Mapping[str, Any]) -> Dict[str, Any]:
        defaults = MetricConfig().to_dict()
        unknown = sorted(set(given) - set(defaults))
        if unknown:
            _fail(f"unknown metric setting(s) {unknown}; valid: {sorted(defaults)}", "metrics")
        merged = {**defaults, **{k: (list(v) if isinstance(v, (list, tuple)) else v) for k, v in given.items()}}
        merged["version"] = METRICS_VERSION
        taus = merged["hallucination_taus"]
        if not taus or not all(isinstance(t, (int, float)) and t > 0 and math.isfinite(t) for t in taus):
            _fail("must be a non-empty list of positive reflectance thresholds", "metrics.hallucination_taus")
        for key in ("hf_sigma", "index_min_sum", "ratio_floor"):
            if not (isinstance(merged[key], (int, float)) and merged[key] > 0):
                _fail("must be a positive number", f"metrics.{key}")
        return merged

    # ------------------------------------------------------------------ views

    def metric_config(self) -> MetricConfig:
        m = dict(self.metrics)
        m.pop("version")
        m["hallucination_taus"] = tuple(m["hallucination_taus"])
        return MetricConfig(**m)

    def tiling_config(self):
        from frame.tiling.plan import TilingConfig

        return TilingConfig(tile_size=self.tiling.tile_size, overlap=self.tiling.overlap, padding_mode=self.tiling.padding_mode, blend_mode=self.tiling.blend_mode)

    # ------------------------------------------------------------------ serialisation

    def to_dict(self) -> Dict[str, Any]:
        return _plain(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"

    def digest(self) -> str:
        return hashlib.sha256(self.to_json().encode("utf-8")).hexdigest()

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "EvalConfig":
        return _build_config(data)

    @classmethod
    def from_json(cls, text: str) -> "EvalConfig":
        try:
            return cls.from_dict(json.loads(text))
        except json.JSONDecodeError as exc:
            raise EvalConfigError(f"not valid JSON ({exc.msg} at line {exc.lineno})") from None

    @classmethod
    def load(cls, path: Union[str, Path]) -> "EvalConfig":
        path = Path(path)
        if not path.is_file():
            raise EvalConfigError(f"config file not found: {path}")
        try:
            return cls.from_dict(json.loads(path.read_text(encoding="utf-8")))
        except json.JSONDecodeError as exc:
            raise EvalConfigError(f"{path}: not valid JSON ({exc.msg} at line {exc.lineno})") from None


def _plain(value: Any) -> Any:
    if is_dataclass(value):
        return {f.name: _plain(getattr(value, f.name)) for f in fields(value)}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if isinstance(value, dict):
        return {k: _plain(v) for k, v in value.items()}
    return value


def _construct(cls: Any, data: Mapping[str, Any], prefix: str, nested: Mapping[str, Any] = {}, lists: Mapping[str, Any] = {}) -> Any:
    if not isinstance(data, Mapping):
        raise EvalConfigError("must be an object", field=prefix.rstrip(".") or None)
    known = {f.name: f for f in fields(cls)}
    unknown = sorted(set(data) - set(known))
    if unknown:
        raise EvalConfigError(f"unknown key(s) {unknown}; valid keys: {sorted(known)}", field=prefix.rstrip(".") or None)
    kwargs: Dict[str, Any] = {}
    for name, f in known.items():
        path = f"{prefix}{name}"
        if name in data:
            value = data[name]
            if name in nested:
                value = _construct(nested[name], value, f"{path}.")
            elif name in lists:
                if not isinstance(value, (list, tuple)):
                    raise EvalConfigError("must be a list", field=path)
                value = tuple(_construct(lists[name], v, f"{path}[{i}].") for i, v in enumerate(value))
            elif name == "pairs" and isinstance(value, (list, tuple)):
                value = tuple(tuple(v) if isinstance(v, (list, tuple)) else v for v in value)
            elif name in ("bands",) and isinstance(value, (list, tuple)):
                value = tuple(value)
            elif isinstance(value, dict):
                value = dict(value)
            kwargs[name] = value
        elif f.default is MISSING and f.default_factory is MISSING:
            raise EvalConfigError("is required", field=path)
    return cls(**kwargs)


def _build_config(data: Mapping[str, Any]) -> EvalConfig:
    return _construct(EvalConfig, data, "", nested={"tiling": TilingSpec, "statistics": StatisticsSpec}, lists={"systems": SystemSpec, "datasets": DatasetSpec})


def resolve_path(value: Union[str, Path], base_dir: Optional[Path]) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() or base_dir is None else Path(base_dir) / path
