"""The reliability configuration: frozen, strict, serialisable (Phase 6). Errors name the dotted field; unknown keys are refused.

Every setting that could influence WHICH evidence is analysed or HOW a result is read is declared here, before any result exists, and is recorded with the
run (config.json, digest): the alignment tolerance, the valid-pixel threshold, the cell sizes, the high-error quantile, the development/test split seed. The dataset and
system safety rules of the Phase 5 evaluation apply unchanged (test splits only, no sen2naipv2, no benchmark contamination).
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import MISSING, dataclass, field, fields
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple, Union

from frame.evaluate.config import DEVICES, _NAME, DatasetSpec, EvalConfig, SystemSpec, TilingSpec, _plain
from frame.evaluate.errors import EvalConfigError
from frame.evaluate.metrics import METRICS_VERSION, MetricConfig
from frame.reliability.errors import ReliabilityConfigError

TRANSFORM_NAMES = ("identity", "hflip", "vflip", "rot90", "rot180", "rot270")
ALIGNMENT_METHODS = ("bicubic_baseline_cross_correlation",)
SYSTEM_KINDS = ("lite", "mamba", "checkpoint")


def _fail(message: str, name: Optional[str] = None):
    raise ReliabilityConfigError(message, field=name)


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_num(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


@dataclass(frozen=True)
class TTASpec:
    """The test-time-augmentation ensemble whose stability is being validated: the deployed one (frame.uncertainty.DEFAULT_TRANSFORMS)."""

    transforms: Tuple[str, ...] = TRANSFORM_NAMES
    seed: int = 42


@dataclass(frozen=True)
class AlignmentSpec:
    """Reference-registration gate. The displacement of the bicubic baseline against the reference is estimated (frame.evaluate.shift.estimate_alignment; a
    system-neutral estimate, so every model is judged on identical evidence). A tile is pixel-level eligible only when, after an optionally applied whole-pixel
    translation (an integer crop of both grids, no resampling, recorded), the remaining displacement is within ``tolerance_hr_px`` and the four quadrants agree
    within ``quadrant_tolerance_hr_px`` (a single global translation is the only misregistration model this phase accepts)."""

    method: str = "bicubic_baseline_cross_correlation"
    tolerance_hr_px: float = 0.5            # 1/8 of a Sentinel-2 pixel; the Phase 5 shift sweep first shows a measurable loss at 1 HR px
    quadrant_tolerance_hr_px: float = 1.0
    max_correction_hr_px: int = 4           # one LR pixel; beyond it the reference is too far off to correct by a recorded global translation
    apply_translation_correction: bool = True
    uncertain_factor: float = 2.0           # residual / spread beyond this many tolerances is 'not eligible'; between one and this is 'uncertain'


@dataclass(frozen=True)
class EligibilitySpec:
    min_valid_fraction: float = 0.5         # of the tile, after the alignment crop, under the strict evaluation mask
    max_nonfinite_fraction: float = 0.0     # non-finite reference pixels tolerated (fraction of the tile); any beyond it excludes the tile
    min_members: int = 2                    # a one-member ensemble has zero spread by construction, which is not stability


@dataclass(frozen=True)
class AnalysisSpec:
    cell_sizes_hr_px: Tuple[int, ...] = (4, 16)          # 4 HR px = one 10 m Sentinel-2 cell; 16 HR px = 40 m
    min_cell_valid_fraction: float = 0.75
    pooled_cells_per_tile: int = 8192                    # seeded uniform subsample of a tile's valid cells for the pooled analyses (cells of a tile are not independent)
    pooled_pixels_per_tile: int = 20000
    coverage_grid: Tuple[float, ...] = (1.0, 0.9, 0.8, 0.7, 0.6, 0.5)
    high_error_quantile: float = 0.9                     # 'high error' = above this quantile of development-cell error
    split_seed: int = 0
    dev_fraction: float = 0.5
    min_units_for_split: int = 12                        # fewer scene units than this cannot be split into development and test: descriptive only
    calibration_bins: int = 10
    displacement_sweep_hr_px: Tuple[int, ...] = (0, 1, 2, 4)   # justification of the gate: the pixel-level correlation with the reference displaced by these amounts


@dataclass(frozen=True)
class BootstrapSpec:
    n_boot: int = 2000
    alpha: float = 0.05
    seed: int = 0


@dataclass(frozen=True)
class ReliabilityConfig:
    name: str
    output_dir: str
    systems: Tuple[SystemSpec, ...]
    datasets: Tuple[DatasetSpec, ...]
    cache_dir: Optional[str] = None                      # per-tile evidence (cell arrays) outside the repository, so the analysis can be repeated without repeating the TTA
    tiling: TilingSpec = field(default_factory=TilingSpec)
    tta: TTASpec = field(default_factory=TTASpec)
    alignment: AlignmentSpec = field(default_factory=AlignmentSpec)
    eligibility: EligibilitySpec = field(default_factory=EligibilitySpec)
    analysis: AnalysisSpec = field(default_factory=AnalysisSpec)
    bootstrap: BootstrapSpec = field(default_factory=BootstrapSpec)
    metrics: Dict[str, Any] = field(default_factory=dict)
    device: str = "auto"

    def __post_init__(self) -> None:
        if not _NAME.match(str(self.name)):
            _fail("must be non-empty and made of letters, digits, '.', '_' or '-'", "name")
        if not isinstance(self.output_dir, str) or not self.output_dir.strip():
            _fail("must be a non-empty path", "output_dir")
        if self.cache_dir is not None and (not isinstance(self.cache_dir, str) or not self.cache_dir.strip()):
            _fail("must be a non-empty path or omitted", "cache_dir")
        if not self.systems:
            _fail("at least one system is required", "systems")
        if not self.datasets:
            _fail("at least one dataset is required", "datasets")
        if len({s.name for s in self.systems}) != len(self.systems):
            _fail("system names must be unique", "systems")
        if len({d.name for d in self.datasets}) != len(self.datasets):
            _fail("dataset names must be unique", "datasets")
        for i, s in enumerate(self.systems):
            if s.kind not in SYSTEM_KINDS:
                _fail(f"stability needs a learned model; must be one of {SYSTEM_KINDS}, got {s.kind!r}", f"systems[{i}].kind")
            try:
                EvalConfig._check_system(i, s)
            except Exception as exc:
                _fail(str(exc), f"systems[{i}]")
        for i, d in enumerate(self.datasets):
            try:
                EvalConfig._check_dataset(i, d)
            except Exception as exc:
                raise ReliabilityConfigError(str(exc).split(": ", 1)[-1], field=f"datasets[{i}]") from exc
        self._check_tta()
        self._check_alignment()
        self._check_eligibility()
        self._check_analysis()
        b = self.bootstrap
        if not _is_int(b.n_boot) or b.n_boot < 100:
            _fail("must be an integer >= 100", "bootstrap.n_boot")
        if not (_is_num(b.alpha) and 0 < b.alpha <= 0.2):
            _fail("must be in (0, 0.2]", "bootstrap.alpha")
        if not _is_int(b.seed) or b.seed < 0:
            _fail("must be an integer >= 0", "bootstrap.seed")
        if not DEVICES.match(str(self.device)):
            _fail("must be 'auto', 'cpu', 'cuda' or 'cuda:N'", "device")
        try:
            from frame.tiling.plan import TilingConfig

            TilingConfig(tile_size=self.tiling.tile_size, overlap=self.tiling.overlap, padding_mode=self.tiling.padding_mode, blend_mode=self.tiling.blend_mode)
        except Exception as exc:
            _fail(str(exc), "tiling")
        object.__setattr__(self, "metrics", EvalConfig._normalise_metrics(self.metrics))

    # ------------------------------------------------------------------ validation

    def _check_tta(self) -> None:
        t = self.tta.transforms
        if not isinstance(t, (list, tuple)) or len(t) < 2:
            _fail("needs at least 2 transforms: a single member has zero spread by construction", "tta.transforms")
        unknown = [x for x in t if x not in TRANSFORM_NAMES]
        if unknown:
            _fail(f"unknown transform(s) {unknown}; valid: {list(TRANSFORM_NAMES)}", "tta.transforms")
        if len(set(t)) != len(t):
            _fail("duplicate transforms would count one view twice", "tta.transforms")
        if "identity" not in t:
            _fail("must include 'identity' (the reference orientation and the single-pass prediction)", "tta.transforms")
        if not _is_int(self.tta.seed) or self.tta.seed < 0:
            _fail("must be an integer >= 0", "tta.seed")

    def _check_alignment(self) -> None:
        a = self.alignment
        if a.method not in ALIGNMENT_METHODS:
            _fail(f"must be one of {ALIGNMENT_METHODS}, got {a.method!r}", "alignment.method")
        if not (_is_num(a.tolerance_hr_px) and a.tolerance_hr_px > 0):
            _fail("must be a positive number of HR pixels", "alignment.tolerance_hr_px")
        if not (_is_num(a.quadrant_tolerance_hr_px) and a.quadrant_tolerance_hr_px >= a.tolerance_hr_px):
            _fail("must be a number >= alignment.tolerance_hr_px", "alignment.quadrant_tolerance_hr_px")
        if not _is_int(a.max_correction_hr_px) or a.max_correction_hr_px < 0:
            _fail("must be a whole number of HR pixels >= 0", "alignment.max_correction_hr_px")
        if not isinstance(a.apply_translation_correction, bool):
            _fail("must be true or false", "alignment.apply_translation_correction")
        if not (_is_num(a.uncertain_factor) and a.uncertain_factor > 1):
            _fail("must be a number > 1", "alignment.uncertain_factor")

    def _check_eligibility(self) -> None:
        e = self.eligibility
        if not (_is_num(e.min_valid_fraction) and 0 < e.min_valid_fraction <= 1):
            _fail("must be in (0, 1]", "eligibility.min_valid_fraction")
        if not (_is_num(e.max_nonfinite_fraction) and 0 <= e.max_nonfinite_fraction < 1):
            _fail("must be in [0, 1)", "eligibility.max_nonfinite_fraction")
        if not _is_int(e.min_members) or e.min_members < 2:
            _fail("must be an integer >= 2", "eligibility.min_members")
        if len(self.tta.transforms) < e.min_members:
            _fail(f"the TTA has {len(self.tta.transforms)} members, fewer than eligibility.min_members={e.min_members}", "eligibility.min_members")

    def _check_analysis(self) -> None:
        a = self.analysis
        sizes = a.cell_sizes_hr_px
        if not isinstance(sizes, (list, tuple)) or not sizes or not all(_is_int(s) and s >= 1 for s in sizes) or len(set(sizes)) != len(sizes):
            _fail("must be a non-empty list of distinct whole numbers >= 1", "analysis.cell_sizes_hr_px")
        if not (_is_num(a.min_cell_valid_fraction) and 0 < a.min_cell_valid_fraction <= 1):
            _fail("must be in (0, 1]", "analysis.min_cell_valid_fraction")
        for key in ("pooled_cells_per_tile", "pooled_pixels_per_tile"):
            if not _is_int(getattr(a, key)) or getattr(a, key) < 100:
                _fail("must be an integer >= 100", f"analysis.{key}")
        grid = a.coverage_grid
        if not isinstance(grid, (list, tuple)) or not grid or not all(_is_num(g) and 0 < g <= 1 for g in grid) or grid[0] != 1.0 or list(grid) != sorted(grid, reverse=True) or len(set(grid)) != len(grid):
            _fail("must be strictly decreasing coverages in (0, 1] starting at 1.0", "analysis.coverage_grid")
        if not (_is_num(a.high_error_quantile) and 0.5 < a.high_error_quantile < 1):
            _fail("must be in (0.5, 1)", "analysis.high_error_quantile")
        if not _is_int(a.split_seed) or a.split_seed < 0:
            _fail("must be an integer >= 0", "analysis.split_seed")
        if not (_is_num(a.dev_fraction) and 0 < a.dev_fraction < 1):
            _fail("must be in (0, 1)", "analysis.dev_fraction")
        if not _is_int(a.min_units_for_split) or a.min_units_for_split < 4:
            _fail("must be an integer >= 4", "analysis.min_units_for_split")
        if not _is_int(a.calibration_bins) or not 3 <= a.calibration_bins <= 50:
            _fail("must be an integer in [3, 50]", "analysis.calibration_bins")
        sweep = a.displacement_sweep_hr_px
        if not isinstance(sweep, (list, tuple)) or not sweep or sweep[0] != 0 or not all(_is_int(s) and s >= 0 for s in sweep) or list(sweep) != sorted(set(sweep)):
            _fail("must be increasing whole numbers of HR pixels starting at 0", "analysis.displacement_sweep_hr_px")

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
    def from_dict(cls, data: Mapping[str, Any]) -> "ReliabilityConfig":
        return _construct(cls, data, "", nested=_NESTED, lists={"systems": SystemSpec, "datasets": DatasetSpec})

    @classmethod
    def from_json(cls, text: str) -> "ReliabilityConfig":
        try:
            return cls.from_dict(json.loads(text))
        except json.JSONDecodeError as exc:
            raise ReliabilityConfigError(f"not valid JSON ({exc.msg} at line {exc.lineno})") from None

    @classmethod
    def load(cls, path: Union[str, Path]) -> "ReliabilityConfig":
        path = Path(path)
        if not path.is_file():
            raise ReliabilityConfigError(f"config file not found: {path}")
        try:
            return cls.from_dict(json.loads(path.read_text(encoding="utf-8")))
        except json.JSONDecodeError as exc:
            raise ReliabilityConfigError(f"{path}: not valid JSON ({exc.msg} at line {exc.lineno})") from None


_TUPLE_FIELDS = ("transforms", "cell_sizes_hr_px", "coverage_grid", "displacement_sweep_hr_px")
_NESTED = {"tiling": TilingSpec, "tta": TTASpec, "alignment": AlignmentSpec, "eligibility": EligibilitySpec, "analysis": AnalysisSpec, "bootstrap": BootstrapSpec}


def _construct(cls: Any, data: Mapping[str, Any], prefix: str, nested: Mapping[str, Any] = {}, lists: Mapping[str, Any] = {}) -> Any:
    """Build a dataclass from a plain mapping, refusing unknown keys and naming the dotted field of every problem."""
    where = prefix.rstrip(".") or None
    if not isinstance(data, Mapping):
        raise ReliabilityConfigError("must be an object", field=where)
    known = {f.name: f for f in fields(cls)}
    unknown = sorted(set(data) - set(known))
    if unknown:
        raise ReliabilityConfigError(f"unknown key(s) {unknown}; valid keys: {sorted(known)}", field=where)
    kwargs: Dict[str, Any] = {}
    for name, f in known.items():
        path = f"{prefix}{name}"
        if name in data:
            value = data[name]
            if name in nested:
                value = _construct(nested[name], value, f"{path}.")
            elif name in lists:
                if not isinstance(value, (list, tuple)):
                    raise ReliabilityConfigError("must be a list", field=path)
                value = tuple(_construct(lists[name], v, f"{path}[{i}].") for i, v in enumerate(value))
            elif name in _TUPLE_FIELDS and isinstance(value, (list, tuple)):
                value = tuple(value)
            elif isinstance(value, dict):
                value = dict(value)
            kwargs[name] = value
        elif f.default is MISSING and f.default_factory is MISSING:
            raise ReliabilityConfigError("is required", field=path)
    try:
        return cls(**kwargs)
    except EvalConfigError as exc:
        if isinstance(exc, ReliabilityConfigError):
            raise
        raise ReliabilityConfigError(str(exc).split(": ", 1)[-1] if exc.field else str(exc), field=f"{prefix}{exc.field}" if exc.field else where) from exc
