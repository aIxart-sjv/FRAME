"""The downstream configuration: frozen, strict, serialisable (Phase 7). Errors name the dotted field; unknown keys are refused.

Every setting that could shape a result is declared here BEFORE any result exists and is recorded with the run (config.json, digest): the NDVI bands and validity rule, the vegetation
threshold and how it was chosen, the region scales, the minimum valid pixels of a region, the retained fractions of the risk-coverage analysis. Nothing is selected or tuned on results:
the threshold is a conventional value, the sensitivity thresholds are listed here and reported next to it, and the coverage points are fixed. The reference gate (registration, valid pixels)
is the Phase 6 gate with its Phase 6 settings; loosening it is possible only by writing a different value here, where it is recorded.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import MISSING, dataclass, field, fields
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple, Union

from frame.downstream.errors import DownstreamConfigError
from frame.evaluate.config import DEVICES, _NAME, DatasetSpec, EvalConfig, SystemSpec, TilingSpec, _plain
from frame.evaluate.errors import EvalConfigError
from frame.evaluate.metrics import MetricConfig
from frame.reliability.config import AlignmentSpec, BootstrapSpec, EligibilitySpec, TTASpec

SYSTEM_KINDS = ("lite", "mamba", "checkpoint")
SUPPORTED_BANDS = ("B02", "B03", "B04", "B08")
RESERVED_NAMES = ("ref", "reference", "bicubic", "lr_native", "texture")      # names the analysis gives to the reference and the baselines
DECISION_RULE = "vegetation iff NDVI >= threshold"
SELECTION_POLICY = "predeclared_conventional_value_not_selected_or_tuned_on_results"


def _fail(message: str, name: Optional[str] = None):
    raise DownstreamConfigError(message, field=name)


def _is_int(v: Any) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


def _is_num(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(float(v))


@dataclass(frozen=True)
class NdviSpec:
    """NDVI = (NIR - Red) / (NIR + Red) on reflectance fractions (DN / reflectance scale), bands selected BY NAME. A pixel enters only if the REFERENCE's Red + NIR reflectance exceeds
    ``min_reflectance_sum`` (decided from the reference alone, so every system is scored on identical pixels) and every system's denominator exceeds ``denominator_epsilon``."""

    red_band: str = "B04"
    nir_band: str = "B08"
    min_reflectance_sum: float = 0.02
    denominator_epsilon: float = 1e-6


@dataclass(frozen=True)
class DecisionSpec:
    """One fixed vegetation / non-vegetation rule. The requirements name no NDVI threshold, so 0.3 (a conventional boundary between bare/sparse and vegetated surfaces) was declared
    before any result and is never selected or tuned on results; the sensitivity thresholds are declared here too and reported beside it, not used to choose between them."""

    ndvi_threshold: float = 0.3
    sensitivity_thresholds: Tuple[float, ...] = (0.2, 0.4)
    region_majority_fraction: float = 0.5           # a REGION is called vegetated when at least this share of its valid pixels is
    rule: str = DECISION_RULE
    selection_policy: str = SELECTION_POLICY


@dataclass(frozen=True)
class RegionSpec:
    """Regions are FRAME's existing fixed square cells on the aligned HR grid (as in frame.reliability): 4 x 4 HR px = 10 m = the footprint of one Sentinel-2 pixel (the requirements'
    '10 m pixel = 16 cells at 2.5 m'), and 16 x 16 HR px = 40 m. ``primary_scale_hr_px`` is the scale of the headline results."""

    scales_hr_px: Tuple[int, ...] = (4, 16)
    primary_scale_hr_px: int = 4
    min_valid_fraction: float = 0.75


@dataclass(frozen=True)
class DownstreamAnalysisSpec:
    retention_grid: Tuple[float, ...] = (1.0, 0.8, 0.6, 0.4)       # keep 100%, then drop the 20%, 40%, 60% most unstable regions
    pooled_regions_per_tile: int = 8192                              # seeded subsample of a tile's regions for the pooled (cluster-bootstrap) correlations


@dataclass(frozen=True)
class DownstreamConfig:
    name: str
    output_dir: str
    systems: Tuple[SystemSpec, ...]
    datasets: Tuple[DatasetSpec, ...]
    cache_dir: Optional[str] = None                                  # per-tile region tables outside the repository
    tiling: TilingSpec = field(default_factory=TilingSpec)
    tta: TTASpec = field(default_factory=TTASpec)
    alignment: AlignmentSpec = field(default_factory=AlignmentSpec)
    eligibility: EligibilitySpec = field(default_factory=EligibilitySpec)
    ndvi: NdviSpec = field(default_factory=NdviSpec)
    decision: DecisionSpec = field(default_factory=DecisionSpec)
    regions: RegionSpec = field(default_factory=RegionSpec)
    analysis: DownstreamAnalysisSpec = field(default_factory=DownstreamAnalysisSpec)
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
            if s.name in RESERVED_NAMES:
                _fail(f"{s.name!r} is reserved for the reference / baselines ({list(RESERVED_NAMES)})", f"systems[{i}].name")
            if s.kind not in SYSTEM_KINDS:
                _fail(f"a stability signal needs a learned model; must be one of {SYSTEM_KINDS}, got {s.kind!r}", f"systems[{i}].kind")
            try:
                EvalConfig._check_system(i, s)
            except Exception as exc:
                _fail(str(exc), f"systems[{i}]")
        for i, d in enumerate(self.datasets):
            try:
                EvalConfig._check_dataset(i, d)
            except Exception as exc:
                raise DownstreamConfigError(str(exc).split(": ", 1)[-1], field=f"datasets[{i}]") from exc
        self._check_tta_alignment_eligibility()
        self._check_ndvi()
        self._check_decision()
        self._check_regions()
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

    def _check_tta_alignment_eligibility(self) -> None:
        # the Phase 6 gate and ensemble are validated by the Phase 6 config; reuse it rather than restate its rules
        from frame.reliability.config import ReliabilityConfig

        try:
            ReliabilityConfig(name="probe", output_dir="probe", systems=self.systems, datasets=self.datasets, tiling=self.tiling, tta=self.tta, alignment=self.alignment,
                              eligibility=self.eligibility, bootstrap=self.bootstrap)
        except EvalConfigError as exc:
            field_name = getattr(exc, "field", None) or ""
            if field_name.split(".")[0] in ("tta", "alignment", "eligibility"):
                _fail(str(exc).split(": ", 1)[-1], field_name)

    def _check_ndvi(self) -> None:
        n = self.ndvi
        if n.red_band not in SUPPORTED_BANDS or n.nir_band not in SUPPORTED_BANDS:
            _fail(f"bands must be among FRAME's RGBN bands {SUPPORTED_BANDS}; got red {n.red_band!r}, nir {n.nir_band!r}", "ndvi.red_band" if n.red_band not in SUPPORTED_BANDS else "ndvi.nir_band")
        if n.red_band == n.nir_band:
            _fail("red and nir bands must differ", "ndvi.nir_band")
        if (n.red_band, n.nir_band) != ("B04", "B08"):
            _fail("NDVI is (B08 - B04) / (B08 + B04); other band pairs are a different index", "ndvi.red_band")
        if not (_is_num(n.min_reflectance_sum) and n.min_reflectance_sum > 0):
            _fail("must be a positive reflectance sum", "ndvi.min_reflectance_sum")
        if not (_is_num(n.denominator_epsilon) and n.denominator_epsilon > 0):
            _fail("must be a positive number", "ndvi.denominator_epsilon")

    def _check_decision(self) -> None:
        d = self.decision
        if not (_is_num(d.ndvi_threshold) and -1 < d.ndvi_threshold < 1):
            _fail("must be a number in (-1, 1)", "decision.ndvi_threshold")
        s = d.sensitivity_thresholds
        if (not isinstance(s, (list, tuple)) or len(s) < 1 or len(s) > 4 or not all(_is_num(v) and -1 < v < 1 for v in s) or len(set(s)) != len(s)
                or d.ndvi_threshold in s):
            _fail("must be 1 to 4 distinct numbers in (-1, 1), different from the primary threshold", "decision.sensitivity_thresholds")
        if not (_is_num(d.region_majority_fraction) and 0 < d.region_majority_fraction <= 1):
            _fail("must be in (0, 1]", "decision.region_majority_fraction")
        if d.rule != DECISION_RULE:
            _fail(f"the decision rule is fixed: {DECISION_RULE!r}", "decision.rule")
        if d.selection_policy != SELECTION_POLICY:
            _fail(f"the threshold policy is fixed: {SELECTION_POLICY!r} (a threshold is never selected or tuned on results)", "decision.selection_policy")

    def _check_regions(self) -> None:
        r = self.regions
        s = r.scales_hr_px
        if not isinstance(s, (list, tuple)) or not s or not all(_is_int(v) and v >= 1 for v in s) or len(set(s)) != len(s):
            _fail("must be a non-empty list of distinct whole numbers >= 1", "regions.scales_hr_px")
        if r.primary_scale_hr_px not in s:
            _fail(f"must be one of the declared scales {list(s)}", "regions.primary_scale_hr_px")
        if not (_is_num(r.min_valid_fraction) and 0 < r.min_valid_fraction <= 1):
            _fail("must be in (0, 1]", "regions.min_valid_fraction")

    def _check_analysis(self) -> None:
        a = self.analysis
        g = a.retention_grid
        if not isinstance(g, (list, tuple)) or len(g) < 2 or not all(_is_num(v) and 0 < v <= 1 for v in g) or g[0] != 1.0 or list(g) != sorted(g, reverse=True) or len(set(g)) != len(g):
            _fail("must be at least two strictly decreasing fractions in (0, 1] starting at 1.0", "analysis.retention_grid")
        if not _is_int(a.pooled_regions_per_tile) or a.pooled_regions_per_tile < 100:
            _fail("must be an integer >= 100", "analysis.pooled_regions_per_tile")

    # ------------------------------------------------------------------ views

    def thresholds(self) -> Tuple[float, ...]:
        """The primary threshold first, then the declared sensitivity thresholds."""
        return (self.decision.ndvi_threshold, *self.decision.sensitivity_thresholds)

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
    def from_dict(cls, data: Mapping[str, Any]) -> "DownstreamConfig":
        return _construct(cls, data, "", nested=_NESTED, lists={"systems": SystemSpec, "datasets": DatasetSpec})

    @classmethod
    def from_json(cls, text: str) -> "DownstreamConfig":
        try:
            return cls.from_dict(json.loads(text))
        except json.JSONDecodeError as exc:
            raise DownstreamConfigError(f"not valid JSON ({exc.msg} at line {exc.lineno})") from None

    @classmethod
    def load(cls, path: Union[str, Path]) -> "DownstreamConfig":
        path = Path(path)
        if not path.is_file():
            raise DownstreamConfigError(f"config file not found: {path}")
        try:
            return cls.from_dict(json.loads(path.read_text(encoding="utf-8")))
        except json.JSONDecodeError as exc:
            raise DownstreamConfigError(f"{path}: not valid JSON ({exc.msg} at line {exc.lineno})") from None


_TUPLE_FIELDS = ("transforms", "scales_hr_px", "retention_grid", "sensitivity_thresholds")
_NESTED = {"tiling": TilingSpec, "tta": TTASpec, "alignment": AlignmentSpec, "eligibility": EligibilitySpec, "ndvi": NdviSpec, "decision": DecisionSpec, "regions": RegionSpec,
           "analysis": DownstreamAnalysisSpec, "bootstrap": BootstrapSpec}


def _construct(cls: Any, data: Mapping[str, Any], prefix: str, nested: Mapping[str, Any] = {}, lists: Mapping[str, Any] = {}) -> Any:
    """Build a dataclass from a plain mapping, refusing unknown keys and naming the dotted field of every problem."""
    where = prefix.rstrip(".") or None
    if not isinstance(data, Mapping):
        raise DownstreamConfigError("must be an object", field=where)
    known = {f.name: f for f in fields(cls)}
    unknown = sorted(set(data) - set(known))
    if unknown:
        raise DownstreamConfigError(f"unknown key(s) {unknown}; valid keys: {sorted(known)}", field=where)
    kwargs: Dict[str, Any] = {}
    for name, f in known.items():
        path = f"{prefix}{name}"
        if name in data:
            value = data[name]
            if name in nested:
                value = _construct(nested[name], value, f"{path}.")
            elif name in lists:
                if not isinstance(value, (list, tuple)):
                    raise DownstreamConfigError("must be a list", field=path)
                value = tuple(_construct(lists[name], v, f"{path}[{i}].") for i, v in enumerate(value))
            elif name in _TUPLE_FIELDS and isinstance(value, (list, tuple)):
                value = tuple(value)
            elif isinstance(value, dict):
                value = dict(value)
            kwargs[name] = value
        elif f.default is MISSING and f.default_factory is MISSING:
            raise DownstreamConfigError("is required", field=path)
    try:
        return cls(**kwargs)
    except EvalConfigError as exc:
        if isinstance(exc, DownstreamConfigError):
            raise
        raise DownstreamConfigError(str(exc).split(": ", 1)[-1] if exc.field else str(exc), field=f"{prefix}{exc.field}" if exc.field else where) from exc
