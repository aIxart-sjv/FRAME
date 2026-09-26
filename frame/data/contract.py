"""The paired LR-HR data contract (Phase 3).

Three layers, deliberately separate:

* `PairRecord`   -- everything that can be known WITHOUT reading pixels: identity,
                    provenance, geography, split, pair type, the LR and HR raster
                    descriptions, degradation metadata. JSON-serialisable; this is
                    one line of a manifest.
* `PairedSample` -- a record plus the loaded tensors (LR, HR, validity masks) and,
                    when it is a crop, the `PatchCoords` it came from.
* `PatchCoords`  -- the exact LR window and the HR window that corresponds to it.

Three kinds of pair are distinguished and never mixed (docs/Requirements 142.txt,
Requirement 3 sections 1-2):

* SYNTHETIC                 LR is generated from HR by a recorded degradation.
* REAL_CROSS_SENSOR         LR and HR are both real observations (different sensors)
                            that the dataset authors co-registered as a pair.
* INDEPENDENT_HR_REFERENCE  a real LR scene whose HR reference is an independent
                            product, possibly absent or of unknown availability
                            (the Indian holdout). The HR status is stated, never assumed.

`scale_factor` is per record: SEN2VENuS is x2 (10 m -> 5 m) and x4 (20 m -> 5 m) in
one dataset, and OpenSR-Test has x2 and x4 subsets. Nothing here bakes in x4.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field, fields
from enum import Enum
from typing import Any, Dict, Mapping, Optional, Tuple

import torch

from frame.data.bands import canonical_bands
from frame.data.errors import ContractError

SCHEMA_VERSION = 1

_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:\-]*")


class PairType(str, Enum):
    SYNTHETIC = "synthetic"
    REAL_CROSS_SENSOR = "real_cross_sensor"
    INDEPENDENT_HR_REFERENCE = "independent_hr_reference"


class HRStatus(str, Enum):
    AVAILABLE = "available"
    UNAVAILABLE = "unavailable"
    UNKNOWN = "unknown"


class Split(str, Enum):
    TRAIN = "train"
    VAL = "val"
    TEST = "test"


class DatasetRole(str, Enum):
    TRAINING_PRIMARY = "training_primary"
    TRAINING_SUPPLEMENTARY = "training_supplementary"
    VALIDATION = "validation"
    INDEPENDENT_BENCHMARK = "independent_benchmark"
    DOMAIN_HOLDOUT = "domain_holdout"
    SMOKE_TEST = "smoke_test"                      # generated synthetic fixtures: exercise the pipeline, say nothing about real imagery


def _enum(cls, value, field_name: str):
    try:
        return cls(value)
    except ValueError:
        raise ContractError(
            f"{field_name}={value!r} is not one of {[m.value for m in cls]}.", code=f"invalid_{field_name}"
        ) from None


def _relative_path(path: Optional[str], what: str) -> Optional[str]:
    if path is None:
        return None
    if not isinstance(path, str) or not path:
        raise ContractError(f"{what} path must be a non-empty string.", code="invalid_path")
    if path.startswith("/") or re.match(r"^[A-Za-z]:[\\/]", path) or path.startswith("~"):
        raise ContractError(
            f"{what} path {path!r} is absolute; manifests hold paths relative to the dataset directory "
            "(see frame.data.config).",
            code="absolute_path",
        )
    if ".." in path.replace("\\", "/").split("/"):
        raise ContractError(f"{what} path {path!r} escapes the dataset directory.", code="absolute_path")
    return path


# ---------------------------------------------------------------------------------------------------------------
# Raster description
# ---------------------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class RasterSpec:
    """What one raster (an LR or an HR image) is, without its pixels."""

    band_names: Tuple[str, ...]
    width: int
    height: int
    pixel_size_m: Optional[float] = None
    path: Optional[str] = None                     # relative to the dataset directory
    dtype: str = "float32"
    reflectance_scale: Optional[float] = None      # stored value / scale = reflectance; None = already reflectance
    nodata: Optional[float] = None
    crs: Optional[str] = None
    transform: Optional[Tuple[float, float, float, float, float, float]] = None   # GDAL-style affine (a,b,c,d,e,f)

    def __post_init__(self) -> None:
        object.__setattr__(self, "band_names", canonical_bands(self.band_names))
        if not self.band_names:
            raise ContractError("A raster needs at least one band.", code="invalid_bands")
        for name in ("width", "height"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ContractError(f"{name} must be a positive integer, got {value!r}.", code="invalid_shape")
        if self.pixel_size_m is not None and not self.pixel_size_m > 0:
            raise ContractError(f"pixel_size_m must be positive, got {self.pixel_size_m!r}.", code="invalid_resolution")
        if self.reflectance_scale is not None and not self.reflectance_scale > 0:
            raise ContractError(f"reflectance_scale must be positive, got {self.reflectance_scale!r}.", code="invalid_scale")
        object.__setattr__(self, "path", _relative_path(self.path, "raster"))
        if self.transform is not None:
            if len(self.transform) != 6:
                raise ContractError(f"transform must have 6 values, got {len(self.transform)}.", code="invalid_transform")
            object.__setattr__(self, "transform", tuple(float(v) for v in self.transform))

    @property
    def georeferenced(self) -> bool:
        return self.crs is not None and self.transform is not None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "band_names": list(self.band_names), "width": self.width, "height": self.height,
            "pixel_size_m": self.pixel_size_m, "path": self.path, "dtype": self.dtype,
            "reflectance_scale": self.reflectance_scale, "nodata": self.nodata, "crs": self.crs,
            "transform": list(self.transform) if self.transform is not None else None,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "RasterSpec":
        known = {f.name for f in fields(cls)}
        unknown = set(data) - known
        if unknown:
            raise ContractError(f"Unknown raster field(s) {sorted(unknown)}.", code="unknown_field")
        try:
            return cls(**{k: (tuple(v) if k in ("band_names", "transform") and v is not None else v) for k, v in data.items()})
        except TypeError as exc:
            raise ContractError(f"Malformed raster description: {exc}.", code="invalid_record") from None


@dataclass(frozen=True)
class DegradationRecord:
    """How a synthetic LR was produced from its HR (see frame.data.degradation)."""

    name: str
    version: str
    config: Dict[str, Any]
    seed: Optional[int]
    harmonisation: str
    parameters_verified: bool     # True only if the numbers equal those of the published process this claims to follow
    origin: str = "frame"         # "frame": generated by frame.data.degradation; "upstream": shipped by the dataset itself

    def to_dict(self) -> Dict[str, Any]:
        return {"name": self.name, "version": self.version, "config": dict(self.config), "seed": self.seed,
                "harmonisation": self.harmonisation, "parameters_verified": self.parameters_verified,
                "origin": self.origin}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "DegradationRecord":
        try:
            return cls(**data)
        except TypeError as exc:
            raise ContractError(f"Malformed degradation record: {exc}.", code="invalid_degradation") from None


# ---------------------------------------------------------------------------------------------------------------
# The record (one manifest line)
# ---------------------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class PairRecord:
    """One LR-HR pair (or an LR scene with an independent HR reference), metadata only."""

    sample_id: str
    dataset: str
    scene_id: str
    split: Split
    pair_type: PairType
    hr_status: HRStatus
    scale_factor: int
    lr: RasterSpec
    hr: Optional[RasterSpec]
    variant: str = "default"
    dataset_revision: Optional[str] = None
    region_id: Optional[str] = None
    source_split: Optional[str] = None             # the split label the dataset itself ships, kept for provenance
    lon: Optional[float] = None                    # WGS84 centroid
    lat: Optional[float] = None
    degradation: Optional[DegradationRecord] = None
    license: Optional[str] = None
    provenance: Dict[str, Any] = field(default_factory=dict)   # upstream ids, dates, temporal separation, locator, ...

    def __post_init__(self) -> None:
        for name in ("sample_id", "dataset", "scene_id", "variant"):
            value = getattr(self, name)
            if not isinstance(value, str) or not _ID.fullmatch(value):
                raise ContractError(f"{name}={value!r} must match {_ID.pattern}.", code=f"invalid_{name}")
        if self.region_id is not None and (not isinstance(self.region_id, str) or not _ID.fullmatch(self.region_id)):
            raise ContractError(f"region_id={self.region_id!r} must match {_ID.pattern}.", code="invalid_region_id")
        object.__setattr__(self, "split", _enum(Split, self.split, "split"))
        object.__setattr__(self, "pair_type", _enum(PairType, self.pair_type, "pair_type"))
        object.__setattr__(self, "hr_status", _enum(HRStatus, self.hr_status, "hr_status"))
        if isinstance(self.scale_factor, bool) or not isinstance(self.scale_factor, int) or self.scale_factor < 1:
            raise ContractError(f"scale_factor must be a positive integer, got {self.scale_factor!r}.", code="invalid_scale")
        if self.lat is not None and not -90.0 <= self.lat <= 90.0:
            raise ContractError(f"lat={self.lat!r} is outside [-90, 90].", code="invalid_geography")
        if self.lon is not None and not -180.0 <= self.lon <= 180.0:
            raise ContractError(f"lon={self.lon!r} is outside [-180, 180].", code="invalid_geography")

        # A pair cannot claim an HR counterpart it does not have, and only an independent
        # reference may legitimately have none.
        if self.hr_status == HRStatus.AVAILABLE and self.hr is None:
            raise ContractError(f"{self.sample_id}: hr_status is 'available' but no HR raster is described.", code="missing_hr")
        if self.hr_status != HRStatus.AVAILABLE and self.hr is not None:
            raise ContractError(f"{self.sample_id}: an HR raster is described but hr_status is {self.hr_status.value!r}.", code="invalid_hr_status")
        if self.pair_type != PairType.INDEPENDENT_HR_REFERENCE and self.hr_status != HRStatus.AVAILABLE:
            raise ContractError(
                f"{self.sample_id}: a {self.pair_type.value} pair must have its HR counterpart (hr_status 'available'); "
                "only an independent HR reference may have it 'unavailable' or 'unknown'.",
                code="missing_hr",
            )
        if self.pair_type == PairType.SYNTHETIC and self.degradation is None:
            raise ContractError(f"{self.sample_id}: a synthetic pair must record its degradation.", code="missing_degradation")
        if self.pair_type != PairType.SYNTHETIC and self.degradation is not None:
            raise ContractError(f"{self.sample_id}: a {self.pair_type.value} pair must not carry degradation metadata.", code="unexpected_degradation")

    @property
    def is_georeferenced(self) -> bool:
        return self.lr.georeferenced and (self.hr is None or self.hr.georeferenced)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "sample_id": self.sample_id, "dataset": self.dataset, "variant": self.variant,
            "dataset_revision": self.dataset_revision, "scene_id": self.scene_id, "region_id": self.region_id,
            "split": self.split.value, "source_split": self.source_split,
            "pair_type": self.pair_type.value, "hr_status": self.hr_status.value, "scale_factor": self.scale_factor,
            "lr": self.lr.to_dict(), "hr": self.hr.to_dict() if self.hr is not None else None,
            "lon": self.lon, "lat": self.lat,
            "degradation": self.degradation.to_dict() if self.degradation is not None else None,
            "license": self.license, "provenance": dict(self.provenance),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "PairRecord":
        data = dict(data)
        version = data.pop("schema_version", None)
        if version != SCHEMA_VERSION:
            raise ContractError(f"Unsupported record schema_version {version!r} (this code reads {SCHEMA_VERSION}).", code="unsupported_schema")
        known = {f.name for f in fields(cls)}
        unknown = set(data) - known
        if unknown:
            raise ContractError(f"Unknown record field(s) {sorted(unknown)}.", code="unknown_field")
        required = [n for n in ("sample_id", "dataset", "scene_id", "split", "pair_type", "hr_status", "scale_factor", "lr") if n not in data]
        if required:
            raise ContractError(f"Missing required field(s) {required}.", code="missing_field")
        data["lr"] = RasterSpec.from_dict(data["lr"])
        data["hr"] = RasterSpec.from_dict(data["hr"]) if data.get("hr") is not None else None
        if data.get("degradation") is not None:
            data["degradation"] = DegradationRecord.from_dict(data["degradation"])
        return cls(**data)


# ---------------------------------------------------------------------------------------------------------------
# Patch coordinates and the loaded sample
# ---------------------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class PatchCoords:
    """An LR window and the HR window covering exactly the same ground.

    The HR window is the LR window multiplied by ``scale``: there is one definition of
    the relationship, here, and nothing crops LR and HR independently.
    """

    lr_row: int
    lr_col: int
    lr_height: int
    lr_width: int
    scale: int

    def __post_init__(self) -> None:
        if min(self.lr_row, self.lr_col) < 0 or min(self.lr_height, self.lr_width, self.scale) < 1:
            raise ContractError(f"Invalid patch coordinates {self}.", code="invalid_patch")

    @property
    def hr_row(self) -> int:
        return self.lr_row * self.scale

    @property
    def hr_col(self) -> int:
        return self.lr_col * self.scale

    @property
    def hr_height(self) -> int:
        return self.lr_height * self.scale

    @property
    def hr_width(self) -> int:
        return self.lr_width * self.scale

    def to_dict(self) -> Dict[str, int]:
        return {"lr_row": self.lr_row, "lr_col": self.lr_col, "lr_height": self.lr_height, "lr_width": self.lr_width,
                "hr_row": self.hr_row, "hr_col": self.hr_col, "hr_height": self.hr_height, "hr_width": self.hr_width,
                "scale": self.scale}


@dataclass(frozen=True)
class PairedSample:
    """A record plus its loaded pixels (float32 reflectance, channels first).

    ``lr_bands`` / ``hr_bands`` name the channels actually present, in order; they are
    a selection of the record's bands when the caller asked for one. Masks are True where
    the pixel is a real observation and False at nodata; nodata pixels are 0.0 in the tensor.
    """

    record: PairRecord
    lr: torch.Tensor
    hr: Optional[torch.Tensor]
    lr_bands: Tuple[str, ...]
    hr_bands: Optional[Tuple[str, ...]]
    lr_mask: Optional[torch.Tensor] = None
    hr_mask: Optional[torch.Tensor] = None
    patch: Optional[PatchCoords] = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "lr_bands", canonical_bands(self.lr_bands))
        if self.hr_bands is not None:
            object.__setattr__(self, "hr_bands", canonical_bands(self.hr_bands))
        self._check_tensor("lr", self.lr, self.lr_bands)
        if (self.hr is None) != (self.hr_bands is None):
            raise ContractError("hr and hr_bands must be given together.", code="invalid_sample")
        if self.hr is None:
            if self.record.hr_status == HRStatus.AVAILABLE:
                raise ContractError(f"{self.record.sample_id}: the record has an HR counterpart but none was loaded.", code="missing_hr")
            return
        self._check_tensor("hr", self.hr, self.hr_bands)
        s = self.record.scale_factor
        if tuple(self.hr.shape[-2:]) != (self.lr.shape[-2] * s, self.lr.shape[-1] * s):
            raise ContractError(
                f"{self.record.sample_id}: HR is {tuple(self.hr.shape[-2:])} but LR {tuple(self.lr.shape[-2:])} x scale {s} "
                f"= {(self.lr.shape[-2] * s, self.lr.shape[-1] * s)}.",
                code="shape_mismatch",
            )
        for name, mask, ref in (("lr_mask", self.lr_mask, self.lr), ("hr_mask", self.hr_mask, self.hr)):
            if mask is not None and (mask.dtype != torch.bool or tuple(mask.shape) != tuple(ref.shape[-2:])):
                raise ContractError(f"{name} must be a bool tensor of shape {tuple(ref.shape[-2:])}.", code="invalid_sample")

    @staticmethod
    def _check_tensor(name: str, tensor: object, bands: Tuple[str, ...]) -> None:
        if not isinstance(tensor, torch.Tensor) or tensor.ndim != 3:
            raise ContractError(f"{name} must be a 3-D (channels, height, width) tensor.", code="invalid_sample")
        if tensor.dtype != torch.float32:
            raise ContractError(f"{name} must be float32 reflectance, got {tensor.dtype}.", code="invalid_sample")
        if tensor.shape[0] != len(bands):
            raise ContractError(f"{name} has {tensor.shape[0]} channel(s) but {len(bands)} band name(s) {list(bands)}.", code="band_mismatch")

    @property
    def sample_id(self) -> str:
        return self.record.sample_id

    @property
    def patch_id(self) -> str:
        if self.patch is None:
            return self.record.sample_id
        return f"{self.record.sample_id}@r{self.patch.lr_row}c{self.patch.lr_col}"

    @property
    def scale_factor(self) -> int:
        return self.record.scale_factor

    def metadata_dict(self) -> Dict[str, Any]:
        """JSON-safe provenance for one sample or patch (what a DataLoader hands the training code)."""
        r = self.record
        return {
            "sample_id": r.sample_id, "patch_id": self.patch_id, "dataset": r.dataset, "variant": r.variant,
            "dataset_revision": r.dataset_revision, "scene_id": r.scene_id, "region_id": r.region_id,
            "split": r.split.value, "pair_type": r.pair_type.value, "hr_status": r.hr_status.value,
            "scale_factor": r.scale_factor, "lr_bands": list(self.lr_bands),
            "hr_bands": list(self.hr_bands) if self.hr_bands is not None else None,
            "lr_pixel_size_m": r.lr.pixel_size_m, "hr_pixel_size_m": r.hr.pixel_size_m if r.hr is not None else None,
            "crs": r.lr.crs, "patch": self.patch.to_dict() if self.patch is not None else None,
            "degradation": r.degradation.to_dict() if r.degradation is not None else None,
            "license": r.license,
        }
