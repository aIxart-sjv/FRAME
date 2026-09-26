"""Dataset / manifest quality control (Phase 3).

Reports problems; never repairs them. A corrupt sample is listed with a machine-readable
code so it can be excluded or investigated, not silently "fixed".

Layers, cheapest first (each is optional beyond the first):

1. record      -- against the dataset's documented profile (frame.data.roles): known dataset and
                  variant, pair type, permitted split (role), scale, bands and their ORDER,
                  pixel sizes, tile sizes; and the LR/HR geometry (CRS, resolution ratio,
                  footprint, dimensions -- frame.data.geo).
2. files       -- with a data root: every referenced file exists; with ``check_headers`` its
                  band count, size, dtype, CRS and transform match what the record declares.
3. pixels      -- with a ``loader``: NaN/Inf, reflectance range, all-nodata, constant images.
4. manifest    -- duplicate ids, unparsable lines, record-count mismatch, split leakage (scene and
                  region), role violations, spatial proximity across splits.

Reflectance bounds are the model contract's (`frame.models.config.MIN/MAX_REFLECTANCE`, derived from
the L2A encoding), so QC and inference agree on what is a plausible value.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import torch

from frame.data.contract import PairedSample, PairRecord, PairType, RasterSpec
from frame.data.errors import (
    CRSMismatchError,
    DataError,
    DimensionMismatchError,
    FootprintMismatchError,
    ResolutionRatioError,
)
from frame.data.geo import same_crs, validate_pair_geometry
from frame.data.issues import ERROR, QCIssue, error, warning
from frame.data.manifest import ManifestContents, manifest_digest, read_manifest
from frame.data.roles import DATASET_PROFILES
from frame.data.splits import check_split_integrity
from frame.models.config import MAX_REFLECTANCE, MIN_REFLECTANCE

DEFAULT_MAX_NODATA_FRACTION = 0.5
# Compared as float32 (the tensors' own precision): the largest legitimate DN, 65535, is 6.5535002 in float32,
# which a double-precision bound would wrongly reject -- the same trap frame.models.contract documents.
_REFLECTANCE_LO = float(torch.tensor(MIN_REFLECTANCE, dtype=torch.float32))
_REFLECTANCE_HI = float(torch.tensor(MAX_REFLECTANCE, dtype=torch.float32))
PathLike = Union[str, Path]
Loader = Callable[[PairRecord], PairedSample]

_GEO_CODES = {
    CRSMismatchError: "crs_mismatch",
    ResolutionRatioError: "resolution_mismatch",
    FootprintMismatchError: "footprint_mismatch",
    DimensionMismatchError: "dimension_mismatch",
}


# ---------------------------------------------------------------------------------------------------------------
# 1. record vs profile and geometry
# ---------------------------------------------------------------------------------------------------------------

def _profile_issues(record: PairRecord) -> List[QCIssue]:
    sid = record.sample_id
    profile = DATASET_PROFILES.get(record.dataset)
    if profile is None:
        return [error("unknown_dataset", f"{sid}: dataset {record.dataset!r} has no profile; known: {sorted(DATASET_PROFILES)}.", sid)]
    if record.variant not in profile.variants:
        return [error("unknown_variant", f"{sid}: {record.dataset} has no variant {record.variant!r}; known: {sorted(profile.variants)}.", sid)]

    spec = profile.variants[record.variant]
    issues: List[QCIssue] = []
    if record.pair_type != spec.pair_type:
        issues.append(error("pair_type_mismatch", f"{sid}: {record.dataset}/{record.variant} pairs are {spec.pair_type.value}, "
                            f"the record says {record.pair_type.value}.", sid))
    if record.split not in spec.allowed_splits:
        issues.append(error("role_violation", f"{sid}: {record.dataset}/{record.variant} ({spec.role.value}) may only be used in "
                            f"{sorted(s.value for s in spec.allowed_splits)}, not {record.split.value!r}.", sid))
    if record.scale_factor != spec.scale_factor:
        issues.append(error("scale_mismatch", f"{sid}: scale factor is x{record.scale_factor}, the profile says x{spec.scale_factor}.", sid))

    # bands must match the documented stored bands AND their order: reordering is done explicitly at load time, never in the manifest
    if record.lr.band_names != spec.lr_bands:
        issues.append(error("band_mismatch", f"{sid}: LR bands {list(record.lr.band_names)} != documented {list(spec.lr_bands)}.", sid))
    if record.hr is not None and record.hr.band_names != spec.hr_bands:
        issues.append(error("band_mismatch", f"{sid}: HR bands {list(record.hr.band_names)} != documented {list(spec.hr_bands)}.", sid))

    for name, raster, size, pixel in (("LR", record.lr, spec.lr_size, spec.lr_pixel_size_m), ("HR", record.hr, spec.hr_size, spec.hr_pixel_size_m)):
        if raster is None:
            continue
        if size is not None and (raster.height, raster.width) != size:
            issues.append(error("shape_mismatch", f"{sid}: {name} is {raster.height}x{raster.width}, the profile says {size[0]}x{size[1]}.", sid))
        if pixel is not None and raster.pixel_size_m is not None and abs(raster.pixel_size_m - pixel) > 1e-6:
            issues.append(error("resolution_mismatch", f"{sid}: {name} pixel size {raster.pixel_size_m} m, the profile says {pixel} m.", sid))
    return issues


def _geometry_issues(record: PairRecord) -> List[QCIssue]:
    if record.hr is None:
        return []
    sid = record.sample_id
    try:
        checked = validate_pair_geometry(record.lr, record.hr, record.scale_factor)
    except DataError as exc:
        return [error(_GEO_CODES.get(type(exc), "geometry_mismatch"), f"{sid}: {exc}", sid)]
    if not checked and record.pair_type != PairType.SYNTHETIC:
        return [warning("not_georeferenced", f"{sid}: no CRS/transform on both rasters, so only the raster sizes were checked.", sid)]
    return []


def validate_record(record: PairRecord) -> List[QCIssue]:
    """Profile and geometry findings for one record (no files, no pixels)."""
    return _profile_issues(record) + _geometry_issues(record)


# ---------------------------------------------------------------------------------------------------------------
# 2. files
# ---------------------------------------------------------------------------------------------------------------

def _file_issues(record: PairRecord, root: Path, check_headers: bool) -> List[QCIssue]:
    issues: List[QCIssue] = []
    sid = record.sample_id
    for name, spec in (("LR", record.lr), ("HR", record.hr)):
        if spec is None or spec.path is None:
            continue
        path = root / record.dataset / spec.path
        if not path.is_file():
            issues.append(error("missing_file", f"{sid}: {name} file not found: {record.dataset}/{spec.path}", sid))
            continue
        if check_headers:
            issues.extend(_header_issues(sid, name, spec, path))
    return issues


def _header_issues(sid: str, name: str, spec: RasterSpec, path: Path) -> List[QCIssue]:
    import rasterio

    try:
        with rasterio.open(path) as src:
            count, width, height, dtype = src.count, src.width, src.height, src.dtypes[0]
            crs = src.crs.to_string() if src.crs else None
            transform = tuple(src.transform)[:6]
    except Exception as exc:  # unreadable / corrupt file
        return [error("unreadable_file", f"{sid}: {name} file {path.name} could not be opened ({type(exc).__name__}: {exc}).", sid)]

    issues: List[QCIssue] = []
    if count != len(spec.band_names):
        issues.append(error("band_count_mismatch", f"{sid}: {name} file has {count} band(s), the record lists {len(spec.band_names)}.", sid))
    if (height, width) != (spec.height, spec.width):
        issues.append(error("shape_mismatch", f"{sid}: {name} file is {height}x{width}, the record says {spec.height}x{spec.width}.", sid))
    if spec.dtype != "float32" and dtype != spec.dtype:
        issues.append(error("dtype_mismatch", f"{sid}: {name} file dtype is {dtype}, the record says {spec.dtype}.", sid))
    if spec.crs is not None and crs is not None and not same_crs(spec.crs, crs):
        issues.append(error("crs_mismatch", f"{sid}: {name} file CRS {crs} differs from the record's {spec.crs}.", sid))
    if spec.transform is not None and any(abs(a - b) > 1e-6 for a, b in zip(spec.transform, transform)):
        issues.append(error("transform_mismatch", f"{sid}: {name} file transform {transform} differs from the record's {spec.transform}.", sid))
    return issues


# ---------------------------------------------------------------------------------------------------------------
# 3. pixels
# ---------------------------------------------------------------------------------------------------------------

def validate_sample(sample: PairedSample, *, max_nodata_fraction: float = DEFAULT_MAX_NODATA_FRACTION) -> List[QCIssue]:
    """Radiometric findings for a loaded sample (NaN/Inf, range, nodata, constant images)."""
    sid = sample.sample_id
    issues: List[QCIssue] = []
    for name, tensor, mask in (("LR", sample.lr, sample.lr_mask), ("HR", sample.hr, sample.hr_mask)):
        if tensor is None:
            continue
        if not bool(torch.isfinite(tensor).all()):
            issues.append(error("non_finite", f"{sid}: {name} contains NaN or Inf.", sid))
            continue
        valid = tensor if mask is None else tensor[:, mask]
        if valid.numel() == 0:
            issues.append(error("no_valid_pixels", f"{sid}: {name} has no valid (non-nodata) pixels.", sid))
            continue
        lo, hi = float(valid.min()), float(valid.max())
        if lo < _REFLECTANCE_LO or hi > _REFLECTANCE_HI:
            issues.append(error("invalid_range", f"{sid}: {name} values span [{lo:.4g}, {hi:.4g}], outside the plausible reflectance range "
                                f"[{MIN_REFLECTANCE}, {MAX_REFLECTANCE:.4f}] (a wrong reflectance scale?).", sid))
        if mask is not None:
            missing = 1.0 - float(mask.float().mean())
            if missing > max_nodata_fraction:
                issues.append(warning("high_nodata", f"{sid}: {name} is {missing:.0%} nodata (threshold {max_nodata_fraction:.0%}).", sid))
        flat_bands = [b for b, (band_lo, band_hi) in enumerate(zip(valid.amin(dim=1).tolist(), valid.amax(dim=1).tolist())) if band_lo == band_hi]
        if flat_bands:
            issues.append(warning("constant_image", f"{sid}: {name} band(s) {flat_bands} are constant over the valid pixels.", sid))
    return issues


# ---------------------------------------------------------------------------------------------------------------
# 4. manifest
# ---------------------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class QCReport:
    total: int
    valid: int
    invalid: int
    issues: Tuple[QCIssue, ...]
    per_split: Dict[str, int]
    per_dataset: Dict[str, int]
    invalid_ids: Tuple[str, ...]
    manifest_digest: Optional[str]
    checks: Dict[str, bool]

    @property
    def ok(self) -> bool:
        """True when there is no error-level finding of any kind (warnings do not fail a manifest)."""
        return not any(i.severity == ERROR for i in self.issues)

    def counts_by_code(self) -> Dict[str, int]:
        return dict(sorted(Counter(i.code for i in self.issues).items()))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ok": self.ok, "total_samples": self.total, "valid_samples": self.valid, "invalid_samples": self.invalid,
            "issue_counts": self.counts_by_code(), "per_split": self.per_split, "per_dataset": self.per_dataset,
            "manifest_digest": self.manifest_digest, "checks_performed": self.checks,
            "invalid_sample_ids": list(self.invalid_ids), "issues": [i.to_dict() for i in self.issues],
        }

    def summary(self) -> str:
        lines = [f"samples: {self.total} total, {self.valid} valid, {self.invalid} invalid  ->  {'OK' if self.ok else 'FAILED'}",
                 f"per split: {self.per_split}", f"per dataset: {self.per_dataset}",
                 "checks: " + ", ".join(f"{k}={'yes' if v else 'no'}" for k, v in self.checks.items())]
        for code, count in self.counts_by_code().items():
            lines.append(f"  {code}: {count}")
        return "\n".join(lines)


def validate_manifest(
    records: Sequence[PairRecord],
    *,
    parse_issues: Iterable[QCIssue] = (),
    data_root: Optional[PathLike] = None,
    check_headers: bool = False,
    loader: Optional[Loader] = None,
    check_split: bool = True,
    require_region_disjoint: bool = True,
    max_nodata_fraction: float = DEFAULT_MAX_NODATA_FRACTION,
) -> QCReport:
    """Run every applicable check over ``records`` and summarise."""
    issues: List[QCIssue] = list(parse_issues)

    counts = Counter(r.sample_id for r in records)
    for sample_id, n in sorted(counts.items()):
        if n > 1:
            issues.append(error("duplicate_id", f"sample_id {sample_id!r} appears {n} times.", sample_id))

    for record in records:
        issues.extend(validate_record(record))
        if data_root is not None:
            issues.extend(_file_issues(record, Path(data_root), check_headers))
        if loader is not None:
            try:
                sample = loader(record)
            except Exception as exc:  # a sample that cannot be loaded is a finding, not a crash
                issues.append(error("load_failed", f"{record.sample_id}: could not be loaded ({type(exc).__name__}: {exc}).", record.sample_id))
            else:
                issues.extend(validate_sample(sample, max_nodata_fraction=max_nodata_fraction))
    if check_split:
        issues.extend(check_split_integrity(records, require_region_disjoint=require_region_disjoint))

    bad_ids = {sid for i in issues if i.severity == ERROR for sid in i.sample_ids}
    unparsable_lines = sum(1 for i in issues if i.severity == ERROR and i.line is not None and not i.sample_ids
                           and i.code != "record_count_mismatch")
    invalid_records = sum(1 for r in records if r.sample_id in bad_ids)   # a duplicated id makes every copy invalid
    return QCReport(
        total=len(records) + unparsable_lines,
        valid=len(records) - invalid_records,
        invalid=invalid_records + unparsable_lines,
        issues=tuple(issues),
        per_split=dict(sorted(Counter(r.split.value for r in records).items())),
        per_dataset=dict(sorted(Counter(f"{r.dataset}/{r.variant}" for r in records).items())),
        invalid_ids=tuple(sorted(bad_ids & {r.sample_id for r in records})),
        manifest_digest=manifest_digest(records) if records else None,
        checks={"records": True, "files": data_root is not None, "headers": data_root is not None and check_headers,
                "pixels": loader is not None, "split_integrity": check_split},
    )


def qc_manifest_file(path: PathLike, **kwargs: Any) -> QCReport:
    """Read a manifest leniently and validate it, so every bad line is reported instead of stopping at the first."""
    contents: ManifestContents = read_manifest(path, strict=False)
    return validate_manifest(contents.records, parse_issues=contents.issues, **kwargs)


def write_report(report: QCReport, path: PathLike) -> None:
    Path(path).write_text(json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
