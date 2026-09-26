"""Evaluation datasets (Phase 5): reference/geometry validation, role safety, the strict evaluation mask, scene units.

    manifest / subset -> records -> role safety -> reference + geometry validation -> per-sample load (RGBN by name) -> evaluation mask

* **Role safety.** Only datasets whose Phase 3 role is an independent benchmark, a domain holdout, or the synthetic smoke set can be evaluated; a benchmark
  manifest with ANY record labelled train/val refuses the whole evaluation (a contamination signal, not something to filter around); a manifest of another
  dataset than the configured kind is refused; the manifest digest must match its header.
* **Reference / geometry validation** (before any pixel is read, and again as shapes at load): Phase 3's record checks (profile facts, CRS, resolution ratio,
  footprint origin within 0.01 HR px, dimensions) and, with ``check_files``, that the files exist. An invalid sample is LISTED with its codes, never scored, never dropped silently.
* **The evaluation mask.** The dataset's own nodata rule (a pixel is nodata iff EVERY band equals the nodata value) is necessary but not sufficient for
  spectral metrics: on the real SEN2NEON tiles up to ~2.6% of "valid" HR pixels have a zero in SOME bands (boundary pixels), and a spectrum with a zero band is
  not a valid spectrum for SAM / ERGAS / indices. The evaluation rule is therefore: **an HR pixel is valid iff it is not all-band nodata AND no evaluated band
  equals the HR nodata value AND all evaluated values are finite.** Both fractions are reported.
* **Scene units.** Tiles of one source scene are spatially correlated, so every sample carries the unit it must be aggregated to: the NEON acquisition
  (SEN2NEON), the source orthophoto (OpenSR-Test spain_*; each SPOT ROI is its own scene), the region (synthetic). ``category`` is only ever the dataset's
  own label (NEON land-cover superclass; the OpenSR-Test subset: crops / urban), never inferred.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import torch

from frame.evaluate.config import DatasetSpec, resolve_path
from frame.evaluate.errors import EvaluationError, ReferenceMismatchError, RoleSafetyError
from frame.models.config import RGBN_BAND_ORDER

RGBN = tuple(RGBN_BAND_ORDER)
MASK_RULE = ("HR pixel valid iff it is not all-band nodata (the dataset's rule) AND none of the evaluated bands equals the HR nodata value AND all evaluated values are finite")
_PNOA = re.compile(r"PNOA_ANUAL_\d{4}_OF_ETRS89_HU\d+_h25_\d+-\d+")
_OPENSR_CATEGORY = {"spain_crops": "crops", "spain_urban": "urban", "spot": "spot_mixed"}
_EXPECTED_DATASET = {"sen2neon": "sen2neon", "opensr_test": "opensr_test", "synthetic_smoke": "synthetic_smoke"}


@dataclass
class EvalSample:
    sample_id: str
    dataset: str
    scene_group: str
    category: Optional[str]
    split: str
    evidence_class: str
    lr: torch.Tensor                    # (C, h, w) float32 reflectance, RGBN
    hr: torch.Tensor                    # (C, H, W)
    lr_mask: torch.Tensor               # (h, w) bool
    hr_mask: torch.Tensor               # (H, W) bool: the strict evaluation mask
    bands: Tuple[str, ...]
    scale: int
    lr_pixel_m: Optional[float]
    hr_pixel_m: Optional[float]
    quality: Dict[str, Any]
    provenance: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DatasetValidation:
    valid_records: List[Any]
    invalid: List[Dict[str, Any]]


def evaluation_mask(hr: torch.Tensor, dataset_mask: torch.Tensor, *, hr_nodata: Optional[float], reflectance_scale: Optional[float]) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """The strict HR evaluation mask (see the module docstring) and its accounting."""
    dataset_mask = dataset_mask.bool()
    strict = dataset_mask & torch.isfinite(hr).all(dim=0)
    nonfinite = dataset_mask & ~torch.isfinite(hr).all(dim=0)
    if hr_nodata is not None:
        value = torch.tensor(float(hr_nodata), dtype=torch.float32) / (torch.tensor(float(reflectance_scale), dtype=torch.float32) if reflectance_scale else 1.0)
        strict = strict & ~(hr == value).any(dim=0)
    total = float(dataset_mask.numel())
    return strict, {
        "rule": MASK_RULE,
        "hr_dataset_nodata_fraction": float(1.0 - dataset_mask.float().mean()),
        "hr_partial_nodata_fraction": float((dataset_mask & ~strict).sum() - nonfinite.sum()) / total,
        "hr_nonfinite_fraction": float(nonfinite.sum()) / total,
        "hr_valid_fraction": float(strict.float().mean()),
        "hr_valid_pixels": int(strict.sum()),
    }


def opensr_source_group(subset: str, roi: str, hr_file: Optional[str]) -> str:
    """The source scene of an OpenSR-Test sample: the orthophoto for spain_* (several ROIs are crops of one image), the ROI itself for SPOT."""
    if subset != "spot" and hr_file:
        match = _PNOA.search(str(hr_file))
        if match:
            return match.group(0)
    return f"{subset}_{roi}"


class EvalDataset:
    def __init__(self, spec: DatasetSpec, records: List[Any], *, ignored: Dict[str, int], manifest_digest: str, role: str, loader: Callable[[Any], Any],
                 data_root: Optional[Path], info: Dict[str, Any]):
        self.spec, self.records, self.ignored, self.manifest_digest, self.role = spec, records, ignored, manifest_digest, role
        self._loader, self.data_root, self.info = loader, data_root, info

    # ------------------------------------------------------------------ identity

    @property
    def name(self) -> str:
        return self.spec.name

    @property
    def kind(self) -> str:
        return self.spec.kind

    @property
    def evidence_class(self) -> str:
        return self.spec.evidence_class

    def scene_group_of(self, record: Any) -> str:
        if self.kind == "sen2neon":
            return record.scene_id
        if self.kind == "opensr_test":
            roi = record.scene_id.split("_", 1)[1] if "_" in record.scene_id else record.scene_id
            return opensr_source_group(self.spec.subset or "", roi, record.provenance.get("hr_file"))
        return record.region_id or record.scene_id

    def category_of(self, record: Any) -> Optional[str]:
        if self.kind == "sen2neon":
            return record.provenance.get("land_cover_superclass") or "unlabelled"
        if self.kind == "opensr_test":
            return _OPENSR_CATEGORY.get(self.spec.subset or "")
        return None

    # ------------------------------------------------------------------ validation

    def validate(self, *, check_files: bool = True) -> DatasetValidation:
        from frame.data.qc import validate_manifest

        report = validate_manifest(self.records, data_root=self.data_root if (check_files and self.data_root is not None) else None, check_split=False)
        problems: Dict[str, Dict[str, List[str]]] = {}
        for issue in report.issues:
            if issue.severity != "error":
                continue
            for sid in issue.sample_ids or ("<manifest>",):
                entry = problems.setdefault(sid, {"codes": [], "messages": []})
                if issue.code not in entry["codes"]:
                    entry["codes"].append(issue.code)
                entry["messages"].append(issue.message)
        valid = [r for r in self.records if r.sample_id not in problems]
        invalid = [{"sample_id": sid, **entry} for sid, entry in sorted(problems.items())]
        return DatasetValidation(valid_records=valid, invalid=invalid)

    # ------------------------------------------------------------------ loading

    def load(self, record: Any) -> EvalSample:
        paired = self._loader(record)
        if paired.hr is None or tuple(paired.hr.shape[-2:]) != (paired.lr.shape[-2] * record.scale_factor, paired.lr.shape[-1] * record.scale_factor):
            raise ReferenceMismatchError(f"{record.sample_id}: HR shape {None if paired.hr is None else tuple(paired.hr.shape)} is not LR {tuple(paired.lr.shape)} x scale {record.scale_factor}.")
        # an adapter without nodata (OpenSR-Test) returns no masks: every pixel is then a real observation
        hr_mask = paired.hr_mask if paired.hr_mask is not None else torch.ones(paired.hr.shape[-2:], dtype=torch.bool)
        lr_mask = paired.lr_mask if paired.lr_mask is not None else torch.ones(paired.lr.shape[-2:], dtype=torch.bool)
        strict, quality = evaluation_mask(paired.hr, hr_mask, hr_nodata=record.hr.nodata, reflectance_scale=record.hr.reflectance_scale)
        quality.update(lr_valid_fraction=float(lr_mask.float().mean()), lr_valid_pixels=int(lr_mask.sum()))
        prov = {"dataset": record.dataset, "variant": record.variant, "dataset_revision": record.dataset_revision, "pair_type": record.pair_type.value,
                "lr_path": record.lr.path, "hr_path": record.hr.path, "crs": record.lr.crs, "license": record.license,
                "hr_reflectance_scale": record.hr.reflectance_scale, "lr_reflectance_scale": record.lr.reflectance_scale,
                "hr_nodata": record.hr.nodata, "lr_nodata": record.lr.nodata, "lon": record.lon, "lat": record.lat}
        if record.degradation is not None:
            prov["degradation"] = record.degradation.to_dict()
        for key in ("neon_acquisition_id", "s2_date", "neon_date", "temporal_difference_days", "hr_source", "lr_grid_caveat", "hr_file", "lr_gee_id"):
            if key in record.provenance:
                prov[key] = record.provenance[key]
        if "locator" in record.provenance:
            prov["locator"] = record.provenance["locator"]
        return EvalSample(
            sample_id=record.sample_id, dataset=record.dataset, scene_group=self.scene_group_of(record), category=self.category_of(record), split=record.split.value,
            evidence_class=self.evidence_class, lr=paired.lr.float(), hr=paired.hr.float(), lr_mask=lr_mask.bool(), hr_mask=strict, bands=tuple(paired.hr_bands or RGBN),
            scale=int(record.scale_factor), lr_pixel_m=record.lr.pixel_size_m, hr_pixel_m=record.hr.pixel_size_m, quality=quality, provenance=prov)


# ---------------------------------------------------------------------------------------------------------------
# construction
# ---------------------------------------------------------------------------------------------------------------

def _check_roles(spec: DatasetSpec, records: Sequence[Any]) -> str:
    from frame.data.contract import DatasetRole, Split
    from frame.data.roles import DATASET_PROFILES

    codes: List[str] = []
    messages: List[str] = []
    roles = set()
    for r in records:
        profile = DATASET_PROFILES.get(r.dataset)
        role = profile.variant(r.variant).role if profile is not None else None
        roles.add(role)
        if role not in (DatasetRole.INDEPENDENT_BENCHMARK, DatasetRole.DOMAIN_HOLDOUT, DatasetRole.SMOKE_TEST):
            codes.append("not_an_evaluation_dataset")
            messages.append(f"{r.sample_id}: dataset {r.dataset!r} has role {getattr(role, 'value', None)!r}, which is not an evaluation role (training and validation data are not benchmarks)")
        if r.dataset != _EXPECTED_DATASET[spec.kind]:
            codes.append("dataset_kind_mismatch")
            messages.append(f"{r.sample_id}: the manifest holds {r.dataset!r} records but the configured kind {spec.kind!r} expects {_EXPECTED_DATASET[spec.kind]!r}")
        if spec.kind != "synthetic_smoke" and r.split != Split.TEST:
            codes.append("role_violation")
            messages.append(f"{r.sample_id}: benchmark record labelled {r.split.value!r}; benchmarks are test-only")
    if codes:
        raise RoleSafetyError("Evaluation refused: " + "; ".join(messages[:4]), codes=tuple(sorted(set(codes))))
    return next(iter(roles)).value if len(roles) == 1 else "mixed"


def build_dataset(spec: DatasetSpec, *, base_dir: Optional[Path] = None) -> EvalDataset:
    """Construct the evaluation dataset for ``spec`` (reads the manifest / the cached subset; opens no pixel)."""
    from frame.data.config import data_root as default_root
    from frame.data.contract import Split
    from frame.data.errors import DataError
    from frame.data.manifest import manifest_digest, read_manifest

    root = resolve_path(spec.data_root, base_dir) if spec.data_root else default_root()
    if spec.kind == "opensr_test":
        from frame.data.adapters import OpenSRTestAdapter
        from frame.data.adapters import opensr_test as opensr_module
        from frame.validation import load_subset

        loaded = load_subset(spec.subset)
        records = opensr_module.records_from_loaded(loaded, spec.subset, hr_variant=spec.hr_variant)
        role = _check_roles(spec, records)
        adapter = OpenSRTestAdapter(records, loaded_subsets={spec.subset: loaded})
        loader = lambda r: adapter.load_pair(r, lr_bands=RGBN)      # noqa: E731
        return EvalDataset(spec, records, ignored={}, manifest_digest=manifest_digest(records), role=role, loader=loader, data_root=None,
                           info={"subset": spec.subset, "hr_variant": spec.hr_variant, "n_in_subset": len(records)})

    path = resolve_path(spec.manifest, base_dir)
    try:
        contents = read_manifest(path)
    except DataError as exc:
        raise EvaluationError(str(exc)) from exc
    digest = manifest_digest(contents.records)
    if contents.header.get("digest") != digest:
        raise EvaluationError(f"Manifest {path} does not match its own digest (header {str(contents.header.get('digest'))[:12]}..., contents {digest[:12]}...): it was edited after it was written.")
    records = sorted(contents.records, key=lambda r: r.sample_id)
    if spec.kind != "synthetic_smoke":
        role = _check_roles(spec, records)                                       # every record is examined: one bad record refuses the whole evaluation
        selected, ignored = records, {}
    else:
        selected = [r for r in records if r.split == Split.TEST]
        ignored = {s: n for s, n in _count_splits(r for r in records if r.split != Split.TEST).items()}
        role = _check_roles(spec, selected) if selected else "smoke_test"
        if not selected:
            raise EvaluationError(f"The manifest {path.name} has no records in the test split; nothing to evaluate (train/val records of a synthetic set are not evaluation data).")
    if spec.kind == "sen2neon":
        from frame.data.adapters import Sen2NeonAdapter

        adapter = Sen2NeonAdapter(selected, data_root=root)
    else:
        from frame.data.adapters import SyntheticSmokeAdapter

        adapter = SyntheticSmokeAdapter(selected, data_root=root)
    loader = lambda r: adapter.load_pair(r, lr_bands=RGBN, hr_bands=RGBN)       # noqa: E731
    return EvalDataset(spec, selected, ignored=ignored, manifest_digest=digest, role=role, loader=loader, data_root=root,
                       info={"manifest": str(spec.manifest), "dataset_revisions": sorted({r.dataset_revision for r in selected if r.dataset_revision})})


def _count_splits(records) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for r in records:
        out[r.split.value] = out.get(r.split.value, 0) + 1
    return dict(sorted(out.items()))
