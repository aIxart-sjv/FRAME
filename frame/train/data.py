"""Manifest -> train / validation datasets, behind the geographic-leakage gate (Phase 4). Main environment only.

    read manifest -> check its digest -> QC (records + split integrity) -> select the train and val splits
    -> check the files of the used records -> PairedPatchDataset x 2

This module never splits anything. The train and validation records are exactly the manifest's own
``split`` assignments (made by `python -m frame.data split`, per scene/region), and training REFUSES to
start if the manifest fails any of:

    scene_leakage    one scene contributes to more than one split        (never optional)
    region_leakage   one region spans splits                             (unless the config explicitly opts
                                                                          into a scene-level split)
    role_violation   a dataset in a split its role forbids -- e.g. SEN2NEON / OpenSR-Test / the Indian holdout in
                     train or val, or a SEN2VENuS record in test

Records in other splits (``test``) may sit in the same manifest; they are never opened. Their files are not even
checked for existence, so a benchmark can be indexed next to training data without any chance of being read.

Imports `frame.data` (rasterio, ...) so it does not run in the isolated Mamba environment; the training core
(frame.train.trainer) has no such dependency and takes ready-made datasets.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from frame.train.config import TrainConfig, resolve_path
from frame.train.errors import LeakageGuardError, TrainDataError

LEAKAGE_CODES = ("scene_leakage", "region_leakage", "role_violation")
_MAX_SHOWN = 4


@dataclass
class TrainingData:
    train: Any                                 # PairedPatchDataset (random crops, augmentation, masks)
    val: Any                                   # PairedPatchDataset (deterministic grid, no augmentation, masks)
    train_records: List[Any]
    val_records: List[Any]
    manifest_digest: str
    scale: int
    band_names: Tuple[str, ...]
    info: Dict[str, Any] = field(default_factory=dict)
    train_eval: Any = None                     # the TRAIN scenes on the validation grid (only when config.evaluate_train; an over-fit check)


def _describe(issues: Sequence[Any]) -> str:
    counts = Counter(i.code for i in issues)
    head = ", ".join(f"{code} x{n}" for code, n in sorted(counts.items()))
    shown = "; ".join(i.message for i in list(issues)[:_MAX_SHOWN])
    return f"{head}. First finding(s): {shown}"


def resolve_manifest_path(config: TrainConfig, base_dir: Optional[Path]) -> Path:
    return resolve_path(config.data.manifest, base_dir)


def prepare_training_data(config: TrainConfig, *, base_dir: Optional[Path] = None) -> TrainingData:
    from frame.data.adapters import ADAPTERS
    from frame.data.config import data_root as default_data_root
    from frame.data.contract import DatasetRole
    from frame.data.errors import ContractError, DataError, PatchError
    from frame.data.loader import PairedPatchDataset, dispatching_loader
    from frame.data.manifest import manifest_digest, read_manifest
    from frame.data.qc import validate_manifest
    from frame.data.roles import DATASET_PROFILES

    d = config.data
    path = resolve_manifest_path(config, base_dir)
    try:
        contents = read_manifest(path)
    except DataError as exc:
        raise TrainDataError(str(exc)) from exc
    digest = manifest_digest(contents.records)
    if contents.header.get("digest") != digest:
        raise TrainDataError(f"Manifest {path} does not match its own digest (header {str(contents.header.get('digest'))[:12]}..., contents {digest[:12]}...): it was edited after it was written.")

    records = [r for r in contents.records if not d.datasets or r.dataset in d.datasets]
    if not records:
        raise TrainDataError(f"The manifest has no records for datasets {list(d.datasets) or 'any'}.")

    # 1. the leakage gate: every record of the (filtered) manifest, whatever its split -- no file access.
    report = validate_manifest(records, check_split=True, require_region_disjoint=d.require_region_disjoint)
    errors = [i for i in report.issues if i.severity == "error"]
    leaks = [i for i in errors if i.code in LEAKAGE_CODES]
    if leaks:
        raise LeakageGuardError(f"Training refused: the manifest violates the geographic-split rules. {_describe(leaks)}", codes=tuple(sorted({i.code for i in leaks})))
    if errors:
        raise TrainDataError(f"Training refused: the manifest fails validation. {_describe(errors)}")

    # 2. defence in depth: whatever the split labels say, benchmark / holdout roles never train or validate.
    train = sorted((r for r in records if r.split.value == d.train_split), key=lambda r: r.sample_id)
    val = sorted((r for r in records if r.split.value == d.val_split), key=lambda r: r.sample_id)
    for name, group in ((d.train_split, train), (d.val_split, val)):
        forbidden = [r for r in group if DATASET_PROFILES[r.dataset].variant(r.variant).role in (DatasetRole.INDEPENDENT_BENCHMARK, DatasetRole.DOMAIN_HOLDOUT)]
        if forbidden:
            raise LeakageGuardError(f"Training refused: benchmark/holdout data in the {name!r} split: {sorted({r.dataset for r in forbidden})}.", codes=("role_violation",))
    for name, group in (("train", train), ("val", val)):
        if not group:
            raise TrainDataError(f"The manifest has no records in the {name} split ({d.train_split if name == 'train' else d.val_split!r}) for the selected datasets.")

    # 3. the used records must be complete pairs of one scale with the requested bands.
    used = train + val
    scales = sorted({r.scale_factor for r in used})
    if len(scales) != 1:
        raise TrainDataError(f"Training and validation records must share one scale factor, got {scales}.")
    for r in used:
        if r.hr is None:
            raise TrainDataError(f"{r.sample_id} has no HR counterpart; it cannot be used for supervised training.")
        for label, wanted, have in (("LR", d.lr_bands, r.lr.band_names), ("HR", d.hr_bands, r.hr.band_names)):
            missing = [b for b in wanted if b not in have]
            if missing:
                raise TrainDataError(f"{r.sample_id}: requested {label} band(s) {missing} are not in the record's bands {list(have)}.")

    # 4. files of the USED records only (test-split files are never touched).
    root = resolve_path(d.data_root, base_dir) if d.data_root else default_data_root()
    file_report = validate_manifest(used, data_root=root, check_split=False)
    file_errors = [i for i in file_report.issues if i.severity == "error"]
    if file_errors:
        raise TrainDataError(f"Training refused: data files are missing or inconsistent. {_describe(file_errors)}")

    # 5. datasets
    adapters = {}
    for name in sorted({r.dataset for r in used}):
        if name not in ADAPTERS:
            raise TrainDataError(f"No adapter for dataset {name!r}; it cannot be read.")
        adapters[name] = ADAPTERS[name]([r for r in used if r.dataset == name], data_root=root)
    load = dispatching_loader(adapters, lr_bands=list(d.lr_bands), hr_bands=list(d.hr_bands))
    try:
        train_ds = PairedPatchDataset(train, load, lr_patch=d.lr_patch, mode="random", patches_per_pair=d.patches_per_pair, seed=config.seed,
                                      min_valid_fraction=d.min_valid_fraction, augment=d.augment, return_masks=True, cache_size=min(len(train), 8))
        val_ds = PairedPatchDataset(val, load, lr_patch=d.val_lr_patch, mode="grid", augment=False, return_masks=True, cache_size=min(len(val), 8))
        train_eval = (PairedPatchDataset(train, load, lr_patch=d.val_lr_patch, mode="grid", augment=False, return_masks=True, cache_size=min(len(train), 8))
                      if config.evaluate_train else None)
    except (PatchError, ContractError) as exc:
        raise TrainDataError(f"The datasets could not be built: {exc}") from exc

    warnings = Counter(i.code for i in report.issues if i.severity == "warning")
    info = {
        "manifest": str(d.manifest), "manifest_digest": digest, "n_manifest_records": len(contents.records),
        "datasets": sorted({r.dataset for r in used}), "train_split": d.train_split, "val_split": d.val_split,
        "n_train_pairs": len(train), "n_val_pairs": len(val), "n_train_patches_per_epoch": len(train_ds), "n_val_patches": len(val_ds),
        "train_scenes": sorted({r.scene_id for r in train}), "val_scenes": sorted({r.scene_id for r in val}),
        "train_regions": sorted({r.region_id for r in train if r.region_id}), "val_regions": sorted({r.region_id for r in val if r.region_id}),
        "ignored_records": dict(sorted(Counter(r.split.value for r in records if r not in used).items())),
        "pair_types": sorted({r.pair_type.value for r in used}), "scale_factor": scales[0],
        "lr_bands": list(d.lr_bands), "hr_bands": list(d.hr_bands), "lr_patch": d.lr_patch, "val_lr_patch": d.val_lr_patch,
        "qc": {"ok": True, "warnings": dict(sorted(warnings.items()))},
    }
    return TrainingData(train=train_ds, val=val_ds, train_eval=train_eval, train_records=train, val_records=val, manifest_digest=digest, scale=scales[0],
                        band_names=tuple(d.hr_bands), info=info)
