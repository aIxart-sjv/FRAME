"""The dataset adapter interface (Phase 3).

Every dataset is exposed the same way -- as a set of `PairRecord`s (its manifest) plus a way to
load one pair's pixels -- so training, evaluation and QC never care which dataset a sample is from:

    list_scenes()          the scene ids (the unit that must not span splits)
    get_scene(scene_id)    that scene's region, split(s) and sample ids
    iter_records()         manifest records, optionally filtered by split / variant
    get_metadata(id)       one record
    load_pair(record)      -> PairedSample (pixels + provenance)
    iter_pairs()           load_pair over iter_records

Adapters are manifest-backed: they do not scan the dataset. Dataset-specific code only builds
records (each adapter module has a `records_from_*` builder) and, where the pixels are not plain
GeoTIFF pairs, overrides `load_pair`. This is not a plugin framework: a dict of five classes.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence, Tuple, Union

from frame.data.config import data_root as default_data_root
from frame.data.contract import PairedSample, PairRecord, Split
from frame.data.errors import ContractError, DatasetUnavailableError
from frame.data.manifest import read_manifest
from frame.data.roles import DatasetProfile, get_profile

PathLike = Union[str, Path]


@dataclass(frozen=True)
class Scene:
    scene_id: str
    dataset: str
    region_id: Optional[str]
    splits: Tuple[Split, ...]
    sample_ids: Tuple[str, ...]


class DatasetAdapter(ABC):
    #: key into frame.data.roles.DATASET_PROFILES
    dataset: str

    def __init__(self, records: Sequence[PairRecord], *, data_root: Optional[PathLike] = None):
        wrong = sorted({r.dataset for r in records if r.dataset != self.dataset})
        if wrong:
            raise ContractError(f"{type(self).__name__} handles {self.dataset!r} records, got {wrong}.", code="wrong_dataset")
        self._records: Dict[str, PairRecord] = {}
        for record in records:
            if record.sample_id in self._records:
                raise ContractError(f"Duplicate sample_id {record.sample_id!r}.", code="duplicate_id")
            self._records[record.sample_id] = record
        self._root = Path(data_root) if data_root is not None else None

    @classmethod
    def from_manifest(cls, path: PathLike, *, data_root: Optional[PathLike] = None) -> "DatasetAdapter":
        """Open the records of this adapter's dataset from a (possibly multi-dataset) manifest."""
        contents = read_manifest(path)
        return cls([r for r in contents.records if r.dataset == cls.dataset], data_root=data_root)

    # ------------------------------------------------------------------ description

    @property
    def profile(self) -> DatasetProfile:
        return get_profile(self.dataset)

    @property
    def data_root(self) -> Path:
        return self._root if self._root is not None else default_data_root()

    @property
    def data_dir(self) -> Path:
        return self.data_root / self.dataset

    def __len__(self) -> int:
        return len(self._records)

    # ------------------------------------------------------------------ the interface

    def list_scenes(self) -> List[str]:
        return sorted({r.scene_id for r in self._records.values()})

    def get_scene(self, scene_id: str) -> Scene:
        members = [r for r in self._records.values() if r.scene_id == scene_id]
        if not members:
            raise ContractError(f"{self.dataset} has no scene {scene_id!r}.", code="unknown_scene")
        return Scene(
            scene_id=scene_id, dataset=self.dataset, region_id=members[0].region_id,
            splits=tuple(sorted({r.split for r in members}, key=lambda s: s.value)),
            sample_ids=tuple(sorted(r.sample_id for r in members)),
        )

    def iter_records(self, *, split: Optional[Split] = None, variant: Optional[str] = None) -> Iterator[PairRecord]:
        for sample_id in sorted(self._records):
            record = self._records[sample_id]
            if (split is None or record.split == Split(split)) and (variant is None or record.variant == variant):
                yield record

    def get_metadata(self, sample_id: str) -> PairRecord:
        try:
            return self._records[sample_id]
        except KeyError:
            raise ContractError(f"{self.dataset} has no sample {sample_id!r}.", code="unknown_sample") from None

    @abstractmethod
    def load_pair(
        self, record: Union[PairRecord, str], *, lr_bands: Optional[Sequence[str]] = None, hr_bands: Optional[Sequence[str]] = None
    ) -> PairedSample:
        """Load one pair's pixels. ``lr_bands`` / ``hr_bands`` select and order channels BY NAME."""

    def iter_pairs(
        self, *, split: Optional[Split] = None, variant: Optional[str] = None,
        lr_bands: Optional[Sequence[str]] = None, hr_bands: Optional[Sequence[str]] = None,
    ) -> Iterator[PairedSample]:
        for record in self.iter_records(split=split, variant=variant):
            yield self.load_pair(record, lr_bands=lr_bands, hr_bands=hr_bands)

    def _resolve(self, record: Union[PairRecord, str]) -> PairRecord:
        return record if isinstance(record, PairRecord) else self.get_metadata(record)

    def _require_dir(self) -> Path:
        if not self.data_dir.is_dir():
            raise DatasetUnavailableError(
                f"{self.dataset} data directory not found: {self.data_dir}. Place the dataset there or set FRAME_DATA_ROOT (docs/DATA.md)."
            )
        return self.data_dir
