"""A minimal paired-patch dataset for training and evaluation (Phase 3).

Just enough to prove the abstraction and to feed a future training loop; it is deliberately not a
training framework (no samplers, no distributed logic, no augmentation library, no tracking).

    item = dataset[i]   ->   {"lr": (C, h, w), "hr": (C', h*s, w*s), "metadata": {...}}

* Patches come from `frame.data.patches`, so LR and HR always cover the same ground: the HR window
  is the LR window x the pair's scale.
* `mode="grid"` enumerates a dense, deterministic patch grid per pair (evaluation). `mode="random"`
  draws ``patches_per_pair`` origins per pair from a generator seeded by ``(seed, epoch, pair, k)``:
  the same index always gives the same patch, and `set_epoch` changes the draw. Random mode redraws
  a patch that is mostly nodata (up to ``max_attempts``); grid mode keeps every patch and reports its
  ``valid_fraction`` in the metadata.
* Channels are chosen BY NAME (``lr_bands`` / ``hr_bands``), never implicitly. All records must have
  the same scale and channel counts so a batch can be stacked; mix datasets by passing several adapters.
* ``return_masks=True`` adds ``lr_mask`` / ``hr_mask`` (bool, (h, w) / (H, W); True = a real observation) to each
  item, moved by the same augmentation as the pixels. Nodata pixels are zeros in ``lr`` / ``hr``, so a loss must
  use the masks to exclude them (frame.train does). Off by default: existing callers see the same items.
* ``augment=True`` applies the SAME flip/rotation to the LR and HR patch (a member of the dihedral
  transforms frame.uncertainty already uses), so the pair stays aligned; the choice is in the metadata.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from frame.data.adapters.base import DatasetAdapter
from frame.data.contract import PairedSample, PairRecord
from frame.data.errors import ContractError, PatchError
from frame.data.patches import DEFAULT_LR_PATCH, dense_origins, extract_patch, make_coords, random_origin, valid_fraction
from frame.uncertainty.transforms import DEFAULT_TRANSFORMS

Loader = Callable[[PairRecord], PairedSample]
MODES = ("grid", "random")


def dispatching_loader(
    adapters: Mapping[str, DatasetAdapter], *, lr_bands: Optional[Sequence[str]] = None, hr_bands: Optional[Sequence[str]] = None
) -> Loader:
    """A loader that sends each record to its dataset's adapter, selecting the same bands for all."""
    def load(record: PairRecord) -> PairedSample:
        try:
            adapter = adapters[record.dataset]
        except KeyError:
            raise ContractError(f"No adapter was given for dataset {record.dataset!r}.", code="unknown_dataset") from None
        return adapter.load_pair(record, lr_bands=lr_bands, hr_bands=hr_bands)
    return load


def collate_pairs(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Stack ``lr`` and ``hr`` (and the masks, when present); keep ``metadata`` as a list of per-sample dicts."""
    out = {"lr": torch.stack([b["lr"] for b in batch]), "hr": torch.stack([b["hr"] for b in batch]),
           "metadata": [b["metadata"] for b in batch]}
    for key in ("lr_mask", "hr_mask"):
        if key in batch[0]:
            out[key] = torch.stack([b[key] for b in batch])
    return out


class PairedPatchDataset(Dataset):
    def __init__(
        self,
        records: Sequence[PairRecord],
        loader: Loader,
        *,
        lr_patch: int = DEFAULT_LR_PATCH,
        mode: str = "random",
        patches_per_pair: int = 1,
        stride: Optional[int] = None,
        seed: int = 0,
        min_valid_fraction: float = 0.8,
        max_attempts: int = 8,
        augment: bool = False,
        cache_size: int = 2,
        return_masks: bool = False,
    ):
        if mode not in MODES:
            raise ContractError(f"mode must be one of {MODES}, got {mode!r}.", code="invalid_mode")
        if not records:
            raise ContractError("PairedPatchDataset needs at least one record.", code="empty_dataset")
        self._records = sorted(records, key=lambda r: r.sample_id)
        scales = {r.scale_factor for r in self._records}
        if len(scales) != 1:
            raise ContractError(f"All records must share one scale factor to be batched, got {sorted(scales)}; build one dataset per scale.",
                                code="inconsistent_scale")
        missing = [r.sample_id for r in self._records if r.hr is None]
        if missing:
            raise ContractError(f"Records without an HR counterpart cannot be used for paired patches: {missing[:3]}.", code="missing_hr")

        self.scale = scales.pop()
        self.lr_patch = lr_patch
        self.mode = mode
        self.patches_per_pair = patches_per_pair
        self.seed = seed
        self.min_valid_fraction = min_valid_fraction  # random mode redraws a patch until at least this fraction of it is valid
        self.max_attempts = max_attempts
        self.augment = augment
        self.return_masks = return_masks
        self._loader = loader
        self._cache: "OrderedDict[int, PairedSample]" = OrderedDict()
        self._cache_size = max(1, cache_size)
        self._epoch = 0

        if mode == "grid":
            self._index: List[Tuple[int, int, int]] = []
            for i, r in enumerate(self._records):
                for row, col in dense_origins(r.lr.height, r.lr.width, lr_patch, stride):
                    self._index.append((i, row, col))
        else:
            for r in self._records:
                if r.lr.height < lr_patch or r.lr.width < lr_patch:
                    raise PatchError(f"{r.sample_id}: LR {r.lr.height}x{r.lr.width} is smaller than the {lr_patch}px patch.")
            self._index = []

    # ------------------------------------------------------------------ Dataset protocol

    def __len__(self) -> int:
        return len(self._index) if self.mode == "grid" else len(self._records) * self.patches_per_pair

    def set_epoch(self, epoch: int) -> None:
        """Change the random draws (random mode); the same (seed, epoch, index) always gives the same patch."""
        self._epoch = int(epoch)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        if not 0 <= index < len(self):
            raise IndexError(index)
        if self.mode == "grid":
            pair_index, row, col = self._index[index]
            sample = self._pair(pair_index)
            patch = extract_patch(sample, make_coords(sample, row, col, self.lr_patch))
            rng = np.random.default_rng([self.seed, self._epoch, index])
        else:
            pair_index, k = divmod(index, self.patches_per_pair)
            sample = self._pair(pair_index)
            rng = np.random.default_rng([self.seed, self._epoch, pair_index, k])
            patch = self._draw(sample, rng)

        lr, hr, augmentation = patch.lr, patch.hr, "identity"
        lr_mask, hr_mask = patch.lr_mask, patch.hr_mask
        if self.augment:
            transform = DEFAULT_TRANSFORMS[int(rng.integers(0, len(DEFAULT_TRANSFORMS)))]
            lr, hr, augmentation = transform.forward(lr), transform.forward(hr), transform.name
            if self.return_masks:
                lr_mask = transform.forward(lr_mask[None])[0] if lr_mask is not None else None
                hr_mask = transform.forward(hr_mask[None])[0] if hr_mask is not None else None
        metadata = patch.metadata_dict()
        metadata.update({"valid_fraction": valid_fraction(patch), "augmentation": augmentation, "epoch": self._epoch})
        item = {"lr": lr, "hr": hr, "metadata": metadata}
        if self.return_masks:
            item["lr_mask"] = lr_mask if lr_mask is not None else torch.ones(lr.shape[-2:], dtype=torch.bool)
            item["hr_mask"] = hr_mask if hr_mask is not None else torch.ones(hr.shape[-2:], dtype=torch.bool)
        return item

    # ------------------------------------------------------------------ internals

    def _pair(self, pair_index: int) -> PairedSample:
        if pair_index in self._cache:
            self._cache.move_to_end(pair_index)
            return self._cache[pair_index]
        sample = self._loader(self._records[pair_index])
        self._cache[pair_index] = sample
        while len(self._cache) > self._cache_size:
            self._cache.popitem(last=False)
        return sample

    def _draw(self, sample: PairedSample, rng: np.random.Generator) -> PairedSample:
        """A random patch, redrawn while too much of it is nodata."""
        best, best_valid = None, -1.0
        height, width = sample.lr.shape[-2:]
        for _ in range(self.max_attempts):
            row, col = random_origin(height, width, self.lr_patch, rng)
            candidate = extract_patch(sample, make_coords(sample, row, col, self.lr_patch))
            fraction = valid_fraction(candidate)
            if fraction >= self.min_valid_fraction:
                return candidate
            if fraction > best_valid:
                best, best_valid = candidate, fraction
        if best_valid <= 0.0:
            raise PatchError(f"{sample.sample_id}: no patch with any valid pixels found in {self.max_attempts} draws.")
        return best  # the least-masked patch seen; its valid_fraction is reported in the metadata
