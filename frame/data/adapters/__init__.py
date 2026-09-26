"""Dataset adapters (Phase 3): one common interface, five datasets, honest statuses.

    sen2neon       REAL     independent 12-band 2.5 m benchmark   (tested on real tiles)
    opensr_test    REAL     independent benchmark                 (tested on the cached 'spot' subset)
    sen2naipv2     FIXTURE  primary synthetic training data       (documented format; no real data read)
    sen2venus      FIXTURE  supplementary real 5 m pairs          (documented format; no real data read)
    india_holdout  PLANNED  Indian geographic holdout             (profile + LR-only record builder; no data)
    synthetic_smoke  GENERATED  FRAME's synthetic smoke dataset     (Phase 4; fixtures only, not a benchmark)

`frame.data.roles.DATASET_PROFILES[...].status` is the authoritative statement of each one.
"""

from typing import Dict, Type

from frame.data.adapters.base import DatasetAdapter, Scene
from frame.data.adapters.india_holdout import IndiaHoldoutAdapter
from frame.data.adapters.synthetic import SyntheticSmokeAdapter
from frame.data.adapters.opensr_test import OpenSRTestAdapter
from frame.data.adapters.sen2naipv2 import Sen2NaipV2Adapter
from frame.data.adapters.sen2neon import Sen2NeonAdapter
from frame.data.adapters.sen2venus import Sen2VenusAdapter
from frame.data.errors import ContractError

ADAPTERS: Dict[str, Type[DatasetAdapter]] = {
    "sen2neon": Sen2NeonAdapter,
    "opensr_test": OpenSRTestAdapter,
    "sen2naipv2": Sen2NaipV2Adapter,
    "sen2venus": Sen2VenusAdapter,
    "india_holdout": IndiaHoldoutAdapter,
    "synthetic_smoke": SyntheticSmokeAdapter,
}


def get_adapter_class(dataset: str) -> Type[DatasetAdapter]:
    try:
        return ADAPTERS[dataset]
    except KeyError:
        raise ContractError(f"No adapter for dataset {dataset!r}; known: {sorted(ADAPTERS)}.", code="unknown_dataset") from None


__all__ = ["ADAPTERS", "DatasetAdapter", "IndiaHoldoutAdapter", "OpenSRTestAdapter", "Scene", "Sen2NaipV2Adapter", "SyntheticSmokeAdapter",
           "Sen2NeonAdapter", "Sen2VenusAdapter", "get_adapter_class"]
