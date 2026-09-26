"""Indian geographic holdout (Phase 3) -- PLANNED: profile and record builder only; there is no data.

What the requirements say qualifies (docs/Requirements 142.txt Requirement 3 sections 28-29, 34 and
Requirement 9 sections 24-27):

* Sentinel-2 L2A scenes over Indian locations that are HELD OUT of training -- whole regions, chosen
  for geographic diversity, not "visually convenient" scenes;
* consistent preprocessing and independent evaluation, and enough scenes; otherwise call it
  "Indian-domain evaluation", never "India-optimised model";
* an HR reference only "where legally/data-access feasible" -- candidates named are Cartosat, Resourcesat,
  aerial/orthophoto data; availability is still to be investigated.

Therefore this module never invents an HR label: a record's ``hr_status`` is ``unknown`` (or
``unavailable``) until a real, co-registered reference exists, and only then does it become a paired
record. An LR-only scene is still useful (model stability, downstream analysis) but supports no
reference-based metric, and QC treats it as valid only for this dataset's independent-reference type.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, Union

from frame.data.adapters.base import DatasetAdapter
from frame.data.adapters.geotiff import read_geotiff_pair
from frame.data.contract import HRStatus, PairedSample, PairRecord, PairType, RasterSpec, Split
from frame.data.errors import ContractError
from frame.data.roles import get_profile

DATASET = "india_holdout"


def lr_only_record(
    *, sample_id: str, scene_id: str, region_id: str, lr: RasterSpec, hr_status: HRStatus = HRStatus.UNKNOWN,
    lon: Optional[float] = None, lat: Optional[float] = None, license: Optional[str] = None, provenance: Optional[Dict[str, Any]] = None,
) -> PairRecord:
    """A Sentinel-2 scene with no HR reference (status stated explicitly; it can never be 'available' here)."""
    if hr_status == HRStatus.AVAILABLE:
        raise ContractError("Use a paired record once an HR reference exists; lr_only_record is for scenes without one.", code="invalid_hr_status")
    profile = get_profile(DATASET)
    spec = profile.variant("default")
    return PairRecord(
        sample_id=sample_id, dataset=DATASET, variant="default", scene_id=scene_id, region_id=region_id, split=Split.TEST,
        pair_type=PairType.INDEPENDENT_HR_REFERENCE, hr_status=hr_status, scale_factor=spec.scale_factor, lr=lr, hr=None,
        lon=lon, lat=lat, license=license or profile.license, provenance=dict(provenance or {}),
    )


class IndiaHoldoutAdapter(DatasetAdapter):
    dataset = DATASET

    def load_pair(self, record: Union[PairRecord, str], *, lr_bands: Optional[Sequence[str]] = None,
                  hr_bands: Optional[Sequence[str]] = None) -> PairedSample:
        record = self._resolve(record)
        return read_geotiff_pair(record, self._require_dir(), lr_bands=lr_bands, hr_bands=hr_bands)
