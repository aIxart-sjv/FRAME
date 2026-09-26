"""SEN2NEON adapter (Phase 3) -- REAL: built from the published metadata, tested on real tiles.

Role (docs/Requirements 142.txt Requirement 3 section 26): the INDEPENDENT 12-band, 2.5 m
validation/test benchmark. It is never training data: its profile allows only the ``test`` split,
so QC rejects a train sample from it.

Facts (dataset card and metadata.csv, read during Phase 3; revision recorded in every record):
LR = observed Sentinel-2 L2A, 12 x 256 x 256 uint16 at 10 m; HR = 12 x 1024 x 1024 uint16 at 2.5 m
(AVIRIS-NG convolved with Sentinel-2 spectral response functions); x4, pixel-aligned; reflectance
scaled by 10,000; LR nodata 65535, HR nodata 0; UTM CRS per tile; bands named B1..B12 in the files'
metadata (canonicalised here to B01..B12).

Geography: a tile id is ``<neon_acquisition_id>__<row>_<col>`` (``2018_MLBS_3__0_2``). Tiles of one
NEON acquisition (one flight) are adjacent, so the SCENE is the acquisition and the REGION is the
NEON site (``MLBS``): splitting per tile would leak.

Caveat kept with every record: the LR stack is a convenience product with all 12 bands sampled onto
one 10 m grid; B1/B9 (60 m) and the 20 m bands were not measured at 10 m.

Storage: the full dataset is 354.6 GB (LR + 2.5 m HR ~30 GB). `fetch_files` downloads only the
files of the records it is given, pinned to a repository revision.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

from frame.data.adapters.base import DatasetAdapter
from frame.data.adapters.geotiff import read_geotiff_pair
from frame.data.bands import canonical_bands
from frame.data.contract import HRStatus, PairedSample, PairRecord, PairType, RasterSpec, Split
from frame.data.errors import ContractError, DatasetUnavailableError

DATASET = "sen2neon"
VARIANT = "2.5m"
HF_REPO = "isp-uv-es/SEN2NEON"
LR_GRID_CAVEAT = (
    "LR is a 12-band stack on one common 10 m grid; bands with native 20/60 m resolution were sampled onto it, "
    "so B1/B9 and the 20 m bands were NOT measured at 10 m."
)
PathLike = Union[str, Path]


def _number(value: Any) -> Optional[float]:
    if value is None or str(value).strip() in ("", "nan", "NaN", "None"):
        return None
    number = float(value)
    return None if math.isnan(number) else number


def _text(value: Any) -> Optional[str]:
    text = str(value).strip() if value is not None else ""
    return None if text in ("", "nan", "NaN", "None") else text


def record_from_row(row: Mapping[str, Any], *, dataset_revision: Optional[str] = None) -> PairRecord:
    """One SEN2NEON metadata row -> a `PairRecord` (test split, canonical bands, provenance kept)."""
    try:
        acquisition = str(row["neon_acquisition_id"])
        parts = acquisition.split("_")
        if len(parts) != 3:
            raise ContractError(f"Unexpected neon_acquisition_id {acquisition!r} (expected <year>_<SITE>_<flight>).", code="invalid_scene_id")
        bands = canonical_bands(json.loads(row["bands"]))
        lr = RasterSpec(
            band_names=bands, width=int(row["lr_width"]), height=int(row["lr_height"]), pixel_size_m=float(row["lr_pixel_size_m"]),
            path=str(row["lr"]), dtype=str(row["stored_dtype"]), reflectance_scale=float(row["reflectance_scale"]),
            nodata=_number(row["lr_nodata"]), crs=str(row["crs"]), transform=tuple(json.loads(row["lr_transform"])),
        )
        hr = RasterSpec(
            band_names=bands, width=int(row["hr_width"]), height=int(row["hr_height"]), pixel_size_m=float(row["hr_pixel_size_m"]),
            path=str(row["hr"]), dtype=str(row["stored_dtype"]), reflectance_scale=float(row["reflectance_scale"]),
            nodata=_number(row["hr_nodata"]), crs=str(row["crs"]), transform=tuple(json.loads(row["hr_transform"])),
        )
        scale = round(float(row["lr_pixel_size_m"]) / float(row["hr_pixel_size_m"]))
        provenance = {k: v for k, v in {
            "neon_acquisition_id": acquisition, "neon_asset_id": _text(row.get("neon_asset_id")), "neon_date": _text(row.get("neon_date")),
            "s2_asset_id": _text(row.get("s2_asset_id")), "s2_date": _text(row.get("s2_date")),
            "temporal_difference_days": _number(row.get("temporal_difference_days")),
            "cloud_score_plus_cdf": _number(row.get("cloud_score_plus_cdf")),
            "neon_nodata_percent": _number(row.get("neon_nodata_percent")), "s2_nodata_percent": _number(row.get("s2_nodata_percent")),
            "land_cover_superclass": _text(row.get("LC_superclass_text")), "land_cover_detail": _text(row.get("LC_detail_text")),
            "hr_source": _text(row.get("hr_source")), "lr_source": _text(row.get("lr_source")),
            "lr_grid_caveat": LR_GRID_CAVEAT,
        }.items() if v is not None}
        return PairRecord(
            sample_id=f"{DATASET}:{row['id']}", dataset=DATASET, variant=VARIANT, dataset_revision=dataset_revision,
            scene_id=acquisition, region_id=parts[1], split=Split.TEST, source_split=_text(row.get("split")),
            pair_type=PairType.REAL_CROSS_SENSOR, hr_status=HRStatus.AVAILABLE, scale_factor=scale, lr=lr, hr=hr,
            lon=_number(row.get("centroid_lon", row.get("lon"))), lat=_number(row.get("centroid_lat", row.get("lat"))),
            license="CC-BY-4.0", provenance=provenance,
        )
    except ContractError:
        raise
    except KeyError as exc:
        raise ContractError(f"SEN2NEON metadata row is missing column {exc}.", code="missing_field") from None
    except (ValueError, json.JSONDecodeError) as exc:
        raise ContractError(f"Malformed SEN2NEON metadata row {row.get('id')!r}: {exc}.", code="invalid_record") from None


def records_from_metadata_csv(
    csv_path: PathLike, *, dataset_revision: Optional[str] = None, ids: Optional[Iterable[str]] = None
) -> List[PairRecord]:
    """All records of ``metadata.csv`` (2,269 tiles), or only those whose tile id is in ``ids``."""
    wanted = set(ids) if ids is not None else None
    with open(csv_path, newline="", encoding="utf-8") as handle:
        rows = [row for row in csv.DictReader(handle) if wanted is None or row["id"] in wanted]
    if wanted is not None and len(rows) != len(wanted):
        missing = wanted - {r["id"] for r in rows}
        raise ContractError(f"Tile id(s) not in the metadata: {sorted(missing)}.", code="unknown_sample")
    return [record_from_row(row, dataset_revision=dataset_revision) for row in rows]


def relative_files(records: Iterable[PairRecord]) -> List[str]:
    """Every LR and HR file the records reference (relative to the dataset directory)."""
    return sorted({spec.path for r in records for spec in (r.lr, r.hr) if spec is not None and spec.path})


def fetch_files(
    records: Sequence[PairRecord], data_root: PathLike, *, revision: str, extra: Sequence[str] = ("metadata.csv", "s2_l2a_10m.sha256")
) -> Tuple[List[Path], int]:
    """Download exactly the files ``records`` reference (plus ``extra``) from the Hub, pinned to ``revision``.

    Returns (local paths, total bytes). Nothing else is fetched: this is how a small real sample is
    taken from a 354.6 GB dataset. Needs `huggingface_hub` and the network.
    """
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise DatasetUnavailableError("huggingface_hub is needed to download SEN2NEON files.") from exc
    target = Path(data_root) / DATASET
    paths = [Path(hf_hub_download(HF_REPO, name, repo_type="dataset", revision=revision, local_dir=str(target)))
             for name in [*extra, *relative_files(records)]]
    return paths, sum(p.stat().st_size for p in paths)


def verify_lr_checksums(records: Sequence[PairRecord], data_root: PathLike, *, checksum_file: str = "s2_l2a_10m.sha256") -> Dict[str, str]:
    """Check the LR files of ``records`` against the dataset's published SHA-256 manifest.

    Returns ``{relative_path: 'ok' | 'mismatch' | 'not_listed' | 'missing'}``. A mismatch is what a
    wrong or outdated LR release (the card documents one, corrected 20 Aug 2026) would look like.
    """
    directory = Path(data_root) / DATASET
    listing = directory / checksum_file
    if not listing.is_file():
        raise DatasetUnavailableError(f"Checksum manifest not found: {listing}")
    expected: Dict[str, str] = {}
    for line in listing.read_text().splitlines():
        if line.strip():
            digest, name = line.split(None, 1)
            expected[name.strip()] = digest
    result: Dict[str, str] = {}
    for record in records:
        rel = record.lr.path
        path = directory / rel
        if not path.is_file():
            result[rel] = "missing"
        elif rel not in expected:
            result[rel] = "not_listed"
        else:
            digest = hashlib.sha256()
            with open(path, "rb") as handle:
                for chunk in iter(lambda: handle.read(1 << 20), b""):
                    digest.update(chunk)
            result[rel] = "ok" if digest.hexdigest() == expected[rel] else "mismatch"
    return result


class Sen2NeonAdapter(DatasetAdapter):
    dataset = DATASET

    def load_pair(self, record: Union[PairRecord, str], *, lr_bands: Optional[Sequence[str]] = None,
                  hr_bands: Optional[Sequence[str]] = None) -> PairedSample:
        record = self._resolve(record)
        return read_geotiff_pair(record, self._require_dir(), lr_bands=lr_bands, hr_bands=hr_bands)
