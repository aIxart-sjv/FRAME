"""FRAME paired LR-HR data layer (Phase 3).

    Dataset adapter -> manifest -> scene / pair -> validation -> paired patch extractor -> LR / HR sample

The layer knows what an LR-HR pair IS, where it came from, which split it belongs to, how it was
generated, whether it is spatially and spectrally valid, and how to hand small aligned patches to
training or evaluation code. It does not train anything and it is not a data platform: datasets stay
outside the repository (`FRAME_DATA_ROOT`), manifests are small JSONL indexes, and no pixel is read
until it is asked for.

Modules
-------
    contract     PairRecord / PairedSample / RasterSpec / PatchCoords; PairType, Split, HRStatus, DatasetRole
    roles        the five dataset profiles (role, permitted splits, bands, scales, licence, honest status)
    bands        canonical band names and explicit, by-name band selection
    geo          LR/HR geometry validation (CRS, resolution ratio, footprint, dimensions)
    patches      aligned paired patch extraction (HR window = LR window x scale)
    degradation  the documented HR -> Sentinel-2-like LR chain, seeded and recorded
    manifest     the JSONL manifest format, byte-reproducible, with a content digest
    splits       geographic (scene / region) split assignment and leakage checks
    qc           record / file / pixel / manifest validation and the machine-readable report
    adapters     SEN2NEON, OpenSR-Test (real); SEN2NAIPv2, SEN2VENuS (format only); India holdout (planned); synthetic_smoke (generated, Phase 4)
    loader       PairedPatchDataset, collate_pairs
    cli          ``python -m frame.data {qc, sen2neon-manifest, split, synthetic}``

See docs/DATA.md.
"""

from frame.data.bands import L2A_BANDS, RGBN_BANDS, canonical_band, canonical_bands, select_bands
from frame.data.config import DATA_ROOT_ENV, data_root, dataset_dir
from frame.data.contract import (
    DatasetRole,
    DegradationRecord,
    HRStatus,
    PairedSample,
    PairRecord,
    PairType,
    PatchCoords,
    RasterSpec,
    Split,
)
from frame.data.degradation import DegradationConfig, degrade, frame_default_v1, synthesize_pair
from frame.data.errors import (
    ContractError,
    DataError,
    DatasetUnavailableError,
    GeoPairError,
    ManifestError,
    PatchError,
    RoleViolationError,
    SplitLeakageError,
)
from frame.data.geo import validate_pair_geometry
from frame.data.loader import PairedPatchDataset, collate_pairs, dispatching_loader
from frame.data.manifest import manifest_digest, read_manifest, write_manifest
from frame.data.patches import dense_origins, extract_patch, make_coords
from frame.data.qc import QCReport, qc_manifest_file, validate_manifest, validate_record, validate_sample
from frame.data.roles import DATASET_PROFILES, get_profile
from frame.data.splits import apply_splits, assert_no_leakage, assign_splits, check_split_integrity

__all__ = [
    "ContractError", "DATASET_PROFILES", "DATA_ROOT_ENV", "DataError", "DatasetRole", "DatasetUnavailableError",
    "DegradationConfig", "DegradationRecord", "GeoPairError", "HRStatus", "L2A_BANDS", "ManifestError", "PairRecord",
    "PairType", "PairedPatchDataset", "PairedSample", "PatchCoords", "PatchError", "QCReport", "RGBN_BANDS", "RasterSpec",
    "RoleViolationError", "Split", "SplitLeakageError", "apply_splits", "assert_no_leakage", "assign_splits",
    "canonical_band", "canonical_bands", "check_split_integrity", "collate_pairs", "data_root", "dataset_dir", "degrade",
    "dense_origins", "dispatching_loader", "extract_patch", "frame_default_v1", "get_profile", "make_coords",
    "manifest_digest", "qc_manifest_file", "read_manifest", "select_bands", "synthesize_pair", "validate_manifest",
    "validate_pair_geometry", "validate_record", "validate_sample", "write_manifest",
]
