"""The small REAL SEN2NEON sample (3 tiles, ~40 MB) -- skipped when it has not been fetched.

Fetch it with ``FRAME_DATA_ROOT=<dir> sen2sr_venv/bin/python experiments/data_smoke/run_smoke.py``.
These tests never download anything.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest
import torch

from frame.data.adapters import Sen2NeonAdapter
from frame.data.adapters import sen2neon
from frame.data.config import data_root, dataset_dir
from frame.data.contract import Split
from frame.data.degradation import degrade, frame_default_v1
from frame.data.loader import PairedPatchDataset
from frame.data.manifest import read_manifest, write_manifest
from frame.data.qc import validate_manifest, validate_record
from frame.preprocessing import RGBN_BANDS

REVISION = "9f076b4f652aa0253127d382dcb6e611250b2e67"
TILES = ["2018_MLBS_3__0_2", "2018_MLBS_3__1_1", "2022_KONZ_7__5_3"]
ROOT = data_root()
NEON_DIR = dataset_dir("sen2neon", ROOT)
CSV = NEON_DIR / "metadata.csv"


def sample_present() -> bool:
    if not CSV.is_file():
        return False
    try:
        records = sen2neon.records_from_metadata_csv(CSV, ids=TILES)
    except Exception:
        return False
    return all((NEON_DIR / f).is_file() for f in sen2neon.relative_files(records)) and (NEON_DIR / "s2_l2a_10m.sha256").is_file()


pytestmark = pytest.mark.skipif(not sample_present(), reason="real SEN2NEON sample not fetched (experiments/data_smoke/run_smoke.py)")


@pytest.fixture(scope="module")
def records():
    return sen2neon.records_from_metadata_csv(CSV, dataset_revision=REVISION, ids=TILES)


@pytest.fixture(scope="module")
def adapter(records):
    return Sen2NeonAdapter(records, data_root=ROOT)


def test_the_published_checksums_of_the_lr_files_match(records):
    assert set(sen2neon.verify_lr_checksums(records, ROOT).values()) == {"ok"}


def test_deep_qc_passes_on_real_files_with_no_findings_at_all(records, adapter):
    report = validate_manifest(records, data_root=ROOT, check_headers=True, loader=adapter.load_pair)
    assert report.ok and (report.total, report.valid, report.invalid) == (3, 3, 0)
    assert report.issues == () and report.checks["headers"] and report.checks["pixels"]


def test_real_tiles_have_the_documented_shapes_georeferencing_and_nodata(adapter, records):
    for record in records:
        s = adapter.load_pair(record)
        assert s.lr.shape == (12, 256, 256) and s.hr.shape == (12, 1024, 1024) and s.lr.dtype == torch.float32
        assert bool(s.lr_mask.all()) and 0.0 < float(s.hr_mask.float().mean()) <= 1.0
        assert float(s.lr[:, s.lr_mask].min()) >= 0.0 and float(s.lr[:, s.lr_mask].max()) < 10.0
        assert float(s.hr[:, ~s.hr_mask].abs().sum()) == 0.0                  # nodata is zeroed in the tensor and carried by the mask
    assert adapter.load_pair(records[0]).hr_mask.float().mean() < 1.0            # the first tile really has NEON nodata


def test_a_real_hr_footprint_shifted_by_one_hr_pixel_is_rejected(records):
    r = records[1]
    t = list(r.hr.transform)
    t[2] += 2.5                                                                   # one HR pixel east
    shifted = dataclasses.replace(r, hr=dataclasses.replace(r.hr, transform=tuple(t)))
    assert validate_record(r) == []
    assert [i.code for i in validate_record(shifted)] == ["footprint_mismatch"]


def test_real_pairs_cannot_be_moved_out_of_the_test_split(records):
    assert all(i.code == "role_violation" for r in records for i in validate_record(dataclasses.replace(r, split=Split.TRAIN)) if i.severity == "error")
    assert sum(1 for r in records for i in validate_record(dataclasses.replace(r, split=Split.TRAIN)) if i.severity == "error") == 3


def test_the_two_tiles_of_one_acquisition_share_a_scene_and_a_site(records):
    a, b, c = records
    assert (a.scene_id, a.region_id) == (b.scene_id, b.region_id) == ("2018_MLBS_3", "MLBS") and (c.scene_id, c.region_id) == ("2022_KONZ_7", "KONZ")


def test_the_manifest_of_the_real_sample_round_trips_byte_for_byte(records, tmp_path):
    d1 = write_manifest(tmp_path / "a.jsonl", records)
    loaded = read_manifest(tmp_path / "a.jsonl")
    d2 = write_manifest(tmp_path / "b.jsonl", loaded.records)
    assert d1 == d2 and (tmp_path / "a.jsonl").read_bytes() == (tmp_path / "b.jsonl").read_bytes() and loaded.records == sorted(records, key=lambda r: r.sample_id)
    assert not any(str(ROOT) in line for line in (tmp_path / "a.jsonl").read_text().splitlines())   # relative paths only


def test_the_committed_sample_manifest_matches_what_the_code_builds_now(records, tmp_path):
    """experiments/data_smoke/metadata/sen2neon_sample.manifest.jsonl must not silently drift from the adapter."""
    from pathlib import Path

    committed = Path(__file__).resolve().parents[2] / "experiments" / "data_smoke" / "metadata" / "sen2neon_sample.manifest.jsonl"
    if not committed.is_file():
        pytest.skip("sample manifest not committed")
    rebuilt = tmp_path / "rebuilt.jsonl"
    write_manifest(rebuilt, records, header={"source": "SEN2NEON metadata.csv", "note": "Phase 3 smoke sample (3 tiles)"})
    assert rebuilt.read_bytes() == committed.read_bytes()


def block_mean(x: np.ndarray, s: int) -> np.ndarray:
    h, w = x.shape
    return x.reshape(h // s, s, w // s, s).mean(axis=(1, 3))


def test_real_lr_tracks_the_hr_best_when_the_grids_are_left_unshifted(adapter, records):
    """A coarse registration sanity check on real data: shifting the HR east by 1 LR pixel lowers the LR/HR correlation."""
    for record in records:
        s = adapter.load_pair(record, lr_bands=["B08"], hr_bands=["B08"])
        lr, hr = s.lr[0].numpy(), s.hr[0].numpy()
        valid = s.lr_mask.numpy() & (block_mean(s.hr_mask.numpy().astype("float32"), 4) == 1.0)

        def corr(hr_image):
            return float(np.corrcoef(lr[:, 3:-3][valid[:, 3:-3]], block_mean(hr_image, 4)[:, 3:-3][valid[:, 3:-3]])[0, 1])

        assert corr(hr) > corr(np.roll(hr, 4, axis=1)) > corr(np.roll(hr, 8, axis=1)), record.sample_id


def test_real_patches_come_out_of_the_loader_aligned_and_reproducible(adapter, records):
    def rgbn(record):
        return adapter.load_pair(record, lr_bands=RGBN_BANDS, hr_bands=RGBN_BANDS)

    a = PairedPatchDataset(records, rgbn, lr_patch=128, mode="random", patches_per_pair=2, seed=5, augment=True, cache_size=1)
    b = PairedPatchDataset(records, rgbn, lr_patch=128, mode="random", patches_per_pair=2, seed=5, augment=True, cache_size=1)
    for i in range(len(a)):
        x, y = a[i], b[i]
        assert x["lr"].shape == (4, 128, 128) and x["hr"].shape == (4, 512, 512) and x["metadata"]["lr_bands"] == list(RGBN_BANDS)
        assert torch.equal(x["hr"], y["hr"]) and x["metadata"]["valid_fraction"] > 0.0


def test_the_full_grid_of_real_tiles_covers_each_lr_raster_exactly_with_edge_clamping(adapter, records):
    ds = PairedPatchDataset(records, adapter.load_pair, lr_patch=128, mode="grid", cache_size=1)
    assert len(ds) == 3 * 4                                                       # 256 = 2 x 128, no clamping needed, no padding
    seen = {(m["sample_id"], m["patch"]["lr_row"], m["patch"]["lr_col"]) for m in (ds[i]["metadata"] for i in range(len(ds)))}
    assert len(seen) == 12 and {(r, c) for _, r, c in seen} == {(0, 0), (0, 128), (128, 0), (128, 128)}


def test_a_synthetic_pair_can_be_made_from_a_real_hr_patch_deterministically(adapter, records):
    s = adapter.load_pair(records[1], lr_bands=RGBN_BANDS, hr_bands=RGBN_BANDS)
    hr = s.hr[:, 256:768, 256:768].contiguous()
    a, rec = degrade(hr, frame_default_v1(4), seed=11)
    b, _ = degrade(hr, frame_default_v1(4), seed=11)
    c, _ = degrade(hr, frame_default_v1(4), seed=12)
    assert a.shape == (4, 128, 128) and torch.equal(a, b) and not torch.equal(a, c)
    assert rec.seed == 11 and rec.parameters_verified is False and rec.origin == "frame"
