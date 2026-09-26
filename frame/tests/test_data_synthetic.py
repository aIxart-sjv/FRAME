"""frame.data.adapters.synthetic -- the controlled synthetic paired dataset used for Phase 4 smoke experiments."""

from __future__ import annotations

import dataclasses
import json

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from frame.data import cli
from frame.data.adapters import ADAPTERS, SyntheticSmokeAdapter
from frame.data.adapters.geotiff import read_raster
from frame.data.adapters.synthetic import DATASET, build_synthetic_dataset, generate_scene
from frame.data.contract import DatasetRole, PairType, Split
from frame.data.degradation import degrade
from frame.data.manifest import read_manifest, write_manifest
from frame.data.qc import validate_manifest, validate_record
from frame.data.roles import DATASET_PROFILES, get_profile
from frame.data.splits import apply_splits, assign_splits, check_split_integrity
from frame.preprocessing import RGBN_BANDS


# ============================================================================== profile


def test_the_synthetic_dataset_has_an_honest_profile_of_its_own_role():
    p = get_profile(DATASET)
    v = p.variant("default")
    assert v.role is DatasetRole.SMOKE_TEST and v.pair_type is PairType.SYNTHETIC and v.scale_factor == 4
    assert (v.lr_size, v.hr_size, v.lr_bands) == ((128, 128), (512, 512), RGBN_BANDS)
    assert v.allowed_splits == {Split.TRAIN, Split.VAL, Split.TEST}
    assert "synthetic" in p.status.lower() and "not evidence" in " ".join(p.caveats).lower()


def test_the_adapter_is_registered_next_to_the_real_datasets():
    assert ADAPTERS[DATASET] is SyntheticSmokeAdapter


# ============================================================================== scenes


def test_a_scene_is_deterministic_per_seed_and_different_across_seeds():
    a, b, c = generate_scene(1), generate_scene(1), generate_scene(2)
    assert a.shape == (4, 512, 512) and a.dtype == torch.float32 and torch.equal(a, b) and not torch.equal(a, c)


def test_scene_values_are_physical_reflectances():
    scene = generate_scene(3)
    assert 0.0 < float(scene.min()) and float(scene.max()) <= 1.0 and torch.isfinite(scene).all()
    assert float(scene[3].mean()) != float(scene[0].mean())            # the bands differ: vegetation-like NIR is not red


def test_a_scene_has_sharp_detail_that_bicubic_from_a_degraded_copy_cannot_recover():
    """Otherwise the smoke experiment could not show learning: the LR must lose information the HR contains."""
    scene = generate_scene(4)
    lr = F.avg_pool2d(scene[None], 4)
    bicubic = F.interpolate(lr, scale_factor=4, mode="bicubic", align_corners=False)[0].clamp(min=0)
    assert float((bicubic - scene).abs().mean()) > 0.004
    edges = float((scene[:, :, 1:] - scene[:, :, :-1]).abs().max())
    assert edges > 0.05


# ============================================================================== the built dataset


@pytest.fixture()
def built(tmp_path):
    records = build_synthetic_dataset(tmp_path, n_regions=3, scenes_per_region=2, seed=7)
    return tmp_path, records


def test_the_dataset_has_the_requested_geography(built):
    _, records = built
    assert len(records) == 6 and {r.region_id for r in records} == {"R00", "R01", "R02"} and len({r.scene_id for r in records}) == 6
    assert all(r.dataset == DATASET and r.sample_id == f"{DATASET}:{r.scene_id}" for r in records)
    assert all(r.split is Split.TRAIN for r in records)                 # unsplit: the Phase 3 splitter assigns the real splits


def test_every_record_passes_the_profile_and_geometry_checks(built):
    _, records = built
    assert all(validate_record(r) == [] for r in records)
    assert validate_manifest(records).ok


def test_the_files_are_real_and_pass_pixel_and_header_qc(built):
    root, records = built
    adapter = SyntheticSmokeAdapter(records, data_root=root)
    report = validate_manifest(records, data_root=root, check_headers=True, loader=adapter.load_pair)
    assert report.ok and report.valid == 6 and report.issues == (), report.summary()


def test_pairs_load_through_the_common_adapter_interface(built):
    root, records = built
    sample = SyntheticSmokeAdapter(records, data_root=root).load_pair(records[0])
    assert sample.lr.shape == (4, 128, 128) and sample.hr.shape == (4, 512, 512) and sample.lr_bands == RGBN_BANDS
    assert bool(sample.lr_mask.all()) and bool(sample.hr_mask.all())


def test_records_carry_a_frame_degradation_record_that_says_it_is_not_the_published_one(built):
    _, records = built
    d = records[0].degradation
    assert d.origin == "frame" and d.parameters_verified is False and d.seed is not None and d.config["scale"] == 4
    assert len({r.degradation.seed for r in records}) == 6              # every scene has its own recorded seed


def test_the_lr_can_be_regenerated_exactly_from_the_stored_hr_and_its_degradation_record(built):
    """The requirement that a synthetic LR be reproducible from its record: same HR file + same recorded config/seed -> identical LR."""
    root, records = built
    from frame.data.degradation import DegradationConfig

    r = records[1]
    hr, _, _ = read_raster(root / DATASET / r.hr.path, r.hr, None)
    lr_file, _, _ = read_raster(root / DATASET / r.lr.path, r.lr, None)
    config = DegradationConfig(**r.degradation.config)
    regenerated, _ = degrade(hr, config, seed=r.degradation.seed)
    dn = torch.round(regenerated * 10_000).clamp(0, 65535)
    assert torch.equal(dn / 10_000, lr_file)


def test_georeferencing_is_exact_x4_and_scenes_do_not_overlap(built):
    _, records = built
    for r in records:
        assert r.lr.transform[0] == 10.0 and r.hr.transform[0] == 2.5 and r.lr.transform[2] == r.hr.transform[2] and r.lr.transform[5] == r.hr.transform[5]
    origins = {(r.lr.transform[2], r.lr.transform[5]) for r in records}
    assert len(origins) == 6


def test_the_same_seed_builds_byte_identical_manifests_and_files(tmp_path):
    a = build_synthetic_dataset(tmp_path / "a", n_regions=2, scenes_per_region=1, seed=3)
    b = build_synthetic_dataset(tmp_path / "b", n_regions=2, scenes_per_region=1, seed=3)
    assert write_manifest(tmp_path / "a.jsonl", a) == write_manifest(tmp_path / "b.jsonl", b)
    assert (tmp_path / "a" / DATASET / a[0].hr.path).read_bytes() == (tmp_path / "b" / DATASET / b[0].hr.path).read_bytes()
    c = build_synthetic_dataset(tmp_path / "c", n_regions=2, scenes_per_region=1, seed=4)
    assert write_manifest(tmp_path / "c.jsonl", c) != write_manifest(tmp_path / "a.jsonl", a)


def test_the_dataset_splits_by_region_with_the_phase_3_splitter_and_stays_leak_free(tmp_path):
    records = build_synthetic_dataset(tmp_path, n_regions=5, scenes_per_region=2, seed=1)
    fractions = {Split.TRAIN: 0.6, Split.VAL: 0.2, Split.TEST: 0.2}
    split = apply_splits(records, assign_splits(records, fractions, seed=0, level="region"), level="region")
    assert not [i for i in check_split_integrity(split) if i.severity == "error"]
    assert {s for s in (r.split for r in split)} == {Split.TRAIN, Split.VAL, Split.TEST}


def test_bad_arguments_are_refused(tmp_path):
    with pytest.raises(ValueError, match="n_regions"):
        build_synthetic_dataset(tmp_path, n_regions=0, scenes_per_region=1, seed=0)
    with pytest.raises(ValueError, match="scenes_per_region"):
        build_synthetic_dataset(tmp_path, n_regions=1, scenes_per_region=0, seed=0)


# ============================================================================== CLI


def test_the_cli_writes_the_files_and_an_unsplit_manifest(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("FRAME_DATA_ROOT", str(tmp_path / "root"))
    out = tmp_path / "manifest.jsonl"
    assert cli.main(["synthetic", "--out", str(out), "--regions", "2", "--scenes-per-region", "2", "--seed", "5"]) == 0
    contents = read_manifest(out)
    assert len(contents.records) == 4 and contents.header["source"].startswith("frame.data synthetic")
    assert (tmp_path / "root" / DATASET / contents.records[0].lr.path).is_file()
    assert "wrote 4 record(s)" in capsys.readouterr().out
    assert cli.main(["qc", str(out), "--data-root", str(tmp_path / "root"), "--headers", "--pixels"]) == 0


def test_the_cli_manifest_feeds_the_split_command(tmp_path, monkeypatch):
    monkeypatch.setenv("FRAME_DATA_ROOT", str(tmp_path / "root"))
    cli.main(["synthetic", "--out", str(tmp_path / "u.jsonl"), "--regions", "5", "--scenes-per-region", "1", "--seed", "0"])
    assert cli.main(["split", str(tmp_path / "u.jsonl"), "--out", str(tmp_path / "s.jsonl"), "--seed", "0", "--fractions", "0.6", "0.2", "0.2"]) == 0
    assert {r.split for r in read_manifest(tmp_path / "s.jsonl").records} == {Split.TRAIN, Split.VAL, Split.TEST}
