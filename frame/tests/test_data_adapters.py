"""frame.data.adapters -- the common interface, the GeoTIFF reader, and each dataset's builder."""

from __future__ import annotations

import csv
import dataclasses
import hashlib
import os
from pathlib import Path

import numpy as np
import pytest
import torch

from frame.data.adapters import ADAPTERS, IndiaHoldoutAdapter, OpenSRTestAdapter, Sen2NaipV2Adapter, Sen2NeonAdapter, Sen2VenusAdapter, get_adapter_class
from frame.data.adapters import india_holdout, opensr_test, sen2naipv2, sen2neon, sen2venus
from frame.data.adapters.geotiff import read_geotiff_pair, read_raster
from frame.data.bands import L2A_BANDS
from frame.data.config import DATA_ROOT_ENV, data_root
from frame.data.contract import HRStatus, PairedSample, PairRecord, PairType, RasterSpec, Split
from frame.data.errors import ContractError, DatasetUnavailableError
from frame.data.manifest import write_manifest
from frame.data.patches import centred_origin, extract_patch, make_coords
from frame.data.qc import validate_manifest, validate_record
from frame.preprocessing import RGBN_BANDS
from frame.tests.data_fixtures import (  # noqa: F401
    TINY,
    TinyAdapter,
    TinyEnv,
    block_mean,
    make_record,
    tiny_env,
    tiny_profile_installed,
    write_pair_files,
    write_raster,
)
from frame.tests.data_real_rows import REAL_ROWS


def errors(issues):
    return [i for i in issues if i.severity == "error"]


# ============================================================================== common interface


def test_the_registry_has_the_five_datasets_and_rejects_unknown_ones():
    assert sorted(ADAPTERS) == ["india_holdout", "opensr_test", "sen2naipv2", "sen2neon", "sen2venus", "synthetic_smoke"]
    assert get_adapter_class("sen2neon") is Sen2NeonAdapter
    with pytest.raises(ContractError):
        get_adapter_class("nope")


def test_every_adapter_shares_one_interface_and_a_matching_profile():
    for name, cls in ADAPTERS.items():
        assert cls.dataset == name
        for method in ("list_scenes", "get_scene", "iter_records", "get_metadata", "load_pair", "iter_pairs"):
            assert callable(getattr(cls, method))


def test_list_scenes_and_get_scene(tiny_env: TinyEnv):
    adapter = tiny_env.adapter()
    assert adapter.list_scenes() == ["A_s1", "A_s2", "B_s1", "B_s2"] and len(adapter) == 8
    scene = adapter.get_scene("B_s2")
    assert scene.region_id == "B" and scene.splits == (Split.VAL,) and len(scene.sample_ids) == 2 and scene.dataset == TINY
    with pytest.raises(ContractError, match="no scene"):
        adapter.get_scene("Z_s9")


def test_iter_records_filters_by_split_and_is_sorted(tiny_env: TinyEnv):
    adapter = tiny_env.adapter()
    train = list(adapter.iter_records(split=Split.TRAIN))
    assert len(train) == 4 and all(r.split is Split.TRAIN for r in train)
    assert [r.sample_id for r in train] == sorted(r.sample_id for r in train)
    assert len(list(adapter.iter_records(split="val"))) == 4 and len(list(adapter.iter_records(variant="default"))) == 8
    assert list(adapter.iter_records(variant="none")) == []


def test_get_metadata_returns_the_record_and_rejects_unknown_ids(tiny_env: TinyEnv):
    adapter = tiny_env.adapter()
    assert adapter.get_metadata(tiny_env.records[0].sample_id) == tiny_env.records[0]
    with pytest.raises(ContractError, match="no sample"):
        adapter.get_metadata("tinyset:nope")


def test_an_adapter_refuses_foreign_or_duplicate_records(tiny_env: TinyEnv):
    other = dataclasses.replace(tiny_env.records[0], dataset="otherset", sample_id="otherset:x")
    with pytest.raises(ContractError, match="handles"):
        TinyAdapter([other], data_root=tiny_env.root)
    with pytest.raises(ContractError, match="Duplicate"):
        TinyAdapter([tiny_env.records[0], tiny_env.records[0]], data_root=tiny_env.root)


def test_from_manifest_keeps_only_this_datasets_records(tiny_env: TinyEnv, tmp_path):
    foreign = dataclasses.replace(tiny_env.records[0], dataset="otherset", sample_id="otherset:x")
    write_manifest(tmp_path / "m.jsonl", tiny_env.records + [foreign])
    adapter = TinyAdapter.from_manifest(tmp_path / "m.jsonl", data_root=tiny_env.root)
    assert len(adapter) == 8


def test_iter_pairs_loads_pixels_with_masks_and_provenance(tiny_env: TinyEnv):
    pairs = list(tiny_env.adapter().iter_pairs(split=Split.VAL))
    assert len(pairs) == 4 and all(isinstance(p, PairedSample) for p in pairs)
    p = pairs[0]
    assert p.lr.shape == (4, 32, 32) and p.hr.shape == (4, 128, 128) and p.lr_bands == RGBN_BANDS
    assert p.lr_mask.dtype == torch.bool and p.metadata_dict()["scene_id"].startswith("B_")


def test_band_selection_by_name_reorders_explicitly_and_matches_the_full_read(tiny_env: TinyEnv):
    adapter = tiny_env.adapter()
    record = tiny_env.records[0]
    full = adapter.load_pair(record)
    picked = adapter.load_pair(record, lr_bands=["B08", "B04"], hr_bands=["B02"])
    assert picked.lr_bands == ("B08", "B04") and picked.hr_bands == ("B02",)
    assert torch.equal(picked.lr[0], full.lr[3]) and torch.equal(picked.lr[1], full.lr[0]) and torch.equal(picked.hr[0], full.hr[2])
    with pytest.raises(ContractError, match="not available"):
        adapter.load_pair(record, lr_bands=["B11"])


def test_the_data_root_comes_from_the_environment_and_a_missing_dataset_says_where_to_put_it(tmp_path, monkeypatch, tiny_env: TinyEnv):
    monkeypatch.setenv(DATA_ROOT_ENV, str(tmp_path / "elsewhere"))
    assert data_root() == tmp_path / "elsewhere"
    adapter = TinyAdapter(tiny_env.records)  # no explicit root -> the environment's
    with pytest.raises(DatasetUnavailableError, match="FRAME_DATA_ROOT"):
        adapter.load_pair(tiny_env.records[0])
    monkeypatch.delenv(DATA_ROOT_ENV)
    assert data_root() == Path.home() / ".cache" / "frame_data"


# ============================================================================== the GeoTIFF reader


def spec_for(env: TinyEnv, i=0, which="lr") -> RasterSpec:
    return getattr(env.records[i], which)


def test_stored_values_become_float32_reflectance_by_the_records_own_scale(tmp_path):
    lr_spec, _ = write_pair_files(tmp_path, TINY, "lr.tif", "hr.tif", seed=1)
    tensor, mask, names = read_raster(tmp_path / TINY / "lr.tif", lr_spec, None)
    assert tensor.dtype == torch.float32 and names == RGBN_BANDS and bool(mask.all())
    assert 0.0 < float(tensor.min()) and float(tensor.max()) < 1.0
    raw = np.round(np.asarray(read_raster(tmp_path / TINY / "lr.tif", dataclasses.replace(lr_spec, reflectance_scale=None), None)[0]))
    assert np.allclose(raw / 10000.0, tensor.numpy(), atol=1e-6)


def test_nodata_is_an_explicit_mask_and_zero_in_the_tensor(tmp_path):
    lr_spec, hr_spec = write_pair_files(tmp_path, TINY, "lr.tif", "hr.tif", lr_nodata=65535, hr_nodata=0, nodata_box=(0, 8, 0, 16))
    lr, lr_mask, _ = read_raster(tmp_path / TINY / "lr.tif", lr_spec, None)
    hr, hr_mask, _ = read_raster(tmp_path / TINY / "hr.tif", hr_spec, None)
    assert not bool(lr_mask[:8, :16].any()) and bool(lr_mask[8:, :].all()) and bool(lr_mask[:, 16:].all())
    assert not bool(hr_mask[:32, :64].any()) and float(lr[:, :8, :16].abs().sum()) == 0.0 and float(hr[:, :32, :64].abs().sum()) == 0.0
    assert float(lr_mask.float().mean()) == pytest.approx(1 - 8 * 16 / 1024)


def test_a_pixel_is_nodata_only_when_every_band_says_so(tmp_path):
    lr_spec, _ = write_pair_files(tmp_path, TINY, "lr.tif", "hr.tif", lr_nodata=65535)
    data = np.full((4, 32, 32), 3000, dtype="uint16")
    data[0, 5, 5] = 65535           # only one band: still a real observation
    data[:, 6, 6] = 65535           # every band: nodata
    write_raster(tmp_path / TINY / "lr.tif", data, transform=lr_spec.transform, crs=lr_spec.crs, dtype="uint16", nodata=65535)
    _, mask, _ = read_raster(tmp_path / TINY / "lr.tif", lr_spec, None)
    assert bool(mask[5, 5]) and not bool(mask[6, 6])


def test_a_genuine_zero_is_not_confused_with_nodata_when_no_nodata_value_is_declared(tmp_path):
    lr_spec, _ = write_pair_files(tmp_path, TINY, "lr.tif", "hr.tif", lr_nodata=None)
    data = np.full((4, 32, 32), 3000, dtype="uint16")
    data[:, 0, 0] = 0
    write_raster(tmp_path / TINY / "lr.tif", data, transform=lr_spec.transform, crs=lr_spec.crs, dtype="uint16", nodata=None)
    tensor, mask, _ = read_raster(tmp_path / TINY / "lr.tif", lr_spec, None)
    assert bool(mask[0, 0]) and float(tensor[:, 0, 0].sum()) == 0.0


def test_nan_in_a_file_is_preserved_never_silently_replaced(tmp_path):
    lr_spec, _ = write_pair_files(tmp_path, TINY, "lr.tif", "hr.tif")
    data = np.full((4, 32, 32), 0.3, dtype="float32")
    data[2, 4, 4] = np.nan
    write_raster(tmp_path / TINY / "lr.tif", data, transform=lr_spec.transform, crs=lr_spec.crs, dtype="float32", nodata=None)
    spec = dataclasses.replace(lr_spec, dtype="float32", reflectance_scale=None)
    tensor, _, _ = read_raster(tmp_path / TINY / "lr.tif", spec, None)
    assert bool(torch.isnan(tensor[2, 4, 4]))


def test_a_file_that_disagrees_with_its_record_is_an_error(tmp_path):
    lr_spec, _ = write_pair_files(tmp_path, TINY, "lr.tif", "hr.tif")
    with pytest.raises(ContractError) as excinfo:
        read_raster(tmp_path / TINY / "lr.tif", dataclasses.replace(lr_spec, band_names=("B04", "B03", "B02")), None)
    assert excinfo.value.code == "band_count_mismatch"
    with pytest.raises(ContractError) as excinfo:
        read_raster(tmp_path / TINY / "lr.tif", dataclasses.replace(lr_spec, width=16), None)
    assert excinfo.value.code == "shape_mismatch"
    with pytest.raises(DatasetUnavailableError):
        read_raster(tmp_path / TINY / "absent.tif", lr_spec, None)


def test_a_record_without_paths_cannot_use_the_geotiff_reader(tmp_path):
    record = make_record("tinyset:x", scene="s", region="r", lr=RasterSpec(band_names=RGBN_BANDS, width=32, height=32),
                         hr=RasterSpec(band_names=RGBN_BANDS, width=128, height=128))
    with pytest.raises(ContractError) as excinfo:
        read_geotiff_pair(record, tmp_path)
    assert excinfo.value.code == "missing_path"


# ============================================================================== SEN2NEON (real metadata rows)


def neon_records():
    return [sen2neon.record_from_row(row, dataset_revision="9f076b4f652aa0253127d382dcb6e611250b2e67") for row in REAL_ROWS.values()]


def test_a_real_sen2neon_row_becomes_a_faithful_record():
    (a, b, c) = neon_records()
    assert a.sample_id == "sen2neon:2018_MLBS_3__0_2" and a.dataset == "sen2neon" and a.variant == "2.5m"
    assert a.scene_id == "2018_MLBS_3" and a.region_id == "MLBS" and c.scene_id == "2022_KONZ_7" and c.region_id == "KONZ"
    assert a.split is Split.TEST and a.source_split == "validation"            # FRAME's role vs the dataset's own label
    assert a.pair_type is PairType.REAL_CROSS_SENSOR and a.hr_status is HRStatus.AVAILABLE and a.scale_factor == 4
    assert a.lr.band_names == a.hr.band_names == L2A_BANDS                       # B1 -> B01 ... B9 -> B09
    assert (a.lr.height, a.lr.width, a.hr.height, a.hr.width) == (256, 256, 1024, 1024)
    assert (a.lr.pixel_size_m, a.hr.pixel_size_m) == (10.0, 2.5) and (a.lr.nodata, a.hr.nodata) == (65535.0, 0.0)
    assert a.lr.dtype == "uint16" and a.lr.reflectance_scale == 10000.0 and a.lr.crs == "EPSG:32617"
    assert a.lr.transform == (10.0, 0.0, 539120.0, 0.0, -10.0, 4146000.0) and a.hr.transform == (2.5, 0.0, 539120.0, 0.0, -2.5, 4146000.0)
    assert (a.lon, a.lat) == pytest.approx((-80.54325020048908, 37.448448298909405))
    assert a.lr.path == "s2_l2a_10m/2018_MLBS_3__0_2.tif" and a.hr.path == "neon_2.5m_linearized/2018_MLBS_3__0_2.tif"
    assert a.dataset_revision == "9f076b4f652aa0253127d382dcb6e611250b2e67" and a.license == "CC-BY-4.0"


def test_sen2neon_provenance_keeps_the_caveats_and_the_acquisition_facts():
    a = neon_records()[0]
    p = a.provenance
    assert p["neon_acquisition_id"] == "2018_MLBS_3" and p["s2_date"] == "2018-07-08" and p["temporal_difference_days"] == 23.0
    assert p["land_cover_superclass"] == "Forest" and p["neon_nodata_percent"] == pytest.approx(48.9368, abs=1e-3)
    assert "NOT measured at 10 m" in p["lr_grid_caveat"] and "AVIRIS-NG" in p["hr_source"]


def test_real_sen2neon_records_pass_the_profile_and_geometry_checks_and_are_test_only():
    for r in neon_records():
        assert validate_record(r) == [], r.sample_id
    assert validate_manifest(neon_records()).ok
    for r in neon_records():
        assert errors(validate_record(dataclasses.replace(r, split=Split.TRAIN)))[0].code == "role_violation"


def test_two_tiles_of_one_acquisition_are_adjacent_scene_mates():
    a, b, _ = neon_records()
    assert a.scene_id == b.scene_id and a.lr.transform != b.lr.transform  # same flight, different ground


def test_a_malformed_metadata_row_is_a_clear_error():
    row = dict(REAL_ROWS["2018_MLBS_3__0_2"])
    with pytest.raises(ContractError) as excinfo:
        sen2neon.record_from_row({k: v for k, v in row.items() if k != "lr_transform"})
    assert excinfo.value.code == "missing_field"
    with pytest.raises(ContractError) as excinfo:
        sen2neon.record_from_row({**row, "lr_transform": "[1, 2"})
    assert excinfo.value.code == "invalid_record"
    with pytest.raises(ContractError) as excinfo:
        sen2neon.record_from_row({**row, "neon_acquisition_id": "bad"})
    assert excinfo.value.code == "invalid_scene_id"


def write_csv(path, rows):
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(next(iter(rows)).keys()))
        writer.writeheader()
        writer.writerows(rows)


def test_records_are_built_from_a_metadata_csv_optionally_restricted_to_some_tiles(tmp_path):
    write_csv(tmp_path / "metadata.csv", list(REAL_ROWS.values()))
    assert len(sen2neon.records_from_metadata_csv(tmp_path / "metadata.csv")) == 3
    picked = sen2neon.records_from_metadata_csv(tmp_path / "metadata.csv", ids=["2022_KONZ_7__5_3"], dataset_revision="abc")
    assert [r.sample_id for r in picked] == ["sen2neon:2022_KONZ_7__5_3"] and picked[0].dataset_revision == "abc"
    with pytest.raises(ContractError, match="not in the metadata"):
        sen2neon.records_from_metadata_csv(tmp_path / "metadata.csv", ids=["2018_MLBS_3__0_2", "nope"])


def test_relative_files_lists_exactly_what_a_sample_download_needs():
    files = sen2neon.relative_files(neon_records())
    assert len(files) == 6 and files == sorted(files) and files[0].startswith("neon_2.5m_linearized/") and files[-1].startswith("s2_l2a_10m/")


def test_checksum_verification_distinguishes_ok_mismatch_missing_and_unlisted(tmp_path):
    (a, b, c) = neon_records()
    directory = tmp_path / "sen2neon"
    (directory / "s2_l2a_10m").mkdir(parents=True)
    good, bad, unlisted = b"good", b"corrupt", b"other"
    for record, content in ((a, good), (b, bad), (c, unlisted)):
        (directory / record.lr.path).write_bytes(content)
    (directory / "s2_l2a_10m.sha256").write_text(
        f"{hashlib.sha256(good).hexdigest()}  {a.lr.path}\n{hashlib.sha256(b'expected').hexdigest()}  {b.lr.path}\n")
    result = sen2neon.verify_lr_checksums([a, b, c], tmp_path)
    assert result == {a.lr.path: "ok", b.lr.path: "mismatch", c.lr.path: "not_listed"}
    (directory / a.lr.path).unlink()
    assert sen2neon.verify_lr_checksums([a], tmp_path)[a.lr.path] == "missing"
    (directory / "s2_l2a_10m.sha256").unlink()
    with pytest.raises(DatasetUnavailableError):
        sen2neon.verify_lr_checksums([a], tmp_path)


def test_fetch_downloads_only_the_named_files_pinned_to_a_revision(tmp_path, monkeypatch):
    import huggingface_hub

    calls = []

    def fake_download(repo, filename, *, repo_type, revision, local_dir):
        calls.append((repo, filename, repo_type, revision))
        target = Path(local_dir) / filename
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"x" * 10)
        return str(target)

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake_download)
    paths, size = sen2neon.fetch_files(neon_records()[:1], tmp_path, revision="REV")
    assert size == 40 and len(paths) == 4  # metadata.csv + checksum manifest + one LR + one HR
    assert {c[1] for c in calls} == {"metadata.csv", "s2_l2a_10m.sha256", "s2_l2a_10m/2018_MLBS_3__0_2.tif", "neon_2.5m_linearized/2018_MLBS_3__0_2.tif"}
    assert {c[0] for c in calls} == {"isp-uv-es/SEN2NEON"} and {c[3] for c in calls} == {"REV"} and {c[2] for c in calls} == {"dataset"}


# ============================================================================== SEN2NAIPv2 (documented format, fixtures)


def naip_row(variant="unet", i=0, **kw):
    row = dict(id=f"roi{i}", variant=variant, scene_id=f"roi{i}", region_id="US-W", split="train", lr_path=f"lr/{i}.tif", hr_path=f"hr/{i}.tif")
    row.update(kw)
    return row


def test_synthetic_sen2naipv2_records_carry_the_upstream_degradation_not_a_frame_one():
    unet, hist = sen2naipv2.record_from_row(naip_row("unet")), sen2naipv2.record_from_row(naip_row("histmatch", 1))
    assert unet.pair_type is PairType.SYNTHETIC and unet.degradation.origin == "upstream" and unet.degradation.harmonisation == "unet"
    assert hist.degradation.harmonisation == "histogram_match" and unet.degradation.name == "sen2naipv2-unet"
    assert unet.degradation.parameters_verified is True and unet.degradation.seed is None
    assert "gaussian_blur" in unet.degradation.config["steps"] and "noise" in unet.degradation.config["steps"]
    assert (unet.lr.height, unet.hr.height, unet.scale_factor) == (130, 520, 4) and unet.lr.band_names == RGBN_BANDS and unet.license == "CC0-1.0"


def test_the_crosssensor_variant_is_real_data_without_a_degradation():
    r = sen2naipv2.record_from_row(naip_row("crosssensor", split="val"))
    assert r.pair_type is PairType.REAL_CROSS_SENSOR and r.degradation is None and errors(validate_record(r)) == []


def test_sen2naipv2_roles_are_enforced_by_qc():
    assert errors(validate_record(sen2naipv2.record_from_row(naip_row("unet", split="test"))))[0].code == "role_violation"        # primary training data
    assert errors(validate_record(sen2naipv2.record_from_row(naip_row("crosssensor", split="train"))))[0].code == "role_violation"  # dev validation only
    assert errors(validate_record(sen2naipv2.record_from_row(naip_row("unet", split="val")))) == []


def test_sen2naipv2_rows_need_the_keys_the_export_must_supply():
    with pytest.raises(ContractError) as excinfo:
        sen2naipv2.record_from_row({k: v for k, v in naip_row().items() if k != "scene_id"})
    assert excinfo.value.code == "missing_field"
    with pytest.raises(ContractError) as excinfo:
        sen2naipv2.record_from_row(naip_row("bogus"))
    assert excinfo.value.code == "unknown_variant"


def test_a_130_by_520_sen2naipv2_pair_reads_and_yields_aligned_128_512_model_patches(tmp_path):
    lr, hr = write_pair_files(tmp_path, "sen2naipv2", "lr/0.tif", "hr/0.tif", lr_hw=(130, 130), seed=3)
    row = naip_row("unet", lr_transform=lr.transform, hr_transform=hr.transform, crs=lr.crs)
    record = sen2naipv2.record_from_row(row)
    assert validate_record(record) == []                                                # incl. exact x4 geometry
    adapter = Sen2NaipV2Adapter([record], data_root=tmp_path)
    sample = adapter.load_pair(record)
    assert sample.lr.shape == (4, 130, 130) and sample.hr.shape == (4, 520, 520)
    r0, c0 = centred_origin(130, 130, 128)
    patch = extract_patch(sample, make_coords(sample, r0, c0, 128))
    assert patch.lr.shape == (4, 128, 128) and patch.hr.shape == (4, 512, 512) and (patch.patch.hr_row, patch.patch.hr_col) == (4, 4)
    # the fixture's LR is the exact block mean of its HR: an aligned HR patch averages back onto its LR patch
    assert torch.allclose(torch.from_numpy(block_mean(patch.hr.numpy(), 4)), patch.lr, atol=2e-4)


# ============================================================================== SEN2VENuS (documented format, fixtures)


def test_the_documented_file_naming_convention():
    assert sen2venus.file_name("FR-LQ1", 12, "2019-05-01", "T31TDH", "b2b3b4b8", "10m") == "FR-LQ1_12_2019-05-01_T31TDH_b2b3b4b8_10m.tif"
    lr, hr = sen2venus.expected_file_names("ARM", 3, "2020-01-02", "T50XXX", "rededge_20m")
    assert lr == "ARM_3_2020-01-02_T50XXX_b5b6b7b8a_20m.tif" and hr == "ARM_3_2020-01-02_T50XXX_b5b6b7b8a_05m.tif"


def venus_row(variant="rgbn_10m", idx=0, **kw):
    row = dict(site="FR-LQ1", date="2019-05-01", mgrs="T31TDH", idx=idx, variant=variant, split="train", lr_path=f"lr/{idx}.tif", hr_path=f"hr/{idx}.tif")
    row.update(kw)
    return row


def test_sen2venus_pairs_have_their_own_scales_and_the_site_is_the_region():
    a, b = sen2venus.record_from_row(venus_row("rgbn_10m")), sen2venus.record_from_row(venus_row("rededge_20m", 1))
    assert (a.scale_factor, a.lr.height, a.hr.height, a.lr.pixel_size_m, a.hr.pixel_size_m) == (2, 128, 256, 10.0, 5.0)
    assert (b.scale_factor, b.lr.height, b.hr.height, b.lr.pixel_size_m, b.hr.pixel_size_m) == (4, 64, 256, 20.0, 5.0)
    assert a.scene_id == "FR-LQ1_2019-05-01_T31TDH" and a.region_id == "FR-LQ1" and a.sample_id == "sen2venus:rgbn_10m:FR-LQ1_0_2019-05-01_T31TDH"
    assert a.pair_type is PairType.REAL_CROSS_SENSOR and errors(validate_record(a)) == [] and errors(validate_record(b)) == []
    assert "NON-COMMERCIAL" in a.license and "Theia" in a.provenance["lr_processing"]


def test_sen2venus_is_supplementary_training_never_test():
    assert errors(validate_record(sen2venus.record_from_row(venus_row(split="test"))))[0].code == "role_violation"
    assert errors(validate_record(sen2venus.record_from_row(venus_row(split="val")))) == []


def test_sen2venus_files_are_in_ascending_band_order_and_reorder_to_rgbn_by_name(tmp_path):
    bands = ("B02", "B03", "B04", "B08")
    lr, hr = write_pair_files(tmp_path, "sen2venus", "lr/0.tif", "hr/0.tif", bands=bands, lr_hw=(128, 128), scale=2, seed=5, dtype="int16", lr_pixel=10.0)
    record = sen2venus.record_from_row(venus_row(lr_transform=lr.transform, hr_transform=hr.transform, crs=lr.crs))
    assert record.lr.band_names == bands and validate_record(record) == []
    adapter = Sen2VenusAdapter([record], data_root=tmp_path)
    native = adapter.load_pair(record)
    rgbn = adapter.load_pair(record, lr_bands=RGBN_BANDS, hr_bands=RGBN_BANDS)
    assert native.lr_bands == bands and rgbn.lr_bands == RGBN_BANDS
    assert torch.equal(rgbn.lr[0], native.lr[2]) and torch.equal(rgbn.lr[2], native.lr[0]) and torch.equal(rgbn.hr[3], native.hr[3])   # B04, B02 swapped; B08 stays last


def test_a_x4_red_edge_pair_reads_at_its_own_scale(tmp_path):
    bands = ("B05", "B06", "B07", "B8A")
    lr, hr = write_pair_files(tmp_path, "sen2venus", "lr/1.tif", "hr/1.tif", bands=bands, lr_hw=(64, 64), scale=4, seed=6, dtype="int16", lr_pixel=20.0)
    record = sen2venus.record_from_row(venus_row("rededge_20m", 1, lr_transform=lr.transform, hr_transform=hr.transform, crs=lr.crs, lr_path="lr/1.tif", hr_path="hr/1.tif"))
    sample = Sen2VenusAdapter([record], data_root=tmp_path).load_pair(record)
    assert sample.lr.shape == (4, 64, 64) and sample.hr.shape == (4, 256, 256) and sample.scale_factor == 4


# ============================================================================== Indian holdout (planned)


def lr_spec_india():
    return RasterSpec(band_names=L2A_BANDS, width=256, height=256, pixel_size_m=10.0, dtype="uint16", reflectance_scale=10000.0,
                      crs="EPSG:32644", transform=(10.0, 0.0, 700000.0, 0.0, -10.0, 1400000.0), path="scenes/tn_001.tif")


def test_an_indian_scene_without_an_hr_reference_states_that_it_does_not_know():
    r = india_holdout.lr_only_record(sample_id="india_holdout:tn_001", scene_id="S2_TN_001", region_id="TamilNadu", lr=lr_spec_india(), lon=78.0, lat=11.0)
    assert r.hr is None and r.hr_status is HRStatus.UNKNOWN and r.pair_type is PairType.INDEPENDENT_HR_REFERENCE and r.split is Split.TEST
    assert validate_record(r) == [] and validate_manifest([r]).ok
    assert PairRecord.from_dict(r.to_dict()) == r


def test_no_hr_label_can_be_invented_for_an_indian_scene():
    with pytest.raises(ContractError) as excinfo:
        india_holdout.lr_only_record(sample_id="india_holdout:x", scene_id="s", region_id="r", lr=lr_spec_india(), hr_status=HRStatus.AVAILABLE)
    assert excinfo.value.code == "invalid_hr_status"
    unavailable = india_holdout.lr_only_record(sample_id="india_holdout:y", scene_id="s", region_id="r", lr=lr_spec_india(), hr_status=HRStatus.UNAVAILABLE)
    assert unavailable.hr_status is HRStatus.UNAVAILABLE


def test_the_indian_holdout_can_never_be_training_data():
    r = india_holdout.lr_only_record(sample_id="india_holdout:tn_001", scene_id="s", region_id="r", lr=lr_spec_india())
    assert errors(validate_record(dataclasses.replace(r, split=Split.TRAIN)))[0].code == "role_violation"


def test_an_lr_only_indian_scene_loads_without_an_hr(tmp_path):
    lr, _ = write_pair_files(tmp_path, "india_holdout", "scenes/tn_001.tif", "unused.tif", bands=L2A_BANDS, lr_hw=(256, 256), scale=4, crs="EPSG:32644",
                             origin=(700000.0, 1400000.0), seed=2)
    r = india_holdout.lr_only_record(sample_id="india_holdout:tn_001", scene_id="s", region_id="r", lr=dataclasses.replace(lr_spec_india(), path="scenes/tn_001.tif"))
    sample = IndiaHoldoutAdapter([r], data_root=tmp_path).load_pair(r)
    assert sample.hr is None and sample.lr.shape == (12, 256, 256) and sample.metadata_dict()["hr_status"] == "unknown"


# ============================================================================== OpenSR-Test (real, cached subset)


OPENSR_CACHE = Path.home() / ".config" / "opensr_test" / "spot.pkl"
needs_opensr = pytest.mark.skipif(not OPENSR_CACHE.exists(), reason="opensr-test 'spot' subset is not cached locally; not downloading")


@pytest.fixture(scope="module")
def spot():
    from frame.validation import load_subset

    return load_subset("spot")


@needs_opensr
def test_opensr_records_describe_the_real_spot_subset(spot):
    records = opensr_test.records_from_loaded(spot, "spot", dataset_revision="e4600b9c")
    assert len(records) == 9 and len({r.sample_id for r in records}) == 9 and len({r.scene_id for r in records}) == 9
    r = records[0]
    assert r.dataset == "opensr_test" and r.variant == "spot" and r.split is Split.TEST and r.pair_type is PairType.REAL_CROSS_SENSOR
    assert r.scale_factor == 4 and r.lr.band_names == L2A_BANDS and r.hr.band_names == RGBN_BANDS
    assert (r.lr.height, r.hr.height) == (128, 512) and (r.lr.pixel_size_m, r.hr.pixel_size_m) == (10.0, 2.5)
    assert r.lr.crs.startswith("EPSG:") and r.lr.transform is not None and -180 <= r.lon <= 180 and -90 <= r.lat <= 90
    assert r.provenance["locator"]["subset"] == "spot" and "hr_band_order_note" in r.provenance
    assert all(validate_record(x) == [] for x in records)


@needs_opensr
def test_the_lr_grid_is_derived_from_the_hr_grid_exactly_as_the_existing_adapter_does(spot):
    from frame.validation.opensr_test_adapter import _derive_lr_transform, _parse_affine

    for i, record in enumerate(opensr_test.records_from_loaded(spot, "spot")):
        hr_t = _parse_affine(str(spot["metadata"].iloc[i]["affine"]))
        assert record.hr.transform == hr_t and record.lr.transform == pytest.approx(_derive_lr_transform(hr_t, 4))


@needs_opensr
def test_the_common_interface_reproduces_the_existing_extract_sample_exactly(spot):
    """Compatibility: selecting RGBN through the new interface gives the SAME tensors the Phase 4 adapter produces."""
    from frame.validation import extract_sample

    records = opensr_test.records_from_loaded(spot, "spot")
    adapter = OpenSRTestAdapter(records, loaded_subsets={"spot": spot})
    for i in (0, 4, 8):
        old = extract_sample(spot, subset="spot", sample_index=i)
        new = adapter.load_pair(records[i], lr_bands=RGBN_BANDS)
        assert new.lr_bands == RGBN_BANDS and torch.equal(new.lr, old.lr_reflectance) and torch.equal(new.hr, old.hr_reflectance)
        assert new.record.scale_factor == old.scale_factor


@needs_opensr
def test_the_new_interface_keeps_all_twelve_lr_bands_for_the_future_cascade(spot):
    records = opensr_test.records_from_loaded(spot, "spot")
    sample = OpenSRTestAdapter(records, loaded_subsets={"spot": spot}).load_pair(records[0])
    assert sample.lr.shape == (12, 128, 128) and sample.lr_bands == L2A_BANDS and sample.hr.shape == (4, 512, 512)


@needs_opensr
def test_opensr_pixels_pass_qc_and_the_manifest_round_trips(spot, tmp_path):
    records = opensr_test.records_from_loaded(spot, "spot")
    adapter = OpenSRTestAdapter(records, loaded_subsets={"spot": spot})
    report = validate_manifest(records, loader=adapter.load_pair)
    assert report.ok and (report.total, report.valid) == (9, 9), report.summary()
    write_manifest(tmp_path / "opensr.jsonl", records)
    assert OpenSRTestAdapter.from_manifest(tmp_path / "opensr.jsonl", loaded_subsets={"spot": spot}).list_scenes() == sorted(r.scene_id for r in records)


@needs_opensr
def test_opensr_benchmark_samples_cannot_be_used_for_training(spot):
    r = opensr_test.records_from_loaded(spot, "spot")[0]
    assert errors(validate_record(dataclasses.replace(r, split=Split.TRAIN)))[0].code == "role_violation"


def test_unsupported_opensr_subsets_are_refused_as_before():
    with pytest.raises(ContractError) as excinfo:
        opensr_test.records_from_loaded({"L2A": None}, "naip")
    assert excinfo.value.code == "unknown_variant"
