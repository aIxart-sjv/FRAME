"""frame.evaluate.datasets -- reference/geometry validation, role safety, the strict evaluation mask, scene units."""

from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
import pytest
import torch

from frame.data.adapters import sen2neon
from frame.data.adapters.synthetic import build_synthetic_dataset
from frame.data.contract import Split
from frame.data.manifest import manifest_digest, write_manifest
from frame.evaluate.config import DatasetSpec
from frame.evaluate.datasets import build_dataset, evaluation_mask, opensr_source_group
from frame.evaluate.errors import EvaluationError, RoleSafetyError
from frame.tests.data_real_rows import REAL_ROWS

REPO = Path(__file__).resolve().parents[2]


def neon_records():
    return [sen2neon.record_from_row(r) for r in REAL_ROWS.values()]


def write(tmp_path, records, name="m.jsonl"):
    write_manifest(tmp_path / name, records)
    return str(tmp_path / name)


def synthetic_manifest(tmp_path):
    records = build_synthetic_dataset(tmp_path / "data", n_regions=3, scenes_per_region=2, seed=0)
    split = {"R00": Split.TRAIN, "R01": Split.VAL, "R02": Split.TEST}
    records = [dataclasses.replace(r, split=split[r.region_id]) for r in records]
    return write(tmp_path, records), records


# ============================================================================== synthetic (held-out test split of the smoke set)


def test_a_synthetic_dataset_yields_only_its_test_split_and_says_synthetic(tmp_path):
    manifest, records = synthetic_manifest(tmp_path)
    ds = build_dataset(DatasetSpec(name="syn", kind="synthetic_smoke", manifest=manifest, data_root=str(tmp_path / "data")))
    assert [r.region_id for r in ds.records] == ["R02", "R02"] and ds.evidence_class == "synthetic" and ds.ignored == {"train": 2, "val": 2}
    assert ds.manifest_digest == manifest_digest(records) and ds.role == "smoke_test"


def test_samples_carry_bands_masks_scene_group_and_provenance(tmp_path):
    manifest, _ = synthetic_manifest(tmp_path)
    ds = build_dataset(DatasetSpec(name="syn", kind="synthetic_smoke", manifest=manifest, data_root=str(tmp_path / "data")))
    s = ds.load(ds.records[0])
    assert s.lr.shape == (4, 128, 128) and s.hr.shape == (4, 512, 512) and s.bands == ("B04", "B03", "B02", "B08") and s.scale == 4
    assert bool(s.hr_mask.all()) and bool(s.lr_mask.all()) and s.scene_group == "R02" and s.category is None and s.evidence_class == "synthetic"
    assert s.sample_id == ds.records[0].sample_id and s.provenance["degradation"]["origin"] == "frame" and s.provenance["dataset"] == "synthetic_smoke"


def test_a_manifest_edited_after_writing_is_refused(tmp_path):
    manifest, _ = synthetic_manifest(tmp_path)
    lines = Path(manifest).read_text().splitlines()
    lines[1] = lines[1].replace('"split":"train"', '"split":"test"')
    Path(manifest).write_text("\n".join(lines) + "\n")
    with pytest.raises(EvaluationError, match="digest"):
        build_dataset(DatasetSpec(name="syn", kind="synthetic_smoke", manifest=manifest, data_root=str(tmp_path / "data")))


def test_a_synthetic_manifest_without_test_records_is_an_error(tmp_path):
    records = [dataclasses.replace(r, split=Split.TRAIN) for r in build_synthetic_dataset(tmp_path / "d", n_regions=2, scenes_per_region=1, seed=0)]
    with pytest.raises(EvaluationError, match="no records"):
        build_dataset(DatasetSpec(name="syn", kind="synthetic_smoke", manifest=write(tmp_path, records), data_root=str(tmp_path / "d")))


# ============================================================================== benchmark role safety


def test_benchmark_records_marked_train_or_val_refuse_the_whole_evaluation(tmp_path):
    for bad in (Split.TRAIN, Split.VAL):
        recs = neon_records()
        recs[1] = dataclasses.replace(recs[1], split=bad)
        with pytest.raises(RoleSafetyError) as excinfo:
            build_dataset(DatasetSpec(name="neon", kind="sen2neon", manifest=write(tmp_path, recs, f"{bad.value}.jsonl")))
        assert "role_violation" in excinfo.value.codes and recs[1].sample_id in str(excinfo.value)


def test_a_training_dataset_cannot_be_evaluated_as_a_benchmark(tmp_path):
    """A manifest of a dataset whose role is training data (here a SEN2NAIPv2 record) is refused even if its split says test."""
    from frame.data.adapters import sen2naipv2

    row = dict(id="roi0", variant="unet", scene_id="roi0", region_id="US-W", split="test", lr_path="lr/0.tif", hr_path="hr/0.tif")
    with pytest.raises(RoleSafetyError) as excinfo:
        build_dataset(DatasetSpec(name="neon", kind="sen2neon", manifest=write(tmp_path, [sen2naipv2.record_from_row(row)])))
    assert "not_an_evaluation_dataset" in excinfo.value.codes


def test_a_manifest_for_a_different_dataset_kind_is_refused(tmp_path):
    manifest, _ = synthetic_manifest(tmp_path)
    with pytest.raises(RoleSafetyError, match="sen2neon"):
        build_dataset(DatasetSpec(name="neon", kind="sen2neon", manifest=manifest))


# ============================================================================== geometry / reference validation


def test_a_shifted_hr_footprint_is_reported_invalid_not_scored(tmp_path):
    recs = neon_records()
    t = list(recs[1].hr.transform)
    t[2] += 2.5                                                        # one HR pixel east
    recs[1] = dataclasses.replace(recs[1], hr=dataclasses.replace(recs[1].hr, transform=tuple(t)))
    ds = build_dataset(DatasetSpec(name="neon", kind="sen2neon", manifest=write(tmp_path, recs)))
    v = ds.validate(check_files=False)
    assert [i["sample_id"] for i in v.invalid] == [recs[1].sample_id] and "footprint_mismatch" in v.invalid[0]["codes"]
    assert len(v.valid_records) == 2 and recs[1].sample_id not in [r.sample_id for r in v.valid_records]


def test_a_crs_or_scale_mismatch_is_reported_with_its_code(tmp_path):
    recs = neon_records()
    recs[0] = dataclasses.replace(recs[0], hr=dataclasses.replace(recs[0].hr, crs="EPSG:32633"))
    ds = build_dataset(DatasetSpec(name="neon", kind="sen2neon", manifest=write(tmp_path, recs)))
    assert "crs_mismatch" in ds.validate(check_files=False).invalid[0]["codes"]


def test_missing_files_are_invalid_samples_with_a_reason(tmp_path):
    ds = build_dataset(DatasetSpec(name="neon", kind="sen2neon", manifest=write(tmp_path, neon_records()), data_root=str(tmp_path / "nowhere")))
    v = ds.validate(check_files=True)
    assert len(v.invalid) == 3 and all("missing_file" in i["codes"] for i in v.invalid) and not v.valid_records


def test_valid_records_pass_without_files_when_files_are_not_checked(tmp_path):
    v = build_dataset(DatasetSpec(name="neon", kind="sen2neon", manifest=write(tmp_path, neon_records()))).validate(check_files=False)
    assert len(v.valid_records) == 3 and v.invalid == []


# ============================================================================== the strict evaluation mask


def test_the_evaluation_mask_also_excludes_pixels_with_a_zero_band():
    """The dataset's nodata rule is 'every band is nodata'; a spectrum with a zero in ONE evaluated band is not a valid spectrum for SAM / ERGAS / indices."""
    hr = torch.full((4, 8, 8), 0.2)
    dataset_mask = torch.ones(8, 8, dtype=torch.bool)
    dataset_mask[0] = False                                   # dataset nodata row
    hr[:, 0] = 0.0
    hr[1, 3, 3] = 0.0                                         # a partial zero at a 'valid' pixel
    hr[:, 5, 5] = 0.0                                         # an all-zero pixel the dataset rule kept (RGBN all zero)
    strict, info = evaluation_mask(hr, dataset_mask, hr_nodata=0.0, reflectance_scale=10_000.0)
    assert not bool(strict[0].any()) and not bool(strict[3, 3]) and not bool(strict[5, 5]) and bool(strict[1, 1])
    assert info["hr_dataset_nodata_fraction"] == pytest.approx(8 / 64) and info["hr_partial_nodata_fraction"] == pytest.approx(2 / 64)
    assert info["hr_valid_fraction"] == pytest.approx(54 / 64) and info["rule"].startswith("HR pixel valid iff")


def test_without_a_declared_nodata_value_only_the_dataset_mask_applies():
    hr = torch.zeros(4, 4, 4)
    strict, info = evaluation_mask(hr, torch.ones(4, 4, dtype=torch.bool), hr_nodata=None, reflectance_scale=10_000.0)
    assert bool(strict.all()) and info["hr_partial_nodata_fraction"] == 0.0


def test_non_finite_reference_values_are_never_valid():
    hr = torch.full((4, 4, 4), 0.2)
    hr[2, 1, 1] = float("nan")
    strict, _ = evaluation_mask(hr, torch.ones(4, 4, dtype=torch.bool), hr_nodata=None, reflectance_scale=10_000.0)
    assert not bool(strict[1, 1]) and int(strict.sum()) == 15


# ============================================================================== scene units


def test_opensr_test_source_scene_is_parsed_from_the_reference_file_name():
    assert opensr_source_group("spain_crops", "ROI_00001", "HR__ROI_00001__PNOA_ANUAL_2021_OF_ETRS89_HU30_h25_0790-2.tif") == "PNOA_ANUAL_2021_OF_ETRS89_HU30_h25_0790-2"
    assert opensr_source_group("spain_urban", "ROI_00014", "HR__ROI_00014__PNOA_ANUAL_2020_OF_ETRS89_HU30_h25_0372-1.tif") == "PNOA_ANUAL_2020_OF_ETRS89_HU30_h25_0372-1"
    assert opensr_source_group("spot", "ROI_0037", "ROI_0037__IMG_SPOT6_MS_201702050755046_ORT_x_R1C1.tif") == "spot_ROI_0037"      # each SPOT ROI is its own scene
    assert opensr_source_group("spain_crops", "ROI_00099", None) == "spain_crops_ROI_00099"                                        # no file name: the ROI is the unit


def test_neon_scene_is_the_acquisition_and_category_is_the_datasets_own_land_cover_label(tmp_path):
    ds = build_dataset(DatasetSpec(name="neon", kind="sen2neon", manifest=write(tmp_path, neon_records())))
    groups = {r.sample_id: ds.scene_group_of(r) for r in ds.records}
    assert groups["sen2neon:2018_MLBS_3__0_2"] == groups["sen2neon:2018_MLBS_3__1_1"] == "2018_MLBS_3" and groups["sen2neon:2022_KONZ_7__5_3"] == "2022_KONZ_7"
    assert ds.category_of(ds.records[0]) in ("Forest", "Rural", "Developed", "Water", "unlabelled")


# ============================================================================== real data (skipped when it has not been fetched)


NEON_MANIFEST = REPO / "experiments" / "evaluation" / "manifests" / "sen2neon_random30_seed0.jsonl"
needs_neon = pytest.mark.skipif(not NEON_MANIFEST.is_file() or not (Path.home() / ".cache" / "frame_data" / "sen2neon" / "metadata.csv").is_file(),
                                reason="the random-30 SEN2NEON sample manifest / files are not present")


@needs_neon
def test_the_real_random_sen2neon_sample_validates_and_reports_both_nodata_fractions():
    ds = build_dataset(DatasetSpec(name="neon", kind="sen2neon", manifest=str(NEON_MANIFEST)))
    v = ds.validate(check_files=True)
    assert len(v.valid_records) == 30 and v.invalid == [] and len({ds.scene_group_of(r) for r in ds.records}) == 28
    seen_partial = False
    for r in ds.records:
        s = ds.load(r)
        assert s.lr.shape == (4, 256, 256) and s.hr.shape == (4, 1024, 1024) and s.hr_mask.dtype == torch.bool
        assert bool(s.lr_mask.all()) and torch.isfinite(s.hr[:, s.hr_mask]).all() and bool((s.hr[:, s.hr_mask] > 0).all())     # no zero band inside the evaluation mask
        seen_partial |= s.quality["hr_partial_nodata_fraction"] > 0
    assert seen_partial                                                       # the stricter rule really matters on this data


OPENSR_CACHE = Path.home() / ".config" / "opensr_test" / "spot.pkl"
needs_opensr = pytest.mark.skipif(not OPENSR_CACHE.exists(), reason="opensr-test 'spot' subset is not cached locally; not downloading")


@needs_opensr
def test_opensr_test_samples_load_with_all_valid_masks_when_the_adapter_provides_none():
    """The OpenSR-Test adapter has no nodata and returns no masks; the evaluation treats that as 'every pixel is a real observation'."""
    ds = build_dataset(DatasetSpec(name="spot", kind="opensr_test", subset="spot"))
    assert len(ds.records) == 9 and ds.evidence_class == "real_cross_sensor" and ds.role == "independent_benchmark" and ds.ignored == {}
    s = ds.load(ds.records[0])
    assert s.lr.shape == (4, 128, 128) and s.hr.shape == (4, 512, 512) and bool(s.hr_mask.all()) and bool(s.lr_mask.all()) and s.quality["hr_valid_fraction"] == 1.0
    assert s.scene_group == "spot_ROI_0037" and s.category == "spot_mixed" and s.bands == ("B04", "B03", "B02", "B08") and s.provenance["locator"]["hr_variant"] == "HRharm"
    from frame.validation import extract_sample, load_subset

    old = extract_sample(load_subset("spot"), subset="spot", sample_index=0)
    assert torch.equal(s.lr, old.lr_reflectance) and torch.equal(s.hr, old.hr_reflectance)              # the same tensors the existing validation layer feeds its models


@needs_opensr
def test_opensr_test_source_scenes_are_derived_from_the_datasets_own_file_names_for_the_spain_subsets():
    if not (Path.home() / ".config" / "opensr_test" / "spain_crops.pkl").exists():
        pytest.skip("spain_crops not cached")
    ds = build_dataset(DatasetSpec(name="crops", kind="opensr_test", subset="spain_crops"))
    groups = {ds.scene_group_of(r) for r in ds.records}
    assert len(ds.records) == 28 and len(groups) == 5 and all(g.startswith("PNOA_ANUAL_") for g in groups) and ds.category_of(ds.records[0]) == "crops"
