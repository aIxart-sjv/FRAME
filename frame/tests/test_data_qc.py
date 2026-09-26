"""frame.data.qc -- record / file / pixel / manifest validation and the machine-readable report."""

from __future__ import annotations

import dataclasses
import json

import numpy as np
import pytest
import torch

from frame.data.contract import PairedSample, PairRecord, PairType, Split
from frame.data.manifest import write_manifest
from frame.data.qc import qc_manifest_file, validate_manifest, validate_record, validate_sample, write_report
from frame.preprocessing import RGBN_BANDS
from frame.tests.data_fixtures import (  # noqa: F401  (fixtures are used by name)
    TINY,
    TinyEnv,
    tiny_env,
    tiny_profile_installed,
    write_raster,
)


def codes(issues):
    return sorted(i.code for i in issues)


def replace_lr(record: PairRecord, **kw) -> PairRecord:
    return dataclasses.replace(record, lr=dataclasses.replace(record.lr, **kw))


def replace_hr(record: PairRecord, **kw) -> PairRecord:
    return dataclasses.replace(record, hr=dataclasses.replace(record.hr, **kw))


# ------------------------------------------------------------------------------ a clean dataset


def test_a_clean_dataset_passes_every_layer(tiny_env: TinyEnv):
    report = validate_manifest(tiny_env.records, data_root=tiny_env.root, check_headers=True, loader=tiny_env.adapter().load_pair)
    assert report.ok and (report.total, report.valid, report.invalid) == (8, 8, 0) and report.issues == ()
    assert report.per_split == {"train": 4, "val": 4} and report.per_dataset == {"tinyset/default": 8}
    assert report.checks == {"records": True, "files": True, "headers": True, "pixels": True, "split_integrity": True}
    assert report.manifest_digest and report.invalid_ids == ()


def test_only_the_layers_that_were_asked_for_are_reported_as_run(tiny_env: TinyEnv):
    report = validate_manifest(tiny_env.records)
    assert report.checks == {"records": True, "files": False, "headers": False, "pixels": False, "split_integrity": True}


# ------------------------------------------------------------------------------ manifest-level


def test_duplicate_ids_make_every_copy_invalid(tiny_env: TinyEnv):
    records = tiny_env.records + [tiny_env.records[0]]
    report = validate_manifest(records)
    assert report.counts_by_code()["duplicate_id"] == 1
    assert report.total == 9 and report.invalid == 2 and report.valid == 7
    assert report.invalid_ids == (tiny_env.records[0].sample_id,)


def test_missing_files_are_reported_per_sample(tiny_env: TinyEnv):
    victim = tiny_env.records[2]
    (tiny_env.root / TINY / victim.lr.path).unlink()
    (tiny_env.root / TINY / tiny_env.records[5].hr.path).unlink()
    report = validate_manifest(tiny_env.records, data_root=tiny_env.root)
    missing = [i for i in report.issues if i.code == "missing_file"]
    assert {i.sample_ids[0] for i in missing} == {victim.sample_id, tiny_env.records[5].sample_id}
    assert report.invalid == 2 and report.valid == 6 and not report.ok
    assert "LR" in [i for i in missing if i.sample_ids[0] == victim.sample_id][0].message


def test_file_checks_are_skipped_without_a_data_root(tiny_env: TinyEnv):
    (tiny_env.root / TINY / tiny_env.records[0].lr.path).unlink()
    assert validate_manifest(tiny_env.records).ok  # nothing looks at the disk


def test_split_leakage_in_a_manifest_marks_the_involved_samples_invalid(tiny_env: TinyEnv):
    leaky = list(tiny_env.records)
    leaky[0] = dataclasses.replace(leaky[0], split=Split.VAL)  # one tile of a scene moved to another split
    report = validate_manifest(leaky)
    assert report.counts_by_code()["scene_leakage"] == 1 and report.counts_by_code()["region_leakage"] == 1
    assert report.invalid > 0 and not report.ok and leaky[0].sample_id in report.invalid_ids
    assert validate_manifest(leaky, check_split=False).ok


def test_role_violation_is_reported(tiny_env: TinyEnv):
    records = [dataclasses.replace(r, split=Split.TEST) for r in tiny_env.records[:2]]  # the tiny profile allows only train/val
    report = validate_manifest(records)
    assert report.counts_by_code()["role_violation"] >= 2 and report.invalid == 2


# ------------------------------------------------------------------------------ profile checks


@pytest.mark.parametrize(
    "mutate,code",
    [
        (lambda r: replace_lr(r, band_names=tuple(reversed(RGBN_BANDS))), "band_mismatch"),        # bands must match the documented ORDER
        (lambda r: replace_lr(r, band_names=("B04", "B03", "B02")), "band_mismatch"),
        (lambda r: replace_hr(r, band_names=("B02", "B03", "B04", "B08")), "band_mismatch"),
        (lambda r: replace_lr(r, pixel_size_m=20.0), "resolution_mismatch"),
        (lambda r: replace_hr(r, pixel_size_m=5.0), "resolution_mismatch"),
        (lambda r: dataclasses.replace(r, scale_factor=2), "scale_mismatch"),
        (lambda r: dataclasses.replace(r, pair_type=PairType.INDEPENDENT_HR_REFERENCE), "pair_type_mismatch"),
    ],
)
def test_records_that_contradict_their_profile_are_flagged(tiny_env: TinyEnv, mutate, code):
    issues = validate_record(mutate(tiny_env.records[0]))
    assert code in codes(issues) and all(i.severity == "error" for i in issues if i.code == code)


def test_wrong_tile_sizes_are_flagged(tiny_env: TinyEnv):
    r = tiny_env.records[0]
    bad = dataclasses.replace(r, lr=dataclasses.replace(r.lr, width=64, height=64), hr=dataclasses.replace(r.hr, width=256, height=256))
    assert "shape_mismatch" in codes(validate_record(bad))


def test_an_unknown_dataset_or_variant_is_flagged(tiny_env: TinyEnv):
    r = tiny_env.records[0]
    assert codes(validate_record(dataclasses.replace(r, dataset="nowhere"))) == ["unknown_dataset"]
    assert codes(validate_record(dataclasses.replace(r, variant="v9"))) == ["unknown_variant"]


# ------------------------------------------------------------------------------ geometry


def test_a_shifted_hr_transform_in_the_record_is_a_footprint_mismatch(tiny_env: TinyEnv):
    r = tiny_env.records[0]
    t = list(r.hr.transform)
    t[2] += 2.5  # one HR pixel east
    assert "footprint_mismatch" in codes(validate_record(replace_hr(r, transform=tuple(t))))


def test_crs_resolution_and_dimension_problems_are_named(tiny_env: TinyEnv):
    r = tiny_env.records[0]
    assert "crs_mismatch" in codes(validate_record(replace_hr(r, crs="EPSG:32631")))
    assert "resolution_mismatch" in codes(validate_record(replace_hr(r, transform=(2.0, 0, r.hr.transform[2], 0, -2.0, r.hr.transform[5]))))
    assert "dimension_mismatch" in codes(validate_record(replace_hr(r, width=120)))


def test_a_real_pair_without_georeferencing_gets_a_warning_not_an_error(tiny_env: TinyEnv):
    r = tiny_env.records[0]
    stripped = dataclasses.replace(r, lr=dataclasses.replace(r.lr, crs=None, transform=None), hr=dataclasses.replace(r.hr, crs=None, transform=None))
    issues = validate_record(stripped)
    assert codes(issues) == ["not_georeferenced"] and issues[0].severity == "warning"


# ------------------------------------------------------------------------------ headers of real files


def rewrite(env: TinyEnv, record: PairRecord, which: str, **kw):
    spec = getattr(record, which)
    data = kw.pop("data", np.full((len(spec.band_names), spec.height, spec.width), 3000, dtype="uint16"))
    write_raster(env.root / TINY / spec.path, data, transform=kw.pop("transform", spec.transform), crs=kw.pop("crs", spec.crs),
                 dtype=kw.pop("dtype", "uint16"), nodata=None)


def header_codes(env: TinyEnv, record: PairRecord):
    return codes([i for i in validate_manifest([record], data_root=env.root, check_headers=True, check_split=False).issues])


def test_a_file_whose_header_disagrees_with_the_record_is_caught(tiny_env: TinyEnv):
    r = tiny_env.records[0]
    rewrite(tiny_env, r, "lr", data=np.full((3, 32, 32), 3000, dtype="uint16"))
    assert "band_count_mismatch" in header_codes(tiny_env, r)
    rewrite(tiny_env, r, "lr", data=np.full((4, 16, 16), 3000, dtype="uint16"))
    assert "shape_mismatch" in header_codes(tiny_env, r)
    rewrite(tiny_env, r, "lr", data=np.full((4, 32, 32), 0.3, dtype="float32"), dtype="float32")
    assert "dtype_mismatch" in header_codes(tiny_env, r)
    rewrite(tiny_env, r, "lr", crs="EPSG:32631")
    assert "crs_mismatch" in header_codes(tiny_env, r)
    rewrite(tiny_env, r, "lr", transform=(10.0, 0, r.lr.transform[2] + 100.0, 0, -10.0, r.lr.transform[5]))
    assert "transform_mismatch" in header_codes(tiny_env, r)


def test_an_unreadable_file_is_reported_not_raised(tiny_env: TinyEnv):
    r = tiny_env.records[1]
    (tiny_env.root / TINY / r.hr.path).write_bytes(b"this is not a tiff")
    report = validate_manifest([r], data_root=tiny_env.root, check_headers=True, check_split=False)
    assert "unreadable_file" in report.counts_by_code() and report.invalid == 1


def test_headers_are_only_compared_when_asked_for(tiny_env: TinyEnv):
    r = tiny_env.records[0]
    rewrite(tiny_env, r, "lr", data=np.full((3, 32, 32), 3000, dtype="uint16"))
    assert validate_manifest([r], data_root=tiny_env.root, check_split=False).ok


# ------------------------------------------------------------------------------ pixels


def one_sample(env: TinyEnv, lr=None, hr=None, lr_mask=None, hr_mask=None) -> PairedSample:
    base = env.adapter().load_pair(env.records[0])
    return dataclasses.replace(base, lr=lr if lr is not None else base.lr, hr=hr if hr is not None else base.hr,
                               lr_mask=lr_mask if lr_mask is not None else base.lr_mask, hr_mask=hr_mask if hr_mask is not None else base.hr_mask)


def test_nan_and_inf_are_errors_never_repaired(tiny_env: TinyEnv):
    bad = tiny_env.adapter().load_pair(tiny_env.records[0]).lr.clone()
    bad[1, 3, 3] = float("nan")
    assert codes(validate_sample(one_sample(tiny_env, lr=bad))) == ["non_finite"]
    bad[1, 3, 3] = float("inf")
    assert codes(validate_sample(one_sample(tiny_env, lr=bad))) == ["non_finite"]


def test_out_of_range_reflectance_signals_a_wrong_scale(tiny_env: TinyEnv):
    """Values still in raw digital numbers (x10000) are flagged; so are impossible negatives."""
    s = tiny_env.adapter().load_pair(tiny_env.records[0])
    assert codes(validate_sample(one_sample(tiny_env, lr=s.lr * 10000.0))) == ["invalid_range"]
    assert codes(validate_sample(one_sample(tiny_env, hr=s.hr - 5.0))) == ["invalid_range"]
    assert "divid" not in validate_sample(one_sample(tiny_env, lr=s.lr * 10000.0))[0].message  # (message asks about the scale)
    assert "scale" in validate_sample(one_sample(tiny_env, lr=s.lr * 10000.0))[0].message


def test_a_mostly_nodata_sample_is_a_warning_and_an_all_nodata_sample_is_an_error(tiny_env: TinyEnv):
    s = tiny_env.adapter().load_pair(tiny_env.records[0])
    mostly = torch.zeros_like(s.lr_mask)
    mostly[:6, :6] = True
    (issue,) = validate_sample(one_sample(tiny_env, lr_mask=mostly))
    assert issue.code == "high_nodata" and issue.severity == "warning"
    (none,) = validate_sample(one_sample(tiny_env, lr_mask=torch.zeros_like(s.lr_mask)))
    assert none.code == "no_valid_pixels" and none.severity == "error"


def test_a_constant_image_is_a_warning(tiny_env: TinyEnv):
    s = tiny_env.adapter().load_pair(tiny_env.records[0])
    (issue,) = validate_sample(one_sample(tiny_env, lr=torch.full_like(s.lr, 0.2)))
    assert issue.code == "constant_image" and issue.severity == "warning"


def test_nodata_pixels_do_not_count_toward_the_range_check(tiny_env: TinyEnv):
    s = tiny_env.adapter().load_pair(tiny_env.records[0])
    lr, mask = s.lr.clone(), s.lr_mask.clone()
    lr[:, 0, 0] = 9999.0
    mask[0, 0] = False
    assert validate_sample(one_sample(tiny_env, lr=lr, lr_mask=mask)) == []


def test_a_sample_that_cannot_be_loaded_is_a_finding_not_a_crash(tiny_env: TinyEnv):
    def flaky(record):
        if record.sample_id == tiny_env.records[3].sample_id:
            raise RuntimeError("disk error")
        return tiny_env.adapter().load_pair(record)

    report = validate_manifest(tiny_env.records, loader=flaky, check_split=False)
    (issue,) = [i for i in report.issues if i.code == "load_failed"]
    assert "disk error" in issue.message and report.invalid == 1 and report.valid == 7


def test_pixel_problems_in_real_files_are_found_end_to_end(tiny_env: TinyEnv):
    r = tiny_env.records[0]
    rewrite(tiny_env, r, "lr", data=np.full((4, 32, 32), 60000, dtype="uint16"))  # 6.0 reflectance: still in range...
    assert validate_manifest([r], loader=tiny_env.adapter().load_pair, check_split=False).ok
    rewrite(tiny_env, r, "lr", data=np.full((4, 32, 32), 65535, dtype="uint16"))  # ...but 6.5535 is the ceiling
    rewrite(tiny_env, r, "hr", data=np.full((4, 128, 128), 1, dtype="uint16"))
    report = validate_manifest([r], loader=tiny_env.adapter().load_pair, check_split=False)
    assert {"constant_image"} <= set(report.counts_by_code())


# ------------------------------------------------------------------------------ report & file entry point


def test_the_report_is_machine_readable_and_complete(tiny_env: TinyEnv, tmp_path):
    records = tiny_env.records + [tiny_env.records[0]]
    (tiny_env.root / TINY / tiny_env.records[4].lr.path).unlink()
    report = validate_manifest(records, data_root=tiny_env.root)
    out = tmp_path / "report.json"
    write_report(report, out)
    data = json.loads(out.read_text())
    assert data["ok"] is False and data["total_samples"] == 9 and data["invalid_samples"] == 3 and data["valid_samples"] == 6
    assert data["issue_counts"] == {"duplicate_id": 1, "missing_file": 1}
    assert data["per_split"] == {"train": 5, "val": 4} and data["checks_performed"]["files"] is True
    assert {i["code"] for i in data["issues"]} == {"duplicate_id", "missing_file"}
    assert data["manifest_digest"] == report.manifest_digest and len(data["invalid_sample_ids"]) == 2
    assert "3 invalid" in report.summary() and "duplicate_id: 1" in report.summary()


def test_warnings_do_not_fail_a_manifest(tiny_env: TinyEnv):
    r = tiny_env.records[0]
    stripped = dataclasses.replace(r, lr=dataclasses.replace(r.lr, crs=None, transform=None), hr=dataclasses.replace(r.hr, crs=None, transform=None))
    report = validate_manifest([stripped])
    assert report.ok and report.valid == 1 and report.counts_by_code() == {"not_georeferenced": 1}


def test_qc_manifest_file_reports_every_bad_line_and_counts_them_as_invalid(tiny_env: TinyEnv, tmp_path):
    write_manifest(tmp_path / "m.jsonl", tiny_env.records)
    lines = (tmp_path / "m.jsonl").read_text().splitlines()
    lines[3] = "{broken"
    lines[5] = json.dumps({**json.loads(lines[5]), "scale_factor": 0})
    (tmp_path / "bad.jsonl").write_text("\n".join(lines) + "\n")
    report = qc_manifest_file(tmp_path / "bad.jsonl")
    assert report.counts_by_code() == {"invalid_json": 1, "invalid_scale": 1}
    assert report.total == 8 and report.invalid == 2 and report.valid == 6 and not report.ok


def test_an_empty_manifest_is_valid_but_empty():
    report = validate_manifest([])
    assert report.ok and (report.total, report.valid, report.invalid) == (0, 0, 0) and report.manifest_digest is None


def test_the_largest_legitimate_reflectance_is_not_flagged_by_float32_rounding(tiny_env: TinyEnv):
    """DN 65535 -> 6.5535 is 6.5535002 in float32; comparing with the double-precision bound would reject it."""
    s = tiny_env.adapter().load_pair(tiny_env.records[0])
    lr = s.lr.clone()
    lr[0, 0, 0] = 65535 / 10000.0
    lr[1, 0, 0] = -0.1
    assert "invalid_range" not in codes(validate_sample(one_sample(tiny_env, lr=lr)))
    lr[0, 0, 0] = 6.56
    assert codes(validate_sample(one_sample(tiny_env, lr=lr))) == ["invalid_range"]
