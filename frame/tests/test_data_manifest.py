"""frame.data.manifest -- the reproducible JSONL manifest format."""

from __future__ import annotations

import json
import random

import pytest

from frame.data.contract import PairRecord, RasterSpec
from frame.data.errors import ManifestError
from frame.data.manifest import MANIFEST_KIND, MANIFEST_VERSION, canonical_json, manifest_digest, read_manifest, write_manifest
from frame.preprocessing import RGBN_BANDS


def rec(i: int, **kw) -> PairRecord:
    base = dict(sample_id=f"tinyset:s{i:03d}", dataset="tinyset", scene_id=f"scene{i // 2}", region_id=f"reg{i // 4}", split="train",
                pair_type="real_cross_sensor", hr_status="available", scale_factor=4,
                lr=RasterSpec(band_names=RGBN_BANDS, width=32, height=32, path=f"lr/{i}.tif"),
                hr=RasterSpec(band_names=RGBN_BANDS, width=128, height=128, path=f"hr/{i}.tif"))
    base.update(kw)
    return PairRecord(**base)


RECORDS = [rec(i) for i in range(8)]


def test_write_then_read_round_trips_every_record(tmp_path):
    write_manifest(tmp_path / "m.jsonl", RECORDS)
    contents = read_manifest(tmp_path / "m.jsonl")
    assert contents.records == sorted(RECORDS, key=lambda r: r.sample_id) and contents.issues == []


def test_the_header_describes_the_manifest(tmp_path):
    digest = write_manifest(tmp_path / "m.jsonl", RECORDS, header={"note": "unit test"})
    header = read_manifest(tmp_path / "m.jsonl").header
    assert header["kind"] == MANIFEST_KIND and header["manifest_version"] == MANIFEST_VERSION
    assert header["n_records"] == 8 and header["digest"] == digest and header["note"] == "unit test"
    assert header["datasets"] == {"tinyset": {"variants": ["default"], "revisions": []}}


def test_output_is_byte_reproducible_whatever_the_input_order(tmp_path):
    shuffled = list(RECORDS)
    random.Random(1).shuffle(shuffled)
    d1 = write_manifest(tmp_path / "a.jsonl", RECORDS)
    d2 = write_manifest(tmp_path / "b.jsonl", shuffled)
    assert d1 == d2 and (tmp_path / "a.jsonl").read_bytes() == (tmp_path / "b.jsonl").read_bytes()


def test_the_manifest_has_no_timestamp_or_machine_specific_content(tmp_path):
    write_manifest(tmp_path / "m.jsonl", RECORDS)
    text = (tmp_path / "m.jsonl").read_text()
    assert str(tmp_path) not in text and "/home/" not in text and "created" not in text


def test_the_digest_changes_when_any_record_changes():
    changed = list(RECORDS)
    changed[3] = rec(3, split="val")
    assert manifest_digest(RECORDS) != manifest_digest(changed)
    assert manifest_digest(RECORDS) == manifest_digest(reversed(RECORDS))


def test_records_are_written_one_per_line_with_sorted_keys(tmp_path):
    write_manifest(tmp_path / "m.jsonl", RECORDS[:2])
    lines = (tmp_path / "m.jsonl").read_text().splitlines()
    assert len(lines) == 3 and "_header" in json.loads(lines[0])
    assert lines[1] == canonical_json(json.loads(lines[1]))


def test_reserved_header_keys_cannot_be_overridden(tmp_path):
    with pytest.raises(ManifestError, match="reserved"):
        write_manifest(tmp_path / "m.jsonl", RECORDS, header={"digest": "x"})


def test_a_missing_file_or_empty_file_or_wrong_kind_is_a_manifest_error(tmp_path):
    with pytest.raises(ManifestError, match="not found"):
        read_manifest(tmp_path / "nope.jsonl")
    (tmp_path / "empty.jsonl").write_text("")
    with pytest.raises(ManifestError, match="empty"):
        read_manifest(tmp_path / "empty.jsonl")
    (tmp_path / "wrong.jsonl").write_text('{"_header": {"kind": "something-else", "manifest_version": 1}}\n')
    with pytest.raises(ManifestError, match="not a"):
        read_manifest(tmp_path / "wrong.jsonl")
    (tmp_path / "nohead.jsonl").write_text(canonical_json(RECORDS[0].to_dict()) + "\n")
    with pytest.raises(ManifestError, match="header"):
        read_manifest(tmp_path / "nohead.jsonl")


def test_an_unsupported_manifest_version_is_refused(tmp_path):
    (tmp_path / "v.jsonl").write_text('{"_header": {"kind": "%s", "manifest_version": 99}}\n' % MANIFEST_KIND)
    with pytest.raises(ManifestError, match="unsupported manifest_version"):
        read_manifest(tmp_path / "v.jsonl")


def _corrupt(tmp_path, transform):
    write_manifest(tmp_path / "m.jsonl", RECORDS)
    lines = (tmp_path / "m.jsonl").read_text().splitlines()
    (tmp_path / "bad.jsonl").write_text("\n".join(transform(lines)) + "\n")
    return tmp_path / "bad.jsonl"


def test_strict_reading_stops_at_the_first_bad_line_and_names_it(tmp_path):
    path = _corrupt(tmp_path, lambda ls: ls[:3] + ["{not json"] + ls[4:])
    with pytest.raises(ManifestError, match="line 4.*not valid JSON"):
        read_manifest(path)


def test_lenient_reading_collects_every_problem(tmp_path):
    def mangle(lines):
        lines = list(lines)
        lines[2] = "{not json"
        lines[4] = json.dumps({**json.loads(lines[4]), "split": "training"})     # invalid enum
        lines[6] = json.dumps({**json.loads(lines[6]), "lr": {**json.loads(lines[6])["lr"], "path": "/abs/x.tif"}})  # absolute path
        return lines

    contents = read_manifest(_corrupt(tmp_path, mangle), strict=False)
    assert len(contents.records) == 5
    by_line = {i.line: i.code for i in contents.issues if i.line}
    assert by_line == {3: "invalid_json", 5: "invalid_split", 7: "absolute_path"}


def test_a_truncated_manifest_is_detected_through_the_header_count(tmp_path):
    path = _corrupt(tmp_path, lambda ls: ls[:-3])
    with pytest.raises(ManifestError, match="declares 8"):
        read_manifest(path)
    codes = [i.code for i in read_manifest(path, strict=False).issues]
    assert codes == ["record_count_mismatch"]


def test_blank_lines_are_ignored(tmp_path):
    path = _corrupt(tmp_path, lambda ls: ls[:2] + [""] + ls[2:])
    assert len(read_manifest(path).records) == 8


def test_an_empty_record_list_is_a_valid_manifest(tmp_path):
    write_manifest(tmp_path / "e.jsonl", [])
    assert read_manifest(tmp_path / "e.jsonl").records == []
