"""``python -m frame.data`` -- qc, split, sen2neon-manifest."""

from __future__ import annotations

import csv
import dataclasses
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from frame.data import cli
from frame.data.adapters import ADAPTERS
from frame.data.contract import Split
from frame.data.manifest import read_manifest, write_manifest
from frame.tests.data_fixtures import (  # noqa: F401
    TINY,
    TinyAdapter,
    TinyEnv,
    build_tiny_records,
    make_record,
    tiny_env,
    tiny_profile_installed,
    write_pair_files,
    write_raster,
)
from frame.tests.data_real_rows import REAL_ROWS

REPO = Path(__file__).resolve().parents[2]


@pytest.fixture()
def tiny_adapter_registered(monkeypatch):
    monkeypatch.setitem(ADAPTERS, TINY, TinyAdapter)


def manifest_for(env: TinyEnv, tmp_path, records=None, name="m.jsonl") -> Path:
    path = tmp_path / name
    write_manifest(path, records if records is not None else env.records)
    return path


def run(*argv) -> int:
    return cli.main([str(a) for a in argv])


# ============================================================================== qc


def test_a_clean_manifest_passes_and_writes_a_machine_readable_report(tiny_env, tmp_path, capsys):
    report_path = tmp_path / "report.json"
    code = run("qc", manifest_for(tiny_env, tmp_path), "--report", report_path)
    out = capsys.readouterr().out
    assert code == 0 and "OK" in out
    report = json.loads(report_path.read_text())
    assert report["ok"] is True and report["total_samples"] == 8 and report["valid_samples"] == 8 and report["invalid_samples"] == 0
    assert report["per_split"] == {"train": 4, "val": 4} and report["per_dataset"] == {f"{TINY}/default": 8}
    assert report["checks_performed"]["headers"] is False and report["manifest_digest"]


def test_the_report_is_deterministic(tiny_env, tmp_path):
    manifest = manifest_for(tiny_env, tmp_path)
    run("qc", manifest, "--report", tmp_path / "a.json")
    run("qc", manifest, "--report", tmp_path / "b.json")
    assert (tmp_path / "a.json").read_bytes() == (tmp_path / "b.json").read_bytes()


def test_one_scene_in_two_splits_fails_qc_with_a_scene_leakage_finding(tiny_env, tmp_path, capsys):
    records = list(tiny_env.records)
    records[1] = dataclasses.replace(records[1], split=Split.VAL)       # tile 1 of scene A_s1 moves to val; tile 0 stays in train
    assert records[0].scene_id == records[1].scene_id and records[0].split != records[1].split
    report_path = tmp_path / "report.json"
    code = run("qc", manifest_for(tiny_env, tmp_path, records), "--report", report_path)
    assert code == 1
    report = json.loads(report_path.read_text())
    assert report["ok"] is False and report["issue_counts"]["scene_leakage"] >= 1
    assert "FAILED" in capsys.readouterr().out


def test_a_shifted_hr_transform_fails_qc(tiny_env, tmp_path):
    lr, hr = write_pair_files(tmp_path, TINY, "lr/shift.tif", "hr/shift.tif", seed=7, hr_origin_shift=(5.0, 0.0))   # half an LR pixel
    bad = make_record("tinyset:shift", scene="Z_s", region="Z", lr=lr, hr=hr)
    report_path = tmp_path / "r.json"
    assert run("qc", manifest_for(tiny_env, tmp_path, tiny_env.records + [bad]), "--report", report_path) == 1
    report = json.loads(report_path.read_text())
    assert report["issue_counts"]["footprint_mismatch"] == 1 and report["invalid_sample_ids"] == ["tinyset:shift"]


def test_warnings_pass_unless_strict(tiny_env, tmp_path):
    ungeo = []
    for r in tiny_env.records:
        ungeo.append(dataclasses.replace(r, lr=dataclasses.replace(r.lr, crs=None, transform=None), hr=dataclasses.replace(r.hr, crs=None, transform=None)))
    manifest = manifest_for(tiny_env, tmp_path, ungeo)
    report_path = tmp_path / "r.json"
    assert run("qc", manifest, "--report", report_path) == 0
    assert json.loads(report_path.read_text())["issue_counts"]["not_georeferenced"] == 8
    assert run("qc", manifest, "--strict") == 1


def test_header_checks_catch_a_file_that_disagrees_with_its_record(tiny_env, tmp_path):
    victim = tiny_env.records[0]
    write_raster(tiny_env.root / TINY / victim.lr.path, np.zeros((4, 16, 16), dtype="uint16"), transform=victim.lr.transform, crs=victim.lr.crs,
                 dtype="uint16", nodata=None)
    manifest = manifest_for(tiny_env, tmp_path)
    assert run("qc", manifest) == 0                                                     # nothing to check without a data root
    assert run("qc", manifest, "--data-root", tiny_env.root) == 0                       # files exist; headers not compared yet
    report_path = tmp_path / "r.json"
    assert run("qc", manifest, "--data-root", tiny_env.root, "--headers", "--report", report_path) == 1
    report = json.loads(report_path.read_text())
    assert report["invalid_sample_ids"] == [victim.sample_id] and report["checks_performed"]["headers"] is True


def test_a_missing_file_is_an_error_once_a_data_root_is_given(tiny_env, tmp_path):
    (tiny_env.root / TINY / tiny_env.records[3].hr.path).unlink()
    report_path = tmp_path / "r.json"
    assert run("qc", manifest_for(tiny_env, tmp_path), "--data-root", tiny_env.root, "--report", report_path) == 1
    assert json.loads(report_path.read_text())["issue_counts"]["missing_file"] == 1


def test_pixel_checks_catch_nan_and_never_repair_it(tiny_env, tmp_path, tiny_adapter_registered):
    victim = tiny_env.records[2]
    data = np.full((4, 32, 32), 0.3, dtype="float32")
    data[1, 3, 3] = np.nan
    write_raster(tiny_env.root / TINY / victim.lr.path, data, transform=victim.lr.transform, crs=victim.lr.crs, dtype="float32", nodata=None)
    records = [dataclasses.replace(r, lr=dataclasses.replace(r.lr, dtype="float32", reflectance_scale=None)) if r is victim else r for r in tiny_env.records]
    report_path = tmp_path / "r.json"
    code = run("qc", manifest_for(tiny_env, tmp_path, records), "--data-root", tiny_env.root, "--pixels", "--report", report_path)
    report = json.loads(report_path.read_text())
    assert code == 1 and report["issue_counts"]["non_finite"] == 1 and report["invalid_sample_ids"] == [victim.sample_id]
    assert report["checks_performed"]["pixels"] is True
    assert np.isnan(np.asarray(__import__("rasterio").open(tiny_env.root / TINY / victim.lr.path).read())).any()   # the file itself is untouched


def test_pixel_checks_on_good_data_pass(tiny_env, tmp_path, tiny_adapter_registered):
    assert run("qc", manifest_for(tiny_env, tmp_path), "--data-root", tiny_env.root, "--headers", "--pixels") == 0


def test_a_dataset_without_an_adapter_is_a_finding_not_a_crash(tiny_env, tmp_path):
    report_path = tmp_path / "r.json"
    code = run("qc", manifest_for(tiny_env, tmp_path), "--data-root", tiny_env.root, "--pixels", "--report", report_path)
    report = json.loads(report_path.read_text())
    assert code == 1 and report["invalid_samples"] == 8


def test_a_missing_or_foreign_manifest_exits_2_with_a_message(tmp_path, capsys):
    assert run("qc", tmp_path / "absent.jsonl") == 2
    assert "Manifest not found" in capsys.readouterr().err
    (tmp_path / "bad.jsonl").write_text('{"hello": 1}\n')
    assert run("qc", tmp_path / "bad.jsonl") == 2


def test_a_corrupt_record_line_is_reported_with_its_line_number(tiny_env, tmp_path):
    manifest = manifest_for(tiny_env, tmp_path)
    lines = manifest.read_text().splitlines()
    lines[3] = "{not json"
    manifest.write_text("\n".join(lines) + "\n")
    report_path = tmp_path / "r.json"
    assert run("qc", manifest, "--report", report_path) == 1
    issues = json.loads(report_path.read_text())["issues"]
    assert any(i["code"] == "invalid_json" and i["line"] == 4 for i in issues)


# ============================================================================== split


def unsplit_manifest(tmp_path) -> Path:
    layout = {f"R{i}": {"s1": 2, "s2": 1} for i in range(6)}
    records = build_tiny_records(tmp_path, layout)
    path = tmp_path / "unsplit.jsonl"
    write_manifest(path, records)
    return path


def test_the_split_command_assigns_whole_regions_and_passes_qc(tmp_path, tiny_profile_installed, capsys):
    out = tmp_path / "split.jsonl"
    assert run("split", unsplit_manifest(tmp_path), "--out", out, "--seed", 3, "--fractions", 0.5, 0.25, 0.25) == 0
    assert "per split" in capsys.readouterr().out
    records = read_manifest(out).records
    by_region = {}
    for r in records:
        by_region.setdefault(r.region_id, set()).add(r.split)
    assert all(len(s) == 1 for s in by_region.values())                   # no region straddles two splits
    assert len({s for splits in by_region.values() for s in splits}) >= 2
    assert run("qc", out) == 0


def test_the_split_command_is_deterministic_and_seed_dependent(tmp_path, tiny_profile_installed):
    source = unsplit_manifest(tmp_path)
    for name, seed in (("a", 5), ("b", 5), ("c", 6)):
        run("split", source, "--out", tmp_path / f"{name}.jsonl", "--seed", seed, "--fractions", 0.5, 0.25, 0.25)
    assert (tmp_path / "a.jsonl").read_bytes() == (tmp_path / "b.jsonl").read_bytes()
    assert (tmp_path / "a.jsonl").read_bytes() != (tmp_path / "c.jsonl").read_bytes()
    header = json.loads((tmp_path / "a.jsonl").read_text().splitlines()[0])["_header"]
    assert header["split_seed"] == 5 and header["split_level"] == "region" and header["split_fractions"] == [0.5, 0.25, 0.25]


def test_scene_level_splitting_keeps_every_scene_whole(tmp_path, tiny_profile_installed):
    out = tmp_path / "scene.jsonl"
    assert run("split", unsplit_manifest(tmp_path), "--out", out, "--level", "scene", "--seed", 1, "--fractions", 0.6, 0.2, 0.2) == 0
    by_scene = {}
    for r in read_manifest(out).records:
        by_scene.setdefault(r.scene_id, set()).add(r.split)
    assert all(len(s) == 1 for s in by_scene.values())


# ============================================================================== sen2neon-manifest


def write_metadata_csv(path):
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(next(iter(REAL_ROWS.values())).keys()))
        writer.writeheader()
        writer.writerows(REAL_ROWS.values())


def test_the_sen2neon_command_builds_a_manifest_from_metadata(tmp_path, capsys):
    write_metadata_csv(tmp_path / "metadata.csv")
    out = tmp_path / "neon.jsonl"
    code = run("sen2neon-manifest", "--metadata-csv", tmp_path / "metadata.csv", "--out", out, "--ids", "2018_MLBS_3__0_2", "2022_KONZ_7__5_3", "--revision", "abc123")
    assert code == 0 and "wrote 2 record(s)" in capsys.readouterr().out
    contents = read_manifest(out)
    assert [r.sample_id for r in contents.records] == ["sen2neon:2018_MLBS_3__0_2", "sen2neon:2022_KONZ_7__5_3"]
    assert all(r.split is Split.TEST and r.dataset_revision == "abc123" for r in contents.records)
    assert contents.header["source"] == "SEN2NEON metadata.csv" and contents.header["digest"]
    assert run("qc", out, "--strict") == 0                                # a real-row manifest is clean, even under --strict


def test_the_sen2neon_manifest_is_byte_reproducible(tmp_path):
    write_metadata_csv(tmp_path / "metadata.csv")
    for name in ("a", "b"):
        run("sen2neon-manifest", "--metadata-csv", tmp_path / "metadata.csv", "--out", tmp_path / f"{name}.jsonl")
    assert (tmp_path / "a.jsonl").read_bytes() == (tmp_path / "b.jsonl").read_bytes()


def test_an_unknown_tile_id_exits_2(tmp_path, capsys):
    write_metadata_csv(tmp_path / "metadata.csv")
    assert run("sen2neon-manifest", "--metadata-csv", tmp_path / "metadata.csv", "--out", tmp_path / "x.jsonl", "--ids", "nope") == 2
    assert "not in the metadata" in capsys.readouterr().err and not (tmp_path / "x.jsonl").exists()


# ============================================================================== entry point


def test_python_dash_m_frame_data_runs_as_documented(tmp_path):
    result = subprocess.run([sys.executable, "-m", "frame.data", "--help"], cwd=REPO, capture_output=True, text=True, timeout=120)
    assert result.returncode == 0
    assert all(word in result.stdout for word in ("qc", "split", "sen2neon-manifest"))
    missing = subprocess.run([sys.executable, "-m", "frame.data", "qc", str(tmp_path / "none.jsonl")], cwd=REPO, capture_output=True, text=True, timeout=120)
    assert missing.returncode == 2 and "Manifest not found" in missing.stderr
