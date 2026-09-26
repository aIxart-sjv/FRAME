"""python -m frame.reliability -- check / run / scene, refusals, exit codes, on real (synthetic-smoke) files; no GPU."""

from __future__ import annotations

import dataclasses
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from frame.data.adapters import sen2neon
from frame.data.adapters.synthetic import build_synthetic_dataset
from frame.data.contract import Split
from frame.data.manifest import write_manifest
from frame.geospatial import write_geotiff
from frame.preprocessing.metadata import RasterMetadata
from frame.reliability import cli
from frame.tests.data_real_rows import REAL_ROWS
from frame.train import checkpoint as ckpt
from frame.train.config import TrainConfig
from frame.train.models import build_model

REPO = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def synth(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("synth")
    records = build_synthetic_dataset(tmp / "data", n_regions=2, scenes_per_region=2, seed=0)
    split = {"R00": Split.TRAIN, "R01": Split.TEST}
    records = [dataclasses.replace(r, split=split[r.region_id]) for r in records]
    write_manifest(tmp / "m.jsonl", records)
    config = TrainConfig.from_dict({"name": "t", "model": {"name": "tiny_cnn", "params": {"width": 8, "depth": 2}}, "data": {"manifest": "m.jsonl"}, "steps": 5, "output_dir": "o", "seed": 1})
    model = build_model(config.model, seed=1)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    ckpt.save_checkpoint(tmp / "tiny.pt", model=model, optimizer=opt, scheduler=torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 1.0), scaler=torch.amp.GradScaler("cpu", enabled=False),
                         step=5, micro_step=5, config=config, manifest_digest="d" * 64, git_revision=None)
    (tmp / "train_summary.json").write_text(json.dumps({"dataset": {"datasets": ["synthetic_smoke"], "train_scenes": ["R00_S00", "R00_S01"], "val_scenes": [], "manifest_digest": "d" * 64}}))
    return tmp


def write_config(synth, tmp_path, name="cfg.json", out="out", **over):
    d = {"name": "rel-cli", "output_dir": str(tmp_path / out), "device": "cpu",
         "systems": [{"name": "tiny", "kind": "checkpoint", "checkpoint": str(synth / "tiny.pt"), "train_summary": str(synth / "train_summary.json")}],
         "datasets": [{"name": "syn", "kind": "synthetic_smoke", "manifest": str(synth / "m.jsonl"), "data_root": str(synth / "data")}],
         "bootstrap": {"n_boot": 100}, "analysis": {"cell_sizes_hr_px": [8], "displacement_sweep_hr_px": [0, 2], "pooled_cells_per_tile": 200, "pooled_pixels_per_tile": 400}}
    d.update(over)
    path = tmp_path / name
    path.write_text(json.dumps(d))
    return path


def run(*argv):
    return cli.main([str(a) for a in argv])


def load(p):
    return json.loads(Path(p).read_text())


# ============================================================================== check


def test_check_validates_and_previews_the_gate_without_running_a_model_or_writing_anything(synth, tmp_path, capsys):
    assert run("check", write_config(synth, tmp_path)) == 0
    out = capsys.readouterr().out
    assert "config OK" in out and "syn" in out and "reference gate preview" in out and "pixel_level_eligible" in out and "tiny" in out and not (tmp_path / "out").exists()


def test_check_refuses_a_benchmark_labelled_train_and_a_model_that_saw_the_scenes(synth, tmp_path, capsys):
    recs = [sen2neon.record_from_row(r) for r in REAL_ROWS.values()]
    recs[0] = dataclasses.replace(recs[0], split=Split.TRAIN)
    write_manifest(tmp_path / "neon.jsonl", recs)
    cfg = write_config(synth, tmp_path, datasets=[{"name": "neon", "kind": "sen2neon", "manifest": str(tmp_path / "neon.jsonl")}])
    assert run("check", cfg) == 2 and "role_violation" in capsys.readouterr().err
    summary = tmp_path / "s.json"
    summary.write_text(json.dumps({"dataset": {"datasets": ["synthetic_smoke"], "train_scenes": ["R01_S00"], "val_scenes": [], "manifest_digest": "d" * 64}}))
    cfg = write_config(synth, tmp_path, name="c2.json", systems=[{"name": "seen", "kind": "checkpoint", "checkpoint": str(synth / "tiny.pt"), "train_summary": str(summary)}])
    assert run("check", cfg) == 2 and "train_eval_overlap" in capsys.readouterr().err


def test_an_invalid_config_names_the_field_and_a_bicubic_system_is_refused(synth, tmp_path, capsys):
    assert run("check", write_config(synth, tmp_path, alignment={"tolerance_hr_px": -1})) == 2
    assert "alignment.tolerance_hr_px" in capsys.readouterr().err
    assert run("check", write_config(synth, tmp_path, name="c3.json", systems=[{"name": "b", "kind": "bicubic"}])) == 2 and "systems" in capsys.readouterr().err
    assert run("check", tmp_path / "absent.json") == 2 and "not found" in capsys.readouterr().err


# ============================================================================== run


def test_run_writes_the_layout_and_reports_the_evidence(synth, tmp_path, capsys):
    code = run("run", write_config(synth, tmp_path))
    out = tmp_path / "out"
    assert sorted(p.name for p in out.iterdir()) == sorted(["README.md", "calibration.json", "config.json", "correlations.json", "detection.json", "metrics.jsonl", "summary.json", "risk_coverage.json"])
    s = load(out / "summary.json")
    d = s["datasets"]["syn"]
    assert d["counts"]["total"] == 2 and sum(d["gate"]["by_evidence_level"].values()) == 2 and s["overview"]["syn"]["tiny"]["n_tiles"] in (0, 1, 2)
    text = capsys.readouterr().out
    assert "not a calibrated uncertainty" in text and code in (0, 1)
    assert s["alignment"]["method"] == "bicubic_baseline_cross_correlation" and s["tta"]["n_members"] == 6


def test_run_refuses_an_existing_output_directory(synth, tmp_path, capsys):
    cfg = write_config(synth, tmp_path)
    assert run("run", cfg) in (0, 1)
    assert run("run", cfg) == 2 and "already contains results" in capsys.readouterr().err


def test_nothing_eligible_is_a_failed_run_with_a_clear_message(tmp_path, capsys, synth):
    recs = [sen2neon.record_from_row(r) for r in REAL_ROWS.values()]
    write_manifest(tmp_path / "neon.jsonl", recs)                              # valid records, files absent
    cfg = write_config(synth, tmp_path, datasets=[{"name": "neon", "kind": "sen2neon", "manifest": str(tmp_path / "neon.jsonl"), "data_root": str(tmp_path / "nowhere")}])
    assert run("run", cfg) == 1
    assert "no eligible evidence" in capsys.readouterr().err
    assert load(tmp_path / "out" / "summary.json")["datasets"]["neon"]["counts"]["invalid"] == 3


def test_python_dash_m_frame_reliability_runs_as_documented():
    r = subprocess.run([sys.executable, "-m", "frame.reliability", "--help"], cwd=REPO, capture_output=True, text=True, timeout=180)
    assert r.returncode == 0 and "check" in r.stdout and "run" in r.stdout and "scene" in r.stdout
    bad = subprocess.run([sys.executable, "-m", "frame.reliability", "run", "no-such.json"], cwd=REPO, capture_output=True, text=True, timeout=180)
    assert bad.returncode == 2 and "not found" in bad.stderr


# ============================================================================== scene


def geotiff(tmp_path, h=100, w=150):
    meta = RasterMetadata(crs="EPSG:32630", transform=(10.0, 0.0, 720285.0, 0.0, -10.0, 4375125.0), bounds=(720285.0, 4375125.0 - 10.0 * h, 720285.0 + 10.0 * w, 4375125.0), resolution_m=10.0,
                          width=w, height=h, band_names=("B04", "B03", "B02", "B08"), acquisition_timestamp=None, nodata_value=None, cloud_mask_coverage=None, sr_variant="unit")
    data = (np.random.default_rng(0).random((4, h, w)) * 3000 + 500).astype("float32")
    write_geotiff(tmp_path / "lr.tif", data, meta)
    return tmp_path / "lr.tif"


def test_scene_checks_a_rectangular_window_and_writes_its_record(synth, tmp_path, capsys):
    cfg = write_config(synth, tmp_path, tiling={"tile_size": 64, "overlap": 16})
    code = run("scene", cfg, "--system", "tiny", "--lr-geotiff", geotiff(tmp_path), "--band-indices", 0, 1, 2, 3, "--window", 0, 90, 0, 140, "--output-dir", tmp_path / "scene")
    report = load(tmp_path / "scene" / "scene_check.json")
    assert code == 0 and report["scene"]["lr_shape"] == [4, 90, 140] and report["scene"]["sr_shape"] == [4, 360, 560] and report["scene"]["tile_count"] >= 4
    assert report["orientation"]["all_members_aligned"] is True and report["georeferencing"]["array_round_trips_exactly"] is True and report["provenance"]["window_rows_cols"] == [0, 90, 0, 140]
    assert (tmp_path / "scene" / "stability_check.tif").is_file() and "scene check" in capsys.readouterr().out


def test_scene_refuses_an_unknown_system_and_an_existing_directory(synth, tmp_path, capsys):
    cfg = write_config(synth, tmp_path, tiling={"tile_size": 64, "overlap": 16})
    assert run("scene", cfg, "--system", "nope", "--lr-geotiff", geotiff(tmp_path), "--output-dir", tmp_path / "s") == 2 and "system" in capsys.readouterr().err
    args = ("scene", cfg, "--system", "tiny", "--lr-geotiff", geotiff(tmp_path), "--band-indices", 0, 1, 2, 3, "--window", 0, 64, 0, 64, "--output-dir", tmp_path / "s2")
    assert run(*args) == 0
    assert run(*args) == 2 and "already contains" in capsys.readouterr().err


def test_scene_runs_the_model_on_the_device_the_system_lives_on(synth, tmp_path, monkeypatch):
    """Found on the real run: the scene command handed a CPU tensor to a CUDA Lite model. The tensor must be moved to the system's device first."""
    from types import SimpleNamespace

    import frame.evaluate.systems as systems_module
    import frame.reliability.scene_check as scene_module

    seen = {}

    class Stop(Exception):
        pass

    fake = SimpleNamespace(tile_model=lambda x: x, device="meta", provenance=lambda: {}, close=lambda: None)
    monkeypatch.setattr(systems_module, "build_system", lambda *a, **k: fake)

    def capture(model, lr, *a, **k):
        seen["device"] = lr.device.type
        raise Stop

    monkeypatch.setattr(scene_module, "check_scene", capture)
    cfg = write_config(synth, tmp_path, tiling={"tile_size": 64, "overlap": 16})
    with pytest.raises(Stop):
        run("scene", cfg, "--system", "tiny", "--lr-geotiff", geotiff(tmp_path), "--band-indices", 0, 1, 2, 3, "--window", 0, 64, 0, 64, "--output-dir", tmp_path / "s3")
    assert seen["device"] == "meta"
