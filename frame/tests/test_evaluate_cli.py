"""python -m frame.evaluate -- check / run, refusals, exit codes, and an end-to-end run on real (synthetic-smoke) files."""

from __future__ import annotations

import dataclasses
import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from frame.data.adapters import sen2neon
from frame.data.adapters.synthetic import build_synthetic_dataset
from frame.data.contract import Split
from frame.data.manifest import write_manifest
from frame.evaluate import cli
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
    path = ckpt.save_checkpoint(tmp / "tiny.pt", model=model, optimizer=opt, scheduler=torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 1.0), scaler=torch.amp.GradScaler("cpu", enabled=False),
                                step=5, micro_step=5, config=config, manifest_digest="d" * 64, git_revision=None)
    (tmp / "train_summary.json").write_text(json.dumps({"dataset": {"datasets": ["synthetic_smoke"], "train_scenes": ["R00_S00", "R00_S01"], "val_scenes": [], "manifest_digest": "d" * 64}}))
    return tmp


def write_config(synth, tmp_path, name="cfg.json", out="out", **over):
    d = {"name": "cli-unit", "output_dir": str(tmp_path / out), "device": "cpu",
         "systems": [{"name": "bicubic", "kind": "bicubic"}, {"name": "tiny_untrained", "kind": "checkpoint", "checkpoint": str(synth / "tiny.pt"), "train_summary": str(synth / "train_summary.json"),
                                                          "group": "tiny"}],
         "datasets": [{"name": "syn", "kind": "synthetic_smoke", "manifest": str(synth / "m.jsonl"), "data_root": str(synth / "data")}],
         "statistics": {"n_boot": 200}, "metrics": {"hallucination_taus": [0.005]}}
    d.update(over)
    path = tmp_path / name
    path.write_text(json.dumps(d))
    return path


def run(*argv):
    return cli.main([str(a) for a in argv])


def load(p):
    return json.loads(Path(p).read_text())


# ============================================================================== check


def test_check_validates_everything_and_evaluates_nothing(synth, tmp_path, capsys):
    assert run("check", write_config(synth, tmp_path)) == 0
    out = capsys.readouterr().out
    assert "syn" in out and "2 evaluable" in out and "bicubic" in out and "tiny_untrained" in out and not (tmp_path / "out").exists()


def test_check_refuses_a_benchmark_labelled_train(tmp_path, capsys):
    recs = [sen2neon.record_from_row(r) for r in REAL_ROWS.values()]
    recs[0] = dataclasses.replace(recs[0], split=Split.TRAIN)
    write_manifest(tmp_path / "neon.jsonl", recs)
    cfg = tmp_path / "c.json"
    cfg.write_text(json.dumps({"name": "x", "output_dir": str(tmp_path / "out"), "systems": [{"name": "bicubic", "kind": "bicubic"}],
                               "datasets": [{"name": "neon", "kind": "sen2neon", "manifest": str(tmp_path / "neon.jsonl")}]}))
    assert run("check", cfg) == 2
    assert "role_violation" in capsys.readouterr().err


def test_check_refuses_a_model_that_saw_the_evaluated_scenes(synth, tmp_path, capsys):
    summary = tmp_path / "s.json"
    summary.write_text(json.dumps({"dataset": {"datasets": ["synthetic_smoke"], "train_scenes": ["R01_S00"], "val_scenes": [], "manifest_digest": "d" * 64}}))
    cfg = write_config(synth, tmp_path, systems=[{"name": "bicubic", "kind": "bicubic"}, {"name": "seen", "kind": "checkpoint", "checkpoint": str(synth / "tiny.pt"), "train_summary": str(summary)}])
    assert run("check", cfg) == 2 and "train_eval_overlap" in capsys.readouterr().err


def test_an_invalid_config_names_the_field(synth, tmp_path, capsys):
    assert run("check", write_config(synth, tmp_path, min_valid_fraction=5)) == 2
    assert "min_valid_fraction" in capsys.readouterr().err
    assert run("check", tmp_path / "absent.json") == 2 and "not found" in capsys.readouterr().err


# ============================================================================== run


def test_run_writes_the_layout_and_an_untrained_checkpoint_matches_bicubic_exactly(synth, tmp_path, capsys):
    assert run("run", write_config(synth, tmp_path)) == 0
    out = tmp_path / "out"
    assert sorted(p.name for p in out.iterdir()) == ["README.md", "aggregates.json", "config.json", "metrics.jsonl", "summary.json"]
    s = load(out / "summary.json")
    assert s["datasets"]["syn"]["counts"] == {"total": 2, "invalid": 0, "valid": 2, "unreadable": 0, "skipped": 0, "evaluated": 2} and s["datasets"]["syn"]["ignored_records"] == {"train": 2}
    a = load(out / "aggregates.json")["datasets"]["syn"]
    diff = a["paired"]["tiny_untrained - bicubic"]
    for metric in ("reference_accuracy.psnr_db", "reference_accuracy.rmse", "reference_accuracy.sam_degrees", "spatial_detail.hf_relative_error"):
        assert diff[metric]["mean_difference"] == pytest.approx(0.0, abs=1e-9), metric          # an untrained tiny_cnn IS the bicubic baseline (single unpadded tile)
    p = s["systems"]["tiny_untrained"]["provenance"]
    assert p["weights"]["checkpoint_sha256"] and p["training"]["overlap_check"].startswith("performed") and p["hard_constraint"] is False
    assert "syn" in capsys.readouterr().out


def test_run_records_the_seed_group_and_evidence_class(synth, tmp_path):
    assert run("run", write_config(synth, tmp_path)) == 0
    s = load(tmp_path / "out" / "summary.json")
    assert list(s["sections"]) == ["synthetic"] and s["seed_groups"]["tiny"]["n_systems"] == 1


def test_run_refuses_an_existing_output_directory(synth, tmp_path, capsys):
    cfg = write_config(synth, tmp_path)
    assert run("run", cfg) == 0
    assert run("run", cfg) == 2 and "already contains results" in capsys.readouterr().err


def test_a_config_relative_output_dir_resolves_against_the_config_file(synth, tmp_path, monkeypatch):
    sub = tmp_path / "exp"
    sub.mkdir()
    cfg = write_config(synth, sub, out="results")
    d = load(cfg)
    d["output_dir"] = "results"
    cfg.write_text(json.dumps(d))
    monkeypatch.chdir(tmp_path)
    assert run("run", cfg) == 0 and (sub / "results" / "summary.json").is_file()


def test_nothing_evaluable_is_a_failed_run_with_a_clear_message(tmp_path, capsys):
    recs = [sen2neon.record_from_row(r) for r in REAL_ROWS.values()]
    write_manifest(tmp_path / "neon.jsonl", recs)                            # valid records, but their files do not exist
    cfg = tmp_path / "c.json"
    cfg.write_text(json.dumps({"name": "x", "output_dir": str(tmp_path / "out"), "device": "cpu", "systems": [{"name": "bicubic", "kind": "bicubic"}],
                               "datasets": [{"name": "neon", "kind": "sen2neon", "manifest": str(tmp_path / "neon.jsonl"), "data_root": str(tmp_path / "nowhere")}]}))
    assert run("run", cfg) == 1
    err = capsys.readouterr().err
    assert "nothing was evaluated" in err
    assert load(tmp_path / "out" / "summary.json")["datasets"]["neon"]["counts"]["invalid"] == 3                # recorded, not dropped


def test_python_dash_m_frame_evaluate_runs_as_documented():
    r = subprocess.run([sys.executable, "-m", "frame.evaluate", "--help"], cwd=REPO, capture_output=True, text=True, timeout=180)
    assert r.returncode == 0 and "run" in r.stdout and "check" in r.stdout
    bad = subprocess.run([sys.executable, "-m", "frame.evaluate", "run", "no-such.json"], cwd=REPO, capture_output=True, text=True, timeout=180)
    assert bad.returncode == 2 and "not found" in bad.stderr


# ============================================================================== shift (spatial-shift sensitivity)


def test_shift_writes_its_own_record_and_states_it_is_not_a_ranking(synth, tmp_path, capsys):
    assert run("shift", write_config(synth, tmp_path), "--dataset", "syn", "--lr-shifts", "0", "0.5", "--output-dir", tmp_path / "shift_out") == 0
    out = tmp_path / "shift_out"
    assert sorted(p.name for p in out.iterdir()) == ["README.md", "aggregates.json", "config.json", "rows.jsonl", "summary.json"]
    s = load(out / "summary.json")
    assert s["conditions"] == ["shift_0_lr_px", "shift_0.5_lr_px", "aligned_to_bicubic"] and s["counts"]["evaluated"] == 2 and s["lr_shifts"] == [0.0, 0.5]
    text = capsys.readouterr().out
    assert "shift sensitivity" in text and "not a ranking" in text.lower()
    assert not (tmp_path / "out").exists()                                                   # the config's own output_dir was not used


def test_shift_can_be_limited_to_named_systems(synth, tmp_path):
    assert run("shift", write_config(synth, tmp_path), "--dataset", "syn", "--lr-shifts", "0", "--systems", "bicubic", "--output-dir", tmp_path / "s") == 0
    assert list(load(tmp_path / "s" / "summary.json")["systems"]) == ["bicubic"]


def test_shift_defaults_to_the_requirements_sweep(synth, tmp_path):
    assert run("shift", write_config(synth, tmp_path), "--dataset", "syn", "--systems", "bicubic", "--output-dir", tmp_path / "s") == 0
    assert load(tmp_path / "s" / "summary.json")["lr_shifts"] == [0.0, 0.25, 0.5, 1.0, 2.0]


def test_shift_refuses_an_unknown_dataset_an_existing_directory_and_a_fractional_hr_shift(synth, tmp_path, capsys):
    cfg = write_config(synth, tmp_path)
    assert run("shift", cfg, "--dataset", "nope", "--output-dir", tmp_path / "s") == 2 and "dataset" in capsys.readouterr().err
    assert run("shift", cfg, "--dataset", "syn", "--lr-shifts", "0", "--systems", "bicubic", "--output-dir", tmp_path / "s") == 0
    assert run("shift", cfg, "--dataset", "syn", "--lr-shifts", "0", "--systems", "bicubic", "--output-dir", tmp_path / "s") == 2 and "already contains" in capsys.readouterr().err
    assert run("shift", cfg, "--dataset", "syn", "--lr-shifts", "0.3", "--systems", "bicubic", "--output-dir", tmp_path / "t") == 2 and "whole HR pixel" in capsys.readouterr().err


def test_shift_keeps_the_role_and_overlap_safety_of_the_main_evaluation(synth, tmp_path, capsys):
    summary = tmp_path / "s.json"
    summary.write_text(json.dumps({"dataset": {"datasets": ["synthetic_smoke"], "train_scenes": ["R01_S00"], "val_scenes": [], "manifest_digest": "d" * 64}}))
    cfg = write_config(synth, tmp_path, systems=[{"name": "bicubic", "kind": "bicubic"}, {"name": "seen", "kind": "checkpoint", "checkpoint": str(synth / "tiny.pt"), "train_summary": str(summary)}])
    assert run("shift", cfg, "--dataset", "syn", "--lr-shifts", "0", "--output-dir", tmp_path / "s") == 2 and "train_eval_overlap" in capsys.readouterr().err
    assert not (tmp_path / "s").exists()                                                     # refused before anything was written


def test_the_help_lists_the_shift_command():
    r = subprocess.run([sys.executable, "-m", "frame.evaluate", "--help"], cwd=REPO, capture_output=True, text=True, timeout=180)
    assert r.returncode == 0 and "shift" in r.stdout
