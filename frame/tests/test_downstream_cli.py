"""python -m frame.downstream -- check / run / smoke, refusals and exit codes, on real (synthetic-smoke) files; no GPU."""

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
from frame.downstream import cli
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
    d = {"name": "ds-cli", "output_dir": str(tmp_path / out), "device": "cpu",
         "systems": [{"name": "tiny", "kind": "checkpoint", "checkpoint": str(synth / "tiny.pt"), "train_summary": str(synth / "train_summary.json")}],
         "datasets": [{"name": "syn", "kind": "synthetic_smoke", "manifest": str(synth / "m.jsonl"), "data_root": str(synth / "data")}], "bootstrap": {"n_boot": 100},
         "analysis": {"pooled_regions_per_tile": 200}}
    d.update(over)
    path = tmp_path / name
    path.write_text(json.dumps(d))
    return path


def run(*argv):
    return cli.main([str(a) for a in argv])


def load(p):
    return json.loads(Path(p).read_text())


# ============================================================================== check


def test_check_validates_and_previews_the_gate_and_the_region_counts_without_running_a_model(synth, tmp_path, capsys):
    assert run("check", write_config(synth, tmp_path)) == 0
    out = capsys.readouterr().out
    assert "config OK" in out and "reference gate preview" in out and "candidate regions" in out and "frame-reliability-gate/1" in out and not (tmp_path / "out").exists()


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


def test_an_invalid_config_names_the_field_and_a_threshold_cannot_be_tuned_by_policy(synth, tmp_path, capsys):
    assert run("check", write_config(synth, tmp_path, decision={"ndvi_threshold": 2.0})) == 2 and "decision.ndvi_threshold" in capsys.readouterr().err
    assert run("check", write_config(synth, tmp_path, name="c3.json", decision={"selection_policy": "tuned_on_results"})) == 2 and "decision.selection_policy" in capsys.readouterr().err
    assert run("check", write_config(synth, tmp_path, name="c4.json", systems=[{"name": "bicubic", "kind": "lite"}])) == 2 and "reserved" in capsys.readouterr().err
    assert run("check", tmp_path / "absent.json") == 2 and "not found" in capsys.readouterr().err


# ============================================================================== run


def test_run_writes_the_layout_and_states_the_unsupported_tasks(synth, tmp_path, capsys):
    code = run("run", write_config(synth, tmp_path))
    out = tmp_path / "out"
    assert sorted(p.name for p in out.iterdir()) == sorted(["README.md", "association.json", "config.json", "downstream_metrics.json", "risk_coverage.json", "summary.json", "tiles.jsonl"])
    text = capsys.readouterr().out
    assert code in (0, 1) and "India" in text or "india_downstream_validation_unavailable" in text
    assert "secondary_landcover_task_deferred_no_supported_reference" in text and "No ranking" in text


def test_run_refuses_an_existing_output_directory(synth, tmp_path, capsys):
    cfg = write_config(synth, tmp_path)
    assert run("run", cfg) in (0, 1)
    assert run("run", cfg) == 2 and "already contains results" in capsys.readouterr().err


def test_nothing_eligible_is_a_failed_run_with_a_clear_message(tmp_path, capsys, synth):
    recs = [sen2neon.record_from_row(r) for r in REAL_ROWS.values()]
    write_manifest(tmp_path / "neon.jsonl", recs)
    cfg = write_config(synth, tmp_path, datasets=[{"name": "neon", "kind": "sen2neon", "manifest": str(tmp_path / "neon.jsonl"), "data_root": str(tmp_path / "nowhere")}])
    assert run("run", cfg) == 1 and "no eligible evidence" in capsys.readouterr().err


# ============================================================================== smoke


def test_smoke_runs_end_to_end_without_data_weights_or_gpu(tmp_path, capsys):
    assert run("smoke", "--output-dir", tmp_path / "smoke", "--scenes", 6) == 0
    out = tmp_path / "smoke"
    assert (out / "README.md").is_file() and load(out / "summary.json")["overview"]["smoke"]["n_tiles"] == 6
    text = capsys.readouterr().out
    assert "mean nothing scientifically" in text and "6 synthetic scenes" in text
    assert load(out / "summary.json")["systems_compared"] == ["lr_native", "bicubic", "toy_smooth", "toy_sharpen"]


def test_smoke_without_an_output_directory_uses_a_temporary_one(capsys):
    assert run("smoke", "--scenes", 5) == 0
    assert "record in" in capsys.readouterr().out


def test_python_dash_m_frame_downstream_runs_as_documented():
    r = subprocess.run([sys.executable, "-m", "frame.downstream", "--help"], cwd=REPO, capture_output=True, text=True, timeout=180)
    assert r.returncode == 0 and "check" in r.stdout and "run" in r.stdout and "smoke" in r.stdout
    bad = subprocess.run([sys.executable, "-m", "frame.downstream", "run", "no-such.json"], cwd=REPO, capture_output=True, text=True, timeout=180)
    assert bad.returncode == 2 and "not found" in bad.stderr
