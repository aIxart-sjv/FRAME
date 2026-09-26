"""python -m frame.train -- the end-to-end command: refusals, a full run, resume, reproducibility."""

from __future__ import annotations

import dataclasses
import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from frame.data.adapters import ADAPTERS, sen2neon
from frame.data.contract import Split
from frame.data.manifest import manifest_digest, write_manifest
from frame.tests.data_fixtures import TINY, TinyAdapter, TinyEnv, tiny_env, tiny_profile_installed  # noqa: F401
from frame.tests.data_real_rows import REAL_ROWS
from frame.train import checkpoint as ckpt
from frame.train import cli

REPO = Path(__file__).resolve().parents[2]


@pytest.fixture(autouse=True)
def tiny_adapter_registered(monkeypatch):
    monkeypatch.setitem(ADAPTERS, TINY, TinyAdapter)


def write_config(tmp_path, env: TinyEnv, name="cfg.json", out="run", **over) -> Path:
    d = {
        "name": "cli-unit", "model": {"name": "tiny_cnn", "params": {"width": 8, "depth": 2}}, "steps": 4, "seed": 3, "batch_size": 2, "device": "cpu",
        "output_dir": str(tmp_path / out), "optim": {"lr": 3e-3}, "log_every": 1,
        "data": {"manifest": str(tmp_path / "m.jsonl"), "data_root": str(env.root), "lr_patch": 16, "val_lr_patch": 32, "patches_per_pair": 2},
    }
    for key, value in over.items():
        d[key] = {**d[key], **value} if isinstance(value, dict) and isinstance(d.get(key), dict) else value
    path = tmp_path / name
    path.write_text(json.dumps(d))
    return path


def manifest(tmp_path, env, records=None):
    return write_manifest(tmp_path / "m.jsonl", records if records is not None else env.records)


def run(*argv) -> int:
    return cli.main([str(a) for a in argv])


def load(path):
    return json.loads(Path(path).read_text())


def metric_lines(out: Path):
    strip = ("step_seconds", "samples_per_second", "peak_vram_mib")
    return [{k: v for k, v in json.loads(line).items() if k not in strip} for line in (out / "metrics.jsonl").read_text().splitlines()]


# ============================================================================== check


def test_check_validates_config_and_data_and_trains_nothing(tiny_env, tmp_path, capsys):
    manifest(tmp_path, tiny_env)
    cfg = write_config(tmp_path, tiny_env)
    assert run("check", cfg) == 0
    out = capsys.readouterr().out
    assert "4 train pair" in out and "4 validation pair" in out and manifest_digest(tiny_env.records) in out
    assert not (tmp_path / "run").exists()


# ============================================================================== refusals (exit code 2, nothing trained)


def test_an_invalid_config_is_refused_with_the_field_named(tiny_env, tmp_path, capsys):
    manifest(tmp_path, tiny_env)
    cfg = write_config(tmp_path, tiny_env, optim={"lr": -1})
    assert run("run", cfg) == 2
    assert "optim.lr" in capsys.readouterr().err and not (tmp_path / "run").exists()


def test_unknown_config_keys_and_missing_files_are_refused(tiny_env, tmp_path, capsys):
    manifest(tmp_path, tiny_env)
    cfg = write_config(tmp_path, tiny_env, stepz=3)
    assert run("run", cfg) == 2 and "stepz" in capsys.readouterr().err
    assert run("run", tmp_path / "absent.json") == 2 and "not found" in capsys.readouterr().err


def test_a_manifest_with_scene_leakage_is_refused_before_anything_is_trained(tiny_env, tmp_path, capsys):
    records = list(tiny_env.records)
    records[1] = dataclasses.replace(records[1], split=Split.VAL)
    manifest(tmp_path, tiny_env, records)
    assert run("run", write_config(tmp_path, tiny_env)) == 2
    err = capsys.readouterr().err
    assert "scene_leakage" in err and not (tmp_path / "run").exists()


def test_a_benchmark_in_the_training_split_is_refused(tiny_env, tmp_path, capsys):
    neon = [dataclasses.replace(sen2neon.record_from_row(r), split=Split.TRAIN) for r in REAL_ROWS.values()]
    manifest(tmp_path, tiny_env, tiny_env.records + neon)
    assert run("run", write_config(tmp_path, tiny_env)) == 2
    assert "role_violation" in capsys.readouterr().err and not (tmp_path / "run").exists()


def test_the_test_split_can_never_be_requested_as_a_training_split(tiny_env, tmp_path, capsys):
    manifest(tmp_path, tiny_env)
    assert run("run", write_config(tmp_path, tiny_env, data={"train_split": "test"})) == 2
    assert "test" in capsys.readouterr().err


def test_a_missing_manifest_is_refused(tiny_env, tmp_path, capsys):
    assert run("run", write_config(tmp_path, tiny_env)) == 2 and "not found" in capsys.readouterr().err


def test_running_into_a_directory_that_already_holds_a_run_is_refused_without_touching_it(tiny_env, tmp_path, capsys):
    manifest(tmp_path, tiny_env)
    cfg = write_config(tmp_path, tiny_env)
    assert run("run", cfg) == 0
    before = (tmp_path / "run" / "summary.json").read_bytes()
    assert run("run", cfg) == 2 and "already contains a run" in capsys.readouterr().err
    assert (tmp_path / "run" / "summary.json").read_bytes() == before and (tmp_path / "run" / "config.json").is_file()


# ============================================================================== a full run


def test_a_full_run_writes_the_experiment_layout_and_records_what_was_used(tiny_env, tmp_path, capsys):
    digest = manifest(tmp_path, tiny_env)
    assert run("run", write_config(tmp_path, tiny_env)) == 0
    out = tmp_path / "run"
    assert sorted(p.name for p in out.iterdir()) == ["README.md", "checkpoint", "config.json", "metrics.jsonl", "summary.json"]
    assert [p.name for p in (out / "checkpoint").iterdir()] == [ckpt.checkpoint_name(4)]

    s = load(out / "summary.json")
    assert s["status"] == "completed" and s["manifest_digest"] == digest and s["seed"] == 3
    assert s["model"]["name"] == "tiny_cnn" and s["model"]["parameters"]["total"] > 0
    assert s["dataset"]["n_train_pairs"] == 4 and s["dataset"]["n_val_pairs"] == 4 and s["dataset"]["datasets"] == [TINY]
    assert s["training"]["steps"] == 4 and s["training"]["device"] == "cpu" and s["training"]["precision"]["effective"] == "off"
    assert {"python", "torch", "platform", "cpu_count", "gpu"} <= set(s["environment"]) and "revision" in s["git"]
    assert {"rmse", "psnr_db"} <= set(s["validation"]["metrics"]) and s["validation"]["step"] == 4 and s["initial_validation"]["step"] == 0
    assert set(s["baselines"]) == {"bicubic"} and s["baselines"]["bicubic"]["n_patches"] == 4
    assert load(out / "config.json")["seed"] == 3 and ckpt.load_checkpoint(out / "checkpoint" / ckpt.checkpoint_name(4))["manifest_digest"] == digest
    assert "Experiment `cli-unit`" in (out / "README.md").read_text() and "seed 3" in (out / "README.md").read_text()
    assert "completed" in capsys.readouterr().out


def test_the_untrained_tiny_cnn_equals_the_bicubic_baseline_up_to_the_clamp(tiny_env, tmp_path):
    """Step 0 IS bicubic (zero-initialised head), so the 'before training' row must track the bicubic row on the same patches."""
    manifest(tmp_path, tiny_env)
    assert run("run", write_config(tmp_path, tiny_env)) == 0
    s = load(tmp_path / "run" / "summary.json")
    assert s["initial_validation"]["metrics"]["rmse"] == pytest.approx(s["baselines"]["bicubic"]["metrics"]["rmse"], rel=0.02)


def test_the_output_directory_can_be_overridden_on_the_command_line(tiny_env, tmp_path):
    manifest(tmp_path, tiny_env)
    assert run("run", write_config(tmp_path, tiny_env), "--output-dir", tmp_path / "elsewhere") == 0
    assert (tmp_path / "elsewhere" / "summary.json").is_file() and not (tmp_path / "run").exists()
    assert load(tmp_path / "elsewhere" / "config.json")["output_dir"] == str(tmp_path / "elsewhere")


def test_no_test_split_record_is_read_by_a_full_run(tiny_env, tmp_path, monkeypatch):
    """Files of test-split records are deleted; the run must not notice (they are never opened)."""
    from frame.data.roles import DATASET_PROFILES
    from frame.tests.data_fixtures import build_tiny_records, tiny_profile

    monkeypatch.setitem(DATASET_PROFILES, TINY, tiny_profile(splits=frozenset({Split.TRAIN, Split.VAL, Split.TEST})))
    root = tmp_path / "data"
    records = build_tiny_records(root, {"A": {"s1": 2}, "B": {"s1": 2}, "C": {"s1": 2}}, splits={"A": Split.TRAIN, "B": Split.VAL, "C": Split.TEST})
    for r in records:
        if r.split is Split.TEST:
            (root / TINY / r.lr.path).unlink()
    write_manifest(tmp_path / "m.jsonl", records)
    env = TinyEnv(root=root, records=records)
    assert run("run", write_config(tmp_path, env)) == 0
    assert load(tmp_path / "run" / "summary.json")["dataset"]["ignored_records"] == {"test": 2}


# ============================================================================== resume


def test_resume_continues_the_run_and_matches_an_uninterrupted_one(tiny_env, tmp_path, capsys):
    manifest(tmp_path, tiny_env)
    assert run("run", write_config(tmp_path, tiny_env, name="whole.json", out="whole", steps=6)) == 0
    assert run("run", write_config(tmp_path, tiny_env, name="a.json", out="split", steps=3)) == 0
    assert run("run", write_config(tmp_path, tiny_env, name="b.json", out="split", steps=6), "--resume", "latest") == 0
    s = load(tmp_path / "split" / "summary.json")
    assert s["training"]["resumed_from_step"] == 3 and s["training"]["steps"] == 6 and s["status"] == "completed"
    train_only = lambda d: [m for m in metric_lines(tmp_path / d) if m["type"] == "train"]
    assert train_only("split") == train_only("whole") and [m["step"] for m in train_only("split")] == [1, 2, 3, 4, 5, 6]
    # the interrupted session honestly validated at its own end (step 3); the uninterrupted run never did -- otherwise identical
    vals = lambda d: {m["step"]: m for m in metric_lines(tmp_path / d) if m["type"] == "val"}
    assert sorted(vals("split")) == [3, 6] and sorted(vals("whole")) == [6] and vals("split")[6] == vals("whole")[6]
    assert s["validation"] == load(tmp_path / "whole" / "summary.json")["validation"]


def test_resume_with_a_changed_config_or_without_a_checkpoint_is_refused(tiny_env, tmp_path, capsys):
    manifest(tmp_path, tiny_env)
    assert run("run", write_config(tmp_path, tiny_env, steps=2)) == 0
    assert run("run", write_config(tmp_path, tiny_env, name="b.json", steps=4, optim={"lr": 5e-2}), "--resume", "latest") == 2
    assert "config" in capsys.readouterr().err
    assert run("run", write_config(tmp_path, tiny_env, name="c.json", out="empty", steps=2), "--resume", "latest") == 2
    assert "no checkpoint" in capsys.readouterr().err.lower()


def test_resume_can_name_a_checkpoint_file_explicitly(tiny_env, tmp_path):
    manifest(tmp_path, tiny_env)
    assert run("run", write_config(tmp_path, tiny_env, steps=2)) == 0
    path = tmp_path / "run" / "checkpoint" / ckpt.checkpoint_name(2)
    assert run("run", write_config(tmp_path, tiny_env, name="b.json", steps=4), "--resume", path) == 0
    assert load(tmp_path / "run" / "summary.json")["training"]["resumed_from_step"] == 2


# ============================================================================== reproducibility


def test_the_same_seed_config_and_data_reproduce_the_same_metrics(tiny_env, tmp_path):
    """Documented tolerance on CPU: exact. (Bitwise equality is NOT claimed for GPUs; see docs/TRAINING.md.)"""
    manifest(tmp_path, tiny_env)
    assert run("run", write_config(tmp_path, tiny_env, name="a.json", out="a")) == 0
    assert run("run", write_config(tmp_path, tiny_env, name="b.json", out="b")) == 0
    assert metric_lines(tmp_path / "a") == metric_lines(tmp_path / "b")
    sa, sb = load(tmp_path / "a" / "summary.json"), load(tmp_path / "b" / "summary.json")
    for key in ("validation", "initial_validation", "baselines", "manifest_digest", "seed"):
        assert sa[key] == sb[key], key
    a, b = (ckpt.load_checkpoint(tmp_path / d / "checkpoint" / ckpt.checkpoint_name(4)) for d in ("a", "b"))
    assert all(torch.equal(a["model"][k], b["model"][k]) for k in a["model"])


def test_a_different_seed_changes_the_result(tiny_env, tmp_path):
    manifest(tmp_path, tiny_env)
    run("run", write_config(tmp_path, tiny_env, name="a.json", out="a", seed=1))
    run("run", write_config(tmp_path, tiny_env, name="b.json", out="b", seed=2))
    assert load(tmp_path / "a" / "summary.json")["validation"] != load(tmp_path / "b" / "summary.json")["validation"]


# ============================================================================== failures are recorded


def test_a_failure_during_training_is_recorded_in_the_summary(tiny_env, tmp_path, monkeypatch, capsys):
    from frame.train import trainer as trainer_module
    from frame.train.errors import NonFiniteLossError

    manifest(tmp_path, tiny_env)

    def boom(self, resume=None):
        raise NonFiniteLossError("Non-finite loss at step 2")

    monkeypatch.setattr(trainer_module.Trainer, "run", boom)
    assert run("run", write_config(tmp_path, tiny_env)) == 1
    s = load(tmp_path / "run" / "summary.json")
    assert s["status"] == "failed" and "Non-finite loss" in s["error"] and s["manifest_digest"]
    assert "Non-finite loss" in capsys.readouterr().err


# ============================================================================== entry point


def test_python_dash_m_frame_train_runs_as_documented():
    result = subprocess.run([sys.executable, "-m", "frame.train", "--help"], cwd=REPO, capture_output=True, text=True, timeout=180)
    assert result.returncode == 0 and "run" in result.stdout and "check" in result.stdout
    bad = subprocess.run([sys.executable, "-m", "frame.train", "run", "no-such-config.json"], cwd=REPO, capture_output=True, text=True, timeout=180)
    assert bad.returncode == 2 and "not found" in bad.stderr


# ============================================================================== the over-fit check (training scenes) is comparable to its baselines


def test_evaluating_the_training_scenes_also_scores_the_baselines_on_them(tiny_env, tmp_path):
    manifest(tmp_path, tiny_env)
    assert run("run", write_config(tmp_path, tiny_env, evaluate_train=True)) == 0
    s = load(tmp_path / "run" / "summary.json")
    te = s["train_evaluation"]
    assert te["model"]["n_patches"] == 4 and set(te["baselines"]) == {"bicubic"} and te["baselines"]["bicubic"]["n_patches"] == 4
    readme = (tmp_path / "run" / "README.md").read_text()
    assert "Training scenes" in readme and "over-fit check" in readme


def test_no_training_scene_evaluation_is_made_unless_asked(tiny_env, tmp_path):
    manifest(tmp_path, tiny_env)
    assert run("run", write_config(tmp_path, tiny_env)) == 0
    assert load(tmp_path / "run" / "summary.json")["train_evaluation"] is None
    assert "Training scenes" not in (tmp_path / "run" / "README.md").read_text()


# ============================================================================== wrong environment


@pytest.mark.parametrize("missing", ["rasterio", "skimage", "opensr_test"])
def test_running_from_an_environment_without_the_data_layer_libraries_is_a_clean_refusal(tiny_env, tmp_path, monkeypatch, capsys, missing):
    """The isolated Mamba environment has torch and mamba_ssm but not rasterio / scikit-image / opensr-test."""
    manifest(tmp_path, tiny_env)

    def unavailable(args):
        raise ModuleNotFoundError(f"No module named {missing!r}", name=missing)

    monkeypatch.setattr(cli, "_cmd_check", unavailable)
    parser = cli.build_parser()
    parser.set_defaults()
    assert run("check", write_config(tmp_path, tiny_env)) == 2
    err = capsys.readouterr().err
    assert "main FRAME environment" in err and missing in err and "Traceback" not in err


def test_an_unrelated_missing_module_is_still_a_bug_with_a_traceback(tiny_env, tmp_path, monkeypatch):
    manifest(tmp_path, tiny_env)
    monkeypatch.setattr(cli, "_cmd_check", lambda args: (_ for _ in ()).throw(ModuleNotFoundError("No module named 'made_up'", name="made_up")))
    with pytest.raises(ModuleNotFoundError):
        run("check", write_config(tmp_path, tiny_env))
