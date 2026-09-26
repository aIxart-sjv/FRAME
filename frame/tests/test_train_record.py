"""frame.train.record -- what an experiment records about itself: environment, git, summary, README."""

from __future__ import annotations

import json
import subprocess

import pytest

from frame.train.config import TrainConfig
from frame.train.record import environment_info, git_info, render_readme, write_summary


def config(**over):
    d = {"name": "unit", "model": {"name": "tiny_cnn"}, "data": {"manifest": "m.jsonl"}, "steps": 5, "output_dir": "experiments/training/unit"}
    d.update(over)
    return TrainConfig.from_dict(d)


def sample_summary():
    val = {"step": 5, "n_patches": 2, "loss": {"total": 0.0123, "reconstruction": 0.0123},
           "metrics": {"psnr_db": 31.234, "ssim": 0.9012, "rmse": 0.0234, "sam_degrees": 1.5, "ergas": None}}
    return {
        "name": "unit", "status": "completed", "model": {"name": "tiny_cnn", "parameters": {"total": 1234, "trainable": 1234, "megabytes": 0.005}},
        "dataset": {"datasets": ["synthetic_smoke"], "n_train_pairs": 6, "n_val_pairs": 2, "train_scenes": ["R00_S00"], "val_scenes": ["R04_S00"],
                    "manifest_digest": "ab" * 32, "ignored_records": {"test": 2}},
        "manifest_digest": "ab" * 32, "seed": 3, "config_digest": "cd" * 32,
        "training": {"steps": 5, "train_seconds": 1.5, "peak_vram_mib": 123.4, "device": "cuda:0", "precision": {"requested": "off", "effective": "off", "note": None}},
        "initial_validation": {**val, "step": 0, "metrics": {**val["metrics"], "psnr_db": 25.0}},
        "validation": val,
        "baselines": {"bicubic": {**val, "metrics": {**val["metrics"], "psnr_db": 26.5}}},
        "environment": {"python": "3.11.9", "torch": "2.14.0", "gpu": {"name": "RTX 3050", "total_memory_mib": 4096}},
        "git": {"revision": "0" * 40, "dirty": True},
    }


# ============================================================================== environment


def test_the_environment_record_covers_software_and_hardware_and_is_json_safe():
    info = environment_info("cpu")
    for key in ("python", "platform", "cpu_count", "torch", "numpy", "cuda_available", "gpu"):
        assert key in info, key
    json.dumps(info)
    assert info["python"].count(".") >= 2 and info["torch"]


def test_the_gpu_is_described_when_present_and_absent_otherwise():
    import torch

    gpu = environment_info("cuda" if torch.cuda.is_available() else "cpu")["gpu"]
    if torch.cuda.is_available():
        assert gpu["name"] and gpu["total_memory_mib"] > 0 and gpu["compute_capability"]
    else:
        assert gpu is None


# ============================================================================== git


def test_git_info_reports_the_revision_and_whether_the_tree_is_dirty(tmp_path):
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(["git", "-c", "user.email=a@b", "-c", "user.name=t", "commit", "--allow-empty", "-q", "-m", "x"], cwd=tmp_path, check=True)
    clean = git_info(tmp_path)
    assert len(clean["revision"]) == 40 and clean["dirty"] is False
    (tmp_path / "f.txt").write_text("x")
    assert git_info(tmp_path)["dirty"] is True


def test_git_info_outside_a_repository_is_none_not_an_error(tmp_path):
    assert git_info(tmp_path) == {"revision": None, "dirty": None}


# ============================================================================== summary file


def test_the_summary_is_written_atomically_as_sorted_json(tmp_path):
    path = write_summary(tmp_path / "out", {"b": 1, "a": {"y": 2, "x": 1}})
    assert path.name == "summary.json" and json.loads(path.read_text()) == {"a": {"x": 1, "y": 2}, "b": 1}
    assert path.read_text().index('"a"') < path.read_text().index('"b"') and [p.name for p in path.parent.iterdir()] == ["summary.json"]
    write_summary(tmp_path / "out", {"status": "completed"})                       # overwrite is fine
    assert json.loads(path.read_text()) == {"status": "completed"}


# ============================================================================== README


def test_the_readme_states_model_data_seed_digest_hardware_and_how_to_reproduce():
    text = render_readme(sample_summary(), config())
    for needle in ("tiny_cnn", "synthetic_smoke", "ab" * 32, "seed", "RTX 3050", "python -m frame.train run", "123.4", "cuda:0"):
        assert needle in text, needle


def test_the_readme_compares_the_trained_model_with_its_baselines_on_the_same_data_without_naming_a_winner():
    text = render_readme(sample_summary(), config())
    assert "bicubic" in text and "26.5" in text and "31.2" in text and "25.0" in text          # trained, baseline, and step-0 numbers
    lowered = text.lower()
    assert "winner" not in lowered.replace("no winner", "").replace("not a winner", "") and "state of the art" not in lowered and "best model" not in lowered
    assert "n/a" in text                                                                       # an uncomputable metric is shown as n/a, not hidden


def test_the_readme_reports_the_training_scene_overfit_check_next_to_its_baselines():
    summary = sample_summary()
    train = summary["validation"]
    summary["train_evaluation"] = {"model": {**train, "metrics": {**train["metrics"], "psnr_db": 40.5}}, "baselines": {"bicubic": {**train, "metrics": {**train["metrics"], "psnr_db": 33.9}}}}
    text = render_readme(summary, config())
    assert "Training scenes" in text and "40.500" in text and "33.900" in text


def test_the_readme_warns_that_synthetic_results_are_not_evidence_about_real_imagery():
    assert "not evidence about real" in render_readme(sample_summary(), config()).lower()
    real = sample_summary()
    real["dataset"]["datasets"] = ["sen2naipv2"]
    assert "not evidence about real" not in render_readme(real, config()).lower()


def test_the_readme_contains_no_machine_specific_home_paths():
    assert "/home/" not in render_readme(sample_summary(), config())
