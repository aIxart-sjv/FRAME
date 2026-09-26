"""The evaluation layer against REAL data and the EXISTING validation pipeline -- skipped when the data or weights are not cached; never downloads.

These tests tie the new layer to what already existed: the same Lite numbers as `frame.validation.run_validation_sample`, the same self-consistency values as
`frame.consistency`, opensr-test's own values, and a real multi-tile SEN2NEON scene through the tile engine with the seam and mask rules.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch

from frame.evaluate.config import DatasetSpec, EvalConfig, SystemSpec
from frame.evaluate.datasets import build_dataset
from frame.evaluate.runner import run_evaluation, score_sample
from frame.evaluate.systems import BicubicSystem, build_system
from frame.tiling.plan import TilingConfig

REPO = Path(__file__).resolve().parents[2]
TILING = TilingConfig(tile_size=128, overlap=32)
LITE_DIR = Path(os.environ.get("SEN2SR_BASELINE_WEIGHTS_DIR", str(Path.home() / ".cache" / "sen2sr_baseline" / "SEN2SRLite_RGBN")))
needs_spot = pytest.mark.skipif(not (Path.home() / ".config" / "opensr_test" / "spot.pkl").exists() or not (LITE_DIR / "model.safetensor").is_file(),
                                reason="opensr-test 'spot' subset or the Lite weights are not cached locally; not downloading")
NEON_MANIFEST = REPO / "experiments" / "evaluation" / "manifests" / "sen2neon_random30_seed0.jsonl"
needs_neon = pytest.mark.skipif(not NEON_MANIFEST.is_file() or not (Path.home() / ".cache" / "frame_data" / "sen2neon" / "metadata.csv").is_file() or not (LITE_DIR / "model.safetensor").is_file(),
                                reason="the random-30 SEN2NEON sample or the Lite weights are not present")


def cfg():
    from frame.evaluate.config import EvalConfig

    return EvalConfig.from_dict({"name": "real", "output_dir": "unused", "systems": [{"name": "bicubic", "kind": "bicubic"}], "datasets": [{"name": "d", "kind": "sen2neon", "manifest": "m.jsonl"}]}).metric_config()


@needs_spot
def test_lite_on_a_real_opensr_sample_reproduces_the_existing_validation_pipeline():
    from frame.validation import extract_sample, load_subset, run_validation_sample

    ds = build_dataset(DatasetSpec(name="spot", kind="opensr_test", subset="spot"))
    sample = ds.load(ds.records[0])
    system = build_system(SystemSpec(name="lite", kind="lite"), TILING, device="cpu")
    result = system.infer(sample.lr)
    baseline = BicubicSystem("bicubic").infer(sample.lr).sr
    row = score_sample(sample, system, result, baseline, cfg(), tiling=TILING)

    report = run_validation_sample(extract_sample(load_subset("spot"), subset="spot", sample_index=0), result.sr)
    ref = report.standard_reference_metrics.sen2sr
    acc = row["reference_accuracy"]
    for key in ("psnr_db", "rmse", "sam_degrees", "ergas"):
        assert acc[key] == pytest.approx(getattr(ref, key), rel=1e-5), key                                     # the same reused implementation
    assert acc["ssim"] == pytest.approx(ref.ssim, abs=0.01)                                                    # SSIM here excludes the window border of the mask
    old_sc = report.phase3_self_consistency.downsample_consistency.overall
    assert row["self_consistency"]["overall"]["mae"] == pytest.approx(old_sc.mean_abs_error, rel=1e-6) and row["self_consistency"]["overall"]["rmse"] == pytest.approx(old_sc.rmse, rel=1e-6)
    native = row["opensr_native"]
    assert native["status"] == "computed" and native["values"]["hallucination"] == pytest.approx(report.opensr_test_metrics.hallucination, rel=1e-5)
    assert native["values"]["omission"] == pytest.approx(report.opensr_test_metrics.omission, rel=1e-5) and native["values"]["improvement"] == pytest.approx(report.opensr_test_metrics.improvement, rel=1e-5)


@needs_neon
def test_a_real_multi_tile_sen2neon_scene_is_evaluated_end_to_end_with_masks_seams_and_provenance(tmp_path):
    ds = build_dataset(DatasetSpec(name="neon", kind="sen2neon", manifest=str(NEON_MANIFEST)))
    with_nodata = []
    for r in ds.records:
        s = ds.load(r)
        if 0.05 < s.quality["hr_dataset_nodata_fraction"] < 0.6:
            with_nodata.append(r)
        if len(with_nodata) == 1:
            break
    ds.records = with_nodata                                                        # one real tile that has nodata
    config = EvalConfig.from_dict({"name": "real", "output_dir": str(tmp_path / "out"), "device": "cpu", "systems": [{"name": "bicubic", "kind": "bicubic"}, {"name": "lite", "kind": "lite"},
                                                                                                                   {"name": "lite_nc", "kind": "lite", "hard_constraint": False}],
                                    "datasets": [{"name": "neon", "kind": "sen2neon", "manifest": str(NEON_MANIFEST)}], "statistics": {"n_boot": 200}})
    summary = run_evaluation(config, datasets=[ds])
    import json

    rows = [json.loads(line) for line in (tmp_path / "out" / "metrics.jsonl").read_text().splitlines() if '"sample_result"' in line]
    assert {r["system"] for r in rows} == {"bicubic", "lite", "lite_nc"} and all(r["status"] == "ok" for r in rows)
    lite = next(r for r in rows if r["system"] == "lite")
    assert lite["tile_count"] == 9 and lite["seam"]["status"] == "ok" and lite["seam"]["ratio"] is not None                   # 256 px LR -> 3 x 3 tiles of 128 with 32 overlap
    assert 0.0 < lite["quality"]["hr_valid_fraction"] < 1.0 and lite["reference_accuracy"]["n_valid_pixels"] == lite["quality"]["hr_valid_pixels"]
    assert lite["opensr_native"]["status"] == "not_computed" and "nodata" in lite["opensr_native"]["reason"]                # opensr-test's metrics take no mask: refused, not zero-scored
    bic = next(r for r in rows if r["system"] == "bicubic")
    assert bic["seam"] is None and bic["detail_analysis"] is None and bic["reference_accuracy"]["n_valid_pixels"] == lite["reference_accuracy"]["n_valid_pixels"]   # every system is scored on the same pixels
    prov = summary["systems"]["lite"]["provenance"]
    assert prov["hard_constraint"] is True and len(prov["weights"]["model.safetensor"]) == 64 and summary["systems"]["lite_nc"]["provenance"]["hard_constraint"] is False
