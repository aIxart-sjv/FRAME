"""frame.evaluate.shift -- spatial-shift sensitivity (requirements 142 section 27): the same SR scored against a reference displaced by known whole HR pixels."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from frame.evaluate import metrics as M
from frame.evaluate.config import EvalConfig
from frame.evaluate.errors import EvaluationError
from frame.evaluate.shift import (
    SWEEP_LR_PIXELS,
    condition_name,
    displace_pair,
    estimate_alignment,
    hr_shifts,
    run_shift_sensitivity,
    shift_metrics,
)
from frame.evaluate.systems import BicubicSystem
from frame.tests.test_evaluate_runner import BANDS, FakeDataset, field_image, make_sample, sharpen_system, read_jsonl


def smooth_hr(seed=0, size=128):
    return field_image(seed, size).astype("float64")


# ============================================================================== geometry of a displacement


def test_displacing_by_zero_changes_nothing():
    hr = smooth_hr()
    mask = np.ones((128, 128), bool)
    sr_c, hr_c, mask_c = displace_pair(hr, hr, mask, 0, 0)
    assert np.array_equal(sr_c, hr) and np.array_equal(hr_c, hr) and mask_c.all()


@pytest.mark.parametrize("dy, dx", [(3, 0), (0, -2), (3, -2), (-4, 5), (1, 1)])
def test_undoing_a_known_displacement_restores_an_exact_match(dy, dx):
    """SR = the reference moved by (dy, dx); displacing the SR by (-dy, -dx) puts every content pixel back, and only the overlap is compared."""
    hr = smooth_hr(1)
    sr = np.roll(hr, shift=(dy, dx), axis=(1, 2))
    mask = np.ones((128, 128), bool)
    sr_c, hr_c, mask_c = displace_pair(sr, hr, mask, -dy, -dx)
    assert sr_c.shape == hr_c.shape == (4, 128 - abs(dy), 128 - abs(dx)) and mask_c.shape == (128 - abs(dy), 128 - abs(dx))
    assert np.array_equal(sr_c, hr_c)                                                   # exact: no resampling is involved


def test_a_displacement_moves_the_sr_content_down_and_right_for_positive_values():
    hr = np.zeros((1, 8, 8))
    hr[0, 2, 3] = 1.0
    sr_c, hr_c, _ = displace_pair(hr.copy(), hr, np.ones((8, 8), bool), 2, 1)          # both spikes start at (2, 3); the SR content then moves +2 rows, +1 column
    assert sr_c.shape == hr_c.shape == (1, 6, 7)
    # in the common (cropped) frame the reference spike is at (0, 2) and the SR spike at (2, 3): the SR is displaced by (+2, +1) relative to the reference
    assert hr_c[0, 0, 2] == 1.0 and sr_c[0, 2, 3] == 1.0 and hr_c.sum() == 1.0 and sr_c.sum() == 1.0


def test_the_mask_is_cropped_with_the_reference_not_with_the_sr():
    hr = smooth_hr(2)
    mask = np.ones((128, 128), bool)
    mask[:10] = False                                                                     # top rows are nodata in the reference
    _, hr_c, mask_c = displace_pair(hr, hr, mask, 4, 0)                                  # reference rows 4.. are compared
    assert mask_c.shape == (124, 128) and not mask_c[:6].any() and mask_c[6:].all()
    _, _, mask_neg = displace_pair(hr, hr, mask, -4, 0)                                  # reference rows ..124 are compared
    assert not mask_neg[:10].any() and mask_neg[10:].all()


def test_a_displacement_as_large_as_the_image_is_refused():
    hr = smooth_hr(3, size=32)
    with pytest.raises(ValueError, match="displacement"):
        displace_pair(hr, hr, np.ones((32, 32), bool), 0, 32)
    with pytest.raises(ValueError, match="displacement"):
        displace_pair(hr, hr, np.ones((32, 32), bool), -40, 0)


def test_the_inputs_are_never_modified():
    hr = smooth_hr(4)
    sr = hr.copy()
    mask = np.ones((128, 128), bool)
    displace_pair(sr, hr, mask, 3, -2)
    assert np.array_equal(sr, hr) and mask.all()


# ============================================================================== the sweep in LR pixels


def test_the_default_sweep_is_the_requirements_list_and_maps_to_whole_hr_pixels():
    assert SWEEP_LR_PIXELS == (0.0, 0.25, 0.5, 1.0, 2.0)
    assert hr_shifts(SWEEP_LR_PIXELS, 4) == (0, 1, 2, 4, 8)


def test_a_shift_that_is_not_a_whole_number_of_hr_pixels_is_refused_rather_than_resampled():
    with pytest.raises(ValueError, match="whole"):
        hr_shifts((0.3,), 4)
    assert hr_shifts((0.5,), 2) == (1,)                                                   # 0.5 LR px at x2 is one HR px


def test_condition_names_are_stable_and_readable():
    assert condition_name(0.25) == "shift_0.25_lr_px" and condition_name(0.0) == "shift_0_lr_px" and condition_name(2.0) == "shift_2_lr_px"


# ============================================================================== estimating the reference's own misregistration


def test_the_alignment_estimate_recovers_a_known_displacement_and_undoes_it():
    hr = smooth_hr(5)
    baseline = np.roll(hr, shift=(3, -2), axis=(1, 2))
    mask = np.ones((128, 128), bool)
    est = estimate_alignment(baseline, hr, mask, M.MetricConfig())
    assert est["status"] == "ok" and (est["dy"], est["dx"]) == (3, -2)                    # the SR's displacement relative to the reference, whole HR px
    assert est["correction"] == (-3, 2)
    sr_c, hr_c, _ = displace_pair(baseline, hr, mask, *est["correction"])
    assert np.array_equal(sr_c, hr_c)


def test_the_alignment_estimate_rounds_to_whole_pixels_and_reports_the_raw_value():
    hr = smooth_hr(6)
    est = estimate_alignment(hr, hr, np.ones((128, 128), bool), M.MetricConfig())
    assert est["status"] == "ok" and est["correction"] == (0, 0) and abs(est["raw_dy"]) < 0.05 and abs(est["raw_dx"]) < 0.05


def test_an_uncomputable_alignment_is_reported_and_means_no_correction():
    hr = np.full((4, 64, 64), 0.2)                                                        # constant: phase correlation is degenerate
    est = estimate_alignment(hr, hr, np.ones((64, 64), bool), M.MetricConfig())
    assert est["correction"] == (0, 0) and est["status"] in ("ok", "not_computable")


# ============================================================================== metrics under a displacement


def test_the_zero_shift_metrics_equal_the_ordinary_metrics():
    sample = make_sample(0)
    hr = sample.hr.numpy().astype("float64")
    sr = BicubicSystem("bicubic").infer(sample.lr).sr.numpy().astype("float64")
    cfg = M.MetricConfig()
    mask = sample.hr_mask.numpy()
    got = shift_metrics(sr, hr, mask, BANDS, 4, cfg, dy=0, dx=0)
    ref = M.reference_accuracy(sr, hr, mask, BANDS, 4, cfg)
    assert got["reference_accuracy.psnr_db"] == pytest.approx(ref["psnr_db"], rel=1e-12)
    assert got["reference_accuracy.sam_degrees"] == pytest.approx(ref["sam_degrees"], rel=1e-12)
    assert set(got) >= {"reference_accuracy.psnr_db", "reference_accuracy.ssim", "reference_accuracy.rmse", "reference_accuracy.sam_degrees", "reference_accuracy.ergas",
                        "indices.NDVI.mae", "spatial_detail.hf_relative_error", "spatial_detail.hf_correlation", "quality.compared_fraction"}


def test_accuracy_falls_as_the_reference_is_displaced_further():
    """An image with structure: the more the reference is displaced, the worse every reference-based number gets (this is the effect the sweep measures)."""
    hr = smooth_hr(7, size=128)
    mask = np.ones((128, 128), bool)
    cfg = M.MetricConfig()
    rows = [shift_metrics(hr.copy(), hr, mask, BANDS, 4, cfg, dy=0, dx=s) for s in (0, 1, 2, 4, 8)]
    psnr = [r["reference_accuracy.psnr_db"] for r in rows]
    assert rows[0]["reference_accuracy.rmse"] < 1e-9                                       # identical images match perfectly at zero displacement
    assert all(a > b for a, b in zip(psnr[1:], psnr[2:])) and psnr[1] > psnr[-1]           # strictly worse with each larger displacement (zero-shift PSNR is infinite/undefined)
    hf = [r["spatial_detail.hf_relative_error"] for r in rows]
    assert hf[0] < 1e-9 and hf[0] < hf[1] < hf[2]                                          # it saturates near sqrt(2) once the detail is uncorrelated, so only the first steps are ordered
    assert hf[1] < 1.0 < hf[2] and all(v > 1.0 for v in hf[2:])                            # 1 HR px still helps a little; from 2 HR px the "detail" is worse than adding none
    assert rows[-1]["quality.compared_fraction"] == pytest.approx((128 - 8) / 128)


def test_a_fully_masked_displacement_yields_missing_values_not_zeros():
    hr = smooth_hr(8)
    mask = np.zeros((128, 128), bool)
    got = shift_metrics(hr, hr, mask, BANDS, 4, M.MetricConfig(), dy=0, dx=2)
    assert got["reference_accuracy.psnr_db"] is None and got["reference_accuracy.sam_degrees"] is None


# ============================================================================== the runner


CONFIG = {"name": "shift_unit", "output_dir": "SET", "systems": [{"name": "bicubic", "kind": "bicubic"}, {"name": "sharpen", "kind": "lite"}],
          "datasets": [{"name": "fake", "kind": "sen2neon", "manifest": "unused.jsonl"}], "statistics": {"n_boot": 100}, "metrics": {"hallucination_taus": [0.005]}}


def cfg(tmp_path, **over):
    d = dict(CONFIG, output_dir=str(tmp_path / "shift_out"), **over)
    return EvalConfig.from_dict(d)


def shift_run(tmp_path, *, datasets=None, lr_shifts=(0.0, 0.5), **kw):
    systems = [BicubicSystem("bicubic"), sharpen_system()]
    summary = run_shift_sensitivity(cfg(tmp_path), dataset="fake", lr_shifts=lr_shifts, systems=systems, datasets=datasets or [FakeDataset()], **kw)
    return summary, tmp_path / "shift_out"


def test_a_run_writes_its_record_and_one_row_per_sample_system_condition(tmp_path):
    summary, out = shift_run(tmp_path)
    assert sorted(p.name for p in out.iterdir()) == ["README.md", "aggregates.json", "config.json", "rows.jsonl", "summary.json"]
    rows = read_jsonl(out / "rows.jsonl")
    conditions = {"shift_0_lr_px", "shift_0.5_lr_px", "aligned_to_bicubic"}
    assert summary["conditions"] == ["shift_0_lr_px", "shift_0.5_lr_px", "aligned_to_bicubic"]
    assert {r["condition"] for r in rows} == conditions and len(rows) == 4 * 2 * 3        # 4 samples x 2 systems x 3 conditions, nothing dropped
    assert {r["dataset"] for r in rows} == {"fake"} and all(r["status"] == "ok" for r in rows)


def test_the_conditions_record_exactly_what_was_displaced(tmp_path):
    _, out = shift_run(tmp_path, lr_shifts=(0.0, 0.5, 1.0))
    rows = read_jsonl(out / "rows.jsonl")
    by = {r["condition"]: r for r in rows if r["system"] == "bicubic" and r["sample_id"] == "fake:s0"}
    assert (by["shift_0_lr_px"]["dy_hr_px"], by["shift_0_lr_px"]["dx_hr_px"]) == (0, 0)
    assert (by["shift_0.5_lr_px"]["dy_hr_px"], by["shift_0.5_lr_px"]["dx_hr_px"]) == (0, 2)
    assert (by["shift_1_lr_px"]["dy_hr_px"], by["shift_1_lr_px"]["dx_hr_px"]) == (0, 4)
    assert by["aligned_to_bicubic"]["dx_hr_px"] is not None and "alignment" in by["aligned_to_bicubic"]


def test_the_alignment_is_estimated_from_the_bicubic_baseline_and_is_the_same_for_every_system(tmp_path):
    """System-neutral: no system chooses its own alignment. The correction used for a sample is identical across systems."""
    _, out = shift_run(tmp_path)
    rows = [r for r in read_jsonl(out / "rows.jsonl") if r["condition"] == "aligned_to_bicubic"]
    for sample_id in {r["sample_id"] for r in rows}:
        corrections = {(r["dy_hr_px"], r["dx_hr_px"]) for r in rows if r["sample_id"] == sample_id}
        assert len(corrections) == 1


def test_aggregates_are_taken_over_scene_units_per_condition_and_against_bicubic(tmp_path):
    _, out = shift_run(tmp_path)
    agg = json.loads((out / "aggregates.json").read_text())
    zero = agg["conditions"]["shift_0_lr_px"]
    assert set(zero["systems"]) == {"bicubic", "sharpen"} and zero["systems"]["bicubic"]["n_units"] == 4
    assert "sharpen - bicubic" in zero["paired"]
    psnr = agg["conditions"]["shift_0.5_lr_px"]["systems"]["sharpen"]["metrics"]["reference_accuracy.psnr_db"]["unit"]
    assert psnr["n"] == 4 and psnr["ci"]["status"] != "ok"                                # 4 units: descriptive only, no interval


def test_the_readme_is_honest_about_what_it_is(tmp_path):
    _, out = shift_run(tmp_path)
    text = (out / "README.md").read_text()
    assert "not a ranking" in text.lower() and "descriptive" in text.lower()
    assert "whole HR pixel" in text and "classification" in text                           # states what was NOT done (downstream classification, area error)
    assert "aligned_to_bicubic" in text and "0.5" in text


def test_the_summary_records_provenance_and_counts(tmp_path):
    summary, out = shift_run(tmp_path)
    assert summary["dataset"] == "fake" and summary["evidence_class"] == "real_cross_sensor" and summary["counts"]["evaluated"] == 4
    assert summary["lr_shifts"] == [0.0, 0.5] and summary["hr_shifts"] == [0, 2] and summary["axis"] == "x (columns)"
    assert set(summary["systems"]) == {"bicubic", "sharpen"} and summary["metric_configuration"]["version"].startswith("frame-eval-metrics")
    assert json.loads((out / "summary.json").read_text())["status"] == "completed"


def test_results_are_never_overwritten(tmp_path):
    shift_run(tmp_path)
    with pytest.raises(EvaluationError, match="already contains"):
        shift_run(tmp_path)


def test_an_unknown_dataset_name_is_refused(tmp_path):
    with pytest.raises(EvaluationError, match="dataset"):
        run_shift_sensitivity(cfg(tmp_path), dataset="nope", systems=[BicubicSystem("bicubic")], datasets=[FakeDataset()])


def test_invalid_and_skipped_samples_are_listed_not_dropped(tmp_path):
    ds = FakeDataset(samples=[make_sample(i, valid_fraction=(0.02 if i == 2 else 1.0)) for i in range(4)],
                     invalid=[{"sample_id": "fake:s3", "codes": ["footprint_mismatch"], "messages": ["HR origin shifted"]}])
    summary, out = shift_run(tmp_path, datasets=[ds])
    c = summary["counts"]
    assert (c["total"], c["invalid"], c["skipped"], c["evaluated"]) == (4, 1, 1, 2) and summary["skipped"][0]["sample_id"] == "fake:s2"
    rows = read_jsonl(out / "rows.jsonl")
    assert {r["sample_id"] for r in rows if r["type"] == "sample_result"} == {"fake:s0", "fake:s1"}
    assert {(r["type"], r["sample_id"]) for r in rows if r["type"] != "sample_result"} == {("sample_skipped", "fake:s2"), ("sample_invalid", "fake:s3")}


def test_an_inference_failure_is_a_row_with_its_error(tmp_path):
    from frame.tests.test_evaluate_runner import Exploding

    summary_dir = tmp_path / "shift_out"
    systems = [BicubicSystem("bicubic"), Exploding("boom")]
    summary = run_shift_sensitivity(cfg(tmp_path, systems=[{"name": "bicubic", "kind": "bicubic"}, {"name": "boom", "kind": "lite"}]), dataset="fake", lr_shifts=(0.0,),
                                    systems=systems, datasets=[FakeDataset()])
    rows = read_jsonl(summary_dir / "rows.jsonl")
    failed = [r for r in rows if r["status"] == "failed"]
    assert len(failed) == 4 and all("simulated inference failure" in r["error"] and r["system"] == "boom" for r in failed) and summary["failures"] == {"boom": 4}
