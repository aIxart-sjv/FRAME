"""frame.reliability.detection / calibration -- AUROC, AUPRC, flagging metrics, isotonic recalibration, reliability tables, interval coverage."""

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats as scipy_stats

from frame.reliability import calibration as C
from frame.reliability import detection as D


# ============================================================================== AUROC


def test_auroc_of_the_textbook_example_is_three_quarters():
    assert D.auroc([0.1, 0.4, 0.35, 0.8], [0, 0, 1, 1])["value"] == pytest.approx(0.75)


def test_auroc_is_one_for_perfect_zero_for_inverted_and_half_for_a_constant_score():
    label = np.array([0] * 30 + [1] * 30)
    score = np.arange(60.0)
    assert D.auroc(score, label)["value"] == pytest.approx(1.0)
    assert D.auroc(-score, label)["value"] == pytest.approx(0.0)
    assert D.auroc(np.ones(60), label)["value"] == pytest.approx(0.5)                      # ties earn half credit: a constant score is chance, not perfect


def test_auroc_matches_the_mann_whitney_statistic_with_ties():
    rng = np.random.default_rng(0)
    score = rng.integers(0, 8, 400).astype(float)
    label = (rng.random(400) < 0.3 + 0.05 * score).astype(int)
    u = scipy_stats.mannwhitneyu(score[label == 1], score[label == 0], alternative="two-sided").statistic
    assert D.auroc(score, label)["value"] == pytest.approx(u / ((label == 1).sum() * (label == 0).sum()))


def test_auroc_of_a_random_score_is_about_half():
    rng = np.random.default_rng(1)
    assert D.auroc(rng.random(20000), rng.random(20000) < 0.2)["value"] == pytest.approx(0.5, abs=0.02)


@pytest.mark.parametrize("label", [np.zeros(10, int), np.ones(10, int)])
def test_a_single_class_has_no_auroc_or_auprc(label):
    for fn in (D.auroc, D.auprc):
        r = fn(np.arange(10.0), label)
        assert r["status"] == "not_computable" and r["value"] is None and "class" in r["reason"]


def test_labels_must_be_binary_and_lengths_must_agree():
    with pytest.raises(ValueError, match="binary"):
        D.auroc([0.1, 0.2, 0.3], [0, 1, 2])
    with pytest.raises(ValueError, match="same length"):
        D.auroc([0.1, 0.2], [0, 1, 1])


def test_non_finite_scores_are_dropped_and_counted():
    r = D.auroc([0.1, np.nan, 0.4, 0.35, 0.8, np.inf], [0, 1, 0, 1, 1, 0])
    assert r["n"] == 4 and r["n_dropped"] == 2 and r["value"] == pytest.approx(0.75)


# ============================================================================== AUPRC


def test_auprc_of_the_textbook_example():
    assert D.auprc([0.1, 0.4, 0.35, 0.8], [0, 0, 1, 1])["value"] == pytest.approx(5 / 6)          # average precision, no interpolation


def test_auprc_perfect_ranking_is_one_and_its_baseline_is_the_prevalence():
    label = np.array([0] * 90 + [1] * 10)
    r = D.auprc(np.arange(100.0), label)
    assert r["value"] == pytest.approx(1.0) and r["prevalence"] == pytest.approx(0.1)


def test_auprc_of_a_random_score_is_about_the_prevalence_and_of_a_constant_score_exactly_so():
    rng = np.random.default_rng(2)
    label = rng.random(30000) < 0.1
    assert D.auprc(rng.random(30000), label)["value"] == pytest.approx(label.mean(), abs=0.015)
    assert D.auprc(np.ones(1000), np.arange(1000) % 10 == 0)["value"] == pytest.approx(0.1)    # one tied group: precision = prevalence, recall reaches 1 at once


def test_auprc_handles_tied_groups_as_one_threshold():
    score = np.array([1.0, 1.0, 1.0, 0.0, 0.0])
    label = np.array([1, 0, 0, 1, 0])
    # thresholds: {3 tied at 1.0}: precision 1/3, recall 1/2; then all: precision 2/5, recall 1  ->  AP = 0.5*(1/3) + 0.5*(2/5)
    assert D.auprc(score, label)["value"] == pytest.approx(0.5 / 3 + 0.5 * 0.4)


# ============================================================================== flagging the least stable fraction


def test_flagging_the_top_fraction_reports_precision_recall_and_lift():
    score = np.arange(100.0)
    label = np.zeros(100, int)
    label[90:] = 1                                                                        # the 10 highest scores are the high-error cases
    r = D.flag_metrics(score, label, fraction=0.1, seed=0)
    assert r["n_flagged"] == 10 and r["precision"] == pytest.approx(1.0) and r["recall"] == pytest.approx(1.0) and r["lift"] == pytest.approx(10.0)
    r2 = D.flag_metrics(score, label, fraction=0.2, seed=0)
    assert r2["precision"] == pytest.approx(0.5) and r2["recall"] == pytest.approx(1.0)


def test_flagging_with_tied_scores_is_seeded_and_not_input_ordered():
    label = np.array([1] * 50 + [0] * 50)
    a = D.flag_metrics(np.zeros(100), label, fraction=0.5, seed=1)
    assert a == D.flag_metrics(np.zeros(100), label, fraction=0.5, seed=1)
    assert a["precision"] != 1.0                                                          # an input-order tie-break would flag exactly the positives


# ============================================================================== isotonic recalibration


def test_isotonic_fit_pools_violators_to_a_monotone_non_decreasing_fit():
    m = C.isotonic_fit([1, 2, 3, 4], [1, 3, 2, 4])
    assert C.isotonic_predict(m, [1, 2, 3, 4]).tolist() == pytest.approx([1.0, 2.5, 2.5, 4.0])
    rng = np.random.default_rng(0)
    x = rng.random(500)
    y = x**2 + 0.1 * rng.standard_normal(500)
    fit = C.isotonic_predict(C.isotonic_fit(x, y), np.sort(x))
    assert (np.diff(fit) >= -1e-12).all()


def test_isotonic_predict_clips_outside_the_fitted_range_and_respects_weights():
    m = C.isotonic_fit([1, 2, 3], [1, 2, 3])
    assert C.isotonic_predict(m, [0, 10]).tolist() == pytest.approx([1.0, 3.0])
    w = C.isotonic_fit([1, 2], [4, 0], weights=[3, 1])
    assert C.isotonic_predict(w, [1, 2]).tolist() == pytest.approx([3.0, 3.0])            # weighted pooled mean (3*4 + 1*0) / 4


def test_isotonic_fit_refuses_empty_or_non_finite_only_input():
    with pytest.raises(ValueError, match="finite"):
        C.isotonic_fit([np.nan], [1.0])


# ============================================================================== reliability table and calibration line


def test_a_perfectly_calibrated_predictor_has_zero_expected_calibration_error():
    rng = np.random.default_rng(3)
    pred = rng.random(5000)
    t = C.reliability_table(pred, pred.copy(), n_bins=10)
    assert t["ece"] == pytest.approx(0.0, abs=1e-12) and len(t["bins"]) == 10 and sum(b["n"] for b in t["bins"]) == 5000
    assert all(b["gap"] == pytest.approx(0.0, abs=1e-12) for b in t["bins"])


def test_a_systematic_over_prediction_appears_as_a_signed_gap_and_ece():
    rng = np.random.default_rng(4)
    pred = rng.random(3000)
    t = C.reliability_table(pred, 0.5 * pred, n_bins=5)
    assert t["ece"] > 0.1 and all(b["gap"] < 0 for b in t["bins"][1:])                     # predicted more error than observed: observed - predicted < 0


def test_the_calibration_line_recovers_known_slope_and_intercept():
    rng = np.random.default_rng(5)
    pred = rng.random(1000)
    line = C.calibration_line(pred, 2.0 * pred + 1.0)
    assert line["slope"] == pytest.approx(2.0) and line["intercept"] == pytest.approx(1.0) and line["r2"] == pytest.approx(1.0)
    assert C.calibration_line(np.ones(10), np.arange(10.0))["status"] == "not_computable"


def test_equal_count_bins_handle_ties_without_empty_bins():
    t = C.reliability_table(np.array([0.0] * 50 + [1.0] * 50), np.arange(100.0), n_bins=10)
    assert sum(b["n"] for b in t["bins"]) == 100 and all(b["n"] > 0 for b in t["bins"])


# ============================================================================== is the spread a calibrated interval?


def test_interval_coverage_of_a_gaussian_predictor_matches_its_nominal_level():
    rng = np.random.default_rng(6)
    mean = np.zeros(200000)
    ref = rng.standard_normal(200000)
    std = np.ones(200000)
    assert C.interval_coverage(mean, std, ref, k=1.0)["coverage"] == pytest.approx(0.6827, abs=0.005)
    assert C.interval_coverage(mean, std, ref, k=2.0)["coverage"] == pytest.approx(0.9545, abs=0.005)
    assert C.interval_coverage(mean, std, ref, k=1.0)["nominal"] == pytest.approx(0.6827, abs=1e-3)


def test_a_spread_far_smaller_than_the_error_covers_almost_nothing():
    rng = np.random.default_rng(7)
    ref = 0.02 * rng.standard_normal(50000)                                                # errors of ~0.02
    r = C.interval_coverage(np.zeros(50000), np.full(50000, 0.001), ref, k=2.0)              # a spread of 0.001
    assert r["coverage"] < 0.1 and r["nominal"] > 0.95


def test_interval_coverage_drops_non_finite_and_reports_counts():
    r = C.interval_coverage(np.array([0.0, 0.0, np.nan]), np.array([1.0, 1.0, 1.0]), np.array([0.5, 5.0, 0.0]), k=1.0)
    assert r["n"] == 2 and r["n_dropped"] == 1 and r["coverage"] == pytest.approx(0.5)


def test_scale_ratio_says_how_many_times_smaller_the_spread_is_than_the_error():
    r = C.scale_ratio(np.full(100, 0.001), np.full(100, 0.02))
    assert r["median_error_over_median_spread"] == pytest.approx(20.0) and r["mean_error_over_mean_spread"] == pytest.approx(20.0)


# ============================================================================== recalibration assessed on held-out evidence


def test_a_monotone_link_learned_on_development_data_is_calibrated_on_test_data_from_the_same_process():
    rng = np.random.default_rng(8)
    x_dev, x_test = rng.random(4000), rng.random(4000)
    obs = lambda x: 3.0 * x + 0.1 * rng.standard_normal(x.size)                            # noqa: E731
    a = C.recalibration_assessment(x_dev, obs(x_dev), x_test, obs(x_test), n_bins=10)
    assert a["status"] == "ok" and a["line"]["slope"] == pytest.approx(1.0, abs=0.05) and abs(a["line"]["intercept"]) < 0.05
    assert a["skill_vs_constant"] > 0.9 and a["reliability"]["ece"] < 0.05


def test_an_uninformative_signal_earns_no_skill_and_a_slope_far_from_one():
    rng = np.random.default_rng(9)
    a = C.recalibration_assessment(rng.random(4000), rng.random(4000), rng.random(4000), rng.random(4000), n_bins=10)
    assert abs(a["skill_vs_constant"]) < 0.02 and a["line"]["status"] in ("ok", "not_computable")
    if a["line"]["status"] == "ok":
        assert abs(a["line"]["slope"] - 1.0) > 0.3 or abs(a["line"]["r2"]) < 0.05


def test_a_relationship_that_reverses_between_development_and_test_is_exposed():
    rng = np.random.default_rng(10)
    x_dev, x_test = rng.random(3000), rng.random(3000)
    a = C.recalibration_assessment(x_dev, x_dev + 0.02 * rng.standard_normal(3000), x_test, 1.0 - x_test + 0.02 * rng.standard_normal(3000), n_bins=10)
    assert a["skill_vs_constant"] < 0 and a["line"]["slope"] < 0


def test_too_little_data_is_not_assessable():
    a = C.recalibration_assessment(np.arange(5.0), np.arange(5.0), np.arange(5.0), np.arange(5.0), n_bins=10)
    assert a["status"] == "not_computable"


def test_the_expected_calibration_error_weights_bins_by_their_size():
    pred = np.arange(23.0)                                                                # 23 points in 5 equal-count bins: sizes 5, 5, 5, 4, 4
    obs = pred.copy()
    obs[:5] += 1.0                                                                        # only the first bin (5 points) is off, by exactly 1
    t = C.reliability_table(pred, obs, n_bins=5)
    assert [b["n"] for b in t["bins"]] == [5, 5, 5, 4, 4] and t["ece"] == pytest.approx(5 / 23)


def test_skill_is_measured_against_the_development_mean_not_a_baseline_that_has_seen_the_test_data():
    rng = np.random.default_rng(11)
    x_dev, x_test = rng.random(2000), rng.random(2000)
    a = C.recalibration_assessment(x_dev, x_dev, x_test, x_test + 5.0, n_bins=10)         # the test errors are shifted by a level the development data never showed
    assert -0.05 < a["skill_vs_constant"] < 0.05                                          # the map and the constant are both equally wrong; a test-mean baseline would make this hugely negative
    assert a["reliability"]["ece"] > 4.0                                                  # and the reliability table shows the shift plainly
