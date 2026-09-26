"""frame.evaluate.stats -- descriptive summaries, cluster-aware aggregation, bootstrap CIs, paired tests (with honest labelling)."""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy import stats as scipy_stats

from frame.evaluate.stats import MIN_UNITS_CI, MIN_UNITS_TEST, bootstrap_ci, describe, group_means, paired_comparison


# ============================================================================== describe


def test_describe_reports_n_mean_median_std_min_max():
    d = describe([1.0, 2.0, 3.0, 4.0, 10.0])
    assert d["n"] == 5 and d["mean"] == pytest.approx(4.0) and d["median"] == 3.0 and d["min"] == 1.0 and d["max"] == 10.0
    assert d["std"] == pytest.approx(np.std([1, 2, 3, 4, 10], ddof=1)) and d["n_missing"] == 0


def test_describe_drops_and_counts_missing_values_instead_of_hiding_them():
    d = describe([1.0, None, float("nan"), 3.0, float("inf")])
    assert d["n"] == 2 and d["n_missing"] == 3 and d["mean"] == 2.0


def test_describe_of_one_or_zero_values_has_no_spread():
    assert describe([5.0])["std"] is None and describe([5.0])["mean"] == 5.0
    e = describe([])
    assert e["n"] == 0 and e["mean"] is None and e["median"] is None and e["std"] is None


# ============================================================================== cluster-aware aggregation


def test_group_means_averages_within_each_unit_first():
    """Tiles of one scene are correlated: they must count as ONE unit, not as independent samples."""
    values = [1.0, 3.0, 10.0, 20.0, 30.0]
    units = ["A", "A", "B", "B", "B"]
    ids, means, sizes = group_means(values, units)
    assert ids == ["A", "B"] and means == [2.0, 20.0] and sizes == [2, 3]


def test_group_means_skips_missing_values_and_drops_units_left_empty():
    ids, means, sizes = group_means([1.0, None, float("nan")], ["A", "B", "B"])
    assert ids == ["A"] and means == [1.0] and sizes == [1]


def test_group_means_requires_matching_lengths():
    with pytest.raises(ValueError, match="same length"):
        group_means([1.0, 2.0], ["A"])


# ============================================================================== bootstrap CI


def test_the_bootstrap_interval_brackets_the_estimate_and_is_reproducible_by_seed():
    x = list(np.linspace(30, 40, 12))
    a, b = bootstrap_ci(x, seed=3, n_boot=2000), bootstrap_ci(x, seed=3, n_boot=2000)
    assert a == b and a["status"] == "ok" and a["ci_low"] < a["estimate"] < a["ci_high"] and a["n"] == 12 and a["level"] == 0.95
    assert bootstrap_ci(x, seed=4, n_boot=2000)["ci_low"] != a["ci_low"]


def test_the_interval_narrows_as_the_number_of_units_grows():
    rng = np.random.default_rng(0)
    small, large = rng.normal(0, 1, 10), rng.normal(0, 1, 200)
    w = lambda x: (lambda c: c["ci_high"] - c["ci_low"])(bootstrap_ci(list(x), seed=0, n_boot=3000))
    assert w(large) < 0.5 * w(small)


def test_a_constant_sample_has_a_degenerate_interval_not_an_error():
    c = bootstrap_ci([2.5] * 8, seed=0, n_boot=500)
    assert c["ci_low"] == c["ci_high"] == 2.5


def test_a_tiny_sample_is_labelled_descriptive_only_and_gets_no_interval():
    c = bootstrap_ci([1.0, 2.0, 3.0], seed=0)
    assert c["status"] == "descriptive_only" and c["ci_low"] is None and c["ci_high"] is None and c["estimate"] == 2.0
    assert str(MIN_UNITS_CI) in c["reason"] and "3" in c["reason"]
    assert bootstrap_ci([1.0] * (MIN_UNITS_CI - 1))["status"] == "descriptive_only" and bootstrap_ci([1.0] * MIN_UNITS_CI)["status"] == "ok"


def test_the_median_statistic_is_supported():
    c = bootstrap_ci([1.0, 2.0, 3.0, 4.0, 100.0, 5.0, 6.0], statistic="median", seed=0, n_boot=1000)
    assert c["estimate"] == 4.0 and c["statistic"] == "median"


def test_missing_values_are_excluded_from_n():
    assert bootstrap_ci([1.0, 2.0, None, 4.0, 5.0, 6.0, float("nan")], seed=0, n_boot=200)["n"] == 5


# ============================================================================== paired comparison


def test_a_consistent_paired_improvement_is_reported_with_interval_test_and_effect_size():
    b = np.linspace(30, 33, 12)
    a = b + np.array([0.2, 0.3, 0.25, 0.4, 0.1, 0.35, 0.3, 0.2, 0.45, 0.15, 0.3, 0.28])
    r = paired_comparison(list(a), list(b), seed=0, n_boot=3000)
    assert r["n"] == 12 and r["mean_difference"] == pytest.approx(np.mean(a - b)) and r["median_difference"] == pytest.approx(np.median(a - b))
    assert r["ci"]["ci_low"] > 0 and r["wilcoxon"]["status"] == "tested" and r["wilcoxon"]["p_value"] < 0.01
    assert r["effect_size_dz"] == pytest.approx(np.mean(a - b) / np.std(a - b, ddof=1)) and r["fraction_a_greater"] == 1.0
    assert r["label"] == "inferential"


def test_the_wilcoxon_p_value_is_scipys_own():
    rng = np.random.default_rng(1)
    a, b = rng.normal(0.3, 1, 15), rng.normal(0, 1, 15)
    r = paired_comparison(list(a), list(b), seed=0, n_boot=200)
    expected = scipy_stats.wilcoxon(a, b, alternative="two-sided")
    assert r["wilcoxon"]["p_value"] == pytest.approx(expected.pvalue) and r["wilcoxon"]["statistic"] == pytest.approx(expected.statistic)


def test_a_no_effect_difference_does_not_manufacture_significance():
    """Built to have NO systematic effect (differences alternate in sign with growing size); a random-noise version would pass only ~95% of the time."""
    b = np.linspace(33, 37, 12)
    d = np.array([0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8, 0.9, -1.0, 1.1, -1.2])
    r = paired_comparison(list(b + d), list(b), seed=0, n_boot=3000)
    assert r["wilcoxon"]["status"] == "tested" and r["wilcoxon"]["p_value"] > 0.5 and r["ci"]["ci_low"] < 0 < r["ci"]["ci_high"]


def test_too_few_pairs_for_any_two_sided_test_is_descriptive_only():
    """With n pairs the smallest attainable exact two-sided p is 2 / 2^n: 0.0625 for n = 5, so no test can reach 0.05 below n = 6."""
    r = paired_comparison([1.0, 2.0, 3.0, 4.0, 5.0], [0.0, 0.0, 0.0, 0.0, 0.0], seed=0)
    assert r["wilcoxon"]["status"] == "descriptive_only" and r["wilcoxon"]["p_value"] is None and r["label"] == "descriptive_only"
    assert str(MIN_UNITS_TEST) in r["wilcoxon"]["reason"] and r["mean_difference"] == 3.0
    ok = paired_comparison([1.0] * 6 + [], [0.0, 0.1, 0.2, 0.3, 0.4, 0.5], seed=0, n_boot=200)
    assert ok["wilcoxon"]["status"] == "tested"


def test_three_pairs_get_the_difference_and_nothing_inferential():
    r = paired_comparison([36.0, 35.0, 37.0], [33.0, 33.5, 34.0], seed=0)
    assert r["label"] == "descriptive_only" and r["ci"]["status"] == "descriptive_only" and r["wilcoxon"]["p_value"] is None
    assert r["mean_difference"] == pytest.approx(np.mean([3.0, 1.5, 3.0])) and r["n"] == 3


def test_identical_pairs_give_zero_difference_and_no_test_statistic():
    r = paired_comparison([1.0] * 8, [1.0] * 8, seed=0, n_boot=200)
    assert r["mean_difference"] == 0.0 and r["wilcoxon"]["status"] == "no_nonzero_differences" and r["wilcoxon"]["p_value"] is None
    assert r["effect_size_dz"] is None


def test_pairs_with_a_missing_value_are_dropped_together_and_counted():
    r = paired_comparison([1.0, 2.0, None, 4.0, 5.0, 6.0, 7.0, 8.0], [0.5, 1.5, 3.0, None, 4.5, 5.5, 6.5, 7.5], seed=0, n_boot=200)
    assert r["n"] == 6 and r["n_dropped"] == 2


def test_lengths_must_match():
    with pytest.raises(ValueError, match="same length"):
        paired_comparison([1.0, 2.0], [1.0], seed=0)


def test_the_interval_is_of_the_paired_difference_not_of_two_separate_means():
    """A large shared between-unit spread must not widen the interval of the paired difference."""
    rng = np.random.default_rng(3)
    base = rng.normal(35, 5, 20)                             # big scene-to-scene variation
    r = paired_comparison(list(base + 0.5), list(base), seed=0, n_boot=2000)
    assert r["ci"]["ci_high"] - r["ci"]["ci_low"] < 1e-6 and r["ci"]["ci_low"] == pytest.approx(0.5)
