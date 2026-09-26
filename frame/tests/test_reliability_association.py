"""frame.reliability.association -- correlation, partial correlation, unit-clustered bootstrap, risk-coverage, and the development/test split."""

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats as scipy_stats

from frame.reliability import association as A


# ============================================================================== correlation


def test_spearman_is_one_for_any_monotone_increasing_relationship_and_minus_one_for_decreasing():
    x = np.linspace(0.1, 3, 40)
    assert A.correlation(x, np.exp(x), "spearman")["value"] == pytest.approx(1.0)
    assert A.correlation(x, -x**3, "spearman")["value"] == pytest.approx(-1.0)
    assert A.correlation(x, np.exp(x), "pearson")["value"] < 0.95                         # Pearson sees the curvature Spearman ignores


def test_pearson_matches_numpy_and_spearman_matches_scipy_with_ties():
    rng = np.random.default_rng(0)
    x = rng.integers(0, 6, 200).astype(float)                                              # many ties
    y = x + rng.standard_normal(200)
    assert A.correlation(x, y, "pearson")["value"] == pytest.approx(np.corrcoef(x, y)[0, 1])
    assert A.correlation(x, y, "spearman")["value"] == pytest.approx(scipy_stats.spearmanr(x, y).statistic)


def test_non_finite_pairs_are_dropped_and_counted_never_treated_as_zero():
    x = np.array([1.0, 2.0, np.nan, 4.0, 5.0, np.inf])
    y = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    r = A.correlation(x, y, "spearman")
    assert r["value"] == pytest.approx(1.0) and r["n"] == 4 and r["n_dropped"] == 2 and r["status"] == "ok"


@pytest.mark.parametrize("x, y, reason", [
    (np.ones(20), np.arange(20.0), "constant"),
    (np.arange(20.0), np.full(20, 0.3), "constant"),
    (np.array([1.0, 2.0]), np.array([1.0, 2.0]), "too_few"),
    (np.array([]), np.array([]), "too_few"),
])
def test_undefined_correlations_are_reported_with_a_reason_and_no_value(x, y, reason):
    r = A.correlation(x, y, "spearman")
    assert r["status"] == "not_computable" and r["value"] is None and reason in r["reason"]


def test_a_length_mismatch_is_an_error_not_a_silent_truncation():
    with pytest.raises(ValueError, match="same length"):
        A.correlation(np.arange(5.0), np.arange(6.0), "spearman")


def test_an_unknown_correlation_kind_is_refused():
    with pytest.raises(ValueError, match="kendall"):
        A.correlation(np.arange(5.0), np.arange(5.0), "kendall")


# ============================================================================== partial correlation (does stability add anything beyond a trivial predictor?)


def test_a_relationship_that_is_entirely_explained_by_a_control_vanishes_when_it_is_controlled_for():
    rng = np.random.default_rng(1)
    c = rng.standard_normal(2000)
    x = c + 0.3 * rng.standard_normal(2000)                       # 'stability' is texture plus noise
    y = c + 0.3 * rng.standard_normal(2000)                       # 'error' is texture plus independent noise
    assert A.correlation(x, y, "spearman")["value"] > 0.8
    assert abs(A.partial_spearman(x, y, [c])["value"]) < 0.1


def test_a_relationship_that_survives_the_control_stays_positive():
    rng = np.random.default_rng(2)
    c = rng.standard_normal(2000)
    x = rng.standard_normal(2000)
    y = 0.5 * c + 0.8 * x + 0.2 * rng.standard_normal(2000)      # x carries information about y that c does not
    assert A.partial_spearman(x, y, [c])["value"] > 0.7


def test_partial_correlation_handles_several_controls_and_bad_inputs():
    rng = np.random.default_rng(3)
    c1, c2, x = rng.standard_normal((3, 500))
    y = c1 + c2 + x
    assert A.partial_spearman(x, y, [c1, c2])["value"] > 0.4
    assert A.partial_spearman(x, y, [np.ones(500)])["status"] in ("ok", "not_computable")      # a constant control carries no information; must not crash
    with pytest.raises(ValueError, match="same length"):
        A.partial_spearman(x, y, [c1[:10]])
    assert A.partial_spearman(x[:3], y[:3], [c1[:3]])["status"] == "not_computable"


# ============================================================================== the unit-clustered bootstrap


def unit_payloads(n_units=10, per_unit=3, seed=0):
    rng = np.random.default_rng(seed)
    return {f"u{i}": list(rng.normal(loc=i * 0.1, scale=0.05, size=per_unit)) for i in range(n_units)}


def test_the_bootstrap_resamples_whole_units_and_brackets_the_estimate():
    units = unit_payloads()
    seen_sizes = set()

    def stat(payloads):
        seen_sizes.add(len(payloads))
        return float(np.mean([v for p in payloads for v in p]))

    r = A.cluster_bootstrap(units, stat, n_boot=300, alpha=0.05, seed=1)
    assert seen_sizes == {10} | {10}                                                        # every replicate has as many units as the data (whole units, with replacement)
    assert r["status"] == "ok" and r["ci_low"] < r["estimate"] < r["ci_high"] and r["n_units"] == 10
    assert r["estimate"] == pytest.approx(np.mean([v for u in units.values() for v in u]))


def test_the_bootstrap_is_reproducible_by_seed_and_differs_across_seeds():
    units = unit_payloads()
    stat = lambda payloads: float(np.mean([v for p in payloads for v in p]))            # noqa: E731
    a = A.cluster_bootstrap(units, stat, n_boot=200, alpha=0.05, seed=5)
    b = A.cluster_bootstrap(units, stat, n_boot=200, alpha=0.05, seed=5)
    c = A.cluster_bootstrap(units, stat, n_boot=200, alpha=0.05, seed=6)
    assert (a["ci_low"], a["ci_high"]) == (b["ci_low"], b["ci_high"]) and (a["ci_low"], a["ci_high"]) != (c["ci_low"], c["ci_high"])


def test_fewer_than_five_units_is_descriptive_only_with_no_interval():
    r = A.cluster_bootstrap(unit_payloads(n_units=4), lambda p: float(np.mean([v for q in p for v in q])), n_boot=200, alpha=0.05, seed=0)
    assert r["status"] == "descriptive_only" and r["ci_low"] is None and r["ci_high"] is None and "5" in r["reason"] and r["estimate"] is not None


def test_replicates_where_the_statistic_is_undefined_are_counted_and_too_many_of_them_withdraw_the_interval():
    units = unit_payloads()
    calls = {"n": 0}

    def flaky(payloads):
        calls["n"] += 1
        return None if calls["n"] % 2 == 0 else float(np.mean([v for p in payloads for v in p]))

    r = A.cluster_bootstrap(units, flaky, n_boot=100, alpha=0.05, seed=0)
    assert r["status"] == "unstable" and r["ci_low"] is None and r["n_failed_replicates"] > 10
    r2 = A.cluster_bootstrap(units, lambda p: None, n_boot=100, alpha=0.05, seed=0)
    assert r2["estimate"] is None and r2["status"] in ("not_computable", "unstable")


def test_a_statistic_that_needs_more_than_the_mean_gets_the_actual_resampled_data():
    units = {f"u{i}": (np.arange(5.0) + i, np.arange(5.0) + i) for i in range(8)}          # payload = (stability, error) arrays of one unit

    def pooled_spearman(payloads):
        x = np.concatenate([p[0] for p in payloads])
        y = np.concatenate([p[1] for p in payloads])
        return A.correlation(x, y, "spearman")["value"]

    r = A.cluster_bootstrap(units, pooled_spearman, n_boot=100, alpha=0.05, seed=0)
    assert r["estimate"] == pytest.approx(1.0) and r["ci_low"] == pytest.approx(1.0)


# ============================================================================== per-unit aggregation of within-tile correlations


def test_unit_means_average_tiles_within_a_unit_first():
    ids, means, counts = A.unit_means([0.2, 0.4, 0.9], ["a", "a", "b"])
    assert ids == ["a", "b"] and means == pytest.approx([0.3, 0.9]) and counts == [2, 1]
    ids2, means2, _ = A.unit_means([0.2, None, float("nan")], ["a", "a", "b"])
    assert ids2 == ["a"] and means2 == pytest.approx([0.2])                                    # undefined values are dropped, not zeroed


# ============================================================================== risk-coverage


def test_a_perfect_instability_ranking_gives_the_oracle_curve_and_full_selective_efficiency():
    err = np.linspace(0.01, 0.10, 100)
    rc = A.risk_coverage(score=err.copy(), error=err, coverage_grid=(1.0, 0.8, 0.5), seed=0)
    assert rc["curve"][0]["coverage"] == 1.0 and rc["curve"][0]["risk"] == pytest.approx(err.mean())
    assert rc["curve"][2]["risk"] == pytest.approx(np.sort(err)[:50].mean())
    assert rc["curve"][2]["risk"] == pytest.approx(rc["oracle_curve"][2]["risk"])
    assert rc["selective_efficiency"] == pytest.approx(1.0) and rc["aurc"] == pytest.approx(rc["aurc_oracle"])


def test_an_inverted_ranking_is_worse_than_random_and_says_so():
    err = np.linspace(0.01, 0.10, 100)
    rc = A.risk_coverage(score=-err, error=err, coverage_grid=(1.0, 0.5), seed=0)
    assert rc["curve"][1]["risk"] == pytest.approx(np.sort(err)[50:].mean()) and rc["curve"][1]["risk"] > rc["random_risk"]
    assert rc["selective_efficiency"] == pytest.approx(-1.0) and rc["risk_reduction_at"]["0.5"] < 0


def test_an_uninformative_score_reduces_no_risk_on_average():
    rng = np.random.default_rng(4)
    err = rng.random(20000)
    rc = A.risk_coverage(score=rng.random(20000), error=err, coverage_grid=(1.0, 0.5), seed=0)
    assert rc["curve"][1]["risk"] == pytest.approx(err.mean(), abs=0.01) and abs(rc["selective_efficiency"]) < 0.05


def test_risk_reduction_is_relative_to_full_coverage_and_the_grid_is_kept():
    err = np.array([0.1, 0.2, 0.3, 0.4])
    rc = A.risk_coverage(score=err, error=err, coverage_grid=(1.0, 0.5), seed=0)
    assert [c["coverage"] for c in rc["curve"]] == [1.0, 0.5] and rc["curve"][1]["n_retained"] == 2
    assert rc["risk_reduction_at"]["0.5"] == pytest.approx(1 - 0.15 / 0.25)                  # (0.25 - 0.15) / 0.25


def test_ties_in_the_score_are_broken_reproducibly_and_never_by_input_order():
    err = np.arange(1.0, 101.0)
    a = A.risk_coverage(score=np.zeros(100), error=err, coverage_grid=(1.0, 0.5), seed=3)
    b = A.risk_coverage(score=np.zeros(100), error=err, coverage_grid=(1.0, 0.5), seed=3)
    assert a["curve"][1]["risk"] == b["curve"][1]["risk"]
    assert abs(a["curve"][1]["risk"] - err.mean()) < 10                                          # not the biased 'first half' (25.5) an input-order tie-break would give
    risks = {A.risk_coverage(score=np.zeros(100), error=err, coverage_grid=(1.0, 0.5), seed=s)["curve"][1]["risk"] for s in range(20)}
    assert len(risks) > 1


def test_a_custom_risk_function_supports_rmse_style_aggregation():
    mse = np.array([1.0, 4.0, 9.0, 16.0])
    rc = A.risk_coverage(score=mse, error=mse, coverage_grid=(1.0, 0.5), seed=0, risk_fn=lambda e: float(np.sqrt(np.mean(e))))
    assert rc["curve"][0]["risk"] == pytest.approx(np.sqrt(7.5)) and rc["curve"][1]["risk"] == pytest.approx(np.sqrt(2.5))


def test_non_finite_items_are_excluded_and_counted():
    rc = A.risk_coverage(score=np.array([1.0, np.nan, 3.0, 4.0]), error=np.array([0.1, 0.2, np.inf, 0.4]), coverage_grid=(1.0,), seed=0)
    assert rc["n"] == 2 and rc["n_dropped"] == 2
    empty = A.risk_coverage(score=np.array([np.nan]), error=np.array([1.0]), coverage_grid=(1.0,), seed=0)
    assert empty["status"] == "not_computable"


# ============================================================================== the development / test split of scene units


def test_the_split_is_seeded_disjoint_and_covers_every_unit():
    units = [f"u{i}" for i in range(20)]
    s = A.dev_test_split(units, seed=1, dev_fraction=0.5, min_units=12)
    assert s["status"] == "ok" and set(s["dev"]) | set(s["test"]) == set(units) and not set(s["dev"]) & set(s["test"])
    assert len(s["dev"]) == 10 and A.dev_test_split(units, seed=1, dev_fraction=0.5, min_units=12) == s
    assert A.dev_test_split(units, seed=2, dev_fraction=0.5, min_units=12)["dev"] != s["dev"]
    assert A.dev_test_split(list(reversed(units)), seed=1, dev_fraction=0.5, min_units=12) == s          # input order does not matter


def test_too_few_units_cannot_be_split_and_the_result_is_descriptive_only():
    s = A.dev_test_split([f"u{i}" for i in range(11)], seed=0, dev_fraction=0.5, min_units=12)
    assert s["status"] == "not_splittable" and s["dev"] == [] and s["test"] == [] and "12" in s["reason"]


# ============================================================================== several statistics from one set of resamples


def test_the_multi_bootstrap_gives_every_statistic_its_own_interval_from_the_same_resamples():
    units = unit_payloads()
    seen = []

    def stats(payloads):
        values = np.array([v for p in payloads for v in p])
        seen.append(len(payloads))
        return {"mean": float(values.mean()), "max": float(values.max()), "broken": None}

    r = A.cluster_bootstrap_multi(units, stats, n_boot=200, alpha=0.05, seed=2)
    assert set(r) == {"mean", "max", "broken"} and len(seen) == 201                              # one call for the estimate and one per replicate, shared by all statistics
    assert r["mean"]["status"] == "ok" and r["mean"]["ci_low"] < r["mean"]["estimate"] < r["mean"]["ci_high"] and r["max"]["status"] == "ok"
    assert r["broken"]["status"] == "not_computable" and r["broken"]["estimate"] is None
    single = A.cluster_bootstrap(units, lambda p: float(np.mean([v for q in p for v in q])), n_boot=200, alpha=0.05, seed=2)
    assert (r["mean"]["ci_low"], r["mean"]["ci_high"]) == (single["ci_low"], single["ci_high"])       # same seed, same resamples, same interval


def test_the_multi_bootstrap_applies_the_unit_count_rule_to_every_statistic():
    r = A.cluster_bootstrap_multi(unit_payloads(n_units=3), lambda p: {"a": 1.0, "b": 2.0}, n_boot=100, alpha=0.05, seed=0)
    assert r["a"]["status"] == r["b"]["status"] == "descriptive_only" and r["a"]["ci_low"] is None and r["a"]["estimate"] == 1.0
