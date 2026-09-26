"""frame.reliability.analysis -- from per-tile evidence to correlation, tile/scene reliability, risk-coverage, high-error detection, calibration, cost and model comparison."""

from __future__ import annotations

import numpy as np
import pytest
from scipy import ndimage

from frame.reliability import analysis as AN
from frame.reliability.association import dev_test_split
from frame.reliability.config import ReliabilityConfig
from frame.reliability.evidence import EnsembleArrays, TileInputs, compute_tile_evidence

BANDS = ("B04", "B03", "B02", "B08")
SIZE = 128
GATE = {"status": "eligible", "reason": None, "detail": "ok", "evidence_level": "pixel_level_eligible", "valid_fraction": 1.0, "nonfinite_fraction": 0.0, "valid_fraction_after_alignment": 1.0,
        "alignment": {"status": "eligible", "reason": None, "correction": [0, 0], "applied": False, "correction_method": "none_required", "method": "bicubic_baseline_cross_correlation",
                      "residual_magnitude": 0.1, "quadrant_spread": 0.2, "crop_rows": [0, SIZE], "crop_cols": [0, SIZE], "raw_magnitude": 0.1}}


def config(**over):
    d = {"name": "an", "output_dir": "o", "systems": [{"name": "lite", "kind": "lite"}, {"name": "mamba", "kind": "mamba"}], "datasets": [{"name": "d", "kind": "sen2neon", "manifest": "m"}],
         "analysis": {"pooled_cells_per_tile": 400, "pooled_pixels_per_tile": 800, "displacement_sweep_hr_px": [0], "min_units_for_split": 12, **over.pop("analysis", {})},
         "bootstrap": {"n_boot": 200, "alpha": 0.05, "seed": 0}, **over}
    return ReliabilityConfig.from_dict(d)


def smooth(seed, sigma=4.0):
    f = ndimage.gaussian_filter(np.random.default_rng(seed).standard_normal((SIZE, SIZE)), sigma, mode="wrap")
    return (f - f.min()) / (f.max() - f.min())


def make_tile(unit, k, level, informative, system="lite", seed_offset=0, seconds=(0.1, 1.0)):
    """A tile whose error and (if informative) spread scale with ``level`` (a tile-level relationship) and with a within-tile difficulty field."""
    seed = 1000 * seed_offset + 37 * int(unit[1:]) + k
    rng = np.random.default_rng(seed)
    hr = (0.25 + 0.05 * rng.standard_normal((4, SIZE, SIZE))).clip(0.02, 0.9)
    difficulty = 0.3 + smooth(seed + 1)
    pred = hr + 0.01 * level * difficulty[None] * rng.standard_normal(hr.shape)
    other = 0.3 + smooth(seed + 2)
    spread_level = level if informative else rng.uniform(0.5, 3.0)
    std = np.broadcast_to((0.0005 * spread_level * (difficulty if informative else other))[None], hr.shape).copy()
    bic = ndimage.gaussian_filter(hr, (0, 2, 2))
    ens = EnsembleArrays(mean=pred, std=std, member0=pred, n_members=6, transform_names=("identity", "hflip", "vflip", "rot90", "rot180", "rot270"), seed=42, seconds_total=seconds[1],
                         seconds_per_member=tuple([seconds[1] / 6] * 6))
    inp = TileInputs(dataset="d", system=system, sample_id=f"{unit}:{k}", scene_unit=unit, category=None, bands=BANDS, scale=4, bicubic=bic, hr=hr, mask=np.ones((SIZE, SIZE), bool), gate=GATE,
                     ensemble=ens)
    return inp


def evidence(n_units, per_unit, informative, cfg, system="lite", seconds=(0.1, 1.0)):
    rows, arrays = [], {}
    rng = np.random.default_rng(5)
    for u in range(n_units):
        unit = f"u{u:02d}"
        for k in range(per_unit):
            level = float(rng.uniform(0.5, 3.0))
            row, arr = compute_tile_evidence(make_tile(unit, k, level, informative, system=system, seconds=seconds), cfg)
            rows.append(row)
            arrays[row["sample_id"]] = arr
    return rows, arrays


@pytest.fixture(scope="module")
def cfg():
    return config()


@pytest.fixture(scope="module")
def informative(cfg):
    rows, arrays = evidence(14, 2, True, cfg)
    return rows, arrays, AN.analyse_system(rows, arrays, cfg)


@pytest.fixture(scope="module")
def uninformative(cfg):
    rows, arrays = evidence(14, 2, False, cfg)
    return rows, arrays, AN.analyse_system(rows, arrays, cfg)


# ============================================================================== counts and structure


def test_the_analysis_counts_tiles_and_scene_units_and_names_its_evidence(informative):
    _, _, a = informative
    assert a["status"] == "ok" and a["n_tiles"] == 28 and a["n_units"] == 14 and a["evidence_level"] == "pixel_level_eligible"
    assert {"pixel_level", "cell_level", "tile_level", "scene_level", "risk_coverage", "high_error_detection", "calibration", "detail_relationship", "tta_cost"} <= set(a)
    assert "not a calibrated uncertainty" in a["interpretation"].lower()


def test_no_eligible_evidence_is_reported_as_such_not_as_a_result(cfg):
    a = AN.analyse_system([], {}, cfg)
    assert a["status"] == "no_eligible_evidence" and a["n_tiles"] == 0 and "pixel_level" not in a


# ============================================================================== pixel and cell level, aggregated over units


def test_within_tile_correlations_are_aggregated_over_scene_units_with_an_interval(informative):
    _, _, a = informative
    p = a["pixel_level"]["spearman_abs_error"]
    assert p["n_tiles"] == 28 and p["n_units"] == 14 and p["unit"]["mean"] > 0.1 and p["share_positive"] > 0.9
    assert p["ci"]["status"] == "ok" and p["ci"]["ci_low"] > 0
    c = a["cell_level"]["4"]["spearman_abs_error"]
    assert c["unit"]["mean"] > p["unit"]["mean"] and c["ci"]["ci_low"] > 0.2                  # the relationship is clearer at 10 m than per pixel


def test_an_uninformative_stability_has_an_interval_that_includes_zero(uninformative):
    _, _, a = uninformative
    for level in (a["pixel_level"]["spearman_abs_error"], a["cell_level"]["4"]["spearman_abs_error"]):
        assert level["ci"]["status"] == "ok" and level["ci"]["ci_low"] < 0.0 < level["ci"]["ci_high"] or abs(level["unit"]["mean"]) < 0.15


def test_the_baseline_predictors_and_the_partial_correlation_are_reported_alongside(informative):
    _, _, a = informative
    c = a["cell_level"]["4"]
    assert {"spearman_texture_abs_error", "spearman_added_detail_abs_error", "partial_spearman_abs_error_given_baselines"} <= set(c)
    assert c["partial_spearman_abs_error_given_baselines"]["unit"]["mean"] > 0.1


def test_the_pooled_cell_correlation_uses_a_unit_clustered_interval(informative):
    _, _, a = informative
    pooled = a["cell_level"]["4"]["pooled_spearman_abs_error"]
    assert pooled["estimate"] > 0.1 and pooled["method"] == "percentile bootstrap over scene units" and pooled["n_units"] == 14 and pooled["status"] == "ok"


# ============================================================================== tile and scene level


def test_tile_level_reliability_relates_the_tile_stability_to_the_tile_error_targets(informative):
    _, _, a = informative
    t = a["tile_level"]
    for target in ("rmse", "mae", "sam_degrees", "ergas"):
        e = t["stability_vs"][target]
        assert e["n_tiles"] == 28 and e["spearman"]["value"] > 0.5 and e["spearman_ci"]["status"] == "ok" and e["spearman_ci"]["ci_low"] > 0
    s = a["scene_level"]["stability_vs"]["rmse"]
    assert s["n_units"] == 14 and s["spearman"]["value"] > 0.5 and s["spearman"]["n"] == 14


def test_tile_level_reliability_of_an_uninformative_stability_is_near_zero(uninformative):
    _, _, a = uninformative
    e = a["tile_level"]["stability_vs"]["rmse"]
    assert abs(e["spearman"]["value"]) < 0.5 and (e["spearman_ci"]["ci_low"] < 0.0 < e["spearman_ci"]["ci_high"] or e["spearman_ci"]["status"] != "ok")


def test_tile_level_baselines_and_the_detail_relationships_are_reported(informative):
    _, _, a = informative
    t = a["tile_level"]
    assert {"stability_vs", "texture_vs", "added_detail_vs", "partial_stability_given_baselines"} <= set(t)
    d = a["detail_relationship"]
    assert {"unsupported_detail", "omission", "supported_synthesis"} <= set(d["tile_level"]) and "4" in d["cell_level"] and "definition" in d


# ============================================================================== risk-coverage


def test_removing_the_most_unstable_tiles_lowers_the_remaining_error_when_stability_is_informative(informative):
    _, _, a = informative
    rc = a["risk_coverage"]["tile_rmse"]
    assert rc["curve"][0]["risk"] == pytest.approx(rc["random_risk"]) and rc["curve"][-1]["risk"] < rc["curve"][0]["risk"]
    assert rc["selective_efficiency"] > 0.3 and rc["ci"]["risk_reduction_at"]["0.8"]["ci_low"] > 0 and rc["n_units"] == 14
    assert all(c["risk"] >= o["risk"] - 1e-12 for c, o in zip(rc["curve"], rc["oracle_curve"]))    # no ranking can beat the oracle
    assert "calibrated" in rc["note"].lower() and "not" in rc["note"].lower()
    assert "texture_baseline" in rc and "added_detail_baseline" in rc


def test_an_uninformative_stability_removes_no_risk(uninformative):
    _, _, a = uninformative
    rc = a["risk_coverage"]["tile_rmse"]
    assert abs(rc["selective_efficiency"]) < 0.5 and rc["ci"]["risk_reduction_at"]["0.8"]["ci_low"] < 0.02


def test_the_cell_level_risk_coverage_is_pooled_over_units_with_a_clustered_interval(informative):
    _, _, a = informative
    rc = a["risk_coverage"]["cell_4_abs_error"]
    assert rc["status"] == "ok" and rc["selective_efficiency"] > 0.05 and rc["ci"]["selective_efficiency"]["status"] == "ok"


# ============================================================================== high-error detection with a development-only threshold


def test_the_high_error_threshold_comes_from_development_units_only_and_is_applied_to_test_units(informative, cfg):
    rows, arrays, a = informative
    d = a["high_error_detection"]
    assert d["status"] == "ok" and d["threshold_source"] == "development units only"
    split = dev_test_split(sorted({r["scene_unit"] for r in rows}), seed=cfg.analysis.split_seed, dev_fraction=cfg.analysis.dev_fraction, min_units=cfg.analysis.min_units_for_split)
    assert d["dev_units"] == split["dev"] and d["test_units"] == split["test"] and not set(d["dev_units"]) & set(d["test_units"])
    dev_err = np.concatenate([arrays[r["sample_id"]]["c4_abs_error"] for r in rows if r["scene_unit"] in split["dev"]])
    assert d["threshold"] == pytest.approx(float(np.quantile(dev_err, cfg.analysis.high_error_quantile)), rel=1e-6)
    test_err = np.concatenate([arrays[r["sample_id"]]["c4_abs_error"] for r in rows if r["scene_unit"] in split["test"]])
    assert d["n_test_cells"] == test_err.size and d["prevalence_test"] == pytest.approx(float((test_err > d["threshold"]).mean()))


def test_an_informative_stability_detects_high_error_cells_better_than_chance(informative):
    _, _, a = informative
    d = a["high_error_detection"]
    assert d["stability"]["auroc"]["value"] > 0.55 and d["stability"]["auroc_ci"]["status"] == "ok" and d["stability"]["auprc"]["value"] > d["prevalence_test"]
    assert {"texture_baseline", "added_detail_baseline"} <= set(d) and d["stability"]["flag"]["0.1"]["lift"] > 1.0


def test_an_uninformative_stability_detects_nothing(uninformative):
    _, _, a = uninformative
    assert 0.4 < a["high_error_detection"]["stability"]["auroc"]["value"] < 0.6


def test_too_few_units_cannot_be_split_so_detection_is_descriptive_only_and_says_so(cfg):
    rows, arrays = evidence(6, 2, True, cfg)
    a = AN.analyse_system(rows, arrays, cfg)
    d = a["high_error_detection"]
    assert d["status"] == "descriptive_only" and "development/test split" in d["reason"] and d["threshold_source"].startswith("same evidence")
    assert d["stability"]["auroc"]["value"] is not None and d["stability"]["auroc_ci"]["status"] in ("descriptive_only", "not_computable")


# ============================================================================== calibration


def test_the_raw_spread_is_reported_as_uncalibrated_with_its_scale_and_coverage(informative):
    _, _, a = informative
    c = a["calibration"]
    assert c["raw"]["scale"]["median_error_over_median_spread"] > 5 and c["raw"]["coverage"]["k2"]["coverage"] < 0.3 and c["raw"]["coverage"]["k2"]["nominal"] > 0.95
    assert c["verdict"] in ("uncalibrated_stability_evidence",) or c["verdict"].startswith("uncalibrated")


def test_recalibration_is_assessed_on_test_units_and_never_promoted_to_calibrated_uncertainty(informative):
    _, _, a = informative
    r = a["calibration"]["recalibration"]
    assert r["status"] == "ok" and r["fitted_on"] == "development units" and r["assessed_on"] == "test units" and r["skill_vs_constant"] > 0.0
    assert r["line_ci"]["slope"]["status"] == "ok" and "reliability" in r
    assert "not a calibrated" in a["calibration"]["statement"].lower() or "uncalibrated" in a["calibration"]["statement"].lower()


def test_calibration_is_not_assessed_when_the_evidence_cannot_be_split(cfg):
    rows, arrays = evidence(6, 2, True, cfg)
    c = AN.analyse_system(rows, arrays, cfg)["calibration"]
    assert c["recalibration"]["status"] == "not_assessable" and c["verdict"] == "uncalibrated_stability_evidence" and "split" in c["recalibration"]["reason"]


# ============================================================================== small samples are descriptive only


def test_four_units_give_no_intervals_and_no_claims(cfg):
    rows, arrays = evidence(4, 2, True, cfg)
    a = AN.analyse_system(rows, arrays, cfg)
    assert a["n_units"] == 4
    assert a["pixel_level"]["spearman_abs_error"]["ci"]["status"] == "descriptive_only"
    assert a["tile_level"]["stability_vs"]["rmse"]["spearman_ci"]["status"] == "descriptive_only"
    assert a["risk_coverage"]["tile_rmse"]["ci"]["selective_efficiency"]["status"] == "descriptive_only"
    assert a["descriptive_only"] is True and "descriptive" in a["interpretation"].lower()


# ============================================================================== determinism


def test_the_analysis_is_deterministic(informative, cfg):
    rows, arrays, a = informative
    assert AN.analyse_system(rows, arrays, cfg) == a


# ============================================================================== cost and model comparison


def test_the_tta_cost_is_summarised_from_the_recorded_latencies(cfg):
    rows, arrays = evidence(3, 1, True, cfg, seconds=(0.2, 1.2))
    cost = AN.analyse_system(rows, arrays, cfg)["tta_cost"]
    assert cost["n_members"] == 6 and cost["single_pass_seconds_median"] == pytest.approx(0.2) and cost["tta_seconds_median"] == pytest.approx(1.2) and cost["tta_over_single_pass"] == pytest.approx(6.0)


def test_two_systems_are_compared_on_the_tiles_they_share_without_ranking(cfg):
    lite, _ = evidence(14, 2, True, cfg, system="lite")
    mamba, _ = evidence(14, 2, False, cfg, system="mamba")
    cmp = AN.compare_systems({"lite": lite, "mamba": mamba}, cfg)
    assert cmp["n_shared_tiles"] == 28 and cmp["n_shared_units"] == 14 and cmp["pairs"][0]["a"] == "lite" and cmp["pairs"][0]["b"] == "mamba"
    row = cmp["pairs"][0]["metrics"]["cells.4.spearman_abs_error.value"]
    assert row["n"] == 14 and row["mean_difference"] > 0 and row["label"] in ("inferential", "descriptive_only")
    assert "not a ranking" in cmp["note"].lower()


def test_comparison_uses_only_tiles_eligible_for_both_systems(cfg):
    lite, _ = evidence(6, 2, True, cfg, system="lite")
    mamba = [r for r in evidence(6, 2, True, cfg, system="mamba")[0] if r["scene_unit"] != "u00"]
    cmp = AN.compare_systems({"lite": lite, "mamba": mamba}, cfg)
    assert cmp["n_shared_units"] == 5 and cmp["n_shared_tiles"] == 10 and cmp["only_in"] == {"lite": 2, "mamba": 0}


# ============================================================================== leakage and definitions, on constructed evidence


def test_recalibration_is_judged_on_test_units_so_a_relationship_that_exists_only_in_development_units_earns_no_skill(cfg):
    units = [f"u{i:02d}" for i in range(14)]
    split = dev_test_split(units, seed=cfg.analysis.split_seed, dev_fraction=cfg.analysis.dev_fraction, min_units=cfg.analysis.min_units_for_split)
    rows, arrays = [], {}
    rng = np.random.default_rng(3)
    for unit in units:
        for k in range(2):
            level = float(rng.uniform(0.5, 3.0))
            row, arr = compute_tile_evidence(make_tile(unit, k, level, informative=unit in split["dev"]), cfg)      # informative on development units, unrelated on test units
            rows.append(row)
            arrays[row["sample_id"]] = arr
    rec = AN.analyse_system(rows, arrays, cfg)["calibration"]["recalibration"]
    assert rec["status"] == "ok" and rec["skill_vs_constant"] < 0.05 and rec["verdict"] == "not_supported"


def fake_rows(pairs, unit_of=lambda i: f"u{i}", **extra):
    """Result-row stubs carrying only the paths the tile-level analysis reads: pairs are (s.x, s.y, s.c)."""
    return [{"scene_unit": unit_of(i), "sample_id": f"t{i}", "s": {"x": x, "y": y, "c": c}, **extra} for i, (x, y, c) in enumerate(pairs)]


def test_the_tile_level_partial_correlation_removes_what_a_trivial_predictor_explains(cfg):
    rng = np.random.default_rng(0)
    c = rng.standard_normal(60)
    x, y = c + 0.3 * rng.standard_normal(60), c + 0.3 * rng.standard_normal(60)
    rows = fake_rows(list(zip(x, y, c)))
    r = AN._tile_association(rows, "s.x", "s.y", cfg, controls=["s.c"])
    assert r["tile"]["spearman"]["value"] > 0.8 and abs(r["tile"]["partial_spearman"]["value"]) < 0.3
    assert r["scene"]["partial_spearman"]["n"] == 60 and r["tile"]["partial_ci"]["status"] == "ok"


def test_the_tile_rmse_risk_is_the_rmse_of_the_retained_tiles_not_the_mean_of_their_rmse(cfg):
    stubs = [{"scene_unit": f"u{i}", "sample_id": f"t{i}", "stability": {"mean": v}, "targets": {"product": {"rmse": v, "sam_degrees": v}},
              "baselines": {"texture_mean": v, "added_detail_mean": v}} for i, v in enumerate([0.01, 0.02, 0.03, 0.04, 0.05, 0.06])]
    rc = AN._risk_coverage_section(stubs, {}, cfg)["tile_rmse"]
    half = [c for c in rc["curve"] if c["coverage"] == 0.5][0]
    assert half["n_retained"] == 3 and half["risk"] == pytest.approx(np.sqrt(np.mean(np.array([0.01, 0.02, 0.03]) ** 2)))
    assert rc["curve"][0]["risk"] == pytest.approx(np.sqrt(np.mean(np.array([0.01, 0.02, 0.03, 0.04, 0.05, 0.06]) ** 2)))


def test_the_first_tile_of_a_run_is_excluded_from_the_latency_medians_because_it_carries_warm_up():
    rows = [{"tta": {"n_members": 6, "transforms": ["identity"], "total_seconds": t, "single_pass_seconds": s}} for t, s in ((100.0, 20.0), (1.2, 0.2))]
    cost = AN._cost_section(rows)
    assert cost["tta_seconds_median"] == pytest.approx(1.2) and cost["single_pass_seconds_median"] == pytest.approx(0.2) and cost["first_tile_tta_seconds"] == 100.0
    assert cost["n_tiles_timed"] == 1


# ============================================================================== statistics across units need more units than a mean of unit means


def test_a_correlation_across_fewer_than_ten_units_is_descriptive_only_because_its_bootstrap_degenerates(cfg):
    """Found on the real run: six OpenSR `spot` tiles ranked identically by stability, texture and error gave a Spearman of 1.000 with an interval [1.000, 1.000]. With so few units the
    bootstrap distribution of a rank correlation is discrete and, for a monotone relationship, a single point: it looks like certainty and is not. A MEAN of per-unit values keeps the 5-unit rule."""
    rows, arrays = evidence(8, 2, True, cfg)
    a = AN.analyse_system(rows, arrays, cfg)
    assert a["n_units"] == 8
    assert a["pixel_level"]["spearman_abs_error"]["ci"]["status"] == "ok"                              # a mean over 8 units: interval allowed
    assert a["tile_level"]["stability_vs"]["rmse"]["spearman_ci"]["status"] == "descriptive_only" and "10" in a["tile_level"]["stability_vs"]["rmse"]["spearman_ci"]["reason"]
    assert a["scene_level"]["stability_vs"]["rmse"]["spearman_ci"]["status"] == "descriptive_only"
    assert a["risk_coverage"]["tile_rmse"]["ci"]["selective_efficiency"]["status"] == "descriptive_only"
    assert a["detail_relationship"]["tile_level"]["unsupported_detail"]["spearman_ci"]["status"] == "descriptive_only"
    assert a["tile_level"]["stability_vs"]["rmse"]["spearman"]["value"] is not None                     # the point estimate is still reported


def test_ten_units_are_enough_for_an_across_unit_interval(cfg):
    rows, arrays = evidence(10, 1, True, cfg)
    a = AN.analyse_system(rows, arrays, cfg)
    assert a["tile_level"]["stability_vs"]["rmse"]["spearman_ci"]["status"] == "ok" and a["risk_coverage"]["tile_rmse"]["ci"]["selective_efficiency"]["status"] == "ok"
