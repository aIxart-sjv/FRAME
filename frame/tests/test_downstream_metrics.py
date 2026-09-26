"""frame.downstream.metrics -- NDVI fidelity and decision utility per region, per scene unit, and against the bicubic baseline; on hand-computable tables."""

from __future__ import annotations

import numpy as np
import pytest

from frame.downstream.config import DownstreamConfig
from frame.downstream.metrics import aggregate_units, decision_from_counts, region_metrics, unit_metrics


def cfg(**over):
    d = {"name": "m", "output_dir": "o", "systems": [{"name": "lite", "kind": "lite"}, {"name": "mamba", "kind": "mamba"}], "datasets": [{"name": "d", "kind": "sen2neon", "manifest": "m"}],
         "bootstrap": {"n_boot": 200}, **over}
    return DownstreamConfig.from_dict(d)


def table(units, ref, systems, n=16, size=4, tp=None, fp=None, fn=None, veg_ref=None, thresholds=("0.3",)):
    """A region table with the given per-region reference / system NDVI means and (optional) decision counts."""
    k = len(ref)
    t = {"row": np.arange(k, dtype=np.int32), "col": np.zeros(k, np.int32), "n_valid": np.full(k, n, np.int32), "scene_unit": np.asarray(units, dtype=object), "size": size,
         "tile_id": np.asarray([f"{u}:t" for u in units], dtype=object), "mean:ref": np.asarray(ref, np.float32)}
    for name, values in systems.items():
        t[f"mean:{name}"] = np.asarray(values, np.float32)
    for th in thresholds:
        t[f"veg:ref@{th}"] = np.asarray(veg_ref if veg_ref is not None else np.zeros(k), np.int32)
        for name in systems:
            t[f"tp:{name}@{th}"] = np.asarray((tp or {}).get(name, np.zeros(k)), np.int32)
            t[f"fp:{name}@{th}"] = np.asarray((fp or {}).get(name, np.zeros(k)), np.int32)
            t[f"fn:{name}@{th}"] = np.asarray((fn or {}).get(name, np.zeros(k)), np.int32)
            t[f"veg:{name}@{th}"] = t[f"tp:{name}@{th}"] + t[f"fp:{name}@{th}"]
    return t


# ============================================================================== NDVI fidelity per region


def test_region_ndvi_errors_are_signed_absolute_and_squared():
    t = table(["u"] * 3, [0.5, 0.2, 0.7], {"a": [0.6, 0.1, 0.7]})
    m = region_metrics(t, "a", "0.3", majority_fraction=0.5)
    assert np.allclose(m["ndvi_error"], [0.1, -0.1, 0.0], atol=1e-6) and np.allclose(m["ndvi_abs_error"], [0.1, 0.1, 0.0], atol=1e-6)
    assert np.allclose(m["ndvi_sq_error"], [0.01, 0.01, 0.0], atol=1e-6)


def test_the_vegetation_fraction_error_and_the_disagreement_rate_come_from_the_counts():
    # region 0: reference 8 vegetated of 16; system flags 6 of those and 4 others -> tp 6, fp 4, fn 2
    t = table(["u"], [0.4], {"a": [0.4]}, tp={"a": [6]}, fp={"a": [4]}, fn={"a": [2]}, veg_ref=[8])
    m = region_metrics(t, "a", "0.3", majority_fraction=0.5)
    assert m["ref_veg_fraction"][0] == 0.5 and m["veg_fraction"][0] == 10 / 16 and m["veg_fraction_abs_error"][0] == pytest.approx(2 / 16)
    assert m["disagreement"][0] == pytest.approx(6 / 16)


def test_the_region_decision_is_the_majority_rule_and_is_inclusive_at_the_fraction():
    t = table(["u"] * 3, [0.4] * 3, {"a": [0.4] * 3}, tp={"a": [8, 4, 0]}, fp={"a": [0, 0, 0]}, fn={"a": [0, 4, 8]}, veg_ref=[8, 8, 8])
    m = region_metrics(t, "a", "0.3", majority_fraction=0.5)
    # region 0: reference 8/16 = 0.5 -> vegetated; system 8/16 -> vegetated: agree.  region 1: system 4/16 -> not vegetated: disagree.  region 2: system 0/16: disagree.
    assert m["region_decision_error"].tolist() == [0, 1, 1]
    assert region_metrics(t, "a", "0.3", majority_fraction=0.6)["region_decision_error"].tolist() == [0, 0, 0]          # at 0.6 the reference (0.5) and the system (0.5 / 0.25 / 0) are all non-vegetated


# ============================================================================== decision utility from confusion counts


def test_decision_metrics_of_a_known_confusion():
    d = decision_from_counts(tp=30, fp=10, fn=20, tn=40)
    assert d["accuracy"] == pytest.approx(0.7) and d["balanced_accuracy"] == pytest.approx(0.7) and d["false_positive_rate"] == pytest.approx(0.2)
    assert d["false_negative_rate"] == pytest.approx(0.4) and d["disagreement_rate"] == pytest.approx(0.3) and d["n"] == 100 and d["reference_prevalence"] == pytest.approx(0.5)


def test_balanced_accuracy_corrects_for_class_imbalance():
    d = decision_from_counts(tp=0, fp=0, fn=10, tn=990)                                    # a system that never says vegetation on a 1% vegetated scene
    assert d["accuracy"] == pytest.approx(0.99) and d["balanced_accuracy"] == pytest.approx(0.5) and d["false_negative_rate"] == pytest.approx(1.0)


def test_a_reference_with_only_one_class_leaves_the_rates_that_need_both_undefined():
    all_veg = decision_from_counts(tp=90, fp=0, fn=10, tn=0)
    assert all_veg["accuracy"] == pytest.approx(0.9) and all_veg["balanced_accuracy"] is None and all_veg["false_positive_rate"] is None and all_veg["false_negative_rate"] == pytest.approx(0.1)
    none_veg = decision_from_counts(tp=0, fp=5, fn=0, tn=95)
    assert none_veg["balanced_accuracy"] is None and none_veg["false_negative_rate"] is None and none_veg["false_positive_rate"] == pytest.approx(0.05)
    agree = decision_from_counts(tp=0, fp=0, fn=0, tn=50)
    assert agree["accuracy"] == 1.0 and agree["disagreement_rate"] == 0.0 and agree["balanced_accuracy"] is None                # everything agrees, and one class only


def test_no_valid_pixel_gives_no_decision_metrics_not_zeros():
    d = decision_from_counts(tp=0, fp=0, fn=0, tn=0)
    assert d["n"] == 0 and d["accuracy"] is None and d["disagreement_rate"] is None


# ============================================================================== per scene unit


def two_unit_table():
    # unit A: 3 regions; unit B: 1 region. System a is exact in A and off by 0.2 in B; system b is off by 0.1 everywhere.
    ref = [0.5, 0.6, 0.4, 0.3]
    return table(["A", "A", "A", "B"], ref, {"a": [0.5, 0.6, 0.4, 0.5], "b": [0.6, 0.7, 0.5, 0.4], "bicubic": [0.5, 0.6, 0.4, 0.3]}, tp={"a": [8, 8, 8, 0], "b": [8, 8, 8, 0], "bicubic": [8, 8, 8, 0]},
                 fp={"a": [0, 0, 0, 16], "b": [0, 0, 0, 0], "bicubic": [0, 0, 0, 0]}, fn={"a": [0, 0, 0, 0], "b": [0, 0, 0, 0], "bicubic": [0, 0, 0, 0]}, veg_ref=[8, 8, 8, 0])


def test_unit_metrics_average_regions_within_a_unit_and_keep_units_apart():
    um = unit_metrics(two_unit_table(), ("a", "b", "bicubic"), ("0.3",), majority_fraction=0.5)
    assert set(um) == {"A", "B"}
    a_A, a_B = um["A"]["a"]["0.3"], um["B"]["a"]["0.3"]
    assert a_A["ndvi_mae"] == pytest.approx(0.0, abs=1e-6) and a_B["ndvi_mae"] == pytest.approx(0.2, abs=1e-6) and a_A["n_regions"] == 3 and a_B["n_regions"] == 1
    assert um["A"]["b"]["0.3"]["ndvi_mae"] == pytest.approx(0.1, abs=1e-6) and um["A"]["b"]["0.3"]["ndvi_bias"] == pytest.approx(0.1, abs=1e-6)
    assert um["A"]["a"]["0.3"]["ndvi_rmse"] == pytest.approx(0.0, abs=1e-6) and um["B"]["a"]["0.3"]["ndvi_median_abs_error"] == pytest.approx(0.2, abs=1e-6)
    assert a_B["decision"]["false_positive_rate"] == pytest.approx(1.0) and a_B["decision"]["balanced_accuracy"] is None            # unit B's reference has no vegetation
    assert a_A["decision"]["accuracy"] == 1.0 and a_A["valid_coverage"] == 1.0 and a_A["reference_prevalence"] == pytest.approx(0.5)


def test_region_correlation_is_reported_only_where_it_is_meaningful():
    rng = np.random.default_rng(0)
    ref = rng.random(60)
    t = table(["A"] * 60 + ["B"] * 5, list(ref) + list(np.full(5, 0.4)), {"a": list(ref + 0.01) + list(np.full(5, 0.41))})
    um = unit_metrics(t, ("a",), ("0.3",), majority_fraction=0.5)
    assert um["A"]["a"]["0.3"]["ndvi_correlation"] == pytest.approx(1.0, abs=1e-3)
    assert um["B"]["a"]["0.3"]["ndvi_correlation"] is None and "correlation_reason" in um["B"]["a"]["0.3"]            # 5 regions, constant reference: no meaningful correlation


# ============================================================================== across units, and against the bicubic baseline


def units_table(n_units, deltas):
    """n_units units of 4 regions; bicubic errs by 0.10, each system by 0.10 + delta (delta per unit, signed)."""
    units, ref, sysv = [], [], {k: [] for k in ("bicubic", "a")}
    for u in range(n_units):
        for _ in range(4):
            units.append(f"u{u}")
            ref.append(0.5)
            sysv["bicubic"].append(0.6)
            sysv["a"].append(0.6 + deltas[u])
    return table(units, ref, sysv)


def test_the_change_relative_to_bicubic_is_a_paired_difference_over_scene_units():
    c = cfg()
    t = units_table(8, [-0.02, -0.03, -0.01, -0.02, -0.04, -0.02, -0.03, -0.01])
    um = unit_metrics(t, ("bicubic", "a"), ("0.3",), majority_fraction=0.5)
    agg = aggregate_units(um, ("bicubic", "a"), ("0.3",), c, baselines=("bicubic",))
    s = agg["0.3"]["systems"]["a"]["ndvi_mae"]
    assert s["unit"]["n"] == 8 and s["unit"]["mean"] == pytest.approx(0.0775, abs=1e-6) and s["ci"]["status"] == "ok"
    d = agg["0.3"]["paired"]["a - bicubic"]["ndvi_mae"]
    assert d["n"] == 8 and d["mean_difference"] == pytest.approx(-0.0225, abs=1e-6) and d["label"] == "inferential" and d["ci"]["ci_high"] < 0
    assert "winner" not in str(agg).lower() and "rank" not in " ".join(k for k in agg["0.3"]).lower()


def test_four_scene_units_are_descriptive_only():
    c = cfg()
    um = unit_metrics(units_table(4, [-0.02, -0.03, -0.01, -0.02]), ("bicubic", "a"), ("0.3",), majority_fraction=0.5)
    agg = aggregate_units(um, ("bicubic", "a"), ("0.3",), c, baselines=("bicubic",))
    s = agg["0.3"]["systems"]["a"]["ndvi_mae"]
    d = agg["0.3"]["paired"]["a - bicubic"]["ndvi_mae"]
    assert s["ci"]["status"] == "descriptive_only" and s["ci"]["ci_low"] is None and d["label"] == "descriptive_only" and d["ci"]["ci_low"] is None
    assert d["mean_difference"] == pytest.approx(-0.02, abs=1e-6)                                    # the number is still reported


def test_a_unit_with_many_regions_does_not_outweigh_a_unit_with_few():
    units = ["A"] * 100 + ["B"] * 2
    t = table(units, [0.5] * 102, {"a": [0.6] * 100 + [0.9] * 2, "bicubic": [0.5] * 102})
    agg = aggregate_units(unit_metrics(t, ("bicubic", "a"), ("0.3",), majority_fraction=0.5), ("bicubic", "a"), ("0.3",), cfg(), baselines=("bicubic",))
    assert agg["0.3"]["systems"]["a"]["ndvi_mae"]["unit"]["mean"] == pytest.approx((0.1 + 0.4) / 2, abs=1e-6)          # the mean of unit means, not the mean of regions (0.106)


def test_the_sensitivity_thresholds_are_reported_next_to_the_primary_one_under_their_own_keys():
    t = table(["A"] * 4, [0.5] * 4, {"a": [0.5] * 4}, thresholds=("0.3", "0.2", "0.4"))
    um = unit_metrics(t, ("a",), ("0.3", "0.2", "0.4"), majority_fraction=0.5)
    assert set(um["A"]["a"]) == {"0.3", "0.2", "0.4"}


def test_a_correlation_across_fewer_than_thirty_regions_is_not_reported_even_if_it_could_be_computed():
    rng = np.random.default_rng(3)
    ref = rng.random(12)
    t = table(["A"] * 12, list(ref), {"a": list(ref + 0.01)})
    um = unit_metrics(t, ("a",), ("0.3",), majority_fraction=0.5)
    assert um["A"]["a"]["0.3"]["ndvi_correlation"] is None and "fewer than 30" in um["A"]["a"]["0.3"]["correlation_reason"]
