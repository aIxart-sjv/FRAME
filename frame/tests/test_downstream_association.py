"""frame.downstream.association -- does the stability relate to downstream error once texture is controlled for? On tables with known relationships."""

from __future__ import annotations

import numpy as np
import pytest

from frame.downstream import association as AS
from frame.downstream.config import DownstreamConfig


def cfg(**over):
    d = {"name": "a", "output_dir": "o", "systems": [{"name": "lite", "kind": "lite"}], "datasets": [{"name": "d", "kind": "sen2neon", "manifest": "m"}],
         "bootstrap": {"n_boot": 200}, "analysis": {"pooled_regions_per_tile": 300}, **over}
    return DownstreamConfig.from_dict(d)


def make_table(n_units, tiles_per_unit=1, regions=400, mode="informative", seed=0):
    """Regions with a texture, a stability and a downstream error (system 'a').

    informative: the error grows with the stability, texture unrelated | null: nothing related | confounded: stability and error are both driven by texture only |
    added: the error is texture + stability (stability carries information beyond texture)
    """
    rng = np.random.default_rng(seed)
    cols = {k: [] for k in ("scene_unit", "tile_id", "texture", "stab", "err", "added")}
    for u in range(n_units):
        for t in range(tiles_per_unit):
            tex, latent, noise = rng.random(regions), rng.random(regions), rng.random(regions)
            if mode == "informative":
                stab, err = latent, latent + 0.4 * noise
            elif mode == "null":
                stab, err = latent, noise
            elif mode == "confounded":
                stab, err = tex + 0.15 * latent, tex + 0.15 * noise
            elif mode == "added":
                stab, err = latent, tex + latent + 0.3 * noise
            else:
                raise ValueError(mode)
            cols["scene_unit"] += [f"u{u}"] * regions
            cols["tile_id"] += [f"u{u}:t{t}"] * regions
            cols["texture"].append(tex); cols["stab"].append(stab); cols["err"].append(err); cols["added"].append(rng.random(regions))
    n = len(cols["scene_unit"])
    err = np.concatenate(cols["err"])
    ref = np.full(n, 0.5)
    table = {"row": np.arange(n, dtype=np.int32), "col": np.zeros(n, np.int32), "n_valid": np.full(n, 16, np.int32), "size": 4, "n_candidate": n, "excluded": {},
             "scene_unit": np.asarray(cols["scene_unit"], dtype=object), "tile_id": np.asarray(cols["tile_id"], dtype=object),
             "mean:ref": ref.astype(np.float32), "mean:a": (ref + err * 0.1).astype(np.float32), "mean:texture": np.concatenate(cols["texture"]).astype(np.float32),
             "mean:stab:a": np.concatenate(cols["stab"]).astype(np.float32), "mean:added:a": np.concatenate(cols["added"]).astype(np.float32)}
    frac = np.clip(err / 2.0, 0, 1)                                                          # the pixel disagreement follows the same error
    dis = np.round(frac * 16).astype(np.int32)
    table["veg:ref@0.3"] = np.full(n, 8, np.int32)
    table["fp:a@0.3"], table["fn:a@0.3"] = dis, np.zeros(n, np.int32)
    table["tp:a@0.3"], table["veg:a@0.3"] = np.full(n, 8, np.int32), 8 + dis
    return table


TARGETS = ("ndvi_abs_error", "disagreement")


# ============================================================================== within-tile association, aggregated over scene units


def test_an_informative_stability_is_associated_with_downstream_error_across_scene_units():
    t = make_table(8, mode="informative")
    a = AS.association(t, "a", "0.3", cfg())
    e = a["targets"]["ndvi_abs_error"]
    assert e["stability"]["unit"]["mean"] > 0.5 and e["stability"]["ci"]["status"] == "ok" and e["stability"]["ci"]["ci_low"] > 0.4 and e["stability"]["share_positive"] == 1.0
    assert abs(e["texture"]["unit"]["mean"]) < 0.15                                        # texture is unrelated in this table
    assert a["targets"]["disagreement"]["stability"]["unit"]["mean"] > 0.3


def test_a_null_stability_has_an_interval_that_includes_zero():
    a = AS.association(make_table(8, mode="null", seed=3), "a", "0.3", cfg())
    ci = a["targets"]["ndvi_abs_error"]["stability"]["ci"]
    assert ci["status"] == "ok" and ci["ci_low"] < 0.0 < ci["ci_high"]


def test_stability_that_only_re_encodes_texture_adds_nothing_once_texture_is_controlled():
    a = AS.association(make_table(8, mode="confounded"), "a", "0.3", cfg())
    e = a["targets"]["ndvi_abs_error"]
    assert e["stability"]["unit"]["mean"] > 0.5 and e["texture"]["unit"]["mean"] > 0.5      # plain correlation is high for both...
    assert abs(e["partial_given_texture"]["unit"]["mean"]) < 0.15                            # ...and vanishes for the stability when texture is controlled for
    assert abs(e["partial_given_texture_and_added_detail"]["unit"]["mean"]) < 0.15


def test_stability_that_carries_information_beyond_texture_keeps_a_partial_correlation():
    e = AS.association(make_table(8, mode="added"), "a", "0.3", cfg())["targets"]["ndvi_abs_error"]
    assert e["partial_given_texture"]["unit"]["mean"] > 0.4 and e["partial_given_texture"]["ci"]["ci_low"] > 0.3


def test_the_texture_only_relationship_is_reported_next_to_the_stability():
    e = AS.association(make_table(6, mode="confounded"), "a", "0.3", cfg())["targets"]["ndvi_abs_error"]
    assert {"stability", "texture", "added_detail", "partial_given_texture", "partial_given_texture_and_added_detail"} <= set(e)


# ============================================================================== scene units, not regions, are the replicates


def test_many_regions_from_few_scene_units_give_no_interval():
    a = AS.association(make_table(4, tiles_per_unit=3, regions=2000, mode="informative"), "a", "0.3", cfg())
    e = a["targets"]["ndvi_abs_error"]["stability"]
    assert a["n_units"] == 4 and a["n_tiles"] == 12 and a["n_regions"] == 4 * 3 * 2000 and e["ci"]["status"] == "descriptive_only" and e["ci"]["ci_low"] is None
    assert e["unit"]["mean"] > 0.5                                                          # the number is reported; the interval is not claimed


def test_tiles_of_one_scene_unit_are_averaged_before_the_unit_summary():
    t = make_table(6, tiles_per_unit=2, mode="informative")
    a = AS.association(t, "a", "0.3", cfg())
    e = a["targets"]["ndvi_abs_error"]["stability"]
    assert e["n_tiles"] == 12 and e["n_units"] == 6


def test_the_pooled_regions_use_a_unit_clustered_interval_and_a_seeded_subsample():
    t = make_table(8, mode="informative")
    a = AS.association(t, "a", "0.3", cfg())
    p = a["targets"]["ndvi_abs_error"]["pooled_stability"]
    assert p["method"] == "percentile bootstrap over scene units" and p["n_units"] == 8 and p["status"] == "ok" and p["estimate"] > 0.4
    assert AS.association(t, "a", "0.3", cfg()) == a                                        # deterministic


def test_tile_and_scene_level_association_across_units_needs_ten_units_for_an_interval():
    few = AS.association(make_table(8, mode="informative"), "a", "0.3", cfg())["targets"]["ndvi_abs_error"]["across_tiles"]
    many = AS.association(make_table(11, mode="informative"), "a", "0.3", cfg())["targets"]["ndvi_abs_error"]["across_tiles"]
    assert few["tile"]["spearman_ci"]["status"] == "descriptive_only" and many["tile"]["spearman_ci"]["status"] == "ok"
    assert many["scene"]["n_units"] == 11


# ============================================================================== risk-coverage on the downstream error


def test_dropping_the_most_unstable_regions_lowers_the_remaining_downstream_error_within_units():
    rc = AS.risk_coverage(make_table(8, mode="informative"), "a", "0.3", cfg())
    e = rc["targets"]["ndvi_abs_error"]
    assert [c["retention"] for c in e["by_retention"]] == [1.0, 0.8, 0.6, 0.4]                # exactly the declared grid, in order
    r = {c["retention"]: c for c in e["by_retention"]}
    assert r[0.8]["stability"]["reduction"]["unit"]["mean"] > 0.03 and r[0.4]["stability"]["reduction"]["unit"]["mean"] > r[0.8]["stability"]["reduction"]["unit"]["mean"]
    assert r[0.8]["stability"]["reduction"]["ci"]["ci_low"] > 0 and r[0.4]["oracle"]["reduction"]["unit"]["mean"] >= r[0.4]["stability"]["reduction"]["unit"]["mean"]
    assert r[1.0]["stability"]["reduction"]["unit"]["mean"] == pytest.approx(0.0)


def test_an_uninformative_ranking_removes_no_downstream_error_and_the_texture_baseline_is_reported_the_same_way():
    e = AS.risk_coverage(make_table(8, mode="null", seed=5), "a", "0.3", cfg())["targets"]["ndvi_abs_error"]
    r = {c["retention"]: c for c in e["by_retention"]}
    assert abs(r[0.6]["stability"]["reduction"]["unit"]["mean"]) < 0.05 and r[0.6]["stability"]["reduction"]["ci"]["ci_low"] < 0.0 < r[0.6]["stability"]["reduction"]["ci"]["ci_high"] + 0.05
    assert {"stability", "texture", "oracle"} <= set(r[0.6])


def test_texture_ranks_the_error_when_the_error_is_texture_driven():
    e = AS.risk_coverage(make_table(8, mode="confounded"), "a", "0.3", cfg())["targets"]["ndvi_abs_error"]
    r = {c["retention"]: c for c in e["by_retention"]}
    assert r[0.6]["texture"]["reduction"]["unit"]["mean"] > 0.1


def test_the_ranking_direction_is_most_unstable_first():
    t = make_table(6, mode="informative")
    inverted = dict(t)
    inverted["mean:stab:a"] = -t["mean:stab:a"]                                             # the least unstable are now 'most unstable'
    good = AS.risk_coverage(t, "a", "0.3", cfg())["targets"]["ndvi_abs_error"]["by_retention"][1]["stability"]["reduction"]["unit"]["mean"]
    bad = AS.risk_coverage(inverted, "a", "0.3", cfg())["targets"]["ndvi_abs_error"]["by_retention"][1]["stability"]["reduction"]["unit"]["mean"]
    assert good > 0 > bad


def test_retention_is_deterministic_and_uses_only_the_declared_grid():
    t = make_table(6, mode="informative")
    a, b = AS.risk_coverage(t, "a", "0.3", cfg()), AS.risk_coverage(t, "a", "0.3", cfg())
    assert a == b
    c2 = AS.risk_coverage(t, "a", "0.3", cfg(analysis={"retention_grid": [1.0, 0.5], "pooled_regions_per_tile": 300}))
    assert [x["retention"] for x in c2["targets"]["ndvi_abs_error"]["by_retention"]] == [1.0, 0.5]


def test_four_units_give_descriptive_risk_coverage_only():
    e = AS.risk_coverage(make_table(4, mode="informative"), "a", "0.3", cfg())["targets"]["ndvi_abs_error"]["by_retention"][1]["stability"]["reduction"]
    assert e["ci"]["status"] == "descriptive_only" and e["unit"]["mean"] > 0


def test_the_pooled_interval_counts_scene_units_not_tiles():
    a = AS.association(make_table(8, tiles_per_unit=2, mode="informative"), "a", "0.3", cfg())
    assert a["targets"]["ndvi_abs_error"]["pooled_stability"]["n_units"] == 8 and a["n_tiles"] == 16


def test_the_across_tile_association_is_computed_from_tile_means():
    from frame.reliability.association import correlation

    t = make_table(11, mode="informative", regions=200)
    a = AS.association(t, "a", "0.3", cfg())["targets"]["ndvi_abs_error"]["across_tiles"]["tile"]["spearman"]["value"]
    units = np.asarray(t["scene_unit"], dtype=object)
    stab = np.asarray([t["mean:stab:a"][np.asarray([u == f"u{i}" for u in units])].astype(float).mean() for i in range(11)])
    err = np.asarray([np.abs(t["mean:a"][np.asarray([u == f"u{i}" for u in units])].astype(float) - 0.5).mean() for i in range(11)])
    assert a == pytest.approx(correlation(stab, err)["value"], abs=1e-9)


# ============================================================================== the stability against the texture ordering, paired over scene units


def _by(table, target="ndvi_abs_error", **over):
    return AS.risk_coverage(table, "a", "0.3", cfg(**over))["targets"][target]["by_retention"]


def test_the_stability_order_is_compared_with_the_texture_order_paired_over_scene_units():
    for entry in _by(make_table(11, mode="informative")):
        d = entry["stability_minus_texture"]
        assert d["n"] == 11 and d["mean_difference"] == pytest.approx(entry["stability"]["reduction"]["unit"]["mean"] - entry["texture"]["reduction"]["unit"]["mean"], abs=1e-12)


def test_the_paired_difference_is_positive_where_the_stability_carries_information_the_texture_does_not():
    d = _by(make_table(11, mode="informative"))[1]["stability_minus_texture"]
    assert d["mean_difference"] > 0.1 and d["ci"]["ci_low"] > 0


def test_the_paired_difference_is_not_positive_where_the_texture_alone_drives_the_error():
    d = _by(make_table(11, mode="confounded"))[1]["stability_minus_texture"]
    assert d["ci"]["ci_low"] <= 0 <= d["ci"]["ci_high"] or d["mean_difference"] < 0


def test_the_paired_difference_is_descriptive_with_fewer_than_five_scene_units():
    d = _by(make_table(4, mode="informative"))[1]["stability_minus_texture"]
    assert d["ci"]["status"] == "descriptive_only" and d["label"] == "descriptive_only"


def test_full_retention_has_no_difference_to_report():
    d = _by(make_table(11, mode="informative"))[0]["stability_minus_texture"]
    assert d["mean_difference"] == pytest.approx(0.0, abs=1e-12)
