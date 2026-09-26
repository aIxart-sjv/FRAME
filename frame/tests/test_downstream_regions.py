"""frame.downstream.regions -- fixed square regions on the aligned grid: deterministic, exactly located, disjoint, and honest about minimum valid pixels."""

from __future__ import annotations

import numpy as np
import pytest

from frame.downstream.regions import candidate_region_count, extract_regions, make_grid, stack_tables

H, W = 24, 32


def counts_only(valid, size=4, correction=(0, 0), min_valid=0.0, **maps):
    grid = make_grid((valid.shape[0], valid.shape[1]), size, correction)
    return grid, extract_regions(maps, valid, grid, min_valid_fraction=min_valid, thresholds=(), systems=(), reference="ref")


# ============================================================================== the grid


def test_the_region_grid_starts_on_a_prediction_pixel_boundary_whatever_the_alignment_crop():
    """10 m regions must be the footprint of ONE Sentinel-2 pixel: after a translation crop of the aligned frame the first region may not start at the frame's edge."""
    g0 = make_grid((H, W), 4, (0, 0))
    assert (g0.origin_row, g0.origin_col, g0.sr_row_start, g0.hr_row_start) == (0, 0, 0, 0) and (g0.n_rows, g0.n_cols) == (6, 8)
    g = make_grid((H, W), 4, (-3, 2))                          # the prediction is displaced by -3 rows: its window starts 3 rows in; the reference's window starts 2 columns in
    assert g.sr_row_start == 3 and g.hr_row_start == 0 and g.sr_col_start == 0 and g.hr_col_start == 2
    assert (g.sr_row_start + g.origin_row) % 4 == 0 and (g.sr_col_start + g.origin_col) % 4 == 0 and g.origin_row == 1 and g.origin_col == 0
    assert g.n_rows == (H - 1) // 4 and g.n_cols == W // 4


@pytest.mark.parametrize("size", [4, 16])
def test_every_region_starts_at_a_multiple_of_the_size_in_prediction_coordinates(size):
    for correction in ((0, 0), (-3, 2), (2, -1), (-1, -3)):
        g = make_grid((70, 90), size, correction)
        assert (g.sr_row_start + g.origin_row) % size == 0 and (g.sr_col_start + g.origin_col) % size == 0


def test_a_grid_too_small_for_one_region_has_none():
    g = make_grid((3, 40), 4, (0, 0))
    assert g.n_rows == 0 and candidate_region_count((3, 40), 4) == 0
    assert candidate_region_count((1024, 1024), 4) == 256 * 256 and candidate_region_count((512, 512), 16) == 32 * 32 and candidate_region_count((1023, 1025), 4) == 255 * 256


# ============================================================================== exact correspondence


def test_a_region_reports_the_mean_of_exactly_the_pixels_it_covers_and_where_they_are():
    yy, xx = np.mgrid[0:H, 0:W].astype(float)
    value = yy * 100 + xx
    valid = np.ones((H, W), bool)
    grid, t = counts_only(valid, value=value)
    assert len(t["row"]) == 6 * 8
    k = int(np.flatnonzero((t["row"] == 2) & (t["col"] == 5))[0])
    block = value[8:12, 20:24]
    assert t["mean:value"][k] == pytest.approx(block.mean()) and t["n_valid"][k] == 16
    assert (t["sr_row"][k], t["sr_col"][k], t["hr_row"][k], t["hr_col"][k]) == (8, 20, 8, 20)


def test_a_single_pixel_lands_in_exactly_one_region_with_its_location_recorded_in_both_grids():
    valid = np.ones((H, W), bool)
    spike = np.zeros((H, W))
    spike[9, 14] = 16.0
    grid, t = counts_only(valid, size=4, correction=(-3, 2), spike=spike)
    hit = np.flatnonzero(t["mean:spike"] > 0)
    assert hit.size == 1
    k = hit[0]
    # aligned pixel (9, 14) is prediction pixel (9 + 3, 14 + 0) = (12, 14) and reference pixel (9 + 0, 14 + 2) = (9, 16)
    assert t["sr_row"][k] <= 12 < t["sr_row"][k] + 4 and t["sr_col"][k] <= 14 < t["sr_col"][k] + 4
    assert t["hr_row"][k] <= 9 < t["hr_row"][k] + 4 and t["hr_col"][k] <= 16 < t["hr_col"][k] + 4
    assert t["sr_row"][k] % 4 == 0 and t["sr_col"][k] % 4 == 0


def test_regions_are_disjoint_and_every_valid_pixel_is_counted_at_most_once():
    rng = np.random.default_rng(0)
    valid = rng.random((H, W)) > 0.2
    grid, t = counts_only(valid, size=4, correction=(1, -2))
    covered = grid.n_rows * grid.n_cols * 16
    assert int(t["n_valid"].sum()) <= int(valid.sum()) and int(t["n_valid"].sum()) <= covered
    cells = {(int(r), int(c)) for r, c in zip(t["row"], t["col"])}
    assert len(cells) == len(t["row"])                                                        # no region twice
    ones = np.ones((H, W))
    _, t1 = counts_only(np.ones((H, W), bool), size=4, correction=(1, -2), ones=ones)
    assert np.allclose(t1["mean:ones"], 1.0) and int(t1["n_valid"].sum()) == grid.n_rows * grid.n_cols * 16


def test_extraction_is_deterministic():
    rng = np.random.default_rng(1)
    v = rng.random((H, W))
    valid = rng.random((H, W)) > 0.1
    _, a = counts_only(valid, value=v)
    _, b = counts_only(valid, value=v)
    assert all(np.array_equal(a[k], b[k]) for k in a)


# ============================================================================== minimum valid pixels


def test_a_region_with_too_few_valid_pixels_is_excluded_and_counted_never_scored():
    valid = np.zeros((8, 8), bool)
    for (r0, c0), n in {(0, 0): 11, (0, 4): 4, (4, 0): 12, (4, 4): 16}.items():         # valid pixels per region: 11, 4, 12 and 16 of 16
        block = np.zeros(16, bool)
        block[:n] = True
        valid[r0:r0 + 4, c0:c0 + 4] = block.reshape(4, 4)
    grid, t = counts_only(valid, min_valid=0.75, value=np.arange(64.0).reshape(8, 8))
    kept = {(int(r), int(c)) for r, c in zip(t["row"], t["col"])}
    assert kept == {(1, 0), (1, 1)}                                                            # 12/16 = 0.75 is exactly at the limit and kept; 11/16 and 4/16 are not
    assert t["excluded"]["region_insufficient_valid_pixels"] == 2 and t["n_candidate"] == 4


def test_a_region_with_no_valid_pixel_has_no_value_rather_than_zero():
    valid = np.zeros((8, 8), bool)
    _, t = counts_only(valid, min_valid=0.1, value=np.ones((8, 8)))
    assert len(t["row"]) == 0 and t["excluded"]["region_insufficient_valid_pixels"] == 4


def test_invalid_pixels_never_contribute_to_a_region_mean():
    valid = np.ones((8, 8), bool)
    valid[0, 0] = False
    value = np.ones((8, 8))
    value[0, 0] = 1e9                                                                          # junk under the mask
    value[7, 7] = np.nan                                                                       # non-finite on a valid pixel: not usable either
    _, t = counts_only(valid, min_valid=0.0, value=value)
    assert np.allclose(t["mean:value"][(t["row"] == 0) & (t["col"] == 0)], 1.0) and np.isfinite(t["mean:value"]).all()


# ============================================================================== the decision counts


def test_decision_counts_partition_the_valid_pixels_and_disagreement_is_false_positives_plus_false_negatives():
    rng = np.random.default_rng(2)
    ref = rng.random((16, 16)) * 0.8
    sys_ = np.clip(ref + rng.normal(0, 0.15, ref.shape), -1, 1)
    valid = rng.random((16, 16)) > 0.1
    grid = make_grid((16, 16), 4, (0, 0))
    t = extract_regions({}, valid, grid, min_valid_fraction=0.0, thresholds=(0.3, 0.2), systems=("a",), reference="ref", ndvi={"ref": ref, "a": sys_})
    for th in ("0.3", "0.2"):
        tp, fp, fn = t[f"tp:a@{th}"], t[f"fp:a@{th}"], t[f"fn:a@{th}"]
        tn = t["n_valid"] - tp - fp - fn
        assert (tn >= 0).all() and (tp + fn == t[f"veg:ref@{th}"]).all() and (tp + fp == t[f"veg:a@{th}"]).all()
    # brute-force check of one region
    k = int(np.flatnonzero((t["row"] == 1) & (t["col"] == 2))[0])
    m = valid[4:8, 8:12]
    r, s = ref[4:8, 8:12][m] >= 0.3, sys_[4:8, 8:12][m] >= 0.3
    assert t["tp:a@0.3"][k] == int((r & s).sum()) and t["fp:a@0.3"][k] == int((~r & s).sum()) and t["fn:a@0.3"][k] == int((r & ~s).sum())
    assert t["mean:ref"][k] == pytest.approx(ref[4:8, 8:12][m].mean()) and t["mean:a"][k] == pytest.approx(sys_[4:8, 8:12][m].mean())


def test_the_threshold_is_inclusive_at_the_boundary():
    ref = np.full((4, 4), 0.3)
    t = extract_regions({}, np.ones((4, 4), bool), make_grid((4, 4), 4, (0, 0)), min_valid_fraction=0.0, thresholds=(0.3,), systems=("a",), reference="ref", ndvi={"ref": ref, "a": ref - 1e-9})
    assert t["veg:ref@0.3"][0] == 16 and t["veg:a@0.3"][0] == 0 and t["fn:a@0.3"][0] == 16          # NDVI >= threshold is vegetation


# ============================================================================== tables of several tiles never mix their identity


def test_stacking_tables_keeps_every_regions_tile_and_scene_unit():
    valid = np.ones((8, 8), bool)
    _, a = counts_only(valid, value=np.ones((8, 8)))
    _, b = counts_only(valid, value=np.full((8, 8), 2.0))
    joined = stack_tables([({"dataset": "d", "scene_unit": "u1", "tile_id": "t1"}, a), ({"dataset": "d", "scene_unit": "u2", "tile_id": "t2"}, b)])
    assert len(joined["row"]) == 8 and list(joined["scene_unit"][:4]) == ["u1"] * 4 and list(joined["scene_unit"][4:]) == ["u2"] * 4
    assert list(joined["tile_id"][:4]) == ["t1"] * 4 and np.allclose(joined["mean:value"][:4], 1.0) and np.allclose(joined["mean:value"][4:], 2.0)
    assert joined["n_candidate"] == 8 and joined["size"] == 4
    _, c16 = counts_only(np.ones((16, 16), bool), size=16, value=np.ones((16, 16)))
    with pytest.raises(ValueError, match="different region sizes"):
        stack_tables([({"tile_id": "a"}, a), ({"tile_id": "b"}, c16)])
