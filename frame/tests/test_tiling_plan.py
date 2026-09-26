"""frame.tiling.plan -- tile planning: coverage, counts, overlap, validation."""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from frame.tiling import InvalidSceneError, InvalidTilingConfigError, TilingConfig, plan_tiles

# (height, width) -> expected (n_rows, n_cols) at the default tile 128 / overlap 32 (stride 96),
# derived by hand: starts 0, 96, 192, ... until start + 128 >= size.
REPRESENTATIVE = {
    (128, 128): (1, 1),
    (128, 256): (1, 3),
    (256, 128): (3, 1),
    (300, 500): (3, 5),
    (511, 777): (5, 8),
}


def _coverage(plan) -> np.ndarray:
    covered = np.zeros((plan.scene_height, plan.scene_width), dtype=np.int32)
    for t in plan.tiles:
        covered[t.row_start : t.row_end, t.col_start : t.col_end] += 1
    return covered


# ----------------------------------------------------------- representative shapes


@pytest.mark.parametrize("shape,grid", REPRESENTATIVE.items())
def test_representative_shapes_have_the_expected_grid_and_tile_count(shape, grid):
    plan = plan_tiles(*shape)
    assert (plan.n_rows, plan.n_cols) == grid
    assert plan.tile_count == grid[0] * grid[1] == len(plan.tiles)


@pytest.mark.parametrize("shape", REPRESENTATIVE)
def test_every_scene_pixel_is_covered_and_no_tile_leaves_the_scene(shape):
    plan = plan_tiles(*shape)
    assert _coverage(plan).min() >= 1  # complete coverage: no border is dropped
    for t in plan.tiles:
        assert 0 <= t.row_start < t.row_end <= shape[0]
        assert 0 <= t.col_start < t.col_end <= shape[1]


@pytest.mark.parametrize("shape", REPRESENTATIVE)
def test_tile_positions_are_unique_and_strictly_increasing(shape):
    plan = plan_tiles(*shape)
    assert len({(t.row_start, t.col_start) for t in plan.tiles}) == plan.tile_count
    assert list(plan.row_starts) == sorted(set(plan.row_starts))
    assert list(plan.col_starts) == sorted(set(plan.col_starts))


def test_processing_order_is_row_major_and_deterministic():
    plan = plan_tiles(300, 500)
    assert [(t.row_index, t.col_index) for t in plan.tiles] == [(r, c) for r in range(3) for c in range(5)]
    assert plan_tiles(300, 500) == plan_tiles(300, 500)


def test_extents_of_the_300x500_scene():
    plan = plan_tiles(300, 500)
    assert plan.row_starts == (0, 96, 192) and plan.col_starts == (0, 96, 192, 288, 384)
    interior = plan.tile_at(0, 0)
    assert (interior.valid_height, interior.valid_width, interior.is_padded) == (128, 128, False)
    bottom_right = plan.tile_at(2, 4)
    assert (bottom_right.row_start, bottom_right.row_end) == (192, 300)
    assert (bottom_right.col_start, bottom_right.col_end) == (384, 500)
    assert (bottom_right.valid_height, bottom_right.valid_width) == (108, 116)
    assert (bottom_right.padded_height, bottom_right.padded_width) == (128, 128)
    assert bottom_right.is_padded


def test_scene_smaller_than_one_tile_is_one_padded_tile():
    plan = plan_tiles(100, 90)
    assert plan.tile_count == 1
    tile = plan.tiles[0]
    assert (tile.valid_height, tile.valid_width, tile.padded_height, tile.padded_width) == (100, 90, 128, 128)
    assert tile.is_padded and not (tile.overlaps_top or tile.overlaps_bottom or tile.overlaps_left or tile.overlaps_right)


def test_sr_coordinates_are_the_input_coordinates_times_the_scale():
    tile = plan_tiles(300, 500).tile_at(2, 4)
    assert (tile.sr_row_start, tile.sr_col_start) == (192 * 4, 384 * 4)
    assert (tile.sr_valid_height, tile.sr_valid_width) == (108 * 4, 116 * 4)
    assert (tile.sr_row_end, tile.sr_col_end) == (300 * 4, 500 * 4)


def test_neighbour_flags_mark_exactly_the_interior_sides():
    plan = plan_tiles(300, 500)
    first, middle, last = plan.tile_at(0, 0), plan.tile_at(1, 2), plan.tile_at(2, 4)
    assert (first.overlaps_top, first.overlaps_left, first.overlaps_bottom, first.overlaps_right) == (False, False, True, True)
    assert all((middle.overlaps_top, middle.overlaps_bottom, middle.overlaps_left, middle.overlaps_right))
    assert (last.overlaps_top, last.overlaps_left, last.overlaps_bottom, last.overlaps_right) == (True, True, False, False)


def test_sr_shape_is_four_times_the_scene():
    for shape in REPRESENTATIVE:
        assert plan_tiles(*shape).sr_shape == (shape[0] * 4, shape[1] * 4)


# ------------------------------------------------------------------------ overlap


def _tiles_along(size: int, overlap: int) -> int:
    """Independent closed-form oracle for the tile count along one axis."""
    stride = 128 - overlap
    return 1 if size <= 128 else -(-(size - 128) // stride) + 1


@pytest.mark.parametrize("overlap,expected_grid", [(0, (3, 4)), (16, (3, 5)), (32, (3, 5)), (48, (4, 6)), (64, (4, 7))])
def test_overlap_zero_default_and_other_valid_values_on_300x500(overlap, expected_grid):
    plan = plan_tiles(300, 500, TilingConfig(overlap=overlap))
    assert (plan.n_rows, plan.n_cols) == expected_grid
    assert (_tiles_along(300, overlap), _tiles_along(500, overlap)) == expected_grid  # closed form agrees
    assert _coverage(plan).min() >= 1


def test_the_planner_matches_the_closed_form_tile_count_everywhere():
    for size, overlap in itertools.product(range(1, 800, 7), [0, 1, 16, 32, 48, 64]):
        assert plan_tiles(size, size, TilingConfig(overlap=overlap)).n_rows == _tiles_along(size, overlap), (size, overlap)


@pytest.mark.parametrize("overlap", [0, 1, 16, 32, 48, 64])
def test_consecutive_tiles_overlap_by_exactly_the_configured_amount(overlap):
    plan = plan_tiles(511, 777, TilingConfig(overlap=overlap))
    for starts in (plan.row_starts, plan.col_starts):
        for a, b in zip(starts, starts[1:]):
            assert a + 128 - b == overlap
    for t in plan.tiles:
        if t.overlaps_bottom:
            assert t.valid_height == 128  # a tile with a successor is fully valid
        if t.overlaps_right:
            assert t.valid_width == 128


@pytest.mark.parametrize("overlap", [0, 16, 32, 64])
def test_the_last_tile_is_always_more_than_the_overlap_wide(overlap):
    """Needed so the two blend ramps of the final seam both lie inside the scene."""
    for size in range(129, 700):
        starts = plan_tiles(size, size, TilingConfig(overlap=overlap)).row_starts
        if len(starts) > 1:
            assert size - starts[-1] > overlap


def test_coverage_holds_for_a_wide_grid_of_sizes_and_overlaps():
    sizes = [1, 2, 63, 64, 127, 128, 129, 130, 223, 224, 225, 256, 300, 511, 777]
    for h, w, ov in itertools.product(sizes, sizes, [0, 1, 16, 32, 64]):
        plan = plan_tiles(h, w, TilingConfig(overlap=ov))
        coverage = _coverage(plan)
        assert coverage.min() >= 1, (h, w, ov)
        assert all(t.padded_height == 128 and t.valid_height >= 1 and t.valid_width >= 1 for t in plan.tiles)


@pytest.mark.parametrize("overlap", [-1, -32, 65, 96, 127, 128, 200])
def test_invalid_overlap_is_rejected(overlap):
    with pytest.raises(InvalidTilingConfigError, match="overlap"):
        TilingConfig(overlap=overlap)


def test_an_overlap_that_gives_a_zero_or_negative_stride_says_so():
    with pytest.raises(InvalidTilingConfigError, match="stride"):
        TilingConfig(overlap=128)
    with pytest.raises(InvalidTilingConfigError, match="stride"):
        TilingConfig(overlap=200)


@pytest.mark.parametrize("overlap", [1.5, True, "32", None])
def test_non_integer_overlap_is_rejected(overlap):
    with pytest.raises(InvalidTilingConfigError, match="integer"):
        TilingConfig(overlap=overlap)  # type: ignore[arg-type]


# ------------------------------------------------------------- other invalid config


@pytest.mark.parametrize("tile_size", [0, -1, -128, 2.5, "128", None])
def test_invalid_tile_size_is_rejected(tile_size):
    with pytest.raises(InvalidTilingConfigError, match="tile_size"):
        TilingConfig(tile_size=tile_size, overlap=0)  # type: ignore[arg-type]


@pytest.mark.parametrize("scale", [0, -4, 1.5])
def test_invalid_scale_is_rejected(scale):
    with pytest.raises(InvalidTilingConfigError, match="scale"):
        TilingConfig(scale=scale)  # type: ignore[arg-type]


def test_unknown_padding_and_blend_modes_are_rejected():
    with pytest.raises(InvalidTilingConfigError, match="padding_mode"):
        TilingConfig(padding_mode="zeros")
    with pytest.raises(InvalidTilingConfigError, match="blend_mode"):
        TilingConfig(blend_mode="hann")


def test_a_non_default_tile_size_is_supported_by_the_planner():
    plan = plan_tiles(100, 100, TilingConfig(tile_size=64, overlap=8))
    assert plan.row_starts == (0, 56) and plan.tiles[0].padded_height == 64
    assert _coverage(plan).min() >= 1


def test_config_describes_itself_for_provenance():
    assert TilingConfig().describe() == {
        "tile_size": 128, "overlap": 32, "stride": 96, "scale": 4, "padding_mode": "reflect", "blend_mode": "linear",
    }


# ------------------------------------------------------------------ invalid scenes


@pytest.mark.parametrize("h,w", [(0, 128), (128, 0), (0, 0), (-1, 128), (128, -5)])
def test_empty_or_negative_scene_sizes_are_rejected(h, w):
    with pytest.raises(InvalidSceneError, match="empty|>= 1"):
        plan_tiles(h, w)


@pytest.mark.parametrize("h,w", [(128.0, 128), (128, "128"), (True, 128)])
def test_non_integer_scene_sizes_are_rejected(h, w):
    with pytest.raises(InvalidSceneError, match="integer"):
        plan_tiles(h, w)  # type: ignore[arg-type]
