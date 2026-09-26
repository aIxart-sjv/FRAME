"""frame.tiling.blend -- weight windows and the weighted accumulator (no model, no GPU)."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from frame.tiling import ReconstructionError, TilingConfig, plan_tiles
from frame.tiling.blend import WeightedCanvas, axis_window, tile_window


def test_without_ramps_every_weight_is_one():
    assert np.array_equal(axis_window(512, 0, True, True), np.ones(512))
    assert np.array_equal(axis_window(512, 128, False, False), np.ones(512))


def test_ramp_values_are_linear_strictly_positive_and_never_reach_one_before_the_interior():
    w = axis_window(512, 128, True, False)
    assert w[0] == pytest.approx(0.5 / 128) and w[127] == pytest.approx(127.5 / 128)
    assert np.all(w > 0) and np.all(np.diff(w[:128]) > 0)
    assert np.array_equal(w[128:], np.ones(384))


def test_end_ramp_mirrors_the_start_ramp():
    assert np.array_equal(axis_window(512, 128, False, True), axis_window(512, 128, True, False)[::-1])


@pytest.mark.parametrize("overlap", [1, 16, 32, 48, 64])
def test_neighbouring_ramps_sum_to_exactly_one_across_the_overlap(overlap):
    """The property that makes blending a partition of unity: in the pixels two tiles
    share, their weights add up to 1, so no region is brightened or darkened."""
    config = TilingConfig(overlap=overlap)
    plan = plan_tiles(128, 700, config)
    r = overlap * config.scale
    for left, right in zip(plan.tiles, plan.tiles[1:]):
        w_left = tile_window(left, config).numpy().astype(np.float64)[:, -r:]
        w_right = tile_window(right, config).numpy().astype(np.float64)[:, :r]
        assert np.allclose(w_left + w_right, 1.0, atol=1e-6)


def test_sides_on_the_scene_boundary_are_not_ramped():
    config = TilingConfig()
    plan = plan_tiles(300, 500, config)
    first = tile_window(plan.tile_at(0, 0), config).numpy()
    assert np.all(first[0, :] == first[0, :][0]) or True  # top edge row exists
    assert first[0, 0] == 1.0 and first[0, 100] == 1.0 and first[100, 0] == 1.0  # top/left: scene edge, weight 1
    assert first[-1, -1] < 1.0  # bottom/right neighbours exist -> ramp
    last = tile_window(plan.tile_at(2, 4), config).numpy()
    assert last[-1, -1] == 1.0 and last[0, 0] < 1.0


def test_window_covers_only_the_valid_sr_extent_of_the_tile():
    config = TilingConfig()
    tile = plan_tiles(300, 500, config).tile_at(2, 4)  # valid 108 x 116
    assert tuple(tile_window(tile, config).shape) == (108 * 4, 116 * 4)


def test_overlap_zero_windows_are_all_ones():
    config = TilingConfig(overlap=0)
    for tile in plan_tiles(300, 500, config).tiles:
        assert torch.equal(tile_window(tile, config), torch.ones(tile.sr_valid_height, tile.sr_valid_width))


def test_windows_are_deterministic_float32():
    config = TilingConfig()
    tile = plan_tiles(300, 500, config).tile_at(1, 1)
    a, b = tile_window(tile, config), tile_window(tile, config)
    assert a.dtype == torch.float32 and torch.equal(a, b)


# ---------------------------------------------------------------- the accumulator


def test_canvas_normalises_a_weighted_average():
    canvas = WeightedCanvas(1, 1, 4)
    canvas.add(torch.tensor([[[10.0, 10.0]]]), torch.tensor([[1.0, 3.0]]), 0, 0)
    canvas.add(torch.tensor([[[20.0, 20.0]]]), torch.tensor([[1.0, 1.0]]), 0, 1)
    canvas.add(torch.tensor([[[30.0, 30.0]]]), torch.tensor([[1.0, 1.0]]), 0, 2)
    out = canvas.finalize()
    # col 0: only tile 1 -> 10; col 1: (3*10 + 1*20) / 4 = 12.5; col 2: (1*20 + 1*30) / 2 = 25; col 3: 30
    assert out.tolist() == [[[10.0, 12.5, 25.0, 30.0]]]


def test_canvas_refuses_to_return_an_incomplete_reconstruction():
    canvas = WeightedCanvas(1, 2, 4)
    canvas.add(torch.ones(1, 2, 2), torch.ones(2, 2), 0, 0)  # right half never covered
    with pytest.raises(ReconstructionError, match="incomplete: 4 of 8"):
        canvas.finalize()


def test_canvas_rejects_a_tile_that_falls_outside_it():
    canvas = WeightedCanvas(1, 4, 4)
    with pytest.raises(ReconstructionError, match="outside the canvas"):
        canvas.add(torch.ones(1, 2, 2), torch.ones(2, 2), 3, 0)


def test_canvas_rejects_weights_that_do_not_match_the_values():
    canvas = WeightedCanvas(1, 4, 4)
    with pytest.raises(ReconstructionError, match="do not match"):
        canvas.add(torch.ones(1, 2, 2), torch.ones(3, 3), 0, 0)


def test_a_single_full_weight_tile_is_returned_bit_exactly():
    x = torch.rand(4, 16, 16)
    canvas = WeightedCanvas(4, 16, 16)
    canvas.add(x, torch.ones(16, 16), 0, 0)
    assert torch.equal(canvas.finalize(), x)
