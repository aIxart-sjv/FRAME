"""frame.tiling.padding -- reflect padding of edge tiles."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from frame.tiling import plan_tiles
from frame.tiling.padding import extract_tile, reflect_indices


def test_reflect_indices_mirror_without_repeating_the_edge():
    assert reflect_indices(3, 8).tolist() == [0, 1, 2, 1, 0, 1, 2, 1]
    assert reflect_indices(4, 9).tolist() == [0, 1, 2, 3, 2, 1, 0, 1, 2]


def test_reflect_indices_identity_when_nothing_to_pad():
    assert reflect_indices(128, 128).tolist() == list(range(128))


def test_a_single_pixel_axis_repeats_that_pixel():
    assert reflect_indices(1, 5).tolist() == [0] * 5


@pytest.mark.parametrize("valid,total", [(2, 10), (3, 7), (5, 128), (64, 128), (100, 128), (127, 128), (1, 3)])
def test_reflect_indices_agree_with_numpy_reflect_padding(valid, total):
    data = np.arange(valid) + 10
    expected = np.pad(data, (0, total - valid), mode="reflect") if valid > 1 else np.full(total, data[0])
    assert data[reflect_indices(valid, total).numpy()].tolist() == expected.tolist()


def test_an_interior_tile_is_exactly_the_scene_slice_with_no_padding():
    scene = torch.arange(4 * 300 * 500, dtype=torch.float32).reshape(4, 300, 500)
    tile = plan_tiles(300, 500).tile_at(1, 2)
    out = extract_tile(scene, tile)
    assert out.shape == (4, 128, 128)
    assert torch.equal(out, scene[:, 96:224, 192:320])


def test_an_edge_tile_is_padded_to_full_size_with_the_valid_part_untouched():
    scene = torch.arange(4 * 300 * 500, dtype=torch.float32).reshape(4, 300, 500)
    tile = plan_tiles(300, 500).tile_at(2, 4)  # valid 108 x 116
    out = extract_tile(scene, tile)
    assert out.shape == (4, 128, 128)
    assert torch.equal(out[:, :108, :116], scene[:, 192:300, 384:500])


def test_padding_mirrors_the_scene_about_its_edge_in_both_axes():
    scene = torch.rand(2, 100, 90)
    tile = plan_tiles(100, 90).tiles[0]
    out = extract_tile(scene, tile).numpy()
    expected = np.pad(scene.numpy(), ((0, 0), (0, 28), (0, 38)), mode="reflect")
    assert np.array_equal(out, expected)


def test_padding_never_invents_values_it_only_reuses_scene_pixels():
    scene = torch.rand(4, 100, 90) + 5.0  # every value is >= 5, so a zero-fill would show up
    out = extract_tile(scene, plan_tiles(100, 90).tiles[0])
    assert float(out.min()) >= 5.0
    assert set(out.flatten().tolist()) <= set(scene.flatten().tolist())


def test_a_scene_smaller_than_a_tile_in_both_axes_still_yields_a_full_tile():
    scene = torch.rand(4, 7, 5)
    out = extract_tile(scene, plan_tiles(7, 5).tiles[0])
    assert out.shape == (4, 128, 128) and torch.equal(out[:, :7, :5], scene)


def test_unsupported_padding_mode_is_rejected():
    with pytest.raises(ValueError, match="padding_mode"):
        extract_tile(torch.rand(4, 10, 10), plan_tiles(10, 10).tiles[0], "zeros")
