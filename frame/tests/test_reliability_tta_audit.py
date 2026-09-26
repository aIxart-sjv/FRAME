"""Audit of the existing TTA stability layer (frame.uncertainty) for Phase 6: what exactly it computes, checked on rectangular scenes and through the tile engine.

The layer is not modified by Phase 6; these tests pin down its behaviour so that the validation rests on a verified quantity:

* the six transforms (identity, hflip, vflip, rot90, rot180, rot270) round-trip exactly on NON-square (C, H, W) tensors, and rot90 / rot270 swap H and W;
* de-augmentation puts every member back in the canonical orientation: a perfectly equivariant model has ZERO spread on any scene (a wrong inverse would show spread);
* the spread is the POPULATION standard deviation over the members (divide by N), per band, and the stability map is its band mean;
* a rectangular scene of any size runs through the Phase 2 tile engine with a full-size, orientation-consistent, seam-free result.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from frame.tiling import TilingConfig, TiledModel
from frame.uncertainty import DEFAULT_TRANSFORMS, HFLIP, IDENTITY, ROT90, ROT180, ROT270, VFLIP, run_stochastic_uncertainty, run_tta_ensemble
from frame.uncertainty.transforms import validate_round_trip

BANDS = ("B04", "B03", "B02", "B08")


def nearest_x4(x: torch.Tensor) -> torch.Tensor:
    """A perfectly D4-equivariant 'super-resolution' model: pixel replication. (1, C, h, w) -> (1, C, 4h, 4w)."""
    return x.repeat_interleave(4, dim=-2).repeat_interleave(4, dim=-1)


def scene(h, w, seed=0):
    return torch.from_numpy(np.random.default_rng(seed).random((4, h, w)).astype("float32"))


# ============================================================================== transforms on rectangular tensors


@pytest.mark.parametrize("h, w", [(5, 9), (9, 5), (1, 7), (16, 16), (130, 257)])
def test_every_default_transform_round_trips_exactly_on_rectangular_tensors(h, w):
    x = scene(h, w)
    for t in DEFAULT_TRANSFORMS:
        validate_round_trip(t, x)
        y = t.forward(x)
        expected_shape = (4, w, h) if t.name in ("rot90", "rot270") else (4, h, w)
        assert tuple(y.shape) == expected_shape, t.name
        assert torch.equal(t.inverse(y), x), t.name


def test_the_transforms_are_the_geometric_operations_they_are_named_for():
    x = torch.arange(2 * 3 * 4, dtype=torch.float32).reshape(2, 3, 4)
    a = x.numpy()
    assert torch.equal(IDENTITY.forward(x), x)
    assert np.array_equal(HFLIP.forward(x).numpy(), a[:, :, ::-1])
    assert np.array_equal(VFLIP.forward(x).numpy(), a[:, ::-1, :])
    assert np.array_equal(ROT90.forward(x).numpy(), np.rot90(a, k=1, axes=(1, 2)))
    assert np.array_equal(ROT180.forward(x).numpy(), np.rot90(a, k=2, axes=(1, 2)))
    assert np.array_equal(ROT270.forward(x).numpy(), np.rot90(a, k=3, axes=(1, 2)))
    assert [t.name for t in DEFAULT_TRANSFORMS] == ["identity", "hflip", "vflip", "rot90", "rot180", "rot270"]


def test_the_ensemble_members_are_the_model_applied_to_the_transformed_input_then_de_transformed():
    x = scene(6, 10, seed=1)
    kernel = torch.tensor([[0.0, 1.0], [0.0, 0.0]])                                          # an asymmetric operation: NOT equivariant

    def model(z):
        return nearest_x4(z) + 0.05 * torch.roll(nearest_x4(z), shifts=(1, 2), dims=(-2, -1)) + kernel.sum()

    r = run_tta_ensemble(model, x, DEFAULT_TRANSFORMS, seed=0, keep_per_member_predictions=True)
    for t, member in zip(DEFAULT_TRANSFORMS, r.per_member_predictions):
        assert torch.allclose(member, t.inverse(model(t.forward(x)[None]).squeeze(0)), atol=0), t.name
    assert r.std_prediction.max() > 0                                                        # a non-equivariant model does disagree with itself


# ============================================================================== what the spread is


@pytest.mark.parametrize("h, w", [(8, 8), (5, 9), (9, 5), (13, 4)])
def test_a_perfectly_equivariant_model_has_zero_spread_on_square_and_rectangular_scenes(h, w):
    x = scene(h, w, seed=2)
    r = run_stochastic_uncertainty(lambda z: nearest_x4(z), x, transforms=DEFAULT_TRANSFORMS, seed=0, band_names=BANDS)
    assert tuple(r.std_prediction.shape) == (4, 4 * h, 4 * w) and float(r.std_prediction.max()) < 1e-6
    assert torch.allclose(r.mean_prediction, nearest_x4(x[None])[0], atol=1e-6)


def test_the_spread_is_the_population_standard_deviation_over_the_members_and_the_stability_is_its_band_mean():
    x = scene(4, 6, seed=3)
    calls = {"n": 0}

    def model(z):
        calls["n"] += 1
        return nearest_x4(z) + (0.0 if calls["n"] == 1 else 0.02)                            # member 1 is 0.02 lower than member 2: a constant offset, so no orientation effect

    r = run_stochastic_uncertainty(model, x, transforms=(IDENTITY, HFLIP), seed=0, band_names=BANDS)
    assert torch.allclose(r.std_prediction, torch.full_like(r.std_prediction, 0.01), atol=1e-6)      # sqrt(((0 - 0.01)^2 + (0.02 - 0.01)^2) / 2) = 0.01, not the N-1 value 0.01414
    assert r.scalar_summary == pytest.approx(0.01, abs=1e-6) and r.n == 2
    assert torch.allclose(r.std_prediction.mean(dim=0), torch.full((16, 24), 0.01), atol=1e-6)


def test_a_position_dependent_model_produces_a_spread_that_lies_where_the_closed_form_says():
    """Rectangular scene, members (identity, vflip): the row ramp is seen as r(i) and r(H-1-i), so the spread at row i is |r(i) - r(H-1-i)| / 2, in the right place."""
    h, w = 3, 5
    x = scene(h, w, seed=4)
    ramp = torch.linspace(0.0, 0.3, 4 * h).reshape(1, 1, 4 * h, 1)

    def model(z):
        return nearest_x4(z) + ramp

    r = run_stochastic_uncertainty(model, x, transforms=(IDENTITY, VFLIP), seed=0, band_names=BANDS)
    expected = (ramp[0, 0, :, 0] - ramp[0, 0, :, 0].flip(0)).abs() / 2
    assert torch.allclose(r.std_prediction[0, :, 0], expected, atol=1e-6) and torch.allclose(r.std_prediction[3, :, 4 * w - 1], expected, atol=1e-6)


def test_a_single_member_ensemble_has_zero_spread_by_construction_which_is_not_stability():
    r = run_stochastic_uncertainty(lambda z: nearest_x4(z) + torch.rand(1) * 0, scene(4, 4), transforms=(IDENTITY,), seed=0, band_names=BANDS)
    assert r.n == 1 and float(r.std_prediction.max()) == 0.0


# ============================================================================== through the tile engine, on a rectangular scene


def test_a_rectangular_scene_through_the_tile_engine_has_full_size_zero_spread_for_an_equivariant_tile_model():
    tiling = TilingConfig(tile_size=128, overlap=32)
    x = scene(200, 300, seed=5)                                                              # 2 x 3 tiles; not a multiple of 128; rot90 makes it 300 x 200
    tiled = TiledModel(lambda t: nearest_x4(t), tiling)
    r = run_stochastic_uncertainty(tiled, x, transforms=DEFAULT_TRANSFORMS, seed=0, band_names=BANDS, keep_per_member_predictions=False)
    assert tuple(r.std_prediction.shape) == (4, 800, 1200) and tuple(r.mean_prediction.shape) == (4, 800, 1200)
    assert torch.isfinite(r.std_prediction).all() and float(r.std_prediction.max()) < 1e-5     # every member, through every tile grid, lands in the same orientation and agrees
    assert torch.allclose(r.mean_prediction, nearest_x4(x[None])[0], atol=1e-5)
    assert len(tiled.runs) == 6 and {run.sr_shape for run in tiled.runs} == {(800, 1200), (1200, 800)}       # the rotated views really were run on the rotated grid
