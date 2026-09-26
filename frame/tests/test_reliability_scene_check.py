"""frame.reliability.scene_check -- the multi-tile / rectangular-scene check of the stability map: coverage, orientation, seams, georeferencing (Phase 6)."""

from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from frame.preprocessing.metadata import RasterMetadata
from frame.reliability.scene_check import check_scene, seam_mask, seam_statistics
from frame.tiling import TilingConfig, plan_tiles

BANDS = ("B04", "B03", "B02", "B08")
TILING = TilingConfig(tile_size=64, overlap=16)


def nearest_x4(x):
    return x.repeat_interleave(4, dim=-2).repeat_interleave(4, dim=-1)


class Skewed(torch.nn.Module):
    def forward(self, x):
        up = F.interpolate(x, scale_factor=4, mode="bicubic", antialias=True).clamp(min=0)
        return up + 0.5 * (up - torch.roll(up, shifts=(1, 2), dims=(-2, -1)))


def lr_scene(h, w, seed=0):
    rng = np.random.default_rng(seed)
    base = torch.from_numpy(rng.random((4, h, w)).astype("float32"))
    return F.avg_pool2d(F.pad(base[None], (2, 2, 2, 2), mode="reflect"), 5, stride=1)[0] * 0.5 + 0.1


def metadata(h, w):
    return RasterMetadata(crs="EPSG:32630", transform=(10.0, 0.0, 720285.0, 0.0, -10.0, 4375125.0), bounds=(720285.0, 4375125.0 - 10.0 * h, 720285.0 + 10.0 * w, 4375125.0), resolution_m=10.0,
                          width=w, height=h, band_names=BANDS, acquisition_timestamp="2023-01-15T10:54:11", nodata_value=None, cloud_mask_coverage=1.0, sr_variant="unit")


def check(model, h=100, w=150, tmp=None, **kw):
    return check_scene(model, lr_scene(h, w), metadata(h, w), TILING, work_dir=tmp, **kw)


# ============================================================================== an equivariant model: everything must be exactly consistent


def test_an_equivariant_model_on_a_rectangular_multi_tile_scene_has_full_size_zero_spread(tmp_path):
    r = check(nearest_x4, tmp=tmp_path)
    assert r["scene"]["lr_shape"] == [4, 100, 150] and r["scene"]["sr_shape"] == [4, 400, 600] and r["scene"]["tile_count"] >= 4
    assert r["coverage"]["std_shape_matches_output"] is True and r["coverage"]["non_finite_std_pixels"] == 0 and r["coverage"]["non_finite_mean_pixels"] == 0
    assert r["stability"]["max_std"] < 1e-5 and r["orientation"]["all_members_aligned"] is True
    assert set(r["orientation"]["members"]) == {"identity", "hflip", "vflip", "rot90", "rot180", "rot270"}


def test_the_views_really_used_a_different_tile_grid_for_the_rotated_scene(tmp_path):
    r = check(nearest_x4, h=100, w=150, tmp=tmp_path)
    assert r["scene"]["view_grids"]["identity"] != r["scene"]["view_grids"]["rot90"]                      # rot90 turns a 100 x 150 scene into 150 x 100: a different grid, different seams


# ============================================================================== a real (non-equivariant) model


def test_orientation_is_preserved_for_every_member_even_when_the_model_is_not_equivariant(tmp_path):
    r = check(Skewed(), tmp=tmp_path)
    assert r["orientation"]["all_members_aligned"] is True and r["stability"]["mean_std"] > 0
    for name, m in r["orientation"]["members"].items():
        assert m["mae_vs_identity"] <= 0.5 * m["mae_if_misoriented"], name                                 # aligned members are far closer to the identity view than a mis-oriented one would be


def test_a_de_transform_that_puts_a_member_in_the_wrong_orientation_is_detected(tmp_path):
    """A transform whose inverse is right on the LR-sized round-trip guard but wrong on the 4x larger SR output: only the orientation check can see it."""
    from frame.uncertainty.transforms import Transform

    lr_width = 64

    def inverse(x):
        return torch.flip(x, dims=(-1,)) if x.shape[-1] == lr_width else x                  # passes the LR round-trip, forgets to flip the SR back

    broken = Transform("hflip", lambda x: torch.flip(x, dims=(-1,)), inverse)
    r = check_scene(Skewed(), lr_scene(64, lr_width), metadata(64, lr_width), TILING, work_dir=tmp_path, transforms=(_identity(), broken))
    assert r["orientation"]["all_members_aligned"] is False and r["orientation"]["members"]["hflip"]["aligned"] is False and r["orientation"]["members"]["identity"]["aligned"] is True


def _identity():
    from frame.uncertainty.transforms import IDENTITY

    return IDENTITY


def test_no_stability_seam_is_introduced_at_the_tile_boundaries_of_either_grid(tmp_path):
    r = check(Skewed(), h=120, w=180, tmp=tmp_path)
    s = r["seams"]
    assert s["canonical_grid"]["n_seam_pixels"] > 0 and s["rot90_grid"]["n_seam_pixels"] > 0
    assert 0.8 < s["canonical_grid"]["seam_over_interior"] < 1.1 and 0.8 < s["rot90_grid"]["seam_over_interior"] < 1.1 and s["no_seam_introduced"] is True
    assert s["tolerance"] == 1.10 and "half_width_px" in s


def test_a_stability_map_with_a_seam_artefact_is_flagged():
    """The seam statistic itself: a spread bump exactly on the tile seams must exceed the tolerance, a flat map must not."""
    plan = plan_tiles(100, 150, TILING)
    mask = seam_mask(plan, TILING, half_width=4)
    flat = np.full(mask.shape, 0.002)
    bumped = flat.copy()
    bumped[mask] *= 1.5
    ok, bad = seam_statistics(flat, mask, tolerance=1.10), seam_statistics(bumped, mask, tolerance=1.10)
    assert ok["seam_over_interior"] == pytest.approx(1.0) and ok["within_tolerance"] is True
    assert bad["seam_over_interior"] == pytest.approx(1.5) and bad["within_tolerance"] is False and bad["n_seam_pixels"] == int(mask.sum())
    assert seam_statistics(flat, np.zeros_like(mask), tolerance=1.10)["seam_over_interior"] is None            # no seams (a single tile): nothing to compare, not a pass


def test_seam_masks_lie_on_the_blend_overlaps_of_the_tile_plan():
    plan = plan_tiles(100, 150, TILING)
    m = seam_mask(plan, TILING, half_width=2)
    assert m.shape == (400, 600) and m.any() and not m.all()
    rows = sorted({t.sr_row_start + 32 for t in plan.tiles if t.sr_row_start > 0})
    assert all(m[r, :].all() for r in rows)


# ============================================================================== georeferencing


def test_the_uncertainty_geotiff_shares_the_grid_of_the_sr_output_and_round_trips(tmp_path):
    r = check(Skewed(), h=100, w=150, tmp=tmp_path)
    g = r["georeferencing"]
    assert g["crs_equal"] is True and g["origin_equal"] is True and g["pixel_size_m"] == pytest.approx(2.5) and g["bounds_equal"] is True
    assert g["shape"] == [400, 600] and g["array_round_trips_exactly"] is True and g["band_names"][-1] == "overall_std"
    assert (tmp_path / "stability_check.tif").is_file()


def test_the_check_records_its_own_timing_and_the_member_count(tmp_path):
    r = check(nearest_x4, tmp=tmp_path)
    assert r["timing"]["n_members"] == 6 and r["timing"]["tta_seconds"] > 0 and r["timing"]["single_pass_seconds_median"] > 0 and r["timing"]["tta_over_single_pass"] > 1
