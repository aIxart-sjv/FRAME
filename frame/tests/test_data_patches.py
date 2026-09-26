"""frame.data.patches -- aligned paired patch extraction (LR window x scale = HR window)."""

from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from frame.data.contract import PairedSample, PairRecord, PatchCoords, RasterSpec
from frame.data.errors import PatchError
from frame.data.patches import (
    centred_origin,
    check_alignment,
    dense_origins,
    extract_patch,
    make_coords,
    random_origin,
    valid_fraction,
)
from frame.preprocessing import RGBN_BANDS


def make_sample(lr_hw=(64, 96), scale=4, bands=RGBN_BANDS, *, lr=None, hr=None, lr_mask=None, hr_mask=None) -> PairedSample:
    """An LR/HR pair where HR is the NEAREST upsample of LR: any correctly aligned HR patch must equal the
    upsample of its LR patch, so a misaligned crop cannot pass unnoticed."""
    h, w = lr_hw
    lr = lr if lr is not None else torch.rand(len(bands), h, w)
    hr = hr if hr is not None else F.interpolate(lr[None], scale_factor=scale, mode="nearest")[0]
    lr_spec = RasterSpec(band_names=bands, width=lr.shape[-1], height=lr.shape[-2], pixel_size_m=10.0)
    hr_spec = RasterSpec(band_names=bands, width=hr.shape[-1], height=hr.shape[-2], pixel_size_m=10.0 / scale)
    rec = PairRecord(sample_id="d:x", dataset="tinyset", scene_id="s", region_id="r", split="train", pair_type="real_cross_sensor",
                     hr_status="available", scale_factor=scale, lr=lr_spec, hr=hr_spec)
    return PairedSample(record=rec, lr=lr, hr=hr, lr_bands=bands, hr_bands=bands, lr_mask=lr_mask, hr_mask=hr_mask)


# ------------------------------------------------------------------------------ correspondence


@pytest.mark.parametrize("scale", [2, 4, 10])
def test_the_hr_patch_is_exactly_the_upsampled_lr_patch_for_every_origin(scale):
    """The heart of the module: LR and HR patches cover the same ground, so with HR = upsample(LR)
    the HR patch must equal the upsample of the LR patch, at every origin we try."""
    sample = make_sample((40, 56), scale)
    patch = 16
    rng = np.random.default_rng(0)
    for _ in range(25):
        r, c = random_origin(40, 56, patch, rng)
        p = extract_patch(sample, make_coords(sample, r, c, patch))
        assert p.lr.shape == (4, patch, patch) and p.hr.shape == (4, patch * scale, patch * scale)
        assert torch.equal(p.hr, F.interpolate(p.lr[None], scale_factor=scale, mode="nearest")[0])


def test_the_documented_128_512_relationship_for_x4():
    sample = make_sample((200, 300), 4)
    p = extract_patch(sample, make_coords(sample, 50, 70, 128))
    assert p.lr.shape == (4, 128, 128) and p.hr.shape == (4, 512, 512)
    assert (p.patch.hr_row, p.patch.hr_col, p.patch.hr_height, p.patch.hr_width) == (200, 280, 512, 512)
    assert torch.equal(p.hr, sample.hr[:, 200:712, 280:792])


def test_x2_pairs_use_their_own_scale_not_four():
    sample = make_sample((64, 64), 2)
    p = extract_patch(sample, make_coords(sample, 8, 16, 32))
    assert p.hr.shape == (4, 64, 64) and p.patch.scale == 2


def test_a_wrongly_offset_hr_would_be_caught_by_the_correspondence_check():
    """Guard for the guard: shifting the HR by one pixel breaks the equality the tests above rely on."""
    sample = make_sample((32, 32), 4)
    shifted = PairedSample(record=sample.record, lr=sample.lr, hr=torch.roll(sample.hr, 1, dims=-1), lr_bands=RGBN_BANDS, hr_bands=RGBN_BANDS)
    p = extract_patch(shifted, make_coords(shifted, 4, 4, 16))
    assert not torch.equal(p.hr, F.interpolate(p.lr[None], scale_factor=4, mode="nearest")[0])


def test_band_order_is_preserved_through_extraction():
    lr = torch.stack([torch.full((32, 32), float(i)) for i in range(4)])
    sample = make_sample(lr=lr)
    p = extract_patch(sample, make_coords(sample, 0, 0, 16))
    assert p.lr_bands == RGBN_BANDS and p.hr_bands == RGBN_BANDS
    assert [float(ch.mean()) for ch in p.lr] == [0.0, 1.0, 2.0, 3.0]
    assert [float(ch.mean()) for ch in p.hr] == [0.0, 1.0, 2.0, 3.0]


def test_masks_are_cropped_with_the_same_windows():
    lr_mask = torch.ones(32, 32, dtype=torch.bool)
    lr_mask[:16, :] = False
    sample = make_sample((32, 32), 4, lr_mask=lr_mask, hr_mask=F.interpolate(lr_mask.float()[None, None], scale_factor=4, mode="nearest")[0, 0] > 0)
    top = extract_patch(sample, make_coords(sample, 0, 0, 16))
    bottom = extract_patch(sample, make_coords(sample, 16, 0, 16))
    assert valid_fraction(top) == 0.0 and valid_fraction(bottom) == 1.0
    assert top.hr_mask.shape == (64, 64) and not bool(top.hr_mask.any())


def test_valid_fraction_is_the_smaller_of_lr_and_hr_and_one_without_masks():
    assert valid_fraction(make_sample()) == 1.0
    lr_mask = torch.ones(64, 96, dtype=torch.bool)
    hr_mask = torch.ones(256, 384, dtype=torch.bool)
    hr_mask[:, :192] = False
    assert valid_fraction(make_sample(lr_mask=lr_mask, hr_mask=hr_mask)) == pytest.approx(0.5)


def test_extraction_is_deterministic_and_copies_the_data():
    sample = make_sample()
    coords = make_coords(sample, 3, 5, 16)
    a, b = extract_patch(sample, coords), extract_patch(sample, coords)
    assert torch.equal(a.lr, b.lr) and torch.equal(a.hr, b.hr)
    a.lr.zero_()
    assert float(sample.lr.abs().sum()) > 0  # the source pair is untouched


# ------------------------------------------------------------------------------ coordinates


def test_dense_origins_cover_a_rectangular_non_multiple_raster_without_padding():
    origins = dense_origins(300, 500, 128)
    rows = sorted({r for r, _ in origins})
    cols = sorted({c for _, c in origins})
    assert rows == [0, 128, 172] and cols == [0, 128, 256, 372]  # last window clamped inside, never padded
    assert len(origins) == 12
    covered = np.zeros((300, 500), dtype=bool)
    for r, c in origins:
        assert r + 128 <= 300 and c + 128 <= 500  # never leaves the raster
        covered[r : r + 128, c : c + 128] = True
    assert covered.all()  # ...yet every pixel is covered


def test_dense_origins_are_row_major_and_deterministic():
    a = dense_origins(130, 130, 128)
    assert a == [(0, 0), (0, 2), (2, 0), (2, 2)] == dense_origins(130, 130, 128)


def test_dense_origins_with_overlap_and_exact_fit():
    assert dense_origins(128, 128, 128) == [(0, 0)]
    assert dense_origins(64, 64, 32, stride=16) == [(r, c) for r in (0, 16, 32) for c in (0, 16, 32)]
    assert dense_origins(64, 96, (32, 48), stride=(32, 48)) == [(0, 0), (0, 48), (32, 0), (32, 48)]


def test_a_patch_larger_than_the_raster_is_an_error():
    with pytest.raises(PatchError, match="does not fit"):
        dense_origins(100, 500, 128)
    with pytest.raises(PatchError):
        random_origin(100, 100, 128, np.random.default_rng(0))
    with pytest.raises(PatchError):
        centred_origin(100, 100, 128)


def test_stride_must_be_positive():
    with pytest.raises(PatchError, match="stride"):
        dense_origins(256, 256, 128, stride=0)


def test_random_origins_stay_inside_and_are_reproducible():
    a = [random_origin(300, 500, 128, np.random.default_rng([1, i])) for i in range(50)]
    assert a == [random_origin(300, 500, 128, np.random.default_rng([1, i])) for i in range(50)]
    assert all(0 <= r <= 172 and 0 <= c <= 372 for r, c in a)
    assert len(set(a)) > 10  # they do vary


def test_centred_origin_gives_the_128_core_of_a_130_tile():
    """SEN2NAIPv2 ships 130x130 LR / 520x520 HR; the 128/512 model crop is the centre, HR offset = 4 x LR offset."""
    assert centred_origin(130, 130, 128) == (1, 1)
    sample = make_sample((130, 130), 4)
    p = extract_patch(sample, make_coords(sample, 1, 1, 128))
    assert p.hr.shape == (4, 512, 512) and (p.patch.hr_row, p.patch.hr_col) == (4, 4)
    assert torch.equal(p.hr, sample.hr[:, 4:516, 4:516])


# ------------------------------------------------------------------------------ errors


def test_windows_outside_the_raster_are_rejected():
    sample = make_sample((32, 32), 4)
    with pytest.raises(PatchError, match="outside"):
        make_coords(sample, 20, 0, 16)
    with pytest.raises(PatchError):
        make_coords(sample, -1, 0, 16)
    with pytest.raises(PatchError, match="outside"):
        extract_patch(sample, PatchCoords(lr_row=20, lr_col=0, lr_height=16, lr_width=16, scale=4))


def test_a_patch_whose_scale_differs_from_the_pairs_is_rejected():
    sample = make_sample((32, 32), 4)
    with pytest.raises(PatchError, match="scale"):
        extract_patch(sample, PatchCoords(lr_row=0, lr_col=0, lr_height=8, lr_width=8, scale=2))


def test_misaligned_rasters_are_rejected_before_any_crop():
    check_alignment((32, 32), (128, 128), 4)
    with pytest.raises(PatchError, match="not LR"):
        check_alignment((32, 32), (120, 128), 4)
    with pytest.raises(PatchError):
        check_alignment((32, 32), (64, 64), 4)


def test_an_lr_only_sample_can_still_be_cropped():
    rec = PairRecord(sample_id="d:lr", dataset="india_holdout", scene_id="s", region_id="r", split="test", pair_type="independent_hr_reference",
                     hr_status="unknown", scale_factor=4, lr=RasterSpec(band_names=RGBN_BANDS, width=64, height=64), hr=None)
    s = PairedSample(record=rec, lr=torch.rand(4, 64, 64), hr=None, lr_bands=RGBN_BANDS, hr_bands=None)
    p = extract_patch(s, make_coords(s, 8, 8, 32))
    assert p.lr.shape == (4, 32, 32) and p.hr is None
