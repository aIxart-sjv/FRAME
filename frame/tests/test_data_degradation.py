"""frame.data.degradation -- the documented HR -> Sentinel-2-like LR chain: seeded, versioned, recorded."""

from __future__ import annotations

import dataclasses
import json

import pytest
import torch
import torch.nn.functional as F

from frame.data.contract import RasterSpec, Split
from frame.data.degradation import (
    DEGRADATION_NAME,
    DEGRADATION_VERSION,
    DegradationConfig,
    add_signal_dependent_noise,
    bilinear_downsample,
    degrade,
    frame_default_v1,
    gaussian_blur,
    histogram_match,
    synthesize_pair,
)
from frame.data.errors import DegradationError
from frame.data.geo import validate_pair_geometry
from frame.preprocessing import RGBN_BANDS


def hr_image(c=4, h=64, w=96, seed=0):
    return 0.05 + 0.4 * torch.rand(c, h, w, generator=torch.Generator().manual_seed(seed))


# ------------------------------------------------------------------------------ dimensions & determinism


@pytest.mark.parametrize("scale,hw", [(4, (64, 96)), (2, (64, 64)), (4, (128, 128)), (10, (100, 200))])
def test_output_dimensions_are_hr_over_scale(scale, hw):
    lr, _ = degrade(hr_image(3, *hw), DegradationConfig(scale=scale, blur_sigma=1.0), seed=0)
    assert lr.shape == (3, hw[0] // scale, hw[1] // scale) and lr.dtype == torch.float32


def test_the_same_seed_gives_bit_identical_output_and_a_different_seed_differs():
    x = hr_image()
    cfg = frame_default_v1()
    a, _ = degrade(x, cfg, seed=11)
    b, _ = degrade(x, cfg, seed=11)
    c, _ = degrade(x, cfg, seed=12)
    assert torch.equal(a, b)
    assert not torch.equal(a, c)


def test_without_noise_the_result_does_not_depend_on_the_seed():
    x = hr_image()
    cfg = DegradationConfig(scale=4, blur_sigma=2.0)
    assert torch.equal(degrade(x, cfg, seed=1)[0], degrade(x, cfg, seed=2)[0])


def test_noise_is_the_only_random_step():
    x = hr_image()
    quiet = DegradationConfig(scale=4, blur_sigma=2.0)
    noisy = DegradationConfig(scale=4, blur_sigma=2.0, noise_a=1e-4, noise_b=1e-5)
    clean, _ = degrade(x, quiet, seed=5)
    a, _ = degrade(x, noisy, seed=5)
    assert not torch.equal(clean, a) and float((a - clean).abs().max()) < 0.1


# ------------------------------------------------------------------------------ metadata


def test_the_record_captures_configuration_seed_and_version():
    cfg = frame_default_v1(scale=4, harmonisation="none")
    _, rec = degrade(hr_image(), cfg, seed=42)
    assert rec.name == DEGRADATION_NAME and rec.version == DEGRADATION_VERSION == "frame-degradation/1"
    assert rec.seed == 42 and rec.harmonisation == "none" and rec.origin == "frame"
    assert rec.config == dataclasses.asdict(cfg)
    json.dumps(rec.to_dict())  # serialisable into a manifest


def test_frame_defaults_are_never_labelled_as_the_published_sen2naipv2_parameters():
    _, rec = degrade(hr_image(), frame_default_v1(), seed=0)
    assert rec.parameters_verified is False


def test_the_recorded_config_regenerates_the_same_lr():
    x = hr_image()
    lr, rec = degrade(x, frame_default_v1(), seed=99)
    again, _ = degrade(x, DegradationConfig(**rec.config), seed=rec.seed)
    assert torch.equal(lr, again)


def test_the_default_blur_follows_the_scale():
    assert frame_default_v1(scale=4).blur_sigma == 2.0 and frame_default_v1(scale=2).blur_sigma == 1.0


# ------------------------------------------------------------------------------ the individual steps


def test_no_spatial_shift_an_impulse_lands_on_the_lr_pixel_that_covers_it():
    hr = torch.zeros(1, 32, 32)
    hr[:, 12:16, 20:24] = 1.0  # exactly the HR block of LR pixel (3, 5) at scale 4
    lr, _ = degrade(hr, DegradationConfig(scale=4, blur_sigma=0.0), seed=0)
    assert divmod(int(lr.flatten().argmax()), 8) == (3, 5)
    assert float(lr.max()) == pytest.approx(1.0) and float(lr.sum()) == pytest.approx(1.0)


def test_a_blurred_impulse_stays_centred_on_the_same_lr_pixel():
    hr = torch.zeros(1, 64, 64)
    hr[:, 26:30, 42:46] = 1.0  # LR pixel (6, 10)
    lr, _ = degrade(hr, DegradationConfig(scale=4, blur_sigma=2.0), seed=0)
    assert divmod(int(lr.flatten().argmax()), 16) == (6, 10)


def test_blur_is_a_normalised_kernel_so_it_preserves_the_mean():
    x = hr_image(2, 48, 48)
    assert float(gaussian_blur(x, 2.0).mean()) == pytest.approx(float(x.mean()), rel=1e-3)
    assert torch.equal(gaussian_blur(x, 0.0), x)
    assert float(gaussian_blur(x, 3.0).std()) < float(x.std())  # it really smooths


def test_bilinear_downsampling_of_a_constant_or_linear_image_is_exact():
    assert torch.allclose(bilinear_downsample(torch.full((1, 16, 16), 0.3), 4), torch.full((1, 4, 4), 0.3))
    ramp = torch.arange(16, dtype=torch.float32).repeat(16, 1)[None]  # value = column index
    down = bilinear_downsample(ramp, 4)[0, 0]
    assert torch.allclose(down, torch.tensor([1.5, 5.5, 9.5, 13.5]))  # centres of the HR blocks: no shift


def test_noise_grows_with_the_signal():
    dark, bright = torch.full((1, 200, 200), 0.02), torch.full((1, 200, 200), 0.4)
    nd = add_signal_dependent_noise(dark, 1e-4, 0.0, 3, 0.0) - dark
    nb = add_signal_dependent_noise(bright, 1e-4, 0.0, 3, 0.0) - bright
    assert float(nb.std()) > 3 * float(nd.std())
    assert float(nb.std()) == pytest.approx((1e-4 * 0.4) ** 0.5, rel=0.1)


def test_reflectance_is_clipped_at_zero():
    out = add_signal_dependent_noise(torch.zeros(1, 50, 50), 0.0, 1e-2, 0, 0.0)
    assert float(out.min()) >= 0.0


def test_histogram_matching_maps_the_distribution_onto_the_reference():
    src = hr_image(2, 40, 40, seed=1)
    ref = 0.2 + 0.1 * torch.rand(2, 30, 30, generator=torch.Generator().manual_seed(9))
    out = histogram_match(src, ref)
    assert out.shape == src.shape
    for b in range(2):
        assert float(out[b].min()) == pytest.approx(float(ref[b].min()), abs=1e-6)
        assert float(out[b].max()) == pytest.approx(float(ref[b].max()), abs=1e-6)
        assert float(out[b].mean()) == pytest.approx(float(ref[b].mean()), abs=2e-3)
    assert torch.equal(torch.argsort(out[0].flatten(), stable=True), torch.argsort(src[0].flatten(), stable=True))  # rank order kept


def test_histogram_harmonisation_needs_a_reference_and_matching_bands():
    with pytest.raises(DegradationError, match="reference"):
        degrade(hr_image(), DegradationConfig(scale=4, blur_sigma=1.0, harmonisation="histogram_match"), seed=0)
    with pytest.raises(DegradationError, match="band"):
        degrade(hr_image(4), DegradationConfig(scale=4, harmonisation="histogram_match"), seed=0, reference=torch.rand(3, 8, 8))


def test_histogram_harmonisation_changes_the_radiometry_and_is_recorded():
    x = hr_image()
    ref = 0.3 + 0.05 * torch.rand(4, 16, 16, generator=torch.Generator().manual_seed(2))
    lr, rec = degrade(x, DegradationConfig(scale=4, blur_sigma=1.0, harmonisation="histogram_match"), seed=0, reference=ref)
    assert rec.harmonisation == "histogram_match"
    assert float(lr.mean()) == pytest.approx(float(ref.mean()), abs=3e-3) and float(lr.mean()) != pytest.approx(float(x.mean()), abs=1e-2)


# ------------------------------------------------------------------------------ invalid input


def test_invalid_configuration_is_rejected():
    for kw in (dict(scale=0), dict(scale=2.5), dict(blur_sigma=-1), dict(noise_a=-1e-4), dict(noise_b=float("nan")), dict(harmonisation="unet")):
        with pytest.raises(DegradationError):
            DegradationConfig(**kw)


def test_sizes_not_divisible_by_the_scale_are_rejected_not_cropped():
    with pytest.raises(DegradationError, match="divisible"):
        degrade(hr_image(4, 65, 64), DegradationConfig(scale=4, blur_sigma=0.0), seed=0)


def test_corrupt_or_malformed_input_is_refused():
    bad = hr_image()
    bad[0, 3, 3] = float("nan")
    with pytest.raises(DegradationError, match="NaN"):
        degrade(bad, frame_default_v1(), seed=0)
    with pytest.raises(DegradationError):
        degrade(hr_image()[0], frame_default_v1(), seed=0)  # 2-D
    with pytest.raises(DegradationError):
        degrade((hr_image() * 1000).to(torch.int32), frame_default_v1(), seed=0)


def test_a_blur_kernel_larger_than_the_image_is_refused():
    with pytest.raises(DegradationError, match="Blur radius"):
        gaussian_blur(torch.rand(1, 6, 6), 5.0)


# ------------------------------------------------------------------------------ synthetic pair


HR_T = (2.5, 0.0, 500000.0, 0.0, -2.5, 4000000.0)


def hr_spec(w=96, h=64):
    return RasterSpec(band_names=RGBN_BANDS, width=w, height=h, pixel_size_m=2.5, crs="EPSG:32630", transform=HR_T)


def test_a_synthetic_pair_is_complete_valid_and_geometrically_consistent():
    hr = hr_image(4, 64, 96)
    s = synthesize_pair(hr, sample_id="d:syn", dataset="tinyset", scene_id="sc", split=Split.TRAIN, hr_spec=hr_spec(), config=frame_default_v1(), seed=3,
                        region_id="r", license="CC0-1.0", provenance={"source": "NAIP-like"})
    assert s.lr.shape == (4, 16, 24) and s.hr.shape == (4, 64, 96) and s.scale_factor == 4
    r = s.record
    assert r.pair_type.value == "synthetic" and r.degradation.seed == 3 and r.provenance == {"source": "NAIP-like"}
    assert r.lr.pixel_size_m == 10.0 and r.lr.transform == pytest.approx((10.0, 0.0, 500000.0, 0.0, -10.0, 4000000.0))
    assert validate_pair_geometry(r.lr, r.hr, r.scale_factor) is True  # the derived LR grid matches the HR grid exactly
    assert s.lr_bands == RGBN_BANDS


def test_a_synthetic_pair_regenerates_from_its_own_record():
    hr = hr_image(4, 64, 96)
    s = synthesize_pair(hr, sample_id="d:syn", dataset="tinyset", scene_id="sc", split=Split.TRAIN, hr_spec=hr_spec(), config=frame_default_v1(), seed=8)
    again, _ = degrade(hr, DegradationConfig(**s.record.degradation.config), seed=s.record.degradation.seed)
    assert torch.equal(s.lr, again)


def test_hr_nodata_propagates_to_a_conservative_lr_mask():
    mask = torch.ones(64, 96, dtype=torch.bool)
    mask[0:8, 0:8] = False  # covers LR pixels (0..1, 0..1) entirely
    mask[20, 40] = False    # one stray HR pixel inside LR pixel (5, 10)
    s = synthesize_pair(hr_image(4, 64, 96), sample_id="d:syn", dataset="tinyset", scene_id="sc", split=Split.TRAIN, hr_spec=hr_spec(),
                        config=frame_default_v1(), seed=0, hr_mask=mask)
    assert s.lr_mask.shape == (16, 24) and not bool(s.lr_mask[:2, :2].any()) and not bool(s.lr_mask[5, 10])
    assert bool(s.lr_mask[8, 8])  # untouched blocks stay valid


def test_a_synthetic_pair_without_georeferencing_still_works():
    spec = RasterSpec(band_names=RGBN_BANDS, width=96, height=64, pixel_size_m=2.5)
    s = synthesize_pair(hr_image(4, 64, 96), sample_id="d:syn", dataset="tinyset", scene_id="sc", split=Split.TRAIN, hr_spec=spec, config=frame_default_v1(), seed=0)
    assert s.record.lr.transform is None and s.record.lr.pixel_size_m == 10.0
