"""frame.train.losses -- each training loss on its own, then the weighted composite."""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn.functional as F

from frame.train.config import LossConfig
from frame.train.errors import LossInputError
from frame.train.losses import CompositeLoss, charbonnier_loss, consistency_loss, l1_loss, spectral_angle_loss

B, C, H, W = 2, 4, 8, 8


def pair(seed=0, offset=0.1):
    g = torch.Generator().manual_seed(seed)
    target = torch.rand(B, C, H, W, generator=g) * 0.5 + 0.1
    return target + offset, target


# ============================================================================== L1


def test_l1_is_exactly_zero_for_identical_tensors():
    _, target = pair()
    assert float(l1_loss(target.clone(), target)) == 0.0


def test_l1_has_the_known_value_for_a_constant_offset():
    pred, target = pair(offset=0.1)
    assert float(l1_loss(pred, target)) == pytest.approx(0.1, abs=1e-6)


def test_l1_gradient_is_sign_over_element_count():
    pred, target = pair()
    pred = pred.clone().requires_grad_()
    l1_loss(pred, target).backward()
    assert torch.allclose(pred.grad, torch.full_like(pred, 1.0 / pred.numel()))          # every pixel is over-predicted by 0.1


def test_l1_ignores_masked_pixels_in_value_and_gradient():
    pred, target = pair()
    mask = torch.ones(B, H, W, dtype=torch.bool)
    mask[:, :4, :] = False
    pred = pred.clone()
    pred[:, :, :4, :] += 5.0                          # huge error, but only where the mask says "invalid"
    pred.requires_grad_()
    loss = l1_loss(pred, target, mask)
    loss.backward()
    assert float(loss.detach()) == pytest.approx(0.1, abs=1e-6)
    assert float(pred.grad[:, :, :4, :].abs().sum()) == 0.0 and float(pred.grad[:, :, 4:, :].abs().sum()) > 0.0


def test_a_fully_masked_batch_gives_zero_loss_and_finite_zero_gradients():
    pred, target = pair()
    pred = pred.clone().requires_grad_()
    loss = l1_loss(pred, target, torch.zeros(B, H, W, dtype=torch.bool))
    loss.backward()
    assert float(loss) == 0.0 and torch.isfinite(pred.grad).all() and float(pred.grad.abs().sum()) == 0.0


def test_losses_reject_mismatched_shapes_and_bad_masks():
    pred, target = pair()
    with pytest.raises(LossInputError, match="shape"):
        l1_loss(pred[:, :3], target)
    with pytest.raises(LossInputError, match="mask"):
        l1_loss(pred, target, torch.ones(B, H + 1, W, dtype=torch.bool))
    with pytest.raises(LossInputError, match="4-D"):
        l1_loss(pred[0], target[0])


def test_losses_are_computed_in_float32_even_for_half_precision_inputs():
    pred, target = pair()
    out = l1_loss(pred.half(), target.half())
    assert out.dtype == torch.float32


# ============================================================================== Charbonnier


def test_charbonnier_is_zero_for_identical_tensors_and_has_a_finite_zero_gradient():
    _, target = pair()
    x = target.clone().requires_grad_()
    loss = charbonnier_loss(x, target)
    loss.backward()
    assert float(loss) == 0.0 and torch.isfinite(x.grad).all() and float(x.grad.abs().max()) == 0.0


def test_charbonnier_matches_its_formula():
    pred, target = pair(offset=0.2)
    eps = 1e-3
    expected = math.sqrt(0.2 ** 2 + eps ** 2) - eps
    assert float(charbonnier_loss(pred, target, eps=eps)) == pytest.approx(expected, rel=1e-4)


def test_charbonnier_approaches_l1_when_the_error_is_much_larger_than_eps():
    pred, target = pair(offset=0.3)
    assert float(charbonnier_loss(pred, target, eps=1e-4)) == pytest.approx(float(l1_loss(pred, target)), abs=2e-4)


def test_charbonnier_respects_the_mask():
    pred, target = pair()
    mask = torch.zeros(B, H, W, dtype=torch.bool)
    mask[:, 2:4, 2:4] = True
    pred = pred.clone()
    pred[:, :, 5:, 5:] += 9.0
    assert float(charbonnier_loss(pred, target, mask)) == pytest.approx(float(charbonnier_loss(pred[:, :, 2:4, 2:4], target[:, :, 2:4, 2:4])), rel=1e-5)


# ============================================================================== spectral angle


def test_the_spectral_term_is_zero_for_identical_spectra_and_for_pure_brightness_changes():
    _, target = pair()
    assert float(spectral_angle_loss(target.clone(), target)) == pytest.approx(0.0, abs=1e-6)
    assert float(spectral_angle_loss(target * 1.7, target)) == pytest.approx(0.0, abs=1e-6)       # same spectral direction


def test_the_spectral_term_is_one_for_orthogonal_spectra():
    target = torch.zeros(1, 2, 1, 1)
    target[0, 0] = 1.0
    pred = torch.zeros(1, 2, 1, 1)
    pred[0, 1] = 1.0
    assert float(spectral_angle_loss(pred, target)) == pytest.approx(1.0, abs=1e-6)


def test_the_spectral_term_grows_with_the_angle():
    target = torch.tensor([1.0, 0.0]).view(1, 2, 1, 1)
    small = torch.tensor([1.0, 0.1]).view(1, 2, 1, 1)
    large = torch.tensor([1.0, 1.0]).view(1, 2, 1, 1)
    assert 0.0 < float(spectral_angle_loss(small, target)) < float(spectral_angle_loss(large, target))


def test_the_spectral_gradient_is_finite_at_identical_and_at_all_zero_spectra():
    _, target = pair()
    x = target.clone().requires_grad_()
    spectral_angle_loss(x, target).backward()
    assert torch.isfinite(x.grad).all()
    zero = torch.zeros(1, 4, 2, 2, requires_grad=True)
    spectral_angle_loss(zero, target[:1, :, :2, :2]).backward()
    assert torch.isfinite(zero.grad).all()


def test_the_spectral_term_needs_more_than_one_band_to_mean_anything():
    with pytest.raises(LossInputError, match="at least 2"):
        spectral_angle_loss(torch.rand(1, 1, 4, 4), torch.rand(1, 1, 4, 4))


def test_the_spectral_term_respects_the_mask():
    pred, target = pair(offset=0.0)
    pred = pred.clone()
    pred[:, :, :4, :] = pred[:, :, :4, :].flip(1)          # scramble the spectra in the top half
    mask = torch.ones(B, H, W, dtype=torch.bool)
    mask[:, :4, :] = False
    assert float(spectral_angle_loss(pred, target, mask)) == pytest.approx(0.0, abs=1e-6)
    assert float(spectral_angle_loss(pred, target)) > 1e-4


# ============================================================================== downsample consistency


def test_consistency_is_zero_when_the_area_average_of_sr_equals_lr():
    lr = torch.rand(B, C, 4, 4)
    sr = lr.repeat_interleave(4, dim=2).repeat_interleave(4, dim=3)      # its 4x4 block means are exactly lr
    assert float(consistency_loss(sr, lr, 4)) == pytest.approx(0.0, abs=1e-7)


def test_consistency_measures_a_constant_radiometric_shift():
    lr = torch.rand(B, C, 4, 4)
    sr = lr.repeat_interleave(4, dim=2).repeat_interleave(4, dim=3) + 0.05
    assert float(consistency_loss(sr, lr, 4)) == pytest.approx(0.05, abs=1e-6)


def test_consistency_ignores_sub_pixel_detail_that_averages_out():
    lr = torch.rand(B, C, 4, 4)
    sr = lr.repeat_interleave(4, dim=2).repeat_interleave(4, dim=3)
    pattern = torch.tensor([[1.0, -1.0], [-1.0, 1.0]]).repeat(8, 8) * 0.1          # zero mean in every 4x4 block
    assert float(consistency_loss(sr + pattern, lr, 4)) == pytest.approx(0.0, abs=1e-6)


def test_consistency_respects_the_lr_mask_and_checks_the_geometry():
    lr = torch.rand(B, C, 4, 4)
    sr = lr.repeat_interleave(4, dim=2).repeat_interleave(4, dim=3)
    sr = sr.clone()
    sr[:, :, :4, :4] += 3.0                                                 # only LR pixel (0, 0) is wrong
    mask = torch.ones(B, 4, 4, dtype=torch.bool)
    mask[:, 0, 0] = False
    assert float(consistency_loss(sr, lr, 4, mask)) == pytest.approx(0.0, abs=1e-6)
    with pytest.raises(LossInputError, match="scale"):
        consistency_loss(sr, lr, 3)


def test_consistency_backpropagates_to_the_sr_tensor():
    lr = torch.rand(1, C, 4, 4)
    sr = (lr.repeat_interleave(4, dim=2).repeat_interleave(4, dim=3) + 0.1).clone().requires_grad_()
    consistency_loss(sr, lr, 4).backward()
    assert torch.isfinite(sr.grad).all() and float(sr.grad.abs().sum()) > 0.0


# ============================================================================== composite


def make_batch():
    lr = torch.rand(B, C, 4, 4)
    hr = torch.rand(B, C, 16, 16) * 0.5 + 0.1
    sr = (hr + 0.05 * torch.randn_like(hr)).clone().requires_grad_()
    return lr, hr, sr


def test_the_default_composite_is_plain_l1():
    lr, hr, sr = make_batch()
    out = CompositeLoss(LossConfig())(sr, hr)
    assert float(out.total) == pytest.approx(float(l1_loss(sr, hr)), rel=1e-6)
    assert set(out.components) == {"reconstruction", "total"}


def test_the_composite_is_the_weighted_sum_of_its_parts():
    lr, hr, sr = make_batch()
    cfg = LossConfig(reconstruction="charbonnier", charbonnier_eps=1e-3, spectral_weight=0.5, consistency_weight=0.25)
    out = CompositeLoss(cfg)(sr, hr, lr=lr, scale=4)
    expected = charbonnier_loss(sr, hr, eps=1e-3) + 0.5 * spectral_angle_loss(sr, hr) + 0.25 * consistency_loss(sr, lr, 4)
    assert float(out.total) == pytest.approx(float(expected), rel=1e-5)
    assert set(out.components) == {"reconstruction", "spectral", "consistency", "total"}
    assert out.components["total"] == pytest.approx(float(out.total), rel=1e-6)
    assert all(isinstance(v, float) for v in out.components.values())


def test_the_composite_backpropagates_through_every_enabled_term():
    lr, hr, sr = make_batch()
    out = CompositeLoss(LossConfig(spectral_weight=1.0, consistency_weight=1.0))(sr, hr, lr=lr, scale=4)
    out.total.backward()
    assert torch.isfinite(sr.grad).all() and float(sr.grad.abs().sum()) > 0.0


def test_a_disabled_term_needs_no_lr_and_an_enabled_consistency_term_does():
    lr, hr, sr = make_batch()
    CompositeLoss(LossConfig(spectral_weight=0.3))(sr, hr)                      # no lr required
    with pytest.raises(LossInputError, match="lr"):
        CompositeLoss(LossConfig(consistency_weight=0.3))(sr, hr)


def test_the_composite_passes_masks_to_the_right_terms():
    lr, hr, sr = make_batch()
    hr_mask = torch.ones(B, 16, 16, dtype=torch.bool)
    hr_mask[:, :8, :] = False
    lr_mask = torch.ones(B, 4, 4, dtype=torch.bool)
    sr2 = sr.detach().clone()
    sr2[:, :, :8, :] += 4.0
    full = CompositeLoss(LossConfig())(sr2, hr).total
    masked = CompositeLoss(LossConfig())(sr2, hr, hr_mask=hr_mask).total
    assert float(masked) < float(full)
    CompositeLoss(LossConfig(consistency_weight=1.0))(sr, hr, lr=lr, hr_mask=hr_mask, lr_mask=lr_mask, scale=4)


def test_an_all_zero_loss_configuration_is_still_a_valid_reconstruction_loss():
    lr, hr, sr = make_batch()
    out = CompositeLoss(LossConfig(spectral_weight=0.0, consistency_weight=0.0))(hr.clone(), hr)
    assert float(out.total) == 0.0


def test_the_reconstruction_choice_is_honoured():
    lr, hr, sr = make_batch()
    a = CompositeLoss(LossConfig(reconstruction="l1"))(sr, hr).total
    b = CompositeLoss(LossConfig(reconstruction="charbonnier", charbonnier_eps=0.1))(sr, hr).total
    assert float(a) != float(b)
