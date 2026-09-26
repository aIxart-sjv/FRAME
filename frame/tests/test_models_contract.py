"""frame.models.contract -- input/output validation for the RGBN 4x models."""

from __future__ import annotations

import pytest
import torch

from frame.models import config
from frame.models.contract import expected_output_shape, validate_input, validate_output
from frame.models.errors import ModelContractError


def _valid(batch: bool = False) -> torch.Tensor:
    x = torch.rand(4, 128, 128, dtype=torch.float32) * 0.5
    return x[None] if batch else x


# ------------------------------------------------------------------ input


@pytest.mark.parametrize("batch", [False, True])
def test_valid_input_is_accepted(batch):
    validate_input(_valid(batch))


def test_batched_input_with_several_tiles_is_accepted():
    validate_input(torch.rand(3, 4, 128, 128) * 0.5)


@pytest.mark.parametrize("channels", [1, 3, 5, 10])
def test_wrong_channel_count_is_rejected(channels):
    with pytest.raises(ModelContractError, match="channels"):
        validate_input(torch.rand(channels, 128, 128))


@pytest.mark.parametrize("size", [(64, 64), (256, 256), (128, 127), (127, 128)])
def test_wrong_spatial_size_is_rejected(size):
    with pytest.raises(ModelContractError, match="128x128"):
        validate_input(torch.rand(4, *size))


@pytest.mark.parametrize("shape", [(128, 128), (1, 1, 4, 128, 128)])
def test_wrong_rank_is_rejected(shape):
    with pytest.raises(ModelContractError, match="shape"):
        validate_input(torch.rand(*shape))


@pytest.mark.parametrize("dtype", [torch.float64, torch.float16, torch.int32, torch.uint8])
def test_non_float32_dtype_is_rejected(dtype):
    x = (torch.rand(4, 128, 128) * 100).to(dtype)
    with pytest.raises(ModelContractError, match="float32"):
        validate_input(x)


def test_non_tensor_is_rejected():
    with pytest.raises(ModelContractError, match="torch.Tensor"):
        validate_input([[0.1]])  # type: ignore[arg-type]


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_non_finite_input_is_rejected(bad):
    x = _valid()
    x[2, 5, 7] = bad
    with pytest.raises(ModelContractError, match="NaN or Inf"):
        validate_input(x)


def test_raw_digital_numbers_are_rejected_with_a_helpful_message():
    """The realistic failure: a scene left in DN (thousands) instead of reflectance."""
    x = torch.rand(4, 128, 128) * 3000.0
    with pytest.raises(ModelContractError, match="divided by 10000"):
        validate_input(x)


def test_reflectance_slightly_above_one_is_accepted():
    """Bright clouds/snow legitimately exceed 1.0 in L2A; only impossible values are rejected."""
    x = _valid()
    x[0, 0, 0] = 1.8
    validate_input(x)


def test_range_bounds_are_inclusive_and_the_l2a_offset_minimum_is_allowed():
    x = _valid()
    x[0, 0, 0] = config.MIN_REFLECTANCE
    x[1, 0, 0] = config.MAX_REFLECTANCE
    validate_input(x)


def test_value_just_below_the_minimum_is_rejected():
    x = _valid()
    x[0, 0, 0] = config.MIN_REFLECTANCE - 0.01
    with pytest.raises(ModelContractError, match="reflectance range"):
        validate_input(x)


# ----------------------------------------------------------------- output


@pytest.mark.parametrize("lead", [(4,), (1, 4), (3, 4)])
def test_expected_output_shape_is_4x_spatially(lead):
    assert expected_output_shape(lead + (128, 128)) == lead + (512, 512)


def test_valid_output_is_accepted():
    validate_output(torch.rand(1, 4, 512, 512), input_shape=(1, 4, 128, 128))
    validate_output(torch.rand(4, 512, 512), input_shape=(4, 128, 128))


def test_output_with_wrong_scale_is_rejected():
    with pytest.raises(ModelContractError, match="expected"):
        validate_output(torch.rand(1, 4, 256, 256), input_shape=(1, 4, 128, 128))


def test_output_with_wrong_channels_is_rejected():
    with pytest.raises(ModelContractError, match="expected"):
        validate_output(torch.rand(1, 3, 512, 512), input_shape=(1, 4, 128, 128))


def test_output_with_wrong_dtype_is_rejected():
    with pytest.raises(ModelContractError, match="float32"):
        validate_output(torch.rand(1, 4, 512, 512).double(), input_shape=(1, 4, 128, 128))


@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
def test_non_finite_output_is_rejected(bad):
    y = torch.rand(1, 4, 512, 512)
    y[0, 0, 0, 0] = bad
    with pytest.raises(ModelContractError, match="NaN or Inf"):
        validate_output(y, input_shape=(1, 4, 128, 128))


def test_errors_are_value_errors_so_pydantic_and_callers_can_catch_them_generically():
    assert issubclass(ModelContractError, ValueError)
