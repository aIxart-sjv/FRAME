"""Unit tests for frame.acquisition.providers.planetary_computer -- in particular
the DN-to-reflectance conversion this provider must apply itself (`cubo`'s real
output was verified, in this project's acquisition audit, to be raw digital
numbers in the thousands, not [0, 1] reflectance -- unlike the CDSE provider,
which requests reflectance directly from the Process API)."""

from __future__ import annotations

import numpy as np
import pytest

from frame.acquisition.errors import ProviderUnavailableError
from frame.acquisition.providers.planetary_computer import retrieve_bands


class _FakeDataArray:
    """A minimal stand-in for the xarray.DataArray `cubo.create` returns."""

    def __init__(self, values: np.ndarray, attrs=None):
        self.values = values
        self.attrs = attrs or {}
        self.sizes = {"time": values.shape[0]}

    def copy(self, data):
        return _FakeDataArray(data, attrs=self.attrs)


def _fake_cubo_create_factory(dn_value: float, n_bands=1, height=4, width=4):
    def fake_create(**kwargs):
        values = np.full((1, n_bands, height, width), dn_value, dtype="float64")
        return _FakeDataArray(values)

    return fake_create


def test_continuous_bands_are_converted_from_raw_dn_to_reflectance():
    # A real-shaped Sentinel-2 L2A digital number (~2500) should become ~0.25 reflectance.
    fake_create = _fake_cubo_create_factory(dn_value=2500.0)
    result = retrieve_bands(
        lat=13.0, lon=80.2, bands=["B04"], resolution_m=10.0, edge_size_px=4,
        start_date="2024-01-01T00:00:00Z", end_date="2024-01-02T00:00:00Z",
        categorical=False, cubo_create=fake_create,
    )
    assert np.allclose(result.values, 0.25)


def test_categorical_bands_are_left_as_raw_class_codes():
    fake_create = _fake_cubo_create_factory(dn_value=4.0)  # SCL class 4 = vegetation
    result = retrieve_bands(
        lat=13.0, lon=80.2, bands=["SCL"], resolution_m=20.0, edge_size_px=4,
        start_date="2024-01-01T00:00:00Z", end_date="2024-01-02T00:00:00Z",
        categorical=True, cubo_create=fake_create,
    )
    assert np.allclose(result.values, 4.0)  # unscaled -- NOT divided by 10000


def test_zero_scenes_raises_provider_unavailable():
    def fake_create(**kwargs):
        return _FakeDataArray(np.zeros((0, 1, 4, 4)))

    with pytest.raises(ProviderUnavailableError):
        retrieve_bands(
            lat=13.0, lon=80.2, bands=["B04"], resolution_m=10.0, edge_size_px=4,
            start_date="2024-01-01T00:00:00Z", end_date="2024-01-02T00:00:00Z",
            cubo_create=fake_create,
        )


def test_a_cubo_exception_raises_provider_unavailable_not_a_crash():
    def failing_create(**kwargs):
        raise RuntimeError("STAC search returned no items")

    with pytest.raises(ProviderUnavailableError):
        retrieve_bands(
            lat=13.0, lon=80.2, bands=["B04"], resolution_m=10.0, edge_size_px=4,
            start_date="2024-01-01T00:00:00Z", end_date="2024-01-02T00:00:00Z",
            cubo_create=failing_create,
        )
