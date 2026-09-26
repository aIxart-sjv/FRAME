"""frame.data.bands -- canonical band names and explicit selection."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from frame.data.bands import L2A_BANDS, RGBN_BANDS, canonical_band, canonical_bands, select_bands
from frame.data.errors import ContractError


@pytest.mark.parametrize("raw,expected", [("B1", "B01"), ("b01", "B01"), ("B01", "B01"), ("B12", "B12"), ("B8A", "B8A"),
                                          ("b8a", "B8A"), (" B4 ", "B04"), ("B9", "B09"), ("B10", "B10")])
def test_band_names_are_canonicalised(raw, expected):
    assert canonical_band(raw) == expected


@pytest.mark.parametrize("bad", ["", "B", "B13", "B0", "B2A", "red", "4", "B123", "BB4"])
def test_invalid_band_names_are_rejected(bad):
    with pytest.raises(ContractError):
        canonical_band(bad)


def test_duplicate_bands_are_rejected():
    with pytest.raises(ContractError, match="Duplicate"):
        canonical_bands(["B4", "B04"])


def test_sen2neon_style_names_map_onto_the_l2a_order():
    """SEN2NEON says B1..B9 (no zero padding); its order equals FRAME's L2A order once canonicalised."""
    neon = ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B9", "B11", "B12"]
    assert canonical_bands(neon) == L2A_BANDS


def test_l2a_order_never_drifts_from_the_opensr_test_table():
    from frame.validation import L2A_BAND_ORDER

    assert L2A_BANDS == tuple(L2A_BAND_ORDER)


def test_rgbn_is_the_models_order_not_the_ascending_file_order():
    assert RGBN_BANDS == ("B04", "B03", "B02", "B08")
    from frame.models.config import RGBN_BAND_ORDER

    assert RGBN_BANDS == RGBN_BAND_ORDER


def test_select_bands_reorders_by_name_explicitly():
    ascending = ["B02", "B03", "B04", "B08"]  # e.g. SEN2VENuS file order
    array = torch.stack([torch.full((2, 2), float(i)) for i in range(4)])  # channel i holds the value i
    rgbn = select_bands(array, ascending, RGBN_BANDS)
    assert [float(c[0, 0]) for c in rgbn] == [2.0, 1.0, 0.0, 3.0]  # B04, B03, B02, B08


def test_select_bands_can_pick_a_subset_and_works_on_numpy():
    array = np.arange(12)[:, None, None] * np.ones((12, 1, 1))
    sub = select_bands(array, L2A_BANDS, ["B8A", "B02"])
    assert sub.shape == (2, 1, 1) and sub[:, 0, 0].tolist() == [8.0, 1.0]


def test_a_band_the_sample_does_not_have_is_an_error_not_a_guess():
    with pytest.raises(ContractError, match="not available"):
        select_bands(torch.zeros(4, 2, 2), RGBN_BANDS, ["B04", "B11"])


def test_channel_count_must_match_the_band_list():
    with pytest.raises(ContractError, match="channel"):
        select_bands(torch.zeros(3, 2, 2), RGBN_BANDS, ["B04"])
