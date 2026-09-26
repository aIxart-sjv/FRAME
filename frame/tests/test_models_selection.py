"""frame.models.selection -- the two selectable model ids."""

from __future__ import annotations

import pytest

from frame.models import config
from frame.models.errors import UnknownModelError
from frame.models.selection import MODEL_SPECS, get_spec, normalize_model_name


def test_both_models_are_selectable():
    assert set(MODEL_SPECS) == {"lite", "mamba"}
    assert normalize_model_name("lite") == "lite"
    assert normalize_model_name("mamba") == "mamba"


@pytest.mark.parametrize("raw", ["LITE", " Mamba ", "Lite", "MAMBA"])
def test_ids_are_case_and_whitespace_insensitive(raw):
    assert normalize_model_name(raw) == raw.strip().lower()


@pytest.mark.parametrize("bad", ["swin", "", "sen2sr", "lite2", None, 3, ["lite"]])
def test_invalid_selection_fails_clearly_and_names_the_supported_ids(bad):
    with pytest.raises(UnknownModelError) as excinfo:
        normalize_model_name(bad)
    message = str(excinfo.value)
    assert "lite" in message and "mamba" in message


def test_unknown_model_error_is_a_value_error():
    assert issubclass(UnknownModelError, ValueError)


def test_specs_carry_user_labels_and_canonical_names():
    assert get_spec("lite").label == "SEN2SR-Lite"
    assert get_spec("mamba").label == "SEN2SR-Mamba"
    assert get_spec("lite").model_name == config.LITE_MODEL_NAME == "SEN2SRLite/NonReference_RGBN_x4"
    assert get_spec("mamba").model_name == config.MAMBA_MODEL_NAME


def test_default_is_the_lite_baseline():
    assert config.DEFAULT_MODEL == "lite"
    assert config.DEFAULT_MODEL in MODEL_SPECS
