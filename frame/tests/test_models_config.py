"""frame.models.config -- the exact Mamba architecture and RGBN band contract.

The architecture and band-order checks are cross-checked against the
artifact's own executable loader and the upstream cascade rather than only
against constants typed into this repo, so a divergence fails loudly.
"""

from __future__ import annotations

import ast
import dataclasses
import json
import re
from pathlib import Path

import pytest

from frame.models import config
from frame.preprocessing import RGBN_BANDS

REPO_ROOT = Path(__file__).resolve().parents[2]
LOAD_PY = REPO_ROOT / "models" / "SEN2SR" / "load.py"
MLM_JSON = REPO_ROOT / "models" / "SEN2SR" / "mlm.json"
REFERENCEX4 = REPO_ROOT / "sen2sr" / "referencex4.py"

EXPECTED_ARCHITECTURE = {
    "img_size": (128, 128),
    "in_channels": 4,
    "out_channels": 4,
    "embed_dim": 96,
    "depths": [8, 8, 8, 8, 8, 8],
    "num_heads": [8, 8, 8, 8, 8, 8],
    "mlp_ratio": 4,
    "upscale": 4,
    "attention_type": "sigmoid_02",
    "upsampler": "pixelshuffle",
    "resi_connection": "1conv",
    "operation_attention": "sum",
}


def test_architecture_kwargs_are_exactly_the_specified_parameters():
    assert config.MAMBA_ARCHITECTURE.kwargs() == EXPECTED_ARCHITECTURE


def test_architecture_kwargs_use_the_types_load_py_uses():
    kwargs = config.MAMBA_ARCHITECTURE.kwargs()
    assert isinstance(kwargs["depths"], list) and isinstance(kwargs["num_heads"], list)
    assert isinstance(kwargs["img_size"], tuple)


def test_architecture_is_immutable():
    with pytest.raises(dataclasses.FrozenInstanceError):
        config.MAMBA_ARCHITECTURE.embed_dim = 1  # type: ignore[misc]


def test_kwargs_returns_a_fresh_copy_each_time():
    first = config.MAMBA_ARCHITECTURE.kwargs()
    first["depths"].append(99)
    assert config.MAMBA_ARCHITECTURE.kwargs()["depths"] == [8] * 6


@pytest.mark.skipif(not LOAD_PY.exists(), reason="models/SEN2SR/load.py not present")
def test_architecture_matches_the_artifacts_own_loader():
    """Both `trainable_model` and `compiled_model` in load.py define
    `sr_parameters`; FRAME's config must equal each of them."""
    tree = ast.parse(LOAD_PY.read_text())
    found = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == "sr_parameters" for t in node.targets):
            found.append(ast.literal_eval(node.value))
    assert len(found) == 2, "expected sr_parameters in both trainable_model and compiled_model"
    for params in found:
        assert params == config.MAMBA_ARCHITECTURE.kwargs()


def test_band_order_matches_preprocessing():
    assert config.RGBN_BAND_ORDER == tuple(RGBN_BANDS) == ("B04", "B03", "B02", "B08")


@pytest.mark.skipif(not (MLM_JSON.exists() and REFERENCEX4.exists()), reason="artifact/upstream sources not present")
def test_band_order_is_what_the_upstream_cascade_feeds_the_rgbn_model():
    """Derive the order from source: the cascade slices its 10-band tensor
    (band order declared in the artifact's mlm.json) with
    ``bands_10m = [2, 1, 0, 6]`` before calling the RGBN model."""
    ten_band_order = json.loads(MLM_JSON.read_text())["properties"]["mlm:input"][0]["bands"]
    assert ten_band_order[:3] == ["B02", "B03", "B04"] and len(ten_band_order) == 10

    match = re.search(r"bands_10m\s*=\s*\[([^\]]*)\]", REFERENCEX4.read_text())
    assert match, "bands_10m not found in sen2sr/referencex4.py"
    indices = [int(i) for i in match.group(1).split(",")]

    assert tuple(ten_band_order[i] for i in indices) == config.RGBN_BAND_ORDER


def test_io_size_contract():
    assert (config.INPUT_CHANNELS, config.INPUT_SIZE, config.SCALE_FACTOR, config.OUTPUT_SIZE) == (4, 128, 4, 512)


def test_reflectance_bounds_follow_the_l2a_encoding():
    assert config.MIN_REFLECTANCE == pytest.approx((0 - 1000) / 10000)
    assert config.MAX_REFLECTANCE == pytest.approx(65535 / 10000)


def test_supported_models_and_default():
    assert config.SUPPORTED_MODELS == ("lite", "mamba")
    assert config.DEFAULT_MODEL == "lite"


def test_artifact_metadata_label_differs_from_executable_architecture():
    """The known inconsistency is recorded explicitly, not hidden."""
    assert config.MAMBA_EXECUTABLE_ARCHITECTURE == "MambaSR"
    assert config.MAMBA_ARTIFACT_METADATA_LABEL == "Swin2SR"


def test_config_module_is_stdlib_only():
    """Imported by both environments; must not pull in torch/numpy."""
    import subprocess
    import sys

    code = "import sys, frame.models.config; print('torch' in sys.modules or 'numpy' in sys.modules)"
    out = subprocess.run([sys.executable, "-c", code], cwd=REPO_ROOT, capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "False"
