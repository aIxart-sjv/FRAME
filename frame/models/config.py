"""Single source of truth for FRAME's model configuration (Phase 1).

Everything model-specific that would otherwise be scattered across the
adapter, the worker, the client, the API and the tests lives here: the exact
SEN2SR-Mamba RGBN architecture, artifact file names/locations, the I/O
contract constants, and the isolated-runtime settings.

Dependency-free on purpose (stdlib only): imported by the main FRAME process
and by the isolated Mamba worker alike.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]

# ---------------------------------------------------------------------------
# Model ids (what a caller selects) and canonical names (what provenance records)
# ---------------------------------------------------------------------------

MODEL_LITE = "lite"
MODEL_MAMBA = "mamba"
SUPPORTED_MODELS: Tuple[str, ...] = (MODEL_LITE, MODEL_MAMBA)
DEFAULT_MODEL = MODEL_LITE

LITE_MODEL_NAME = "SEN2SRLite/NonReference_RGBN_x4"
MAMBA_MODEL_NAME = "SEN2SR/MambaSR_RGBN_x4"

# ---------------------------------------------------------------------------
# RGBN I/O contract (shared by Lite and Mamba)
# ---------------------------------------------------------------------------

#: Channel order of the RGBN tensor FRAME feeds the model: B04 (red), B03
#: (green), B02 (blue), B08 (NIR). This is NOT Sentinel-2's B02,B03,B04,B08
#: file order; it is the order the SEN2SR RGBN models consume. Verified
#: against the upstream cascade (sen2sr/referencex4.py slices the 10-band
#: tensor with ``bands_10m = [2, 1, 0, 6]`` into exactly this order) and
#: against the Lite artifact's own manifest (bands B04, B03, B02, B08).
#: frame/tests/test_models_config.py re-derives it from those sources.
RGBN_BAND_ORDER: Tuple[str, ...] = ("B04", "B03", "B02", "B08")

INPUT_CHANNELS = len(RGBN_BAND_ORDER)
INPUT_SIZE = 128            # the model's trained patch size (img_size)
SCALE_FACTOR = 4            # 10 m -> 2.5 m
OUTPUT_SIZE = INPUT_SIZE * SCALE_FACTOR

#: Accepted input value range: surface reflectance as a fraction. The bounds
#: come from the Sentinel-2 L2A encoding, not from tuning: the largest value
#: a uint16 DN can produce is 65535 / 10000, and the smallest a
#: BOA_ADD_OFFSET-corrected DN can produce is (0 - 1000) / 10000. Raw digital
#: numbers (thousands) fall far outside it, which is the point: a missed
#: ``/ 10000`` is rejected instead of silently producing a garbage SR image.
MIN_REFLECTANCE = -0.1
MAX_REFLECTANCE = 65535.0 / 10000.0

# ---------------------------------------------------------------------------
# SEN2SR-Mamba RGBN artifact
# ---------------------------------------------------------------------------

MAMBA_WEIGHTS_DIR = Path(os.environ.get("FRAME_MAMBA_WEIGHTS_DIR", str(REPO_ROOT / "models" / "SEN2SR")))
MAMBA_SR_WEIGHTS_FILENAME = "sr_model.safetensor"
MAMBA_HARD_CONSTRAINT_FILENAME = "sr_hard_constraint.safetensor"
MAMBA_EXAMPLE_DATA_FILENAME = "example_data.safetensor"

#: Facts about the shipped ``sr_model.safetensor``, recorded from the verified
#: load (see docs/MAMBA_INTEGRATION.md). Used by tests as regression guards.
MAMBA_EXPECTED_PARAMETER_COUNT = 13_759_444
MAMBA_EXPECTED_STATE_TENSORS = 1228


@dataclass(frozen=True)
class MambaArchitecture:
    """The exact ``MambaSR`` constructor arguments for the RGBN 10 m -> 2.5 m
    model. Identical to ``sr_parameters`` in ``models/SEN2SR/load.py``;
    frame/tests/test_models_config.py parses that file and fails if the two
    ever diverge."""

    img_size: Tuple[int, int] = (INPUT_SIZE, INPUT_SIZE)
    in_channels: int = INPUT_CHANNELS
    out_channels: int = INPUT_CHANNELS
    embed_dim: int = 96
    depths: Tuple[int, ...] = (8, 8, 8, 8, 8, 8)
    num_heads: Tuple[int, ...] = (8, 8, 8, 8, 8, 8)
    mlp_ratio: int = 4
    upscale: int = SCALE_FACTOR
    attention_type: str = "sigmoid_02"
    upsampler: str = "pixelshuffle"
    resi_connection: str = "1conv"
    operation_attention: str = "sum"

    def kwargs(self) -> Dict[str, Any]:
        """Constructor kwargs, in the exact types ``load.py`` passes (lists
        for ``depths``/``num_heads``, a tuple for ``img_size``)."""
        return {
            "img_size": tuple(self.img_size),
            "in_channels": self.in_channels,
            "out_channels": self.out_channels,
            "embed_dim": self.embed_dim,
            "depths": list(self.depths),
            "num_heads": list(self.num_heads),
            "mlp_ratio": self.mlp_ratio,
            "upscale": self.upscale,
            "attention_type": self.attention_type,
            "upsampler": self.upsampler,
            "resi_connection": self.resi_connection,
            "operation_attention": self.operation_attention,
        }


MAMBA_ARCHITECTURE = MambaArchitecture()

#: Label the artifact's own ``mlm.json`` carries. It describes the artifact's
#: PRIMARY (10-band cascade) component, a Swin2SR; the RGBN MambaSR used here
#: is listed there only as an auxiliary asset. See docs/MAMBA_INTEGRATION.md.
MAMBA_ARTIFACT_METADATA_LABEL = "Swin2SR"
MAMBA_EXECUTABLE_ARCHITECTURE = "MambaSR"

# ---------------------------------------------------------------------------
# Isolated runtime (worker subprocess)
# ---------------------------------------------------------------------------

#: Interpreter of the dedicated Mamba environment. The main FRAME environment
#: (torch 2.14) and this one (torch 2.6.0+cu118, the build the prebuilt
#: mamba-ssm CUDA extension was compiled against) are deliberately NOT merged.
MAMBA_PYTHON = Path(os.environ.get("FRAME_MAMBA_PYTHON", str(REPO_ROOT / "sen2sr_mamba_venv" / "bin" / "python")))
MAMBA_WORKER_MODULE = "frame.models.mamba_worker"

#: Generous because the first request in a fresh worker pays CUDA context +
#: kernel start-up; measured values are in docs/MAMBA_INTEGRATION.md.
MAMBA_STARTUP_TIMEOUT_S = float(os.environ.get("FRAME_MAMBA_STARTUP_TIMEOUT_S", "180"))
MAMBA_REQUEST_TIMEOUT_S = float(os.environ.get("FRAME_MAMBA_REQUEST_TIMEOUT_S", "120"))
