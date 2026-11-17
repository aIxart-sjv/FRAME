"""Model selection: the two ids FRAME accepts, ``lite`` and ``mamba`` (Phase 1).

Deliberately tiny -- two selectable entries, no registry framework. It answers only
"which ids exist, what are they called, and is this string one of them".
Actually loading a model stays with the caller (frame.api.services.model for
the API), so selection never imports either model's runtime.

Step G (Requirements 142 Requirement 4) adds real capability metadata to `ModelSpec` and a
second, deliberately separate dict, `CAPABILITY_ONLY_SPECS`, for a model that is real and
verified but not selectable (currently LDSR-S2) -- see that dict's docstring for why it is kept
out of `MODEL_SPECS`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

from frame.models import config
from frame.models.errors import UnknownModelError

#: Every currently-integrated learned model shares this exact I/O contract (Step G audit,
#: Requirements 142 Requirement 4): 4-band RGBN in, 4-band RGBN out, 4x scale, 10 m -> 2.5 m.
#: No model here accepts or produces a genuine 12-band 2.5 m output -- see docs/CLAIMS.md.
RGBN_INPUT_RESOLUTION_M = 10.0
RGBN_OUTPUT_RESOLUTION_M = 2.5

#: Honest description of the one uncertainty mechanism the two live models share -- never
#: "calibrated": Phase 6 (docs/RELIABILITY.md) measured it as weakly informative and
#: texture-confounded. Kept as one named constant so every ModelSpec that uses TTA says the
#: identical thing, rather than four slightly different hand-written strings drifting apart.
TTA_STABILITY_DESCRIPTION = "TTA stability -- reconstruction-variation diagnostic (uncalibrated; see docs/RELIABILITY.md)"


@dataclass(frozen=True)
class ModelSpec:
    id: str
    label: str          # what a user sees
    model_name: str     # canonical name recorded in provenance
    description: str
    # Real capability metadata (Step G, Requirements 142 Requirement 4) -- every field here is a
    # verified fact about the actual executable/checkpoint, not an aspiration. See
    # docs/REGISTRY.md for how each was verified and frame/tests/test_models_registry.py for the
    # checks that keep this from drifting from frame.models.config's own single source of truth.
    architecture: str = "unknown"
    input_bands: Tuple[str, ...] = ()
    output_bands: Tuple[str, ...] = ()
    input_resolution_m: Optional[float] = None
    output_resolution_m: Optional[float] = None
    scale_factor: Optional[int] = None
    uncertainty: bool = False
    uncertainty_method: Optional[str] = None
    parameter_count: Optional[int] = None
    environment: str = "main"  # "main" | "isolated:<worker-module>" -- which real environment actually runs it
    #: Set only for a verified-but-not-deployable model (Step G Section 7/19): a fixed, honest
    #: reason `GET /models`/`GET /health` report regardless of device, distinct from Mamba's
    #: per-request CUDA-availability check. None for every model actually reachable via `/sr/run`.
    always_unavailable_reason: Optional[str] = None


MODEL_SPECS: Dict[str, ModelSpec] = {
    config.MODEL_LITE: ModelSpec(
        id=config.MODEL_LITE,
        label="SEN2SR-Lite",
        model_name=config.LITE_MODEL_NAME,
        description="Lightweight CNN baseline (RGBN, 10 m to 2.5 m pixel grid).",
        architecture="CNNSR (fused-conv inference build; frame.train.models._LITE_ARCHITECTURE)",
        input_bands=config.RGBN_BAND_ORDER,
        output_bands=config.RGBN_BAND_ORDER,
        input_resolution_m=RGBN_INPUT_RESOLUTION_M,
        output_resolution_m=RGBN_OUTPUT_RESOLUTION_M,
        scale_factor=config.SCALE_FACTOR,
        uncertainty=True,
        uncertainty_method=TTA_STABILITY_DESCRIPTION,
        environment="main",
    ),
    config.MODEL_MAMBA: ModelSpec(
        id=config.MODEL_MAMBA,
        label="SEN2SR-Mamba",
        model_name=config.MAMBA_MODEL_NAME,
        description="Larger state-space model (RGBN, 10 m to 2.5 m pixel grid). Requires a CUDA GPU.",
        architecture=config.MAMBA_EXECUTABLE_ARCHITECTURE,
        input_bands=config.RGBN_BAND_ORDER,
        output_bands=config.RGBN_BAND_ORDER,
        input_resolution_m=RGBN_INPUT_RESOLUTION_M,
        output_resolution_m=RGBN_OUTPUT_RESOLUTION_M,
        scale_factor=config.SCALE_FACTOR,
        uncertainty=True,
        uncertainty_method=TTA_STABILITY_DESCRIPTION,
        parameter_count=13_759_444,
        environment="isolated:frame.models.mamba_worker",
    ),
}


#: A model that is real and was verified end-to-end (Step G Section 18: real checkpoint, real
#: Sentinel-2 input, real RGBN 4x output, real uncertainty map -- all on CPU, since it OOMs on
#: this project's declared dev GPU even at minimal input size) but is **not** wired into
#: `/sr/run` and can never be selected there. Deliberately kept out of `MODEL_SPECS` so
#: `normalize_model_name`/`get_spec` (the `/sr/run` selection boundary) cannot resolve it --
#: adding it there would let a request silently fall through to Lite's loader, which is exactly
#: the "no silent substitution" failure Requirements 142 Requirement 4 forbids. `GET /models` and
#: `GET /health` still report it, honestly marked unavailable, so its real, verified capability is
#: visible without pretending it is deployable here.
CAPABILITY_ONLY_SPECS: Dict[str, ModelSpec] = {
    "ldsrs2": ModelSpec(
        id="ldsrs2",
        label="LDSR-S2",
        model_name="opensr-model/opensr-ldsrs2_v1_0_0 (SRLatentDiffusion)",
        description="Latent-diffusion RGBN SR baseline (opensr-model / LDSR-S2). Verified with real weights and "
        "real Sentinel-2 data; not available through this API.",
        architecture="opensr_model.srmodel.SRLatentDiffusion (VAE encoder/decoder + DDIM diffusion UNet; "
        "ldsrs2_venv, Python 3.14, separate torch build)",
        input_bands=config.RGBN_BAND_ORDER,
        output_bands=config.RGBN_BAND_ORDER,
        input_resolution_m=RGBN_INPUT_RESOLUTION_M,
        output_resolution_m=RGBN_OUTPUT_RESOLUTION_M,
        scale_factor=config.SCALE_FACTOR,
        uncertainty=True,
        uncertainty_method="Stochastic diffusion resampling (repeated independent DDIM samples, per-pixel std "
        "across variations) -- a different mechanism from the TTA stability above; uncalibrated, no calibration "
        "evidence was assessed.",
        parameter_count=168_955_116,
        environment="isolated:ldsrs2_venv (verification script only -- no API worker exists)",
        always_unavailable_reason=(
            "Verified genuinely executable (real checkpoint load, real RGBN 4x forward pass and uncertainty map "
            "on real Sentinel-2 data), but not deployable on this project's declared dev GPU: CUDA inference "
            "raises torch.OutOfMemoryError on the RTX 3050 (4 GB) even at a 70x71 px input, because the model's "
            "internal 4x bilinear pre-upsample before VAE-encoding makes the encoder's self-attention step too "
            "large for the available ~3.4-3.7 GiB usable VRAM. CPU inference completes but took ~96 s for that "
            "same 70x71 px patch (20 s for a 20-step forward pass, 75 s for a 4-sample uncertainty map) -- roughly "
            "60-100x the per-tile latency of the registered RGBN models -- and no isolated worker was built for "
            "it. See docs/REGISTRY.md."
        ),
    ),
}


def normalize_model_name(value: object) -> str:
    """Canonical model id for ``value`` (case/whitespace-insensitive), or
    `UnknownModelError` naming the supported ids."""
    if isinstance(value, str) and value.strip().lower() in MODEL_SPECS:
        return value.strip().lower()
    raise UnknownModelError(f"Unknown model {value!r}; supported models are: {', '.join(config.SUPPORTED_MODELS)}.")


def get_spec(model_id: str) -> ModelSpec:
    return MODEL_SPECS[normalize_model_name(model_id)]
