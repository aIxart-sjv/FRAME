"""Model selection: the two ids FRAME accepts, ``lite`` and ``mamba`` (Phase 1).

Deliberately tiny -- two entries, no registry framework. It answers only
"which ids exist, what are they called, and is this string one of them".
Actually loading a model stays with the caller (frame.api.services.model for
the API), so selection never imports either model's runtime.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

from frame.models import config
from frame.models.errors import UnknownModelError


@dataclass(frozen=True)
class ModelSpec:
    id: str
    label: str          # what a user sees
    model_name: str     # canonical name recorded in provenance
    description: str


MODEL_SPECS: Dict[str, ModelSpec] = {
    config.MODEL_LITE: ModelSpec(
        id=config.MODEL_LITE,
        label="SEN2SR-Lite",
        model_name=config.LITE_MODEL_NAME,
        description="Lightweight CNN baseline (RGBN, 10 m to 2.5 m pixel grid).",
    ),
    config.MODEL_MAMBA: ModelSpec(
        id=config.MODEL_MAMBA,
        label="SEN2SR-Mamba",
        model_name=config.MAMBA_MODEL_NAME,
        description="Larger state-space model (RGBN, 10 m to 2.5 m pixel grid). Requires a CUDA GPU.",
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
