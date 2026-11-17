"""The reproducible domain-shift experiment protocol (Step H, Requirements 142 Requirement 12).

Requirement 12 asks for a defined protocol -- training domain, test domain, geographic
separation, season/date, land-cover type, model, metrics, stability/evidence -- not just "an
Indian AOI" (the requirement's own words: "An Indian AOI alone is not evidence of Indian
generalisation."). `DomainShiftProtocol` is that definition, filled in with the real, verified
facts of one run; `frame.generalization.runner` executes it.

Honesty constraint this module exists to enforce: FRAME's own training work (Phase 4,
docs/TRAINING.md) never touched the published SEN2SR-Lite/Mamba weights that `/sr/run` actually
serves -- Phase 4 trained only a small synthetic-data smoke model to test the training loop. The
deployed weights are third-party (`tacofoundation/sen2sr`), and their original training data and
geography are not documented in the artifact manifest this project inspected (docs/REGISTRY.md
Sec 1). So "training domain: SEN2NAIPv2" would be a fabricated claim about the model actually
running -- `training_domain_known=False` records that honestly instead.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

from frame.data.contract import HRStatus

#: What frame.consistency/frame.uncertainty can measure on a scene with NO HR reference: NOT
#: reference accuracy (PSNR/SSIM/RMSE/SAM/ERGAS -- Requirement 8's metrics all need one).
NO_REFERENCE_METRICS: Tuple[str, ...] = ("self_consistency", "ndvi_self_consistency", "ndwi_self_consistency", "tta_stability")

TRAINING_DOMAIN_NOTE = (
    "The deployed SEN2SR-Lite/Mamba weights are third-party (tacofoundation/sen2sr); their original "
    "training data and geography are not documented in the artifact manifest FRAME inspected "
    "(docs/REGISTRY.md Sec 1) and are not established here. FRAME's own training work (docs/TRAINING.md) "
    "never touched these weights -- it trained a separate, small synthetic-data smoke model only. "
    "Naming a training domain for the model actually serving /sr/run would be a fabricated claim."
)


@dataclass(frozen=True)
class DomainShiftProtocol:
    """One experiment's protocol definition -- filled in with real facts, not a template with
    blanks. See the module docstring for why `training_domain_known` exists."""

    protocol_id: str
    training_domain: str
    training_domain_known: bool
    test_domain: str
    geographic_separation: bool
    geographic_separation_note: str
    temporal_note: str
    land_cover_note: str
    model_id: str
    metrics: Tuple[str, ...]
    hr_reference_status: HRStatus
    hr_reference_note: str
    stability_method: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "protocol_id": self.protocol_id,
            "training_domain": self.training_domain,
            "training_domain_known": self.training_domain_known,
            "test_domain": self.test_domain,
            "geographic_separation": self.geographic_separation,
            "geographic_separation_note": self.geographic_separation_note,
            "temporal_note": self.temporal_note,
            "land_cover_note": self.land_cover_note,
            "model_id": self.model_id,
            "metrics": list(self.metrics),
            "hr_reference_status": self.hr_reference_status.value,
            "hr_reference_note": self.hr_reference_note,
            "stability_method": self.stability_method,
        }


def indian_holdout_protocol(
    *,
    model_id: str,
    scene_id: str,
    bbox: Tuple[float, float, float, float],
    acquisition_datetime: Optional[str],
    land_cover_note: str = "not established -- no land-cover label exists for this scene (see frame.data.adapters.india_holdout)",
) -> DomainShiftProtocol:
    """The protocol for one real Indian held-out Sentinel-2 scene run through a registered RGBN
    model, with no Indian HR reference (Requirements 142 Requirement 9: none has been obtained --
    see docs/TRACEABILITY.md Sec 9). Every field is either a fact about this specific run
    (scene id, bbox, timestamp) or an explicit, sourced limitation -- never a placeholder.
    """
    return DomainShiftProtocol(
        protocol_id=f"india_holdout/{scene_id}",
        training_domain=TRAINING_DOMAIN_NOTE,
        training_domain_known=False,
        test_domain=f"Real Sentinel-2 L2A over India, scene {scene_id}, bbox {bbox}",
        geographic_separation=True,
        geographic_separation_note=(
            "The test scene is Indian; the reference datasets this project has real evidence from "
            "(SEN2NEON: North America; OpenSR-Test: Spain/SPOT) are geographically disjoint from it. "
            "This establishes geographic separation of the EVIDENCE datasets, not of the deployed "
            "model's unknown training set (see training_domain_known)."
        ),
        temporal_note=f"acquisition_datetime={acquisition_datetime!r}" if acquisition_datetime else "acquisition_datetime unavailable",
        land_cover_note=land_cover_note,
        model_id=model_id,
        metrics=NO_REFERENCE_METRICS,
        hr_reference_status=HRStatus.UNAVAILABLE,
        hr_reference_note=(
            "No Indian HR reference (Cartosat, Resourcesat, AVIRIS-NG or otherwise) has been obtained or "
            "validated by this project (docs/TRACEABILITY.md Sec 9). Reference-based accuracy "
            "(PSNR/SSIM/RMSE/SAM/ERGAS) is therefore not computed for this scene -- not omitted by "
            "oversight, structurally unavailable. Substituting a non-Indian HR reference here would "
            "misrepresent it as Indian-reference evidence, which Requirements 142 explicitly forbids."
        ),
        stability_method="TTA stability -- reconstruction-variation diagnostic (uncalibrated; see docs/RELIABILITY.md)",
    )
