"""Real execution of one domain-shift protocol (Step H, Requirements 142 Requirement 12).

Composes the same real, already-tested primitives every earlier phase uses -- nothing here is
reimplemented:

    frame.acquisition.search / retrieve / harmonise / select_rgbn_adapter   (Step F/G, real CDSE)
      -> frame.api.services.model.get_model                                (the real registered model)
      -> frame.tiling.TiledModel + frame.uncertainty.run_stochastic_uncertainty  (real SR + real TTA stability)
      -> frame.consistency.run_consistency_diagnostics                     (real self-consistency, NDVI, NDWI)

No HR-reference metric is computed here: Requirement 9's Indian HR reference does not exist
(docs/TRACEABILITY.md Sec 9), and this module never substitutes one -- see
`frame.generalization.protocol.indian_holdout_protocol`'s `hr_reference_note`.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np
import torch

from frame.acquisition.contract import AcquisitionQuery
from frame.acquisition.harmonise import harmonise, select_rgbn_adapter
from frame.acquisition.providers.cdse import CDSEAuth
from frame.acquisition.retrieve import retrieve_scene
from frame.acquisition.search import search_scenes
from frame.api.services.model import get_model, resolve_device
from frame.consistency import run_consistency_diagnostics
from frame.data.bands import L2A_BANDS
from frame.generalization.errors import NoIndianSceneFoundError
from frame.generalization.protocol import DomainShiftProtocol, indian_holdout_protocol
from frame.geospatial import RGBN_SCALE_FACTOR
from frame.models.selection import normalize_model_name
from frame.tiling import TiledModel, TilingConfig
from frame.uncertainty import DEFAULT_TRANSFORMS, run_stochastic_uncertainty


def _index_summary(comparison) -> Dict[str, Any]:
    return {
        "status": comparison.status.value,
        "n_valid": comparison.valid_pixel_count,
        "mean_abs_discrepancy": comparison.mean_abs_discrepancy,
        "rmse": comparison.rmse,
        "max_abs_discrepancy": comparison.max_abs_discrepancy,
    }


def _band_summary(disc) -> Dict[str, Any]:
    return {
        "status": disc.status.value, "n_samples": disc.sample_count, "mae": disc.mean_abs_error, "rmse": disc.rmse,
        "bias": disc.bias, "correlation": disc.correlation, "normalized_rmse": disc.normalized_rmse,
    }


@dataclass(frozen=True)
class DomainShiftRecord:
    """One executed run of a `DomainShiftProtocol`. Every numeric field here came from a real
    model call and real diagnostics on real retrieved pixels -- there is no synthetic or
    fabricated fallback path in this module."""

    protocol: DomainShiftProtocol
    scene_id: str
    provider: str
    cloud_cover: Optional[float]
    input_shape: tuple
    output_shape: tuple
    excluded_bands: list
    self_consistency_overall: Dict[str, Any]
    ndvi_self_consistency: Dict[str, Any]
    ndwi_self_consistency: Dict[str, Any]
    stability_scalar_summary: float
    stability_definition: str
    valid_pixel_fraction: float
    device: str
    seed: int
    inference_seconds: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "protocol": self.protocol.to_dict(),
            "scene_id": self.scene_id,
            "provider": self.provider,
            "cloud_cover": self.cloud_cover,
            "input_shape": list(self.input_shape),
            "output_shape": list(self.output_shape),
            "excluded_bands": list(self.excluded_bands),
            "self_consistency_overall": self.self_consistency_overall,
            "ndvi_self_consistency": self.ndvi_self_consistency,
            "ndwi_self_consistency": self.ndwi_self_consistency,
            "stability_scalar_summary": self.stability_scalar_summary,
            "stability_definition": self.stability_definition,
            "valid_pixel_fraction": self.valid_pixel_fraction,
            "device": self.device,
            "seed": self.seed,
            "inference_seconds": self.inference_seconds,
        }


def run_domain_shift_experiment(
    query: AcquisitionQuery,
    *,
    model_name: str,
    device: str = "auto",
    seed: int = 0,
    cdse_auth: Optional[CDSEAuth] = None,
) -> DomainShiftRecord:
    """Execute one real Indian domain-shift protocol run: real discovery, real retrieval, real
    12-band harmonisation, the real RGBN adapter, real inference through a registered model, and
    real (no-reference) fidelity/stability diagnostics.

    Raises `NoIndianSceneFoundError` if the query matches no real scene -- never falls back to a
    different AOI or a synthetic one. Raises `UnknownModelError`/`ModelUnavailableError`
    (frame.models.errors) exactly as `/sr/run` would for the same model id.
    """
    model_id = normalize_model_name(model_name)
    resolved_device = resolve_device(device)

    candidates = search_scenes(query)
    if not candidates:
        raise NoIndianSceneFoundError(f"No Sentinel-2 scene matched bbox={query.bbox}, {query.start_date}..{query.end_date}.")
    candidate = candidates[0]

    retrieved = retrieve_scene(candidate, query, bands=L2A_BANDS, resolution_m=10.0, include_scl=True, cdse_auth=cdse_auth or CDSEAuth())
    product = harmonise(retrieved)
    rgbn_array, band_names, excluded_bands = select_rgbn_adapter(product)

    mask = product.validity_mask.array if product.validity_mask is not None else np.ones((product.height, product.width), dtype=bool)

    model = get_model(resolved_device, model_name=model_id)
    x = torch.from_numpy(rgbn_array.astype("float32"))

    t0 = time.time()
    tiled_model = TiledModel(model, TilingConfig())
    uncertainty_result = run_stochastic_uncertainty(
        tiled_model, x, transforms=DEFAULT_TRANSFORMS, seed=seed, band_names=band_names, keep_per_member_predictions=False
    )
    inference_seconds = time.time() - t0

    consistency = run_consistency_diagnostics(
        x, uncertainty_result.mean_prediction, mask, band_names=band_names, scale_factor=RGBN_SCALE_FACTOR
    )

    protocol = indian_holdout_protocol(
        model_id=model_id, scene_id=candidate.id, bbox=query.bbox, acquisition_datetime=product.provenance.matched_acquisition_datetime
    )
    providers = set(retrieved.provenance.band_sources.values())

    return DomainShiftRecord(
        protocol=protocol,
        scene_id=candidate.id,
        provider=", ".join(sorted(providers)),
        cloud_cover=candidate.cloud_cover,
        input_shape=tuple(x.shape),
        output_shape=tuple(uncertainty_result.mean_prediction.shape),
        excluded_bands=list(excluded_bands),
        self_consistency_overall=_band_summary(consistency.downsample_consistency.overall),
        ndvi_self_consistency=_index_summary(consistency.ndvi_comparison),
        ndwi_self_consistency=_index_summary(consistency.ndwi_comparison),
        stability_scalar_summary=uncertainty_result.scalar_summary,
        stability_definition=uncertainty_result.scalar_summary_definition,
        valid_pixel_fraction=float(mask.mean()),
        device=resolved_device,
        seed=seed,
        inference_seconds=inference_seconds,
    )
