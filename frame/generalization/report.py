"""Machine-readable and human-readable output for one `DomainShiftRecord` (Step H Section 21).

Two files, matching the convention every other layer (frame.evaluate, frame.reliability,
frame.downstream) already uses: `record.json` (reusable) and `README.md` (what a person reads).
There is no aggregation here -- unlike those layers, a domain-shift run is currently one scene,
not a scene-unit population, so no confidence interval or ranking is computed or implied.
"""

from __future__ import annotations

import json
from pathlib import Path

from frame.generalization.runner import DomainShiftRecord


def _n(x, d: int = 4) -> str:
    return "n/a" if x is None else f"{x:.{d}f}"


def render_readme(record: DomainShiftRecord) -> str:
    p = record.protocol
    sc = record.self_consistency_overall
    out = [
        "# FRAME domain-shift record (Requirements 142 Requirement 12)",
        "",
        "This is evidence about ONE real Indian Sentinel-2 scene, not a validated claim of Indian "
        "generalisation. Read every number here against the limitations below before repeating it.",
        "",
        f"* **Scene**: `{record.scene_id}` (provider: {record.provider}, cloud cover {_n(record.cloud_cover, 1)}%)",
        f"* **Model**: `{p.model_id}`",
        f"* **Test domain**: {p.test_domain}",
        f"* **Training domain**: {p.training_domain}",
        f"* **Geographic separation**: {p.geographic_separation} -- {p.geographic_separation_note}",
        f"* **Temporal**: {p.temporal_note}",
        f"* **Land cover**: {p.land_cover_note}",
        f"* **HR reference status**: {p.hr_reference_status.value} -- {p.hr_reference_note}",
        "",
        "## What was measured (no HR reference exists for this scene)",
        "",
        f"* Self-consistency (SR downsampled to 10 m vs. the original Sentinel-2 observation): "
        f"MAE {_n(sc['mae'])}, RMSE {_n(sc['rmse'])}, bias {_n(sc['bias'])}, "
        f"correlation {_n(sc['correlation'])} (n={sc['n_samples']}).",
        f"* NDVI self-consistency: {record.ndvi_self_consistency['status']}, "
        f"mean |Δ| {_n(record.ndvi_self_consistency['mean_abs_discrepancy'])}.",
        f"* NDWI self-consistency: {record.ndwi_self_consistency['status']}, "
        f"mean |Δ| {_n(record.ndwi_self_consistency['mean_abs_discrepancy'])}.",
        f"* {p.stability_method}: scalar summary {_n(record.stability_scalar_summary, 6)} "
        f"({record.stability_definition}).",
        f"* Valid-pixel fraction: {_n(record.valid_pixel_fraction, 3)}.",
        f"* Bands excluded from the SR model (not RGBN, harmonised context only): {record.excluded_bands}.",
        "",
        "## What this record is NOT",
        "",
        "* NOT reference accuracy: no PSNR/SSIM/RMSE/SAM/ERGAS against an independent HR reference -- "
        "none exists for this scene (Requirements 142 Requirement 9; docs/TRACEABILITY.md Sec 9).",
        "* NOT a downstream-task result: no Indian land-cover, crop, urban, flood or burn-scar label exists here.",
        "* NOT evidence the deployed model was trained for, or excludes, Indian imagery: its training "
        "domain is not established (see `training_domain` above).",
        "* NOT a multi-scene generalisation claim: this is one scene. A defensible generalisation claim "
        "needs many geographically diverse Indian scenes, which this record does not attempt.",
        "",
        f"Inference: {record.device}, seed {record.seed}, {record.inference_seconds:.2f} s.",
    ]
    return "\n".join(out) + "\n"


def write_record(record: DomainShiftRecord, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "record.json").write_text(json.dumps(record.to_dict(), indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    (out_dir / "README.md").write_text(render_readme(record), encoding="utf-8")
