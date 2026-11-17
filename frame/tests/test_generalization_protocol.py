"""Step H: frame.generalization.protocol / report -- the honesty guarantees around one
Indian domain-shift record (Requirements 142 Requirement 12): no fabricated training domain, no
fabricated HR reference, and the human-readable report says both plainly.
"""

from __future__ import annotations

from frame.data.contract import HRStatus
from frame.generalization.protocol import NO_REFERENCE_METRICS, indian_holdout_protocol
from frame.generalization.report import render_readme
from frame.generalization.runner import DomainShiftRecord


def _protocol():
    return indian_holdout_protocol(
        model_id="lite", scene_id="S2TEST", bbox=(80.0, 13.0, 80.1, 13.1), acquisition_datetime="2024-03-30T04:56:59Z"
    )


def test_training_domain_is_reported_unknown_not_fabricated():
    p = _protocol()
    assert p.training_domain_known is False
    assert "third-party" in p.training_domain
    assert "SEN2NAIPv2" not in p.training_domain  # must never claim the deployed weights trained on FRAME's own dataset


def test_hr_reference_is_unavailable_and_says_why():
    p = _protocol()
    assert p.hr_reference_status is HRStatus.UNAVAILABLE
    assert "no Indian HR reference" in p.hr_reference_note.lower() or "cartosat" in p.hr_reference_note.lower()


def test_metrics_never_include_reference_accuracy():
    """No HR reference exists for this protocol, so its declared metric list must not include
    anything that needs one (PSNR/SSIM/RMSE/SAM/ERGAS are frame.evaluate.metrics.reference_accuracy's keys)."""
    forbidden = {"psnr", "ssim", "sam", "ergas", "reference_accuracy"}
    for metric in NO_REFERENCE_METRICS:
        assert not any(f in metric.lower() for f in forbidden)


def test_geographic_separation_is_true_but_does_not_imply_training_domain_knowledge():
    p = _protocol()
    assert p.geographic_separation is True
    assert "unknown" in p.geographic_separation_note.lower() or "training_domain_known" in p.geographic_separation_note


def _record():
    p = _protocol()
    return DomainShiftRecord(
        protocol=p, scene_id="S2TEST", provider="copernicus-cdse", cloud_cover=5.0,
        input_shape=(4, 16, 16), output_shape=(4, 64, 64), excluded_bands=["B01", "B05", "B06", "B07", "B8A", "B09", "B11", "B12"],
        self_consistency_overall={"status": "COMPUTABLE", "n_samples": 1024, "mae": 0.001, "rmse": 0.002, "bias": 0.0001, "correlation": 0.99, "normalized_rmse": 0.01},
        ndvi_self_consistency={"status": "COMPUTABLE", "n_valid": 256, "mean_abs_discrepancy": 0.002, "rmse": 0.003, "max_abs_discrepancy": 0.01},
        ndwi_self_consistency={"status": "COMPUTABLE", "n_valid": 256, "mean_abs_discrepancy": 0.002, "rmse": 0.003, "max_abs_discrepancy": 0.01},
        stability_scalar_summary=0.0015, stability_definition="Mean per-pixel standard deviation across the TTA ensemble.",
        valid_pixel_fraction=1.0, device="cpu", seed=0, inference_seconds=0.5,
    )


def test_record_round_trips_through_to_dict():
    record = _record()
    d = record.to_dict()
    assert d["scene_id"] == "S2TEST"
    assert d["protocol"]["hr_reference_status"] == "unavailable"
    assert d["excluded_bands"] == record.excluded_bands


def test_readme_states_what_it_is_not():
    readme = render_readme(_record())
    assert "NOT reference accuracy" in readme
    assert "NOT a downstream-task result" in readme
    assert "NOT a multi-scene generalisation claim" in readme
    assert "S2TEST" in readme
