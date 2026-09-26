"""frame.reliability.evidence -- one tile's stability, error targets, baselines and within-tile associations, on controlled synthetic scenes."""

from __future__ import annotations

import numpy as np
import pytest
from scipy import ndimage

from frame.evaluate import metrics as M
from frame.reliability.config import ReliabilityConfig
from frame.reliability.eligibility import EXCLUDED
from frame.reliability.evidence import EnsembleArrays, TileInputs, compute_tile_evidence, load_evidence, save_evidence

BANDS = ("B04", "B03", "B02", "B08")
SIZE = 256


def cfg(**analysis):
    d = {"name": "ev", "output_dir": "o", "systems": [{"name": "lite", "kind": "lite"}], "datasets": [{"name": "d", "kind": "sen2neon", "manifest": "m"}],
         "analysis": {"pooled_cells_per_tile": 500, "pooled_pixels_per_tile": 1000, **analysis}}
    return ReliabilityConfig.from_dict(d)


def smooth(seed, size=SIZE, sigma=8.0):
    rng = np.random.default_rng(seed)
    f = ndimage.gaussian_filter(rng.standard_normal((size, size)), sigma, mode="wrap")
    return (f - f.min()) / (f.max() - f.min())


def scenario(informative=True, seed=0, size=SIZE):
    """A reference, and a prediction whose error magnitude follows a smooth 'difficulty' field s(x); the ensemble spread follows s(x) (informative) or an unrelated field."""
    rng = np.random.default_rng(seed)
    hr = (0.25 + 0.05 * rng.standard_normal((4, size, size))).clip(0.02, 0.9)
    difficulty = 0.2 + smooth(seed + 1, size)
    pred = hr + 0.02 * difficulty[None] * rng.standard_normal(hr.shape)
    stab_field = difficulty if informative else 0.2 + smooth(seed + 99, size)
    std = np.broadcast_to(0.001 * stab_field[None], hr.shape).copy()
    bicubic = ndimage.gaussian_filter(hr, (0, 2, 2))
    return hr, pred, std, bicubic


GATE = {"status": "eligible", "reason": None, "detail": "ok", "evidence_level": "pixel_level_eligible", "valid_fraction": 1.0, "nonfinite_fraction": 0.0, "valid_fraction_after_alignment": 1.0,
        "alignment": {"status": "eligible", "reason": None, "correction": [0, 0], "applied": False, "correction_method": "none_required", "method": "bicubic_baseline_cross_correlation",
                      "residual_magnitude": 0.1, "quadrant_spread": 0.2, "crop_rows": [0, SIZE], "crop_cols": [0, SIZE], "raw_magnitude": 0.1}}


def inputs(hr, pred, std, bicubic, mask=None, gate=None, n=6, member0=None, sample_id="s1", **kw):
    ens = EnsembleArrays(mean=pred, std=std, member0=pred if member0 is None else member0, n_members=n, transform_names=("identity", "hflip", "vflip", "rot90", "rot180", "rot270")[:n],
                         seed=42, seconds_total=1.5, seconds_per_member=tuple([0.25] * n))
    return TileInputs(dataset="d", system="lite", sample_id=sample_id, scene_unit="u1", category="Forest", bands=BANDS, scale=4, bicubic=bicubic, hr=hr,
                      mask=np.ones(hr.shape[-2:], bool) if mask is None else mask, gate=gate or GATE, ensemble=ens, **kw)


def run(inp, **analysis):
    return compute_tile_evidence(inp, cfg(**analysis))


# ============================================================================== association of a controlled scene


def test_an_informative_stability_correlates_with_error_more_strongly_as_the_scale_grows():
    hr, pred, std, bic = scenario(True)
    row, arrays = run(inputs(hr, pred, std, bic))
    assert row["status"] == "eligible" and row["evidence_level"] == "pixel_level_eligible"
    px = row["pixel"]["spearman_abs_error"]["value"]
    c4, c16 = row["cells"]["4"]["spearman_abs_error"]["value"], row["cells"]["16"]["spearman_abs_error"]["value"]
    assert 0.02 < px < c4 < c16 and c16 > 0.7                                                # error is noisy per pixel, and averages out into the smooth difficulty field over cells


def test_an_uninformative_stability_has_no_association_with_error():
    hr, pred, std, bic = scenario(False, seed=3)
    row, _ = run(inputs(hr, pred, std, bic))
    assert abs(row["pixel"]["spearman_abs_error"]["value"]) < 0.15 and abs(row["cells"]["4"]["spearman_abs_error"]["value"]) < 0.15      # smooth fields: few independent regions, so a few hundredths of chance correlation


def test_the_baseline_predictors_are_reported_next_to_the_stability_and_the_partial_shows_what_stability_adds():
    hr, pred, std, bic = scenario(True, seed=5)
    row, _ = run(inputs(hr, pred, std, bic))
    px = row["pixel"]
    assert px["spearman_texture_abs_error"]["status"] == "ok" and px["spearman_added_detail_abs_error"]["status"] == "ok"
    assert px["partial_spearman_abs_error_given_baselines"]["status"] == "ok" and px["partial_spearman_abs_error_given_baselines"]["n_controls"] == 2
    assert px["partial_spearman_abs_error_given_baselines"]["value"] > 0.02                    # the difficulty field is not texture: stability keeps its signal


def test_the_targets_are_computed_for_the_ensemble_mean_and_for_the_single_pass_separately():
    hr, pred, std, bic = scenario(True, seed=6)
    single = pred + 0.01                                                                     # the identity member is worse than the ensemble mean
    row, _ = run(inputs(hr, pred, std, bic, member0=single))
    assert row["targets"]["product"]["rmse"] < row["targets"]["single_pass"]["rmse"]
    ref = M.reference_accuracy(pred, hr, np.ones((SIZE, SIZE), bool), BANDS, 4)
    assert row["targets"]["product"]["rmse"] == pytest.approx(ref["rmse"], rel=1e-4) and "definition" in row["targets"]


def test_the_stability_summary_is_the_band_mean_spread_over_the_valid_pixels():
    hr, pred, std, bic = scenario(True, seed=7)
    mask = np.ones((SIZE, SIZE), bool)
    mask[:, :100] = False
    row, _ = run(inputs(hr, pred, std, bic, mask=mask))
    s = row["stability"]
    assert s["mean"] == pytest.approx(float(std[0][:, 100:].mean()), rel=1e-9) and s["p90"] >= s["median"] and s["definition"].startswith("Mean per-pixel standard deviation")
    assert row["pixel"]["n"] == SIZE * (SIZE - 100)


# ============================================================================== alignment is applied, and recorded


def test_the_recorded_alignment_crop_is_applied_to_every_grid_before_any_error_is_computed():
    hr, pred, std, bic = scenario(True, seed=8)
    displaced_ref = np.roll(hr, shift=(2, -1), axis=(1, 2))                                  # the reference is registered 2 rows down and 1 column left of the prediction
    unfixed, _ = run(inputs(displaced_ref, pred, std, bic))
    gate = {**GATE, "alignment": {**GATE["alignment"], "correction": [2, -1], "applied": True, "correction_method": "integer_translation_crop", "crop_rows": [2, SIZE], "crop_cols": [0, SIZE - 1]}}
    fixed, arrays = run(inputs(displaced_ref, pred, std, bic, gate=gate))
    assert fixed["targets"]["product"]["rmse"] < 0.5 * unfixed["targets"]["product"]["rmse"]
    assert fixed["pixel"]["n"] == (SIZE - 2) * (SIZE - 1) and fixed["alignment"]["correction"] == [2, -1] and fixed["alignment"]["applied"] is True


def test_the_pixel_level_association_decays_as_the_reference_is_displaced_which_is_why_only_registered_evidence_is_admitted():
    hr, pred, std, bic = scenario(True, seed=9)
    row, _ = run(inputs(hr, pred, std, bic), displacement_sweep_hr_px=[0, 1, 2, 4])
    sweep = row["pixel"]["by_reference_displacement"]
    assert set(sweep) == {"0", "1", "2", "4"}
    assert sweep["0"]["value"] > 0.2 and sweep["0"]["value"] > sweep["1"]["value"] + 0.1 and sweep["0"]["value"] - sweep["4"]["value"] > 0.15


def test_misregistration_alone_manufactures_an_association_with_a_texture_like_stability():
    """A prediction that IS the reference has no error. Displace the reference and error appears exactly where the image has edges, which is where a texture-driven stability is high:
    the two correlate although the model is perfect. This is the confound the eligibility gate exists to keep out."""
    rng = np.random.default_rng(9)
    hr = ndimage.gaussian_filter(rng.standard_normal((4, SIZE, SIZE)), (0, 2.0, 2.0)) * 0.08 + 0.25
    edges = np.hypot(*np.gradient(ndimage.gaussian_filter(hr.mean(0), 1.0)))
    std = np.broadcast_to((0.0005 + 0.002 * edges / edges.max())[None], hr.shape).copy()
    bic = ndimage.gaussian_filter(hr, (0, 2.5, 2.5))
    registered, _ = run(inputs(hr, hr.copy(), std, bic))
    assert registered["pixel"]["spearman_abs_error"]["status"] == "not_computable"           # perfect prediction, registered reference: no error at all
    displaced_reference = np.roll(hr, shift=(0, 2), axis=(1, 2))
    row, _ = run(inputs(displaced_reference, hr.copy(), std, bic))
    assert row["pixel"]["spearman_abs_error"]["value"] > 0.15 and row["cells"]["4"]["spearman_abs_error"]["value"] > 0.25 and row["targets"]["product"]["rmse"] > 0      # from nothing


# ============================================================================== exclusions after inference, and edge cases


def test_a_non_finite_prediction_on_a_valid_pixel_excludes_the_tile_with_no_numbers():
    hr, pred, std, bic = scenario(True, seed=10)
    pred = pred.copy()
    pred[1, 40, 40] = np.nan
    out = run(inputs(hr, pred, std, bic))
    assert isinstance(out, dict) and out["status"] == EXCLUDED and out["reason"] == "prediction_not_finite" and "targets" not in out and "pixel" not in out


def test_junk_on_masked_pixels_changes_nothing():
    hr, pred, std, bic = scenario(True, seed=11)
    mask = np.ones((SIZE, SIZE), bool)
    mask[:60] = False
    clean, _ = run(inputs(hr, pred, std, bic, mask=mask))
    hr2, pred2 = hr.copy(), pred.copy()
    hr2[:, :60] = 99.0
    pred2[:, :60] = np.nan
    dirty, _ = run(inputs(hr2, pred2, std, bic, mask=mask))
    assert dirty["targets"] == clean["targets"] and dirty["pixel"]["spearman_abs_error"] == clean["pixel"]["spearman_abs_error"]


def test_a_constant_stability_map_gives_undefined_correlations_with_a_reason_but_still_reports_the_tile():
    hr, pred, std, bic = scenario(True, seed=12)
    row, arrays = run(inputs(hr, pred, np.full_like(std, 0.002), bic))
    r = row["pixel"]["spearman_abs_error"]
    assert r["status"] == "not_computable" and r["value"] is None and "constant" in r["reason"]
    assert row["stability"]["mean"] == pytest.approx(0.002) and row["targets"]["product"]["rmse"] > 0 and row["status"] == "eligible"


def test_a_zero_error_prediction_is_undefined_not_perfect_correlation():
    hr, _, std, bic = scenario(True, seed=13)
    row, _ = run(inputs(hr, hr.copy(), std, bic))
    assert row["pixel"]["spearman_abs_error"]["status"] == "not_computable" and row["targets"]["product"]["rmse"] == pytest.approx(0.0)


# ============================================================================== pooled arrays, provenance, determinism, cache


def test_the_pooled_arrays_are_seeded_capped_and_aligned_across_variables():
    hr, pred, std, bic = scenario(True, seed=14)
    row, a1 = run(inputs(hr, pred, std, bic))
    _, a2 = run(inputs(hr, pred, std, bic))
    assert row["cells"]["4"]["n_cells"] == (SIZE // 4) ** 2 and len(a1["c4_stability"]) == 500 and len(a1["p_stability"]) == 1000
    assert all(np.array_equal(a1[k], a2[k]) for k in a1)                                       # the subsample is seeded by the sample, not by the process
    assert len({len(a1[k]) for k in a1 if k.startswith("c4_")}) == 1 and len({len(a1[k]) for k in a1 if k.startswith("p_")}) == 1
    _, other = run(inputs(hr, pred, std, bic, sample_id="another"))
    assert not np.array_equal(other["c4_stability"], a1["c4_stability"])


def test_the_row_carries_the_provenance_the_analysis_needs():
    hr, pred, std, bic = scenario(True, seed=15)
    row, _ = run(inputs(hr, pred, std, bic))
    assert row["tta"]["n_members"] == 6 and row["tta"]["transforms"][0] == "identity" and row["tta"]["seed"] == 42 and row["tta"]["single_pass_seconds"] == pytest.approx(0.25)
    assert row["tta"]["total_seconds"] == pytest.approx(1.5) and row["alignment"]["method"] == "bicubic_baseline_cross_correlation"
    assert row["metric_version"].startswith("frame-eval-metrics") and "strict" in row["valid_pixel_rule"].lower() or "valid" in row["valid_pixel_rule"].lower()
    assert row["dataset"] == "d" and row["system"] == "lite" and row["sample_id"] == "s1" and row["scene_unit"] == "u1" and row["category"] == "Forest"


def test_the_evidence_round_trips_through_the_cache(tmp_path):
    hr, pred, std, bic = scenario(True, seed=16)
    row, arrays = run(inputs(hr, pred, std, bic))
    save_evidence(tmp_path / "s1", row, arrays)
    row2, arrays2 = load_evidence(tmp_path / "s1")
    assert row2 == row and set(arrays2) == set(arrays) and all(np.array_equal(arrays[k], arrays2[k]) for k in arrays)
    with pytest.raises(FileNotFoundError):
        load_evidence(tmp_path / "missing")


def test_detail_fractions_relate_the_prediction_to_the_reference_per_tile():
    hr, pred, std, bic = scenario(True, seed=17)
    row, _ = run(inputs(hr, pred, std, bic))
    d = row["detail"]
    assert {"supported_synthesis", "unsupported_detail", "omission", "neutral"} <= set(d) and sum(d[k] for k in ("supported_synthesis", "unsupported_detail", "omission", "neutral")) == pytest.approx(1.0)
    ref = M.hallucination_analysis(bic, pred, hr, np.ones((SIZE, SIZE), bool), M.MetricConfig())["0.005"]
    assert d["unsupported_detail"] == pytest.approx(ref["unsupported_detail"], abs=1e-9) and d["omission"] == pytest.approx(ref["omission"], abs=1e-9)


# ============================================================================== is the raw spread an interval? (per tile; aggregated by the analysis)


def test_the_raw_spread_covers_almost_none_of_the_error_because_it_is_orders_of_magnitude_smaller():
    hr, pred, std, bic = scenario(True, seed=18)                                            # errors ~0.02, spread ~0.001
    row, _ = run(inputs(hr, pred, std, bic))
    c = row["calibration_raw"]
    assert c["coverage"]["k1"]["coverage"] < 0.15 and c["coverage"]["k2"]["coverage"] < 0.25 and c["coverage"]["k2"]["nominal"] == pytest.approx(0.9545, abs=1e-3)
    assert c["scale"]["median_error_over_median_spread"] > 8 and c["n_elements"] == 4 * SIZE * SIZE


def test_a_spread_as_large_as_the_error_covers_about_what_a_gaussian_would():
    hr, pred, std, bic = scenario(True, seed=19)
    rng = np.random.default_rng(0)
    wide = np.abs(pred - hr).mean() * 1.25 * np.ones_like(std)                              # a spread comparable to the error
    row, _ = run(inputs(hr, pred, wide, bic))
    assert row["calibration_raw"]["coverage"]["k2"]["coverage"] > 0.85 and rng is not None
