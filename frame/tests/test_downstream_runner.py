"""frame.downstream.runner -- the whole flow on fake datasets and fake models: gate, NDVI, regions, metrics, association, provenance, exclusions. No GPU."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from scipy import ndimage

from frame.downstream.config import DownstreamConfig
from frame.downstream.runner import run_downstream
from frame.evaluate.errors import EvaluationError, RoleSafetyError
from frame.tests.test_evaluate_runner import FakeDataset, make_sample
from frame.tests.test_reliability_runner import Failing, NaNs, Skewed, system


def veg_sample(i, **kw):
    """A sample with vegetated (NIR-bright) patches, so NDVI crosses the 0.3 threshold inside it."""
    s = make_sample(i, **kw)
    rng = np.random.default_rng(100 + i)
    patch = ndimage.gaussian_filter(rng.standard_normal((128, 128)), 9)
    patch = (patch - patch.min()) / (patch.max() - patch.min())
    hr = s.hr.clone()
    hr[3] = hr[3] + 0.4 * torch.from_numpy(patch).float()
    s.hr, s.lr = hr, F.avg_pool2d(hr[None], 4)[0]
    return s


def config(tmp_path, names=("a", "b"), **over):
    d = {"name": "ds-unit", "output_dir": str(tmp_path / "out"), "device": "cpu", "systems": [{"name": n, "kind": "lite"} for n in names],
         "datasets": [{"name": "fake", "kind": "sen2neon", "manifest": "unused.jsonl"}], "bootstrap": {"n_boot": 100}, "analysis": {"pooled_regions_per_tile": 300}, **over}
    return DownstreamConfig.from_dict(d)


def run(tmp_path, *, datasets=None, systems=None, names=("a", "b"), **over):
    systems = systems or [system(n) for n in names]
    ds = datasets or [FakeDataset(samples=[veg_sample(i) for i in range(6)])]
    return run_downstream(config(tmp_path, names=[s.name for s in systems], **over), systems=systems, datasets=ds), tmp_path / "out"


def load(p):
    return json.loads(Path(p).read_text())


def tiles(out):
    return [json.loads(l) for l in (out / "tiles.jsonl").read_text().splitlines() if l.strip()]


@pytest.fixture(scope="module")
def standard(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("ds")
    summary, out = run(tmp, cache_dir=str(tmp / "cache"))
    return summary, out, tmp / "cache"


# ============================================================================== a whole run


def test_a_run_writes_the_experiment_layout(standard):
    summary, out, _ = standard
    assert sorted(p.name for p in out.iterdir()) == sorted(["README.md", "association.json", "config.json", "downstream_metrics.json", "risk_coverage.json", "summary.json", "tiles.jsonl"])
    assert summary["status"] == "completed" and len(tiles(out)) == 6 and {t["status"] for t in tiles(out)} == {"analysed"}


def test_the_systems_compared_are_the_reference_the_lr_baselines_and_every_model(standard):
    _, out, _ = standard
    m = load(out / "downstream_metrics.json")["datasets"]["fake"]
    systems = set(m["scales"]["4"]["0.3"]["systems"])
    assert systems == {"lr_native", "bicubic", "a", "b"}
    assert set(m["scales"]["4"]["0.3"]["paired"]) >= {"a - bicubic", "b - bicubic", "a - lr_native", "b - lr_native", "a - b"}


def test_results_are_reported_at_every_declared_scale_and_threshold(standard):
    _, out, _ = standard
    m = load(out / "downstream_metrics.json")["datasets"]["fake"]
    assert set(m["scales"]) == {"4", "16"} and set(m["scales"]["4"]) >= {"0.3", "0.2", "0.4"}
    a = load(out / "association.json")["datasets"]["fake"]["scales"]["4"]
    assert set(a) == {"a", "b"} and a["a"]["threshold"] == "0.3"                                  # associations are at the primary threshold only


# ============================================================================== only eligible evidence enters


def excluded_dataset():
    ok = [veg_sample(i) for i in range(5)]
    shifted = veg_sample(5)
    shifted.hr = torch.roll(shifted.hr, shifts=(14, 9), dims=(1, 2))                                # registered 14 rows / 9 columns away: beyond the correctable range
    sparse = veg_sample(6, valid_fraction=0.3)
    return FakeDataset(samples=ok + [shifted, sparse])


def test_an_ineligible_reference_cannot_enter_the_primary_analysis(tmp_path, monkeypatch):
    """The regression that matters: a tile whose reference is misregistered (or too sparse) must not have ANY quantity derived from that reference: no NDVI, no region, no error."""
    import frame.downstream.runner as runner

    ndvi_calls, region_calls = [], []
    real_ndvi, real_regions = runner.ndvi_map, runner.extract_regions
    monkeypatch.setattr(runner, "ndvi_map", lambda *a, **k: (ndvi_calls.append(1), real_ndvi(*a, **k))[1])
    monkeypatch.setattr(runner, "extract_regions", lambda *a, **k: (region_calls.append(1), real_regions(*a, **k))[1])
    summary, out = run(tmp_path, datasets=[excluded_dataset()], names=("a",), cache_dir=str(tmp_path / "cache"))
    rows = {t["sample_id"]: t for t in tiles(out)}
    assert rows["fake:s5"]["status"] == "excluded_from_downstream_primary_analysis" and rows["fake:s5"]["reason"] == "reference_alignment_invalid" and rows["fake:s5"]["evidence_level"] == "not_eligible"
    assert rows["fake:s6"]["reason"] == "insufficient_valid_pixels"
    assert len(region_calls) == 5 * 2 and len(ndvi_calls) == 5 * 4                                  # only the five eligible tiles: two scales each, and four NDVI sources (reference, lr, bicubic, 'a')
    cached = {p.name.split("__")[0] for p in (tmp_path / "cache" / "ds-unit" / "fake").glob("*.npz")}
    assert cached == {f"fake_s{i}" for i in range(5)}                                             # no region table exists for an excluded tile
    for t in (rows["fake:s5"], rows["fake:s6"]):
        assert "regions" not in t or all(v["analysed"] == 0 for v in t["regions"].values())
    assert summary["datasets"]["fake"]["units_eligible"] == [f"scene{i}" for i in range(5)]


def test_only_regions_of_eligible_tiles_appear_in_the_results(tmp_path):
    summary, out = run(tmp_path, datasets=[excluded_dataset()], names=("a",), cache_dir=str(tmp_path / "cache"))
    a = load(out / "association.json")["datasets"]["fake"]["scales"]["4"]["a"]
    assert a["n_tiles"] == 5 and a["n_units"] == 5


def test_the_region_counts_reconcile_candidates_analysed_and_every_exclusion(tmp_path):
    summary, out = run(tmp_path, datasets=[excluded_dataset()], names=("a",))
    r = summary["datasets"]["fake"]["regions"]["4"]
    assert r["candidate"] == 7 * (128 // 4) ** 2 and r["candidate_in_eligible_tiles"] == 5 * 32 * 32
    assert r["candidate"] == r["analysed"] + r["excluded_tile_level"]["regions"] + sum(r["excluded_region_level"].values())
    assert r["excluded_tile_level"]["by_reason"] == {"insufficient_valid_pixels": 1024, "reference_alignment_invalid": 1024}
    assert summary["datasets"]["fake"]["gate"]["by_evidence_level"] == {"not_eligible": 2, "pixel_level_eligible": 5}


def test_a_model_that_fails_on_a_tile_excludes_the_tile_for_every_system_so_comparisons_share_evidence(tmp_path):
    summary, out = run(tmp_path, systems=[system("a"), system("b", Failing(fail_on_call=1, persistent=True))], names=("a", "b"))
    rows = tiles(out)
    assert {t["status"] for t in rows} == {"excluded_from_downstream_primary_analysis"} and {t["reason"] for t in rows} == {"model_failure"} and "b" in rows[0]["detail"]
    assert summary["overview"]["fake"]["status"] == "no_eligible_evidence"


def test_a_non_finite_prediction_excludes_the_tile(tmp_path):
    _, out = run(tmp_path, systems=[system("a", NaNs())], names=("a",))
    assert {t["reason"] for t in tiles(out)} == {"prediction_not_finite"}


# ============================================================================== provenance


def test_the_summary_records_what_is_needed_to_reproduce_and_to_interpret_the_run(standard):
    summary, out, _ = standard
    assert len(summary["config_digest"]) == 64 and "revision" in summary["code"]["git"] and "dirty" in summary["code"]["git"] and summary["run_id"].startswith("ds-unit-")
    assert summary["ndvi"]["formula"] == "NDVI = (NIR - Red) / (NIR + Red) = (B08 - B04) / (B08 + B04)" and summary["ndvi"]["bands"] == {"red": "B04", "nir": "B08", "band_order_of_inputs": ["B04", "B03", "B02", "B08"]}
    assert "reflectance" in summary["ndvi"]["input_scaling"].lower() and "strict" in summary["ndvi"]["valid_pixel_rule"].lower()
    dec = summary["decision"]
    assert dec["ndvi_threshold"] == 0.3 and dec["sensitivity_thresholds"] == [0.2, 0.4] and dec["selection_policy"] == "predeclared_conventional_value_not_selected_or_tuned_on_results"
    assert summary["regions"]["definition"].startswith("Fixed square cells") and summary["regions"]["scales_hr_px"] == [4, 16] and summary["regions"]["min_valid_fraction"] == 0.75
    assert summary["reference_gate"]["version"].startswith("frame-reliability-gate/") and summary["reference_gate"]["alignment"]["tolerance_hr_px"] == 0.5
    assert summary["tta"]["transforms"][0] == "identity" and summary["tta"]["seed"] == 42 and summary["bootstrap"]["seed"] == 0
    d = summary["datasets"]["fake"]
    assert d["manifest_digest"] == "f" * 64 and d["role"] == "independent_benchmark" and d["split"] == "test" and d["scene_units_all"] == [f"scene{i}" for i in range(6)]
    assert summary["systems"]["a"]["provenance"]["model_name"] == "unit-a"


def test_the_status_of_the_tasks_that_are_not_supported_is_recorded_explicitly(standard):
    summary, _, _ = standard
    assert summary["secondary_tasks"]["landcover"] == "secondary_landcover_task_deferred_no_supported_reference"
    assert summary["indian_data"]["status"] == "india_downstream_validation_unavailable"
    assert any("not land cover" in n.lower() for n in summary["notes"]) and any("calibrated" in n.lower() for n in summary["notes"])


def test_each_tile_row_carries_its_registration_and_its_region_counts(standard):
    _, out, _ = standard
    t = tiles(out)[0]
    assert t["alignment"]["status"] == "eligible" and t["alignment"]["correction"] == [0, 0] and t["gate_version"].startswith("frame-reliability-gate/")
    assert t["candidate_regions"] == {"4": 1024, "16": 64} and t["regions"]["4"]["analysed"] > 0 and set(t["regions_sha256"]) == {"4", "16"}


def test_no_output_uses_ranking_language(standard):
    _, out, _ = standard
    text = " ".join((out / n).read_text().lower() for n in ("README.md", "summary.json", "downstream_metrics.json"))
    for word in ("winner", "best system", "ranking", "outperform"):
        assert word not in text.replace("no ranking", "").replace("not a ranking", "").replace("nothing is ranked", ""), word


# ============================================================================== determinism, safety, cache


def strip_timing(x):
    if isinstance(x, dict):
        return {k: strip_timing(v) for k, v in x.items() if k not in ("seconds", "tta_seconds", "single_pass_seconds", "finished_utc", "run_id")}
    if isinstance(x, list):
        return [strip_timing(v) for v in x]
    return x


def test_two_runs_of_the_same_configuration_give_identical_results(tmp_path):
    _, out1 = run(tmp_path / "1")
    _, out2 = run(tmp_path / "2")
    for name in ("downstream_metrics.json", "association.json", "risk_coverage.json"):
        assert strip_timing(load(out1 / name)) == strip_timing(load(out2 / name)), name
    assert strip_timing(tiles(out1)) == strip_timing(tiles(out2))


def test_results_are_never_overwritten(tmp_path):
    run(tmp_path)
    with pytest.raises(EvaluationError, match="already contains"):
        run(tmp_path)


def test_a_model_that_saw_the_evaluated_scenes_is_refused_before_anything_is_written(tmp_path):
    seen = system("a")
    seen.training_info = {"training_datasets": ["sen2neon"], "train_scenes": [], "val_scenes": []}
    with pytest.raises(RoleSafetyError, match="independent benchmark"):
        run(tmp_path, systems=[seen], names=("a",))
    assert not (tmp_path / "out").exists()


class Counting(Skewed):
    def __init__(self):
        super().__init__()
        self.calls = 0

    def forward(self, x):
        self.calls += 1
        return super().forward(x)


def test_cached_region_tables_are_reused_without_running_the_model_and_give_identical_results(tmp_path):
    cache = str(tmp_path / "cache")
    ds = lambda: FakeDataset(samples=[veg_sample(i) for i in range(4)])                         # noqa: E731
    m1, m2 = Counting(), Counting()
    first = run_downstream(config(tmp_path, names=["a"], cache_dir=cache), systems=[system("a", m1)], datasets=[ds()])
    second = run_downstream(config(tmp_path, names=["a"], cache_dir=cache, output_dir=str(tmp_path / "out2")), systems=[system("a", m2)], datasets=[ds()], reuse_regions=True)
    assert m1.calls == 4 * 6 and m2.calls == 0
    assert strip_timing(load(tmp_path / "out" / "downstream_metrics.json")) == strip_timing(load(tmp_path / "out2" / "downstream_metrics.json"))
    assert second["datasets"]["fake"]["reused_tiles"] == 4 and first["datasets"]["fake"]["reused_tiles"] == 0


def test_a_changed_declared_setting_invalidates_the_cache(tmp_path):
    cache = str(tmp_path / "cache")
    ds = lambda: FakeDataset(samples=[veg_sample(i) for i in range(3)])                         # noqa: E731
    run_downstream(config(tmp_path, names=["a"], cache_dir=cache), systems=[system("a", Skewed())], datasets=[ds()])
    m2 = Counting()
    second = run_downstream(config(tmp_path, names=["a"], cache_dir=cache, output_dir=str(tmp_path / "out2"), decision={"ndvi_threshold": 0.35}), systems=[system("a", m2)], datasets=[ds()],
                            reuse_regions=True)
    assert m2.calls == 3 * 6 and second["datasets"]["fake"]["reused_tiles"] == 0


# ============================================================================== the recorded alignment is applied before anything is compared


def test_a_recorded_translation_correction_is_applied_and_regions_stay_on_prediction_pixel_boundaries(tmp_path):
    s = veg_sample(0)
    s.hr = torch.roll(s.hr, shifts=(2, -1), dims=(1, 2))                                        # registered 2 rows down and 1 column left of the prediction: correctable
    summary, out = run(tmp_path, datasets=[FakeDataset(samples=[s] + [veg_sample(i) for i in range(1, 6)])], names=("a",), cache_dir=str(tmp_path / "cache"))
    t = next(r for r in tiles(out) if r["sample_id"] == "fake:s0")
    assert t["status"] == "analysed" and t["alignment"]["applied"] is True and t["alignment"]["correction"] == [2, -1]
    with np.load(tmp_path / "cache" / "ds-unit" / "fake" / "fake_s0__s4.npz") as z:
        assert (z["sr_row"] % 4 == 0).all() and (z["sr_col"] % 4 == 0).all()                       # a 10 m region is one prediction (Sentinel-2) pixel
        assert ((z["hr_row"] - z["sr_row"]) == 2).all() and ((z["hr_col"] - z["sr_col"]) == -1).all()      # the reference window is offset by the recorded correction
    g = t["regions"]["4"]["grid"]
    assert (g["sr_row_start"], g["hr_row_start"], g["sr_col_start"], g["hr_col_start"]) == (0, 2, 1, 0)


def test_the_gate_decides_from_the_reference_and_the_bicubic_baseline_before_any_model_runs(tmp_path):
    """No model is ever called for an ineligible tile (they would only cost time and could leak the reference into a comparison)."""
    m = Counting()
    run(tmp_path, datasets=[excluded_dataset()], systems=[system("a", m)], names=("a",))
    assert m.calls == 5 * 6                                                                     # five eligible tiles x six views; the two ineligible tiles ran nothing


def unit_bicubic_mae(out, scale="4"):
    m = load(out / "downstream_metrics.json")["datasets"]["fake"]["unit_metrics"][scale]
    return [m[u]["bicubic"]["0.3"]["ndvi_mae"] for u in sorted(m)]


def test_a_recorded_translation_leaves_the_comparison_unchanged_because_everything_is_cropped_to_the_registered_overlap(tmp_path):
    """The same scene, once registered and once with its reference displaced by a correctable (2, -1) HR px: after the recorded crop the bicubic NDVI error must be the same (within the
    edge cells lost to the crop). Without the crop the displaced reference would change every region's NDVI."""
    plain = [veg_sample(i) for i in range(5)]
    _, out_a = run(tmp_path / "a", datasets=[FakeDataset(samples=plain)], names=("a",))
    displaced = [veg_sample(i) for i in range(5)]
    displaced[0].hr = torch.roll(displaced[0].hr, shifts=(2, -1), dims=(1, 2))
    _, out_b = run(tmp_path / "b", datasets=[FakeDataset(samples=displaced)], names=("a",))
    a, b = unit_bicubic_mae(out_a), unit_bicubic_mae(out_b)
    assert abs(a[0] - b[0]) < 0.15 * a[0] and a[1:] == b[1:]
    t = next(r for r in tiles(out_b) if r["sample_id"] == "fake:s0")
    assert t["alignment"]["applied"] is True and t["status"] == "analysed"


class ZeroPatch(Skewed):
    """A model that returns an all-zero patch: NDVI is undefined there (0 / 0) for THIS system only. The patch is the centre of the 128 x 128 scene, so it is symmetric under every TTA view
    and survives the ensemble mean."""

    def forward(self, x):
        y = super().forward(x)
        y[..., 48:80, 48:80] = 0.0
        return y


def test_every_system_is_scored_on_identical_pixels_so_one_systems_undefined_ndvi_removes_those_regions_for_all(tmp_path):
    summary, out = run(tmp_path, systems=[system("a", Skewed()), system("b", ZeroPatch())], names=("a", "b"), datasets=[FakeDataset(samples=[veg_sample(i) for i in range(5)])],
                       cache_dir=str(tmp_path / "cache"))
    with np.load(tmp_path / "cache" / "ds-unit" / "fake" / "fake_s0__s4.npz") as z:
        inside = (z["sr_row"] >= 48) & (z["sr_row"] <= 76) & (z["sr_col"] >= 48) & (z["sr_col"] <= 76)
        assert not inside.any() and len(z["row"]) == 32 * 32 - 64                                    # the 64 cells fully inside the patch are gone for every system, including model 'a' and bicubic
        assert (z["n_valid"] >= 12).all()
    assert summary["datasets"]["fake"]["regions"]["4"]["excluded_region_level"]["region_insufficient_valid_pixels"] >= 5 * 64


def test_a_different_model_never_reuses_another_models_region_tables(tmp_path):
    from frame.evaluate.systems import CallableSystem
    from frame.tests.test_reliability_runner import TILING

    cache = str(tmp_path / "cache")
    ds = lambda: FakeDataset(samples=[veg_sample(i) for i in range(3)])                         # noqa: E731
    run_downstream(config(tmp_path, names=["a"], cache_dir=cache), systems=[system("a", Skewed())], datasets=[ds()])
    other = Counting()
    swapped = CallableSystem("a", other, TILING, hard_constraint=False, provenance={"model_name": "unit-a", "weights": {"sha": "different"}}, device="cpu", kind="lite")
    second = run_downstream(config(tmp_path, names=["a"], cache_dir=cache, output_dir=str(tmp_path / "out2")), systems=[swapped], datasets=[ds()], reuse_regions=True)
    assert other.calls == 3 * 6 and second["datasets"]["fake"]["reused_tiles"] == 0
