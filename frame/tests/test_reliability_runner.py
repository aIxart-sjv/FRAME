"""frame.reliability.runner -- the whole flow on fake datasets and fake models: gate, TTA, evidence, analysis, exclusions, failures, provenance, safety. No GPU."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from frame.evaluate.errors import EvaluationError, RoleSafetyError
from frame.evaluate.systems import CallableSystem
from frame.reliability.config import ReliabilityConfig
from frame.reliability.eligibility import EXCLUDED
from frame.reliability.evidence import load_evidence
from frame.reliability.runner import run_reliability
from frame.tests.test_evaluate_runner import FakeDataset, field_image, make_sample
from frame.tiling.plan import TilingConfig

TILING = TilingConfig(tile_size=128, overlap=32)
BANDS = ("B04", "B03", "B02", "B08")


class Skewed(torch.nn.Module):
    """A NON-equivariant x4 model (an asymmetric sharpening), so the six views genuinely disagree."""

    def __init__(self, k=0.5):
        super().__init__()
        self.k = k

    def forward(self, x):
        up = F.interpolate(x, scale_factor=4, mode="bicubic", antialias=True).clamp(min=0)
        return up + self.k * (up - torch.roll(up, shifts=(1, 2), dims=(-2, -1)))


class Failing(Skewed):
    def __init__(self, fail_on_call, persistent=False):
        super().__init__()
        self.fail_on_call, self.persistent, self.calls = fail_on_call, persistent, 0

    def forward(self, x):
        self.calls += 1
        if self.calls == self.fail_on_call or (self.persistent and self.calls >= self.fail_on_call):
            raise RuntimeError("worker died")
        return super().forward(x)


class NaNs(Skewed):
    def forward(self, x):
        y = super().forward(x)
        y[..., 5, 5] = float("nan")
        return y


def system(name, model=None, **kw):
    return CallableSystem(name, model or Skewed(), TILING, hard_constraint=False, provenance={"model_name": f"unit-{name}", "weights": None}, device="cpu", kind="lite")


def config(tmp_path, names=("a", "b"), **over):
    d = {"name": "rel-unit", "output_dir": str(tmp_path / "out"), "device": "cpu", "systems": [{"name": n, "kind": "lite"} for n in names],
         "datasets": [{"name": "fake", "kind": "sen2neon", "manifest": "unused.jsonl"}], "bootstrap": {"n_boot": 100},
         "analysis": {"pooled_cells_per_tile": 200, "pooled_pixels_per_tile": 400, "displacement_sweep_hr_px": [0, 2], "cell_sizes_hr_px": [4, 8], **over.pop("analysis", {})}, **over}
    return ReliabilityConfig.from_dict(d)


def run(tmp_path, *, datasets=None, systems=None, names=("a", "b"), **over):
    systems = systems or [system(n) for n in names]
    return run_reliability(config(tmp_path, names=[s.name for s in systems], **over), systems=systems, datasets=datasets or [FakeDataset(samples=[make_sample(i) for i in range(6)])]), tmp_path / "out"


def rows_of(out):
    return [json.loads(l) for l in (out / "metrics.jsonl").read_text().splitlines() if l.strip()]


@pytest.fixture(scope="module")
def standard(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("rel")
    summary, out = run(tmp)
    return summary, out


# ============================================================================== a whole run


def test_a_run_writes_the_experiment_layout(standard):
    summary, out = standard
    assert sorted(p.name for p in out.iterdir()) == sorted(["README.md", "calibration.json", "config.json", "correlations.json", "detection.json", "metrics.jsonl", "risk_coverage.json", "summary.json"])
    assert summary["status"] == "completed" and json.loads((out / "summary.json").read_text())["status"] == "completed"
    assert summary["overview"]["fake"]["a"]["n_tiles"] == 6 and summary["overview"]["fake"]["a"]["n_units"] == 6


def test_every_sample_ends_as_exactly_one_recorded_outcome_per_system(standard):
    _, out = standard
    rows = rows_of(out)
    eligible = [r for r in rows if r["type"] == "tile_result" and r["status"] == "eligible"]
    assert len(eligible) == 6 * 2 and {(r["system"], r["sample_id"]) for r in eligible} == {(s, f"fake:s{i}") for s in "ab" for i in range(6)}
    assert all(r["evidence_level"] == "pixel_level_eligible" and r["alignment"]["status"] == "eligible" for r in eligible)


def test_the_readme_states_the_limits_and_lists_the_evidence(standard):
    _, out = standard
    text = (out / "README.md").read_text()
    assert "not a calibrated uncertainty" in text and "descriptive" in text.lower() and "eligibility gate" in text.lower() and "Evidence" in text
    assert "Risk-coverage" in text and "not** calibrated selective prediction" in text and "uncalibrated stability evidence" in text and "`fake`" in text


# ============================================================================== exclusions are explicit and carry no numbers


def excluded_dataset():
    samples = [make_sample(i) for i in range(6)]
    shifted = make_sample(6)
    shifted.hr = torch.roll(shifted.hr, shifts=(14, 9), dims=(1, 2))                                      # registered 14 rows / 9 columns away: beyond the correctable range
    sparse = make_sample(7, valid_fraction=0.3)
    nodata = make_sample(8, valid_fraction=0.0)
    return FakeDataset(samples=samples + [shifted, sparse, nodata])


def test_misregistered_and_sparse_references_are_excluded_with_machine_readable_reasons_and_never_run(tmp_path):
    ds = excluded_dataset()
    model = Skewed()
    calls = {"n": 0}
    orig = model.forward

    def counting(x):
        calls["n"] += 1
        return orig(x)

    model.forward = counting
    summary, out = run(tmp_path, datasets=[ds], systems=[system("a", model)], names=("a",))
    rows = rows_of(out)
    ex = {r["sample_id"]: r for r in rows if r["type"] == "tile_result" and r["status"] == EXCLUDED}
    assert ex["fake:s6"]["reason"] == "reference_alignment_invalid" and ex["fake:s6"]["evidence_level"] == "not_eligible" and ex["fake:s6"]["alignment"]["reason"] == "reference_alignment_invalid"
    assert ex["fake:s7"]["reason"] == "insufficient_valid_pixels" and ex["fake:s8"]["reason"] == "insufficient_valid_pixels"
    assert all("targets" not in r and "pixel" not in r and r["system"] is None for r in ex.values())     # no invented numbers; the gate is system-independent
    assert calls["n"] == 6 * 6                                                                          # only the six eligible tiles were run: six views each, never the excluded ones
    d = summary["datasets"]["fake"]
    assert d["gate"]["excluded_by_reason"] == {"insufficient_valid_pixels": 2, "reference_alignment_invalid": 1} and d["gate"]["by_evidence_level"]["pixel_level_eligible"] == 6
    assert len(d["tiles"]) == 9 and {t["sample_id"] for t in d["tiles"]} == {f"fake:s{i}" for i in range(9)}


def test_the_summary_counts_reconcile_and_list_invalid_and_unreadable_samples(tmp_path):
    ds = FakeDataset(samples=[make_sample(i) for i in range(5)], invalid=[{"sample_id": "fake:s4", "codes": ["footprint_mismatch"], "messages": ["HR origin shifted"]}], load_errors=["fake:s3"])
    summary, out = run(tmp_path, datasets=[ds], names=("a",))
    c = summary["datasets"]["fake"]["counts"]
    assert c == {"total": 5, "invalid": 1, "unreadable": 1, "reference_gate_evaluated": 3}
    types = {(r["type"], r["sample_id"]) for r in rows_of(out) if r["type"] in ("sample_invalid", "sample_unreadable")}
    assert types == {("sample_invalid", "fake:s4"), ("sample_unreadable", "fake:s3")}


# ============================================================================== model and TTA failures


def test_a_model_that_fails_immediately_is_a_model_failure_of_that_system_only(tmp_path):
    summary, out = run(tmp_path, systems=[system("a"), system("b", Failing(fail_on_call=1, persistent=True))])
    rows = rows_of(out)
    failed = [r for r in rows if r["status"] == EXCLUDED and r["system"] == "b"]
    assert len(failed) == 6 and {r["reason"] for r in failed} == {"model_failure"} and "worker died" in failed[0]["detail"]
    assert summary["overview"]["fake"]["a"]["n_tiles"] == 6 and summary["overview"]["fake"]["b"]["status"] == "no_eligible_evidence"
    assert summary["datasets"]["fake"]["per_system"]["b"]["excluded_after_gate"] == {"model_failure": 6}


def test_a_model_that_dies_part_way_through_the_ensemble_is_a_tta_member_failure_naming_the_member(tmp_path):
    _, out = run(tmp_path, systems=[system("a", Failing(fail_on_call=4))], names=("a",))
    failed = [r for r in rows_of(out) if r["status"] == EXCLUDED]
    assert failed and {r["reason"] for r in failed} == {"tta_member_failure"} and "member 3 (rot90)" in failed[0]["detail"]


def test_a_non_finite_prediction_excludes_the_tile_and_is_never_scored(tmp_path):
    summary, out = run(tmp_path, systems=[system("a", NaNs())], names=("a",))
    failed = [r for r in rows_of(out) if r["status"] == EXCLUDED]
    assert len(failed) == 6 and {r["reason"] for r in failed} == {"prediction_not_finite"} and all("targets" not in r for r in failed)
    assert summary["overview"]["fake"]["a"]["status"] == "no_eligible_evidence"


# ============================================================================== provenance


def test_the_summary_records_what_is_needed_to_interpret_and_reproduce_the_run(standard):
    summary, out = standard
    assert summary["alignment"]["method"] == "bicubic_baseline_cross_correlation" and summary["alignment"]["tolerance_hr_px"] == 0.5 and summary["alignment"]["apply_translation_correction"] is True
    assert summary["tta"]["transforms"] == ["identity", "hflip", "vflip", "rot90", "rot180", "rot270"] and summary["tta"]["n_members"] == 6 and summary["tta"]["seed"] == 42
    assert summary["metric_configuration"]["version"].startswith("frame-eval-metrics") and "revision" in summary["code"]["git"] and len(summary["config_digest"]) == 64
    d = summary["datasets"]["fake"]
    assert d["manifest_digest"] == "f" * 64 and d["role"] == "independent_benchmark" and d["split"] == "test" and d["evidence_class"] == "real_cross_sensor"
    assert d["scene_units_eligible"] == [f"scene{i}" for i in range(6)]
    assert summary["systems"]["a"]["provenance"]["model_name"] == "unit-a" and summary["systems"]["a"]["provenance"]["tiling"]
    row = next(r for r in rows_of(out) if r["status"] == "eligible")
    assert row["alignment"]["correction"] == [0, 0] and row["tta"]["n_members"] == 6 and "strict" in row["valid_pixel_rule"] or "valid" in row["valid_pixel_rule"]


def test_the_result_rows_carry_the_stability_definition_and_the_error_target_definitions(standard):
    _, out = standard
    row = next(r for r in rows_of(out) if r["status"] == "eligible")
    assert row["stability"]["definition"].startswith("Mean per-pixel standard deviation") and "TTA ensemble mean" in row["targets"]["definition"]


# ============================================================================== determinism, safety, cache


def strip_timing(rows):
    out = []
    for r in rows:
        r = json.loads(json.dumps(r))
        if "tta" in r:
            for k in ("total_seconds", "seconds_per_member", "single_pass_seconds"):
                r["tta"].pop(k, None)
        out.append(r)
    return out


def test_two_runs_of_the_same_configuration_give_identical_evidence(tmp_path):
    _, out1 = run(tmp_path / "1")
    _, out2 = run(tmp_path / "2")
    assert strip_timing(rows_of(out1)) == strip_timing(rows_of(out2))
    def stable(path):
        data = json.loads(path.read_text())
        for cmp in data["comparison"].values():
            for pair in cmp["pairs"]:
                pair["metrics"] = {k: v for k, v in pair["metrics"].items() if not k.startswith("tta.")}         # latencies are measurements, not results
        return data

    assert stable(out1 / "correlations.json") == stable(out2 / "correlations.json")
    for name in ("risk_coverage.json", "detection.json", "calibration.json"):
        assert json.loads((out1 / name).read_text()) == json.loads((out2 / name).read_text()), name


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


def test_the_pooled_arrays_are_cached_outside_the_run_directory_and_round_trip(tmp_path):
    cache = tmp_path / "cache"
    _, out = run(tmp_path, cache_dir=str(cache), names=("a",))
    stem = cache / "rel-unit" / "fake" / "a" / "fake_s0"
    row, arrays = load_evidence(stem)
    assert row["sample_id"] == "fake:s0" and "c4_stability" in arrays and len(list(out.iterdir())) == 8
    assert not any(p.suffix == ".npz" for p in out.iterdir())


# ============================================================================== reusing evidence (the ensemble is the expensive part)


class Counting(Skewed):
    def __init__(self):
        super().__init__()
        self.calls = 0

    def forward(self, x):
        self.calls += 1
        return super().forward(x)


def two_runs(tmp_path, second_over=None):
    cache = str(tmp_path / "cache")
    ds = lambda: FakeDataset(samples=[make_sample(i) for i in range(4)])              # noqa: E731
    m1, m2 = Counting(), Counting()
    first = run_reliability(config(tmp_path, names=["a"], cache_dir=cache), systems=[system("a", m1)], datasets=[ds()])
    over = {"output_dir": str(tmp_path / "out2"), "cache_dir": cache, **(second_over or {})}
    second = run_reliability(config(tmp_path, names=["a"], **over), systems=[system("a", m2)], datasets=[ds()], reuse_evidence=True)
    return first, second, m1, m2, tmp_path / "out", tmp_path / "out2"


def test_cached_evidence_is_reused_without_running_the_model_and_gives_identical_results(tmp_path):
    first, second, m1, m2, out1, out2 = two_runs(tmp_path)
    assert m1.calls == 4 * 6 and m2.calls == 0                                                        # four tiles x six views the first time, nothing the second
    assert strip_timing(rows_of(out1)) == strip_timing(rows_of(out2))
    assert json.loads((out1 / "risk_coverage.json").read_text()) == json.loads((out2 / "risk_coverage.json").read_text())
    assert second["datasets"]["fake"]["per_system"]["a"]["reused_evidence"] == 4 and first["datasets"]["fake"]["per_system"]["a"]["reused_evidence"] == 0


def test_a_setting_that_changes_the_evidence_invalidates_the_cache_entry(tmp_path):
    _, second, _, m2, _, _ = two_runs(tmp_path, second_over={"analysis": {"cell_sizes_hr_px": [4, 16], "pooled_cells_per_tile": 200, "pooled_pixels_per_tile": 400, "displacement_sweep_hr_px": [0, 2]}})
    assert m2.calls == 4 * 6 and second["datasets"]["fake"]["per_system"]["a"]["reused_evidence"] == 0


def test_a_different_model_never_reuses_another_models_evidence(tmp_path):
    cache = str(tmp_path / "cache")
    ds = lambda: FakeDataset(samples=[make_sample(i) for i in range(3)])              # noqa: E731
    run_reliability(config(tmp_path, names=["a"], cache_dir=cache), systems=[system("a", Skewed(0.5))], datasets=[ds()])
    other = Counting()
    other.k = 0.9                                                                                   # same system NAME, different behaviour: the provenance (hence the digest) differs only if declared
    swapped = CallableSystem("a", other, TILING, hard_constraint=False, provenance={"model_name": "unit-a", "weights": {"sha": "different"}}, device="cpu", kind="lite")
    run_reliability(config(tmp_path, names=["a"], output_dir=str(tmp_path / "out2"), cache_dir=cache), systems=[swapped], datasets=[ds()], reuse_evidence=True)
    assert other.calls == 3 * 6


# ============================================================================== the ensemble step is reusable by later phases


def test_the_tta_step_is_a_public_function_that_other_analyses_reuse():
    from frame.reliability.runner import run_tta_for_tile

    sample = make_sample(0)
    result = run_tta_for_tile(system("a"), sample, ("identity", "hflip", "vflip"), seed=7)
    assert result.n == 3 and result.seed == 7 and tuple(result.transform_names) == ("identity", "hflip", "vflip") and result.mean_prediction.shape == sample.hr.shape
    failed = run_tta_for_tile(system("b", Failing(fail_on_call=2, persistent=True)), sample, ("identity", "hflip", "vflip"), seed=7)
    assert failed[0] == "tta_member_failure" and "member 1 (hflip)" in failed[1]
