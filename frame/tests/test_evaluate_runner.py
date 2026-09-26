"""frame.evaluate.runner -- the evaluation flow: validation, inference, mask-aware scoring, aggregation, statistics, provenance, safety."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from scipy import ndimage

from frame.evaluate.config import EvalConfig, SystemSpec
from frame.evaluate.datasets import DatasetValidation, EvalSample
from frame.evaluate.errors import EvaluationError, RoleSafetyError
from frame.evaluate.runner import flatten_row, run_evaluation, score_sample
from frame.evaluate.systems import BicubicSystem, CallableSystem, System
from frame.tiling.plan import TilingConfig

BANDS = ("B04", "B03", "B02", "B08")
TILING = TilingConfig(tile_size=128, overlap=32)


def assert_close(a, b, rel=1e-9, path=""):
    """Recursive numeric comparison of nested result dicts."""
    if isinstance(a, dict):
        assert set(a) == set(b), (path, set(a) ^ set(b))
        for k in a:
            assert_close(a[k], b[k], rel, f"{path}.{k}")
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b), path
        for i, (x, y) in enumerate(zip(a, b)):
            assert_close(x, y, rel, f"{path}[{i}]")
    elif isinstance(a, float) or isinstance(b, float):
        assert a == pytest.approx(b, rel=rel, abs=1e-12), (path, a, b)
    else:
        assert a == b, (path, a, b)


def field_image(seed, size=128):
    rng = np.random.default_rng(seed)
    x = ndimage.gaussian_filter(rng.standard_normal((4, size, size)), (0, 1.2, 1.2))
    return (0.22 + 0.06 * x / x.std()).astype("float32").clip(0.02, 0.9)


def make_sample(i, *, dataset="fake", group=None, category=None, valid_fraction=1.0, evidence="real_cross_sensor"):
    hr = torch.from_numpy(field_image(i))
    lr = F.avg_pool2d(hr[None], 4)[0]
    mask = torch.ones(128, 128, dtype=torch.bool)
    if valid_fraction < 1.0:
        mask[: int(128 * (1 - valid_fraction))] = False
    quality = {"hr_valid_fraction": float(mask.float().mean()), "hr_dataset_nodata_fraction": float(1 - mask.float().mean()), "hr_partial_nodata_fraction": 0.0, "hr_nonfinite_fraction": 0.0,
               "hr_valid_pixels": int(mask.sum()), "rule": "test rule", "lr_valid_fraction": 1.0, "lr_valid_pixels": 32 * 32}
    return EvalSample(sample_id=f"{dataset}:s{i}", dataset=dataset, scene_group=group or f"scene{i}", category=category, split="test", evidence_class=evidence, lr=lr, hr=hr,
                      lr_mask=torch.ones(32, 32, dtype=torch.bool), hr_mask=mask, bands=BANDS, scale=4, lr_pixel_m=10.0, hr_pixel_m=2.5, quality=quality,
                      provenance={"dataset": dataset, "hr_reflectance_scale": 10000.0})


class FakeDataset:
    """Just enough of EvalDataset for the runner."""

    def __init__(self, name="fake", kind="sen2neon", evidence="real_cross_sensor", samples=None, invalid=None, load_errors=()):
        self.samples = samples if samples is not None else [make_sample(i, dataset=name, evidence=evidence) for i in range(4)]
        self.name, self.kind, self.evidence_class, self.role = name, kind, evidence, "independent_benchmark"
        self.records = [SimpleNamespace(sample_id=s.sample_id, scene_id=s.scene_group) for s in self.samples]
        self.ignored, self.manifest_digest, self.info = {}, "f" * 64, {"note": "fake"}
        self.spec = SimpleNamespace(split="test", subset=None, hr_variant=None)
        self._invalid, self._load_errors = invalid or [], set(load_errors)
        self._by_id = {s.sample_id: s for s in self.samples}

    def validate(self, *, check_files=True):
        bad = {i["sample_id"] for i in self._invalid}
        return DatasetValidation(valid_records=[r for r in self.records if r.sample_id not in bad], invalid=list(self._invalid))

    def load(self, record):
        if record.sample_id in self._load_errors:
            raise OSError("simulated unreadable file")
        return self._by_id[record.sample_id]

    def scene_group_of(self, record):
        return self._by_id[record.sample_id].scene_group

    def category_of(self, record):
        return self._by_id[record.sample_id].category


class Sharpen(torch.nn.Module):
    def forward(self, x):
        up = F.interpolate(x, scale_factor=4, mode="bicubic", antialias=True).clamp(min=0)
        blur = F.avg_pool2d(F.pad(up, (2, 2, 2, 2), mode="reflect"), 5, stride=1)
        return up + 0.6 * (up - blur)


def sharpen_system(name="sharpen"):
    return CallableSystem(name, Sharpen(), TILING, hard_constraint=False, provenance={"model_name": "unit-sharpen", "weights": None}, device="cpu", kind="callable")


class Exploding(System):
    kind, hard_constraint = "callable", False

    def __init__(self, name="boom", fail_on=("fake:s1",)):
        self.name, self._fail = name, set(fail_on)
        self._current = None

    def infer(self, lr):
        raise RuntimeError("simulated inference failure")

    def provenance(self):
        return {"name": self.name, "kind": "callable"}


def config(tmp_path, systems=("bicubic", "sharpen"), **over):
    d = {"name": "unit", "output_dir": str(tmp_path / "out"), "systems": [{"name": "bicubic", "kind": "bicubic"}] + [{"name": n, "kind": "lite"} for n in systems if n != "bicubic"],
         "datasets": [{"name": "fake", "kind": "sen2neon", "manifest": "unused.jsonl"}], "statistics": {"n_boot": 200}, "metrics": {"hallucination_taus": [0.005, 0.01]}}
    d.update(over)
    return EvalConfig.from_dict(d)


def run(tmp_path, *, systems=None, datasets=None, **over):
    systems = systems or [BicubicSystem("bicubic"), sharpen_system()]
    cfg = config(tmp_path, systems=[s.name for s in systems], **over)
    return run_evaluation(cfg, systems=systems, datasets=datasets or [FakeDataset()]), tmp_path / "out"


def read_jsonl(path):
    return [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]


def agg(out):
    return json.loads((out / "aggregates.json").read_text())


@pytest.fixture(scope="module")
def standard(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("standard")
    summary, out = run(tmp)
    return summary, out


# ============================================================================== one sample, scored


def score(sr_kind="sharpen", sample=None, **kw):
    sample = sample or make_sample(0)
    bic = BicubicSystem("bicubic").infer(sample.lr)
    system = BicubicSystem("bicubic") if sr_kind == "bicubic" else sharpen_system()
    result = system.infer(sample.lr)
    return score_sample(sample, system, result, bic.sr, config(Path("/tmp/x")).metric_config(), **kw)


def test_a_scored_sample_holds_separate_groups_and_is_json_safe():
    row = score()
    for group in ("reference_accuracy", "per_band", "indices", "band_ratios", "spatial_detail", "detail_analysis", "self_consistency", "opensr_native", "quality"):
        assert group in row, group
    assert set(row["per_band"]) == set(BANDS) and "NDVI" in row["indices"] and row["status"] == "ok"
    json.dumps(row, allow_nan=False)                                     # no NaN / Infinity anywhere
    assert not ({"psnr_db", "ssim", "sam_degrees", "ergas"} & set(row["self_consistency"]["overall"]))          # self-consistency is not reference accuracy


def test_the_baseline_system_gets_no_detail_analysis_because_it_is_the_baseline():
    row = score("bicubic")
    assert row["detail_analysis"] is None and "baseline" in row["detail_analysis_note"]


def test_junk_under_the_reference_nodata_mask_changes_nothing():
    sample = make_sample(0, valid_fraction=0.5)
    clean = score(sample=sample)
    sample2 = make_sample(0, valid_fraction=0.5)
    sample2.hr[:, :64] = 9.0                                              # the masked half of the reference is garbage
    dirty = score(sample=sample2)
    for group in ("reference_accuracy", "per_band", "indices", "band_ratios", "spatial_detail", "self_consistency"):
        assert_close(dirty[group], clean[group], rel=1e-9), group                     # (SSIM's running sums differ at the 13th digit even far from the junk)
    assert clean["reference_accuracy"]["valid_fraction"] == pytest.approx(0.5)


def test_an_sr_of_the_wrong_shape_is_rejected():
    sample = make_sample(0)
    result = BicubicSystem("bicubic").infer(sample.lr)
    result.sr = result.sr[:, :100, :100]
    from frame.evaluate.errors import ReferenceMismatchError

    with pytest.raises(ReferenceMismatchError):
        score_sample(sample, BicubicSystem("bicubic"), result, result.sr, config(Path("/tmp/x")).metric_config())


def test_flatten_row_gives_one_number_per_dotted_metric_name():
    flat = flatten_row(score())
    assert flat["reference_accuracy.psnr_db"] > 0 and "per_band.B08.rmse" in flat and "indices.NDVI.mae" in flat and "self_consistency.overall.mae" in flat
    assert "spatial_detail.phase_shift_px.magnitude" in flat and "detail_analysis.0.005.supported_synthesis" in flat and all(isinstance(v, (int, float)) or v is None for v in flat.values())


# ============================================================================== a whole run


def test_a_run_writes_the_experiment_layout(standard):
    summary, out = standard
    assert sorted(p.name for p in out.iterdir()) == ["README.md", "aggregates.json", "config.json", "metrics.jsonl", "summary.json"]
    assert summary["status"] == "completed" and json.loads((out / "summary.json").read_text())["status"] == "completed"


def test_every_sample_system_pair_has_exactly_one_row_and_nothing_is_dropped(standard):
    _, out = standard
    rows = [r for r in read_jsonl(out / "metrics.jsonl") if r["type"] == "sample_result"]
    assert len(rows) == 4 * 2 and {(r["dataset"], r["system"], r["sample_id"]) for r in rows} == {("fake", s, f"fake:s{i}") for s in ("bicubic", "sharpen") for i in range(4)}
    assert all(r["status"] == "ok" and r["evidence_class"] == "real_cross_sensor" for r in rows)


def test_every_row_names_the_configured_dataset_so_two_datasets_of_one_kind_stay_distinguishable(tmp_path):
    """Found on the real run: sample rows carried the dataset KIND ("sen2neon"), so the random-30 rows could not be told from the Phase 3 sample's rows in metrics.jsonl."""
    a = FakeDataset(name="neon_a", kind="sen2neon", samples=[make_sample(i, dataset="sen2neon") for i in range(3)])
    b = FakeDataset(name="neon_b", kind="sen2neon", samples=[make_sample(i + 10, dataset="sen2neon") for i in range(2)],
                    invalid=[{"sample_id": "sen2neon:s99", "codes": ["x"], "messages": ["y"]}])
    b.records.append(SimpleNamespace(sample_id="sen2neon:s99", scene_id="scene99"))
    cfg_dict = {"name": "unit", "output_dir": str(tmp_path / "out"), "systems": [{"name": "bicubic", "kind": "bicubic"}], "statistics": {"n_boot": 100},
                "datasets": [{"name": "neon_a", "kind": "sen2neon", "manifest": "unused.jsonl"}, {"name": "neon_b", "kind": "sen2neon", "manifest": "unused.jsonl"}]}
    run_evaluation(EvalConfig.from_dict(cfg_dict), systems=[BicubicSystem("bicubic")], datasets=[a, b])
    rows = read_jsonl(tmp_path / "out" / "metrics.jsonl")
    results = [r for r in rows if r["type"] == "sample_result"]
    assert {r["dataset"] for r in results} == {"neon_a", "neon_b"}
    assert sum(r["dataset"] == "neon_a" for r in results) == 3 and sum(r["dataset"] == "neon_b" for r in results) == 2
    assert {r["dataset_kind"] for r in results} == {"sen2neon"}                                       # the kind is still recorded, under its own name
    assert all("dataset" in r for r in rows)                                                        # invalid / skipped / unreadable rows already named the dataset


def test_the_summary_counts_reconcile_and_list_what_was_not_scored(tmp_path):
    ds = FakeDataset(samples=[make_sample(i, valid_fraction=(0.02 if i == 2 else 1.0)) for i in range(5)], invalid=[{"sample_id": "fake:s4", "codes": ["footprint_mismatch"], "messages": ["HR origin shifted"]}],
                     load_errors=["fake:s3"])
    summary, out = run(tmp_path, datasets=[ds])
    c = summary["datasets"]["fake"]["counts"]
    assert c == {"total": 5, "invalid": 1, "valid": 4, "unreadable": 1, "skipped": 1, "evaluated": 2}
    assert [s["sample_id"] for s in summary["datasets"]["fake"]["skipped"]] == ["fake:s2"] and "insufficient_valid_reference" in summary["datasets"]["fake"]["skipped"][0]["reason"]
    assert summary["datasets"]["fake"]["invalid"][0]["sample_id"] == "fake:s4" and summary["datasets"]["fake"]["unreadable"][0]["sample_id"] == "fake:s3"
    rows = read_jsonl(out / "metrics.jsonl")
    assert {r["type"] for r in rows} == {"sample_result", "sample_skipped", "sample_invalid", "sample_unreadable"}
    assert agg(out)["datasets"]["fake"]["systems"]["bicubic"]["metrics"]["reference_accuracy.rmse"]["sample"]["n"] == 2


def test_an_inference_failure_is_recorded_per_sample_and_does_not_stop_the_run(tmp_path):
    summary, out = run(tmp_path, systems=[BicubicSystem("bicubic"), Exploding()])
    rows = read_jsonl(out / "metrics.jsonl")
    failed = [r for r in rows if r["type"] == "sample_result" and r["status"] == "failed"]
    assert len(failed) == 4 and all(r["system"] == "boom" and "simulated inference failure" in r["error"] for r in failed)
    assert summary["datasets"]["fake"]["failures"] == {"boom": 4} and "boom" not in agg(out)["datasets"]["fake"]["systems"]
    assert [r for r in rows if r["system"] == "bicubic" and r["status"] == "ok"] and len(summary["datasets"]["fake"]["failure_examples"]) >= 1


def test_an_unavailable_system_is_listed_with_its_reason_and_the_run_continues(tmp_path):
    cfg = config(tmp_path, systems=["bicubic", "mamba_x"], datasets=[{"name": "fake", "kind": "sen2neon", "manifest": "u.jsonl"}])
    cfg = EvalConfig.from_dict({**cfg.to_dict(), "systems": [{"name": "bicubic", "kind": "bicubic"}, {"name": "mamba_x", "kind": "mamba"}]})
    from frame.evaluate.errors import SystemUnavailableError

    def factory(*a, **k):
        raise SystemUnavailableError("no GPU on this machine")

    summary = run_evaluation(cfg, datasets=[FakeDataset()], system_factory=lambda spec, tiling, **kw: BicubicSystem("bicubic") if spec.kind == "bicubic" else factory())
    assert summary["systems_unavailable"] == [{"name": "mamba_x", "kind": "mamba", "reason": "no GPU on this machine"}]
    assert summary["systems"]["mamba_x"]["available"] is False and summary["systems"]["bicubic"]["available"] is True


# ============================================================================== aggregation and statistics


def test_correlated_tiles_of_one_scene_are_one_statistical_unit(tmp_path):
    samples = [make_sample(0, group="A"), make_sample(1, group="A"), make_sample(2, group="B"), make_sample(3, group="C")]
    ds = FakeDataset(samples=samples)
    _, out = run(tmp_path, datasets=[ds])
    rows = [r for r in read_jsonl(out / "metrics.jsonl") if r["system"] == "bicubic" and r["type"] == "sample_result"]
    m = agg(out)["datasets"]["fake"]["systems"]["bicubic"]["metrics"]["reference_accuracy.rmse"]
    assert m["sample"]["n"] == 4 and m["unit"]["n"] == 3
    by = {r["sample_id"]: r["reference_accuracy"]["rmse"] for r in rows}
    expected_unit_mean = np.mean([np.mean([by["fake:s0"], by["fake:s1"]]), by["fake:s2"], by["fake:s3"]])
    assert m["unit"]["mean"] == pytest.approx(expected_unit_mean) and m["unit"]["ci"]["status"] == "descriptive_only"          # 3 units < 5


def test_intervals_and_tests_appear_only_when_there_are_enough_units(tmp_path):
    ds = FakeDataset(samples=[make_sample(i) for i in range(7)])
    _, out = run(tmp_path, datasets=[ds])
    a = agg(out)["datasets"]["fake"]
    m = a["systems"]["sharpen"]["metrics"]["reference_accuracy.rmse"]["unit"]
    assert m["n"] == 7 and m["ci"]["status"] == "ok" and m["ci"]["ci_low"] <= m["mean"] <= m["ci"]["ci_high"]
    pair = a["paired"]["sharpen - bicubic"]["reference_accuracy.rmse"]
    assert pair["n"] == 7 and pair["wilcoxon"]["status"] == "tested" and pair["ci"]["status"] == "ok" and pair["label"] == "inferential"


def test_with_few_scenes_paired_differences_are_descriptive_only(standard):
    _, out = standard                                                     # 4 units
    pair = agg(out)["datasets"]["fake"]["paired"]["sharpen - bicubic"]["reference_accuracy.rmse"]
    assert pair["n"] == 4 and pair["label"] == "descriptive_only" and pair["wilcoxon"]["p_value"] is None and pair["mean_difference"] is not None


def test_the_reference_system_compared_with_itself_has_no_pair(standard):
    _, out = standard
    assert "bicubic - bicubic" not in agg(out)["datasets"]["fake"]["paired"]


def test_extra_pairs_from_the_config_are_computed(tmp_path):
    cfg = config(tmp_path, systems=["bicubic", "sharpen"], statistics={"n_boot": 200, "pairs": [["sharpen", "bicubic"]]})
    run_evaluation(cfg, systems=[BicubicSystem("bicubic"), sharpen_system()], datasets=[FakeDataset()])
    assert "sharpen - bicubic" in agg(tmp_path / "out")["datasets"]["fake"]["paired"]


def test_per_band_error_is_aggregated_for_every_band(standard):
    _, out = standard
    metrics = agg(out)["datasets"]["fake"]["systems"]["sharpen"]["metrics"]
    for band in BANDS:
        assert f"per_band.{band}.rmse" in metrics and f"per_band.{band}.mae" in metrics and f"per_band.{band}.bias" in metrics


def test_categories_are_reported_only_when_the_dataset_supplies_them(tmp_path):
    with_cats = FakeDataset(samples=[make_sample(i, category=("Forest" if i < 2 else "Rural")) for i in range(4)])
    summary, out = run(tmp_path / "a", datasets=[with_cats])
    assert summary["datasets"]["fake"]["categories"] == {"Forest": 2, "Rural": 2}
    assert set(agg(out)["datasets"]["fake"]["by_category"]) == {"Forest", "Rural"}
    none_cat = FakeDataset()
    summary2, out2 = run(tmp_path / "b", datasets=[none_cat])
    assert summary2["datasets"]["fake"]["categories"] == {} and "by_category" not in agg(out2)["datasets"]["fake"]


# ============================================================================== evidence classes stay apart


def test_synthetic_and_real_results_are_never_combined(tmp_path):
    real = FakeDataset("neonlike", "sen2neon", "real_cross_sensor")
    synth = FakeDataset("synlike", "synthetic_smoke", "synthetic", samples=[make_sample(i, dataset="synlike", evidence="synthetic") for i in range(4)])
    summary, out = run(tmp_path, datasets=[real, synth])
    assert set(summary["sections"]) == {"real_cross_sensor", "synthetic"} and list(summary["sections"]["real_cross_sensor"]) == ["neonlike"] and list(summary["sections"]["synthetic"]) == ["synlike"]
    a = agg(out)
    assert set(a) == {"metric_configuration", "statistics", "datasets"} and set(a["datasets"]) == {"neonlike", "synlike"}          # only per-dataset blocks: nothing spans datasets
    assert not ({"overall", "pooled", "combined", "all", "total"} & set(summary)) and not ({"overall", "pooled", "combined", "all"} & set(a["datasets"]["neonlike"]))
    assert all(set(block) == {"headline", "paired_headline"} for datasets in summary["sections"].values() for block in datasets.values())


# ============================================================================== provenance


def test_the_summary_identifies_what_was_evaluated_and_with_what(standard):
    summary, _ = standard
    assert len(summary["config_digest"]) == 64 and "git" in summary["code"] and "torch" in summary["environment"]
    assert summary["metric_configuration"]["version"].startswith("frame-eval-metrics/") and summary["tiling"]["tile_size"] == 128
    sp = summary["systems"]["sharpen"]["provenance"]
    assert sp["hard_constraint"] is False and sp["input_bands"] == list(BANDS) and "value_convention" in sp and sp["tiling"]["overlap"] == 32 and sp["model_name"] == "unit-sharpen"
    d = summary["datasets"]["fake"]
    assert d["manifest_digest"] == "f" * 64 and d["sample_ids"] == ["fake:s0", "fake:s1", "fake:s2", "fake:s3"] and d["scene_units"] == ["scene0", "scene1", "scene2", "scene3"]
    assert d["role"] == "independent_benchmark" and d["split"] == "test"


def test_the_evaluation_matrix_has_a_row_per_system_and_dataset_with_the_required_columns(standard):
    summary, _ = standard
    matrix = summary["evaluation_matrix"]
    assert {(r["system"], r["dataset"]) for r in matrix} == {("bicubic", "fake"), ("sharpen", "fake")}
    required = {"system", "model", "dataset", "evidence_class", "scenes", "split", "bands", "lr_resolution_m", "hr_resolution_m", "hard_constraint", "tiling", "weights", "metric_configuration",
                "n_samples", "n_evaluated"}
    assert all(required <= set(r) for r in matrix)
    row = next(r for r in matrix if r["system"] == "sharpen")
    assert row["hard_constraint"] is False and row["lr_resolution_m"] == 10.0 and row["hr_resolution_m"] == 2.5 and row["bands"] == list(BANDS) and row["n_evaluated"] == 4


def test_two_runs_of_the_same_configuration_give_identical_metrics(tmp_path):
    _, a = run(tmp_path / "a")
    _, b = run(tmp_path / "b")
    strip = lambda rows: [{k: v for k, v in r.items() if k not in ("seconds",)} for r in rows]
    assert strip(read_jsonl(a / "metrics.jsonl")) == strip(read_jsonl(b / "metrics.jsonl")) and agg(a) == agg(b)


# ============================================================================== multi-seed groups


def test_systems_sharing_a_group_are_also_summarised_across_seeds(tmp_path):
    systems = [BicubicSystem("bicubic")] + [CallableSystem(f"t{k}", Sharpen(), TILING, hard_constraint=False, provenance={"group": "tiny"}, device="cpu") for k in range(3)]
    cfg = EvalConfig.from_dict({**config(tmp_path, systems=["bicubic"]).to_dict(), "systems": [{"name": "bicubic", "kind": "bicubic"}] +
                                [{"name": f"t{k}", "kind": "checkpoint", "checkpoint": "x.pt", "group": "tiny"} for k in range(3)]})
    summary = run_evaluation(cfg, systems=systems, datasets=[FakeDataset()])
    g = summary["seed_groups"]["tiny"]
    assert g["systems"] == ["t0", "t1", "t2"] and g["n_systems"] == 3 and g["datasets"]["fake"]["reference_accuracy.rmse"]["std"] == pytest.approx(0.0, abs=1e-12)


# ============================================================================== safety, hygiene, wording


def test_a_run_refuses_to_overwrite_an_existing_output_directory(tmp_path):
    run(tmp_path)
    with pytest.raises(EvaluationError, match="already contains results"):
        run(tmp_path)


def test_a_role_violation_refuses_the_run_before_anything_is_written(tmp_path):
    class Refusing(FakeDataset):
        def validate(self, *, check_files=True):
            raise RoleSafetyError("benchmark labelled train", codes=("role_violation",))

    with pytest.raises(RoleSafetyError):
        run(tmp_path, datasets=[Refusing()])
    assert not (tmp_path / "out").exists()


def test_a_system_that_saw_the_evaluation_scenes_refuses_the_run(tmp_path):
    system = sharpen_system("seen")
    system.training_info = {"training_datasets": ["synthetic_smoke"], "train_scenes": ["scene1"], "val_scenes": []}
    with pytest.raises(RoleSafetyError) as excinfo:
        run(tmp_path, systems=[BicubicSystem("bicubic"), system])
    assert "train_eval_overlap" in excinfo.value.codes and not (tmp_path / "out").exists()


def test_the_readme_reports_without_ranking(standard):
    _, out = standard
    text = (out / "README.md").read_text()
    lowered = text.lower()
    for word in ("winner", "best model", "outperform", "superior", "state of the art", "ranking:", "rank 1"):
        assert word not in lowered.replace("no ranking", "").replace("not a ranking", "").replace("no winner", ""), word
    assert "no ranking" in lowered and "descriptive only" in lowered and "self-consistency" in lowered and "not accuracy against" in lowered
    assert "/home/" not in text


def test_the_readme_separates_self_consistency_from_reference_accuracy(standard):
    _, out = standard
    text = (out / "README.md").read_text()
    assert text.index("Reference accuracy") < text.index("Self-consistency") and "NOT accuracy against an HR reference" in text


def test_an_output_directory_holding_only_the_input_config_is_fresh_but_one_with_results_is_not(tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    (out / "config.json").write_text("{}")                         # the config may live next to where results go
    cfg = config(tmp_path)
    summary = run_evaluation(cfg, systems=[BicubicSystem("bicubic"), sharpen_system()], datasets=[FakeDataset()])
    assert summary["status"] == "completed"
    with pytest.raises(EvaluationError, match="already contains results"):
        run_evaluation(cfg, systems=[BicubicSystem("bicubic"), sharpen_system()], datasets=[FakeDataset()])


def test_the_readme_has_a_per_band_table_for_the_four_rgbn_bands(standard):
    _, out = standard
    text = (out / "README.md").read_text()
    assert "Per-band reflectance error" in text
    header = next(line for line in text.splitlines() if "B02" in line and "B08" in line and line.startswith("| system"))
    for band in BANDS:
        assert band in header


def test_the_readme_lists_dataset_defined_categories_as_descriptive_only(tmp_path):
    ds = FakeDataset(samples=[make_sample(i, category=("Forest" if i < 2 else "Rural")) for i in range(4)])
    _, out = run(tmp_path, datasets=[ds])
    text = (out / "README.md").read_text()
    assert "By dataset-defined category" in text and "Forest" in text and "Rural" in text and "descriptive only" in text.split("By dataset-defined category")[1].split("###")[0].lower()


def test_the_readme_explains_that_a_registration_shift_makes_pixelwise_scores_pessimistic_and_points_to_the_shift_experiment(standard):
    text = (standard[1] / "README.md").read_text()
    assert "Registration" in text and "shift" in text and "python -m frame.evaluate shift" in text


def test_the_readme_states_the_preprocessing_and_lists_the_evaluation_matrix(standard):
    _, out = standard
    text = (out / "README.md").read_text()
    assert "Preprocessing (all systems)" in text and "B04, B03, B02, B08" in text and "reflectance_scale" in text and "no BOA offset" in text
    assert "## Evaluation matrix" in text
    matrix = text.split("## Evaluation matrix")[1].split("\n## ")[0]
    assert "| system |" in matrix and "fake" in matrix and "4/4" in matrix and "10 m" in matrix and "2.5 m" in matrix
