"""frame.evaluate.config -- strict, serialisable evaluation configuration."""

from __future__ import annotations

import copy
import json

import pytest

from frame.evaluate.config import EvalConfig
from frame.evaluate.errors import EvalConfigError


def base():
    return {
        "name": "unit",
        "output_dir": "out",
        "systems": [{"name": "bicubic", "kind": "bicubic"}, {"name": "lite", "kind": "lite"}],
        "datasets": [{"name": "neon", "kind": "sen2neon", "manifest": "m.jsonl"}],
    }


def cfg(**over) -> EvalConfig:
    d = base()
    for k, v in over.items():
        d[k] = v
    return EvalConfig.from_dict(d)


def test_a_minimal_config_gets_documented_defaults():
    c = cfg()
    assert c.tiling.tile_size == 128 and c.tiling.overlap == 32 and c.statistics.reference_system == "bicubic" and c.statistics.n_boot == 10_000
    assert c.min_valid_fraction == 0.05 and c.device == "auto" and c.bands == ("B04", "B03", "B02", "B08") and c.metrics["hallucination_taus"][0] == 0.005
    assert [s.name for s in c.systems] == ["bicubic", "lite"] and c.datasets[0].evidence_class == "real_cross_sensor" and c.datasets[0].split == "test"


def test_the_config_round_trips_through_json_exactly():
    c = cfg(seed=3, tiling={"tile_size": 128, "overlap": 16})
    assert EvalConfig.from_json(c.to_json()) == c and EvalConfig.from_dict(json.loads(c.to_json())) == c
    assert c.to_json() == cfg(seed=3, tiling={"tile_size": 128, "overlap": 16}).to_json() and c.digest() == cfg(seed=3, tiling={"tile_size": 128, "overlap": 16}).digest()


def test_unknown_keys_are_refused():
    d = base()
    d["stat"] = {}
    with pytest.raises(EvalConfigError, match="stat"):
        EvalConfig.from_dict(d)
    d = base()
    d["datasets"][0]["subsett"] = "x"
    with pytest.raises(EvalConfigError, match="subsett"):
        EvalConfig.from_dict(d)


def test_from_dict_does_not_mutate_its_input():
    d = base()
    before = copy.deepcopy(d)
    EvalConfig.from_dict(d)
    assert d == before


def test_files_are_loaded_with_clear_errors(tmp_path):
    (tmp_path / "c.json").write_text(json.dumps(base()))
    assert EvalConfig.load(tmp_path / "c.json") == cfg()
    with pytest.raises(EvalConfigError, match="not found"):
        EvalConfig.load(tmp_path / "absent.json")
    (tmp_path / "bad.json").write_text("{nope")
    with pytest.raises(EvalConfigError, match="not valid JSON"):
        EvalConfig.load(tmp_path / "bad.json")


@pytest.mark.parametrize(
    "mutate, field",
    [
        (lambda d: d.update(name=""), "name"),
        (lambda d: d.update(name="has space"), "name"),
        (lambda d: d.update(output_dir=""), "output_dir"),
        (lambda d: d.update(systems=[]), "systems"),
        (lambda d: d.update(datasets=[]), "datasets"),
        (lambda d: d["systems"].append({"name": "bicubic", "kind": "bicubic"}), "systems"),            # duplicate name
        (lambda d: d["systems"].__setitem__(0, {"name": "x", "kind": "gan"}), "systems[0].kind"),
        (lambda d: d["systems"].__setitem__(1, {"name": "ck", "kind": "checkpoint"}), "systems[1].checkpoint"),
        (lambda d: d["systems"][1].update(checkpoint="a.pt"), "systems[1].checkpoint"),                # a checkpoint on a non-checkpoint kind
        (lambda d: d["datasets"].append({"name": "neon", "kind": "sen2neon", "manifest": "m.jsonl"}), "datasets"),
        (lambda d: d["datasets"][0].update(kind="sen2naipv2"), "datasets[0].kind"),                    # in-distribution for SEN2SR: not an independent benchmark
        (lambda d: d["datasets"][0].pop("manifest"), "datasets[0].manifest"),
        (lambda d: d["datasets"][0].update(split="train"), "datasets[0].split"),
        (lambda d: d["datasets"][0].update(split="val"), "datasets[0].split"),
        (lambda d: d["datasets"].append({"name": "o", "kind": "opensr_test", "subset": "naip"}), "datasets[1].subset"),
        (lambda d: d["datasets"].append({"name": "o", "kind": "opensr_test"}), "datasets[1].subset"),
        (lambda d: d["datasets"].append({"name": "o", "kind": "opensr_test", "subset": "spot", "hr_variant": "HRx"}), "datasets[1].hr_variant"),
        (lambda d: d["datasets"].append({"name": "s", "kind": "synthetic_smoke", "manifest": "m.jsonl", "split": "train"}), "datasets[1].split"),
        (lambda d: d.update(tiling={"tile_size": 128, "overlap": 100}), "tiling"),
        (lambda d: d.update(min_valid_fraction=0.0), "min_valid_fraction"),
        (lambda d: d.update(min_valid_fraction=1.5), "min_valid_fraction"),
        (lambda d: d.update(statistics={"n_boot": 10}), "statistics.n_boot"),
        (lambda d: d.update(statistics={"alpha": 0.5}), "statistics.alpha"),
        (lambda d: d.update(statistics={"reference_system": "ghost"}), "statistics.reference_system"),
        (lambda d: d.update(device="tpu"), "device"),
        (lambda d: d.update(bands=["B04", "B04"]), "bands"),
        (lambda d: d.update(bands=["B02", "B03", "B04", "B08"]), "bands"),                              # the systems consume FRAME's RGBN order
        (lambda d: d.update(metrics={"hallucination_taus": []}), "metrics.hallucination_taus"),
        (lambda d: d.update(metrics={"hf_sigma": 0}), "metrics.hf_sigma"),
        (lambda d: d.update(metrics={"bogus": 1}), "metrics"),
        (lambda d: d.update(seed=-1), "seed"),
    ],
)
def test_invalid_configurations_name_the_offending_field(mutate, field):
    d = base()
    mutate(d)
    with pytest.raises(EvalConfigError) as excinfo:
        EvalConfig.from_dict(d)
    assert excinfo.value.field == field, (field, str(excinfo.value))


def test_a_checkpoint_system_needs_its_checkpoint_and_may_name_its_training_summary():
    c = cfg(systems=[{"name": "bicubic", "kind": "bicubic"}, {"name": "t0", "kind": "checkpoint", "checkpoint": "a.pt", "train_summary": "s.json", "group": "tiny"}])
    assert c.systems[1].checkpoint == "a.pt" and c.systems[1].train_summary == "s.json" and c.systems[1].group == "tiny"


def test_lite_may_be_evaluated_with_and_without_its_hard_constraint_as_separate_named_systems():
    c = cfg(systems=[{"name": "bicubic", "kind": "bicubic"}, {"name": "lite", "kind": "lite"}, {"name": "lite_nc", "kind": "lite", "hard_constraint": False}])
    assert [s.hard_constraint for s in c.systems] == [None, True, False]


def test_hard_constraint_false_is_only_supported_where_it_can_be_run():
    for kind in ("mamba", "bicubic"):
        with pytest.raises(EvalConfigError, match="hard_constraint"):
            cfg(systems=[{"name": "bicubic", "kind": "bicubic"}, {"name": "x", "kind": kind, "hard_constraint": False}])


def test_synthetic_and_real_datasets_carry_their_evidence_class():
    c = cfg(datasets=[{"name": "neon", "kind": "sen2neon", "manifest": "m.jsonl"}, {"name": "syn", "kind": "synthetic_smoke", "manifest": "s.jsonl"},
                      {"name": "spot", "kind": "opensr_test", "subset": "spot"}])
    assert [d.evidence_class for d in c.datasets] == ["real_cross_sensor", "synthetic", "real_cross_sensor"] and c.datasets[2].hr_variant == "HRharm"


def test_the_metric_configuration_is_normalised_and_versioned():
    c = cfg(metrics={"hf_sigma": 3.0})
    assert c.metrics["hf_sigma"] == 3.0 and c.metrics["version"].startswith("frame-eval-metrics/") and c.metrics["index_min_sum"] == 0.02


def test_extra_paired_comparisons_name_two_configured_systems():
    c = cfg(statistics={"pairs": [["lite", "bicubic"]]})
    assert c.statistics.pairs == (("lite", "bicubic"),) and EvalConfig.from_json(c.to_json()) == c
    for bad in ([["lite", "ghost"]], [["lite"]], [["lite", "lite"]]):
        with pytest.raises(EvalConfigError) as excinfo:
            cfg(statistics={"pairs": bad})
        assert excinfo.value.field == "statistics.pairs"
