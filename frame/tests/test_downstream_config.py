"""frame.downstream.config -- everything that shapes the downstream analysis is declared before any result and recorded (Phase 7)."""

from __future__ import annotations

import copy
import json

import pytest

from frame.downstream.config import DownstreamConfig
from frame.downstream.errors import DownstreamConfigError


def base():
    return {"name": "ds-unit", "output_dir": "out", "systems": [{"name": "lite", "kind": "lite"}], "datasets": [{"name": "neon", "kind": "sen2neon", "manifest": "m.jsonl"}]}


def cfg(**over):
    d = base()
    for dotted, value in over.items():
        target = d
        *parents, leaf = dotted.split("__")
        for p in parents:
            target = target.setdefault(p, {})
        target[leaf] = value
    return DownstreamConfig.from_dict(d)


# ----------------------------------------------------------------------------- the predeclared analysis


def test_the_defaults_are_the_predeclared_analysis():
    c = cfg()
    assert (c.ndvi.red_band, c.ndvi.nir_band, c.ndvi.min_reflectance_sum, c.ndvi.denominator_epsilon) == ("B04", "B08", 0.02, 1e-6)
    d = c.decision
    assert d.ndvi_threshold == 0.3 and d.sensitivity_thresholds == (0.2, 0.4) and d.region_majority_fraction == 0.5 and d.rule == "vegetation iff NDVI >= threshold"
    assert d.selection_policy == "predeclared_conventional_value_not_selected_or_tuned_on_results"
    r = c.regions
    assert r.scales_hr_px == (4, 16) and r.primary_scale_hr_px == 4 and r.min_valid_fraction == 0.75
    a = c.analysis
    assert a.retention_grid == (1.0, 0.8, 0.6, 0.4) and a.pooled_regions_per_tile == 8192
    assert c.tta.transforms == ("identity", "hflip", "vflip", "rot90", "rot180", "rot270")
    assert c.alignment.tolerance_hr_px == 0.5 and c.eligibility.min_valid_fraction == 0.5          # the Phase 6 gate, unchanged
    assert c.metrics["version"].startswith("frame-eval-metrics")


def test_the_gate_settings_default_to_exactly_the_phase_6_ones():
    from frame.reliability.config import AlignmentSpec, EligibilitySpec

    c = cfg()
    assert c.alignment == AlignmentSpec() and c.eligibility == EligibilitySpec()


def test_the_config_round_trips_through_json_and_has_a_stable_digest():
    c = cfg(decision__ndvi_threshold=0.35)
    assert DownstreamConfig.from_json(c.to_json()) == c
    assert c.digest() == cfg(decision__ndvi_threshold=0.35).digest() != cfg().digest()
    assert list(json.loads(c.to_json())) == sorted(json.loads(c.to_json()))


def test_from_dict_does_not_mutate_its_input_and_refuses_unknown_and_missing_keys():
    d = base()
    before = copy.deepcopy(d)
    DownstreamConfig.from_dict(d)
    assert d == before
    d["decision"] = {"ndvi_treshold": 0.3}
    with pytest.raises(DownstreamConfigError, match="ndvi_treshold"):
        DownstreamConfig.from_dict(d)
    d2 = base()
    del d2["systems"]
    with pytest.raises(DownstreamConfigError, match="systems"):
        DownstreamConfig.from_dict(d2)


def test_unparseable_and_absent_files_are_config_errors(tmp_path):
    (tmp_path / "bad.json").write_text("{nope")
    with pytest.raises(DownstreamConfigError, match="not valid JSON"):
        DownstreamConfig.load(tmp_path / "bad.json")
    with pytest.raises(DownstreamConfigError, match="not found"):
        DownstreamConfig.load(tmp_path / "absent.json")
    (tmp_path / "ok.json").write_text(json.dumps(base()))
    assert DownstreamConfig.load(tmp_path / "ok.json") == cfg()


# ----------------------------------------------------------------------------- what may be analysed


def test_a_system_without_a_stability_signal_cannot_be_analysed():
    with pytest.raises(DownstreamConfigError, match="systems"):
        DownstreamConfig.from_dict({**base(), "systems": [{"name": "b", "kind": "bicubic"}]})


def test_the_dataset_safety_rules_apply_unchanged():
    with pytest.raises(DownstreamConfigError, match="sen2naipv2"):
        DownstreamConfig.from_dict({**base(), "datasets": [{"name": "n", "kind": "sen2naipv2", "manifest": "m"}]})
    with pytest.raises(DownstreamConfigError, match="test"):
        DownstreamConfig.from_dict({**base(), "datasets": [{"name": "n", "kind": "sen2neon", "manifest": "m", "split": "train"}]})


def test_the_bands_are_the_ones_frame_supports_and_must_differ():
    assert cfg(ndvi__red_band="B04", ndvi__nir_band="B08").ndvi.nir_band == "B08"
    for over in ({"ndvi__red_band": "B08"}, {"ndvi__nir_band": "B02"}, {"ndvi__red_band": "B11"}):
        with pytest.raises(DownstreamConfigError, match="ndvi"):
            cfg(**over)


@pytest.mark.parametrize(
    "overrides, field",
    [
        ({"ndvi__min_reflectance_sum": 0}, "ndvi.min_reflectance_sum"),
        ({"ndvi__denominator_epsilon": -1}, "ndvi.denominator_epsilon"),
        ({"decision__ndvi_threshold": 1.0}, "decision.ndvi_threshold"),
        ({"decision__ndvi_threshold": float("nan")}, "decision.ndvi_threshold"),
        ({"decision__sensitivity_thresholds": [0.3]}, "decision.sensitivity_thresholds"),
        ({"decision__sensitivity_thresholds": [0.2, 0.2]}, "decision.sensitivity_thresholds"),
        ({"decision__sensitivity_thresholds": [0.1, 0.2, 0.4, 0.5, 0.6]}, "decision.sensitivity_thresholds"),
        ({"decision__region_majority_fraction": 0}, "decision.region_majority_fraction"),
        ({"decision__region_majority_fraction": 1.5}, "decision.region_majority_fraction"),
        ({"decision__rule": "ndvi > threshold"}, "decision.rule"),
        ({"decision__selection_policy": "tuned"}, "decision.selection_policy"),
        ({"regions__scales_hr_px": []}, "regions.scales_hr_px"),
        ({"regions__scales_hr_px": [4, 4]}, "regions.scales_hr_px"),
        ({"regions__scales_hr_px": [0]}, "regions.scales_hr_px"),
        ({"regions__primary_scale_hr_px": 8}, "regions.primary_scale_hr_px"),
        ({"regions__min_valid_fraction": 0}, "regions.min_valid_fraction"),
        ({"analysis__retention_grid": [0.8, 0.6]}, "analysis.retention_grid"),
        ({"analysis__retention_grid": [1.0, 1.2]}, "analysis.retention_grid"),
        ({"analysis__retention_grid": [1.0]}, "analysis.retention_grid"),
        ({"analysis__pooled_regions_per_tile": 10}, "analysis.pooled_regions_per_tile"),
        ({"bootstrap__n_boot": 10}, "bootstrap.n_boot"),
        ({"eligibility__min_valid_fraction": 0}, "eligibility.min_valid_fraction"),
        ({"alignment__tolerance_hr_px": -1}, "alignment.tolerance_hr_px"),
        ({"tta__transforms": ["identity"]}, "tta.transforms"),
        ({"device": "tpu"}, "device"),
        ({"name": "has space"}, "name"),
        ({"output_dir": ""}, "output_dir"),
    ],
)
def test_invalid_values_name_the_offending_field(overrides, field):
    with pytest.raises(DownstreamConfigError) as info:
        cfg(**overrides)
    assert info.value.field == field, (overrides, str(info.value))


def test_the_primary_scale_must_be_one_of_the_declared_scales():
    assert cfg(regions__scales_hr_px=[8, 32], regions__primary_scale_hr_px=8).regions.primary_scale_hr_px == 8


def test_a_system_cannot_take_a_name_reserved_for_the_reference_or_a_baseline():
    for name in ("ref", "bicubic", "lr_native", "texture"):
        with pytest.raises(DownstreamConfigError) as info:
            DownstreamConfig.from_dict({**base(), "systems": [{"name": name, "kind": "lite"}]})
        assert info.value.field == "systems[0].name"
