"""frame.reliability.config -- the strict, serialisable configuration of the stability-vs-error experiment (Phase 6)."""

from __future__ import annotations

import copy
import json

import pytest

from frame.reliability.config import ReliabilityConfig
from frame.reliability.errors import ReliabilityConfigError


def base():
    return {
        "name": "rel-unit",
        "output_dir": "out",
        "systems": [{"name": "lite", "kind": "lite"}],
        "datasets": [{"name": "neon", "kind": "sen2neon", "manifest": "m.jsonl"}],
    }


def cfg(**over):
    d = base()
    for dotted, value in over.items():
        target = d
        *parents, leaf = dotted.split("__")
        for p in parents:
            target = target.setdefault(p, {})
        target[leaf] = value
    return ReliabilityConfig.from_dict(d)


# ----------------------------------------------------------------------------- defaults are the pre-declared ones


def test_the_defaults_are_the_predeclared_settings():
    c = cfg()
    assert c.tta.transforms == ("identity", "hflip", "vflip", "rot90", "rot180", "rot270") and c.tta.seed == 42
    a = c.alignment
    assert (a.tolerance_hr_px, a.quadrant_tolerance_hr_px, a.max_correction_hr_px, a.apply_translation_correction) == (0.5, 1.0, 4, True)
    assert a.method == "bicubic_baseline_cross_correlation" and a.uncertain_factor == 2.0
    assert c.eligibility.min_valid_fraction == 0.5 and c.eligibility.min_members == 2
    an = c.analysis
    assert an.cell_sizes_hr_px == (4, 16) and an.high_error_quantile == 0.9 and an.min_units_for_split == 12 and an.dev_fraction == 0.5
    assert an.coverage_grid[0] == 1.0 and list(an.coverage_grid) == sorted(an.coverage_grid, reverse=True)
    assert c.metrics["version"].startswith("frame-eval-metrics")


def test_the_config_round_trips_through_json_and_has_a_stable_digest():
    c = cfg(alignment__tolerance_hr_px=0.4)
    assert ReliabilityConfig.from_json(c.to_json()) == c
    assert c.digest() == cfg(alignment__tolerance_hr_px=0.4).digest() != cfg().digest()
    assert list(json.loads(c.to_json())) == sorted(json.loads(c.to_json()))


def test_from_dict_does_not_mutate_its_input():
    d = base()
    before = copy.deepcopy(d)
    ReliabilityConfig.from_dict(d)
    assert d == before


def test_unknown_and_missing_keys_are_refused_by_name():
    d = base()
    d["alignment"] = {"tolerence_hr_px": 0.5}
    with pytest.raises(ReliabilityConfigError, match="tolerence_hr_px"):
        ReliabilityConfig.from_dict(d)
    d = base()
    del d["datasets"]
    with pytest.raises(ReliabilityConfigError, match="datasets"):
        ReliabilityConfig.from_dict(d)


def test_unparseable_and_absent_files_are_config_errors(tmp_path):
    (tmp_path / "bad.json").write_text("{nope")
    with pytest.raises(ReliabilityConfigError, match="not valid JSON"):
        ReliabilityConfig.load(tmp_path / "bad.json")
    with pytest.raises(ReliabilityConfigError, match="not found"):
        ReliabilityConfig.load(tmp_path / "absent.json")
    (tmp_path / "ok.json").write_text(json.dumps(base()))
    assert ReliabilityConfig.load(tmp_path / "ok.json") == cfg()


# ----------------------------------------------------------------------------- what may be analysed


def test_a_system_without_a_learned_model_cannot_have_stability():
    with pytest.raises(ReliabilityConfigError, match="systems"):
        ReliabilityConfig.from_dict({**base(), "systems": [{"name": "b", "kind": "bicubic"}]})


def test_the_dataset_safety_rules_of_the_evaluation_apply_unchanged():
    with pytest.raises(ReliabilityConfigError, match="sen2naipv2"):
        ReliabilityConfig.from_dict({**base(), "datasets": [{"name": "n", "kind": "sen2naipv2", "manifest": "m"}]})
    with pytest.raises(ReliabilityConfigError, match="test"):
        ReliabilityConfig.from_dict({**base(), "datasets": [{"name": "n", "kind": "sen2neon", "manifest": "m", "split": "train"}]})


def test_names_must_be_unique():
    with pytest.raises(ReliabilityConfigError, match="systems"):
        ReliabilityConfig.from_dict({**base(), "systems": [{"name": "a", "kind": "lite"}, {"name": "a", "kind": "lite"}]})
    with pytest.raises(ReliabilityConfigError, match="datasets"):
        ReliabilityConfig.from_dict({**base(), "datasets": [{"name": "a", "kind": "sen2neon", "manifest": "m"}, {"name": "a", "kind": "sen2neon", "manifest": "m"}]})


# ----------------------------------------------------------------------------- TTA


def test_the_tta_transform_set_is_validated():
    assert cfg(tta__transforms=["identity", "hflip"]).tta.transforms == ("identity", "hflip")
    for bad, match in (([], "at least"), (["identity"], "at least"), (["hflip", "vflip"], "identity"), (["identity", "identity", "hflip"], "duplicate"), (["identity", "shear"], "shear")):
        with pytest.raises(ReliabilityConfigError, match=match) as info:
            cfg(tta__transforms=bad)
        assert info.value.field == "tta.transforms"


# ----------------------------------------------------------------------------- invalid numbers name their field


@pytest.mark.parametrize(
    "overrides, field",
    [
        ({"alignment__tolerance_hr_px": 0}, "alignment.tolerance_hr_px"),
        ({"alignment__tolerance_hr_px": -1}, "alignment.tolerance_hr_px"),
        ({"alignment__tolerance_hr_px": float("nan")}, "alignment.tolerance_hr_px"),
        ({"alignment__quadrant_tolerance_hr_px": 0.25}, "alignment.quadrant_tolerance_hr_px"),
        ({"alignment__max_correction_hr_px": -1}, "alignment.max_correction_hr_px"),
        ({"alignment__max_correction_hr_px": 2.5}, "alignment.max_correction_hr_px"),
        ({"alignment__uncertain_factor": 1.0}, "alignment.uncertain_factor"),
        ({"alignment__method": "sift"}, "alignment.method"),
        ({"alignment__apply_translation_correction": "yes"}, "alignment.apply_translation_correction"),
        ({"eligibility__min_valid_fraction": 0}, "eligibility.min_valid_fraction"),
        ({"eligibility__min_valid_fraction": 1.5}, "eligibility.min_valid_fraction"),
        ({"eligibility__min_members": 1}, "eligibility.min_members"),
        ({"eligibility__max_nonfinite_fraction": -0.1}, "eligibility.max_nonfinite_fraction"),
        ({"analysis__cell_sizes_hr_px": []}, "analysis.cell_sizes_hr_px"),
        ({"analysis__cell_sizes_hr_px": [4, 4]}, "analysis.cell_sizes_hr_px"),
        ({"analysis__cell_sizes_hr_px": [0]}, "analysis.cell_sizes_hr_px"),
        ({"analysis__min_cell_valid_fraction": 0}, "analysis.min_cell_valid_fraction"),
        ({"analysis__coverage_grid": [0.9, 0.5]}, "analysis.coverage_grid"),
        ({"analysis__coverage_grid": [1.0, 1.2]}, "analysis.coverage_grid"),
        ({"analysis__high_error_quantile": 0.4}, "analysis.high_error_quantile"),
        ({"analysis__high_error_quantile": 1.0}, "analysis.high_error_quantile"),
        ({"analysis__dev_fraction": 0}, "analysis.dev_fraction"),
        ({"analysis__dev_fraction": 1}, "analysis.dev_fraction"),
        ({"analysis__min_units_for_split": 3}, "analysis.min_units_for_split"),
        ({"analysis__pooled_cells_per_tile": 10}, "analysis.pooled_cells_per_tile"),
        ({"analysis__calibration_bins": 2}, "analysis.calibration_bins"),
        ({"analysis__displacement_sweep_hr_px": [1, 2]}, "analysis.displacement_sweep_hr_px"),
        ({"bootstrap__n_boot": 10}, "bootstrap.n_boot"),
        ({"bootstrap__alpha": 0.5}, "bootstrap.alpha"),
        ({"device": "tpu"}, "device"),
        ({"name": "has space"}, "name"),
        ({"output_dir": ""}, "output_dir"),
    ],
)
def test_invalid_values_name_the_offending_field(overrides, field):
    with pytest.raises(ReliabilityConfigError) as info:
        cfg(**overrides)
    assert info.value.field == field, (overrides, str(info.value))


def test_the_displacement_sweep_starts_at_zero():
    assert cfg().analysis.displacement_sweep_hr_px[0] == 0
    assert cfg(analysis__displacement_sweep_hr_px=[0, 2]).analysis.displacement_sweep_hr_px == (0, 2)
