"""The README of a downstream run: what was measured, what was found, what was not shown; no ranking, no unsupported claims."""

from __future__ import annotations

import pytest

from frame.tests.test_downstream_runner import excluded_dataset, run


@pytest.fixture(scope="module")
def readme(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("rep")
    summary, out = run(tmp, datasets=[excluded_dataset()])
    return (out / "README.md").read_text(), summary


def test_the_readme_separates_what_was_measured_from_what_was_found_and_what_was_not_shown(readme):
    text, _ = readme
    for heading in ("## What was measured", "## What was found", "## What was not shown", "## Limitations"):
        assert heading in text
    assert text.index("## What was measured") < text.index("## What was found") < text.index("## What was not shown") < text.index("## Limitations")


def test_what_was_measured_states_the_region_the_metric_the_threshold_and_the_gate(readme):
    text, _ = readme
    assert "10 m" in text and "one Sentinel-2 pixel" in text and "NDVI = (NIR - Red) / (NIR + Red)" in text and "B04" in text and "B08" in text
    assert "0.3" in text and "not selected or tuned on results" in text or "never selected or tuned" in text
    assert "registration" in text.lower() and "frame-reliability-gate/1" in text
    for system in ("lr_native", "bicubic", "`a`", "`b`"):
        assert system in text


def test_the_evidence_table_reports_candidates_eligible_excluded_and_the_reasons(readme):
    text, summary = readme
    assert "candidate regions" in text.lower() and "reference_alignment_invalid" in text and "insufficient_valid_pixels" in text and "eligible scene units" in text.lower()
    assert "1024" in text                                                                          # the excluded tiles' candidate regions are counted, not hidden


def test_the_change_relative_to_the_bicubic_baseline_is_a_paired_difference_with_its_label(readme):
    text, _ = readme
    assert "Change relative to bicubic" in text and "a - bicubic" in text and "descriptive" in text.lower()


def test_the_stability_association_shows_the_texture_only_relationship_and_the_partial(readme):
    text, _ = readme
    assert "texture" in text and "partial" in text.lower() and "stability" in text and "descriptive only" in text


def test_the_risk_coverage_uses_the_declared_retention_points(readme):
    text, _ = readme
    assert "Risk-coverage" in text and "80%" in text and "60%" in text and "40%" in text and "not calibrated selective prediction" in text.lower()


def test_the_unsupported_claims_are_stated_as_not_shown(readme):
    text, _ = readme
    section = text[text.index("## What was not shown"):text.index("## Limitations")]
    for phrase in ("calibrated", "causal", "Indian", "land cover", "india_downstream_validation_unavailable", "secondary_landcover_task_deferred_no_supported_reference"):
        assert phrase.lower() in section.lower(), phrase


def test_the_limitations_carry_forward_the_phase_6_ones(readme):
    text, _ = readme
    section = text[text.index("## Limitations"):]
    for phrase in ("scene units", "registration", "North America", "Indian", "texture"):
        assert phrase.lower() in section.lower(), phrase


def test_nothing_in_the_readme_ranks_the_systems(readme):
    text, _ = readme
    lowered = text.lower()
    for word in ("winner", "best system", "outperform", "superior", "is better than"):
        assert word not in lowered, word
    assert "no ranking" in lowered


def test_the_risk_coverage_compares_the_stability_order_with_the_texture_order_paired_over_units(readme):
    text, _ = readme
    section = text[text.index("**Risk-coverage"):text.index("## What was not shown")]
    assert "stability − texture" in section and "paired over scene units" in section
